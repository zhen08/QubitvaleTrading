"""Paper 账本 v2 —— 事件溯源（event-sourced）设计。

Review 修复（2026-07-13）：
  R2  cash/positions 不再依赖 state.json 快照——每次加载都从 trades.parquet
      （不可变追加日志）重放重建；任意时点崩溃后重载即一致。state.json 只存
      元数据（initial_capital/start_date/last_settled）+ 人类可读快照。
  R2  "当日调仓已完成"由 runs.parquet 注册表判定（record_run 在全部成交后写入），
      不再用"当天存在任一成交"推断；崩溃在半途 → 无 completed 记录 → 重跑会
      基于重放后的真实持仓补齐残余差额（阈值内自然 no-op）。
  R3  execute() 内置现金护栏：买入名义超过可用现金/(1+费率) 自动缩量，现金
      永不为负（现货无借款）。
  R4  所有落盘走原子写（tmp + os.replace）。

文件（data/store/paper/）：
  trades.parquet  事件日志（唯一事实来源）[ts, day, symbol, side, qty, price, notional, fee, mode, reason]
  runs.parquet    调仓完成注册表 [day, mode, n_trades, completed_at, note]
  equity.parquet  每日权益（settled 覆盖 intraday）
  state.json      元数据 + 派生快照（仅供人读，加载时以重放结果为准）
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger("qvt.ledger")

TRADE_COLS = ["ts", "day", "symbol", "side", "qty", "price", "notional", "fee", "mode", "reason"]


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _atomic_json(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class Ledger:
    dir: Path
    initial_capital: float
    start_date: str
    last_settled: str | None = None
    cash: float = 0.0
    positions: dict[str, float] = field(default_factory=dict)

    # ---------------- 加载 = 元数据 + 事件重放 ----------------

    @classmethod
    def load_or_init(cls, store: Path, initial_capital: float, start_date: str,
                     book: str | None = None) -> "Ledger":
        """book：多账本命名空间（data/store/paper/<book>/）。None = 旧版根目录
        （仅测试/兼容用；生产引擎总是传策略名）。"""
        d = (store / "paper" / book) if book else (store / "paper")
        d.mkdir(parents=True, exist_ok=True)
        p = d / "state.json"
        if p.exists():
            meta = json.loads(p.read_text(encoding="utf-8"))
            led = cls(dir=d, initial_capital=float(meta["initial_capital"]),
                      start_date=meta["start_date"], last_settled=meta.get("last_settled"))
        else:
            led = cls(dir=d, initial_capital=float(initial_capital), start_date=start_date)
        led._replay()
        if p.exists():
            snap_cash = meta.get("cash")
            if snap_cash is not None and abs(float(snap_cash) - led.cash) > 1e-6:
                log.warning("state snapshot stale (snapshot cash=%.4f, replayed=%.4f) — "
                            "rebuilt from trade log", float(snap_cash), led.cash)
        led.save()
        return led

    def _replay(self) -> None:
        """从 trades 事件日志重建 cash/positions（唯一事实来源）。"""
        self.cash = self.initial_capital
        self.positions = {}
        t = self._read("trades.parquet")
        if t is None or not len(t):
            return
        for r in t.itertuples():
            qty = float(r.qty)
            if r.side == "buy":
                self.cash -= float(r.notional) + float(r.fee)
                self.positions[r.symbol] = self.positions.get(r.symbol, 0.0) + qty
            else:
                self.cash += float(r.notional) - float(r.fee)
                self.positions[r.symbol] = self.positions.get(r.symbol, 0.0) - qty
        self.positions = {k: v for k, v in self.positions.items() if abs(v) > 1e-12}

    def save(self) -> None:
        _atomic_json(
            {"initial_capital": self.initial_capital, "start_date": self.start_date,
             "last_settled": self.last_settled,
             "cash": round(self.cash, 8), "positions": self.positions,  # 派生快照，仅供人读
             "updated_at": str(pd.Timestamp.now(tz="UTC"))},
            self.dir / "state.json")

    def _read(self, name: str) -> pd.DataFrame | None:
        p = self.dir / name
        return pd.read_parquet(p) if p.exists() else None

    # ---------------- 成交（含现金护栏） ----------------

    def execute(self, *, ts: pd.Timestamp, day: str, symbol: str, target_qty: float,
                price: float, fee_bps: float, mode: str, reason: str,
                min_trade_usdt: float = 0.0) -> dict | None:
        cur = self.positions.get(symbol, 0.0)
        dq = target_qty - cur
        if dq < 0:
            dq = max(dq, -cur)                             # 现货长平：不可卖空
        fee_rate = fee_bps / 1e4
        if dq > 0:                                          # R3: 买入按可用现金封顶
            max_notional = max(0.0, self.cash) / (1.0 + fee_rate)
            want = dq * price
            if want > max_notional + 1e-9:
                dq = max_notional / price
                reason = f"{reason} [cash-capped]"
        notional = abs(dq) * price
        if notional < max(min_trade_usdt, 1e-9):
            return None
        fee = notional * fee_rate
        if dq > 0:
            self.cash -= notional + fee
        else:
            self.cash += notional - fee
        assert self.cash > -1e-6, f"negative cash after trade: {self.cash}"
        new_qty = cur + dq
        self.positions.pop(symbol, None) if abs(new_qty) < 1e-12 else \
            self.positions.__setitem__(symbol, new_qty)

        row = {"ts": str(ts), "day": day, "symbol": symbol,
               "side": "buy" if dq > 0 else "sell", "qty": round(abs(dq), 10),
               "price": price, "notional": round(notional, 4), "fee": round(fee, 6),
               "mode": mode, "reason": reason}
        trades = self._read("trades.parquet")
        trades = pd.concat([trades, pd.DataFrame([row])], ignore_index=True) \
            if trades is not None else pd.DataFrame([row])
        _atomic_parquet(trades, self.dir / "trades.parquet")   # 事件先落盘
        self.save()                                            # 快照随后（仅供人读）
        log.info("trade %-4s %-8s qty=%.6g @ %.2f ($%.2f, fee $%.4f) [%s] %s",
                 row["side"], symbol, row["qty"], price, notional, fee, mode, reason)
        return row

    # ---------------- 调仓完成注册表（R2 幂等基元） ----------------

    def record_run(self, day: str, mode: str, n_trades: int, note: str = "") -> None:
        runs = self._read("runs.parquet")
        row = pd.DataFrame([{"day": day, "mode": mode, "n_trades": n_trades,
                             "completed_at": str(pd.Timestamp.now(tz="UTC")), "note": note}])
        runs = pd.concat([runs, row], ignore_index=True) if runs is not None else row
        _atomic_parquet(runs, self.dir / "runs.parquet")

    def run_completed(self, day: str, mode: str | None = None) -> bool:
        runs = self._read("runs.parquet")
        if runs is None or not len(runs):
            return False
        m = runs["day"] == day
        if mode:
            m &= runs["mode"] == mode
        return bool(m.any())

    def rebalanced_on(self, day: str, mode: str | None = None) -> bool:
        """（保留供报表用；幂等判断请用 run_completed）"""
        t = self._read("trades.parquet")
        if t is None or not len(t):
            return False
        m = t["day"] == day
        if mode:
            m &= t["mode"] == mode
        return bool(m.any())

    # ---------------- 估值 ----------------

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(q * prices.get(s, 0.0) for s, q in self.positions.items())

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {s: 0.0 for s in self.positions}
        return {s: q * prices.get(s, 0.0) / eq for s, q in self.positions.items()}

    def mark(self, day: str, prices: dict[str, float], note: str) -> float:
        eq = self.equity(prices)
        df = self._read("equity.parquet")
        row = pd.DataFrame([{"day": day, "equity": round(eq, 4),
                             "cash": round(self.cash, 4),
                             "positions_value": round(eq - self.cash, 4), "note": note}])
        if df is not None and len(df):
            df = pd.concat([df[df["day"] != day], row], ignore_index=True).sort_values("day")
        else:
            df = row
        _atomic_parquet(df, self.dir / "equity.parquet")
        return eq

    def equity_series(self, settled_only: bool = False) -> pd.Series:
        """settled_only=True 只取 note=='settled' 的日度权益（第二轮 review：
        intraday 标记不得混入日度统计样本）。"""
        df = self._read("equity.parquet")
        if df is None or not len(df):
            return pd.Series(dtype=float)
        if settled_only and "note" in df.columns:
            df = df[df["note"] == "settled"]
        return pd.Series(df["equity"].to_numpy(),
                         index=pd.to_datetime(df["day"], utc=True)).sort_index()

    def trades_df(self) -> pd.DataFrame:
        t = self._read("trades.parquet")
        return t if t is not None else pd.DataFrame(columns=TRADE_COLS)
