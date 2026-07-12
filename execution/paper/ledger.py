"""Paper 账本：现金/持仓/成交/权益的持久化与会计。

文件（data/store/paper/）：
  state.json      现金、持仓、起始资金、最后结算日
  trades.parquet  逐笔成交 [ts, day, symbol, side, qty, price, notional, fee, mode, reason]
  equity.parquet  每日权益 [day, equity, cash, positions_value, note]（intraday 标记会被 settled 覆盖）
会计规则：买入 cash -= notional + fee；卖出 cash += notional − fee；fee = notional × fee_bps/1e4。
只做多（现货长平），qty ≥ 0。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger("qvt.ledger")

TRADE_COLS = ["ts", "day", "symbol", "side", "qty", "price", "notional", "fee", "mode", "reason"]


@dataclass
class Ledger:
    dir: Path
    cash: float
    positions: dict[str, float]
    initial_capital: float
    start_date: str
    last_settled: str | None

    # ---------- 持久化 ----------

    @classmethod
    def load_or_init(cls, store: Path, initial_capital: float, start_date: str) -> "Ledger":
        d = store / "paper"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "state.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            return cls(dir=d, cash=float(s["cash"]),
                       positions={k: float(v) for k, v in s["positions"].items()},
                       initial_capital=float(s["initial_capital"]),
                       start_date=s["start_date"], last_settled=s.get("last_settled"))
        led = cls(dir=d, cash=float(initial_capital), positions={},
                  initial_capital=float(initial_capital), start_date=start_date,
                  last_settled=None)
        led.save()
        log.info("paper ledger initialized: $%.2f from %s", initial_capital, start_date)
        return led

    def save(self) -> None:
        payload = {"cash": round(self.cash, 8), "positions": self.positions,
                   "initial_capital": self.initial_capital,
                   "start_date": self.start_date, "last_settled": self.last_settled,
                   "updated_at": str(pd.Timestamp.now(tz="UTC"))}
        (self.dir / "state.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")

    def _read(self, name: str) -> pd.DataFrame | None:
        p = self.dir / name
        return pd.read_parquet(p) if p.exists() else None

    # ---------- 成交 ----------

    def execute(self, *, ts: pd.Timestamp, day: str, symbol: str, target_qty: float,
                price: float, fee_bps: float, mode: str, reason: str,
                min_trade_usdt: float = 0.0) -> dict | None:
        cur = self.positions.get(symbol, 0.0)
        dq = target_qty - cur
        notional = abs(dq) * price
        if notional < max(min_trade_usdt, 1e-9):
            return None
        fee = notional * fee_bps / 1e4
        if dq > 0:
            self.cash -= notional + fee
        else:
            self.cash += notional - fee
        new_qty = cur + dq
        if abs(new_qty) < 1e-12:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_qty
        row = {"ts": str(ts), "day": day, "symbol": symbol,
               "side": "buy" if dq > 0 else "sell", "qty": round(abs(dq), 10),
               "price": price, "notional": round(notional, 4), "fee": round(fee, 6),
               "mode": mode, "reason": reason}
        trades = self._read("trades.parquet")
        trades = pd.concat([trades, pd.DataFrame([row])], ignore_index=True) \
            if trades is not None else pd.DataFrame([row])
        trades.to_parquet(self.dir / "trades.parquet", index=False)
        log.info("trade %-4s %-8s qty=%.6g @ %.2f ($%.2f, fee $%.4f) [%s] %s",
                 row["side"], symbol, row["qty"], price, notional, fee, mode, reason)
        return row

    def rebalanced_on(self, day: str, mode: str | None = None) -> bool:
        trades = self._read("trades.parquet")
        if trades is None or not len(trades):
            return False
        m = trades["day"] == day
        if mode:
            m &= trades["mode"] == mode
        return bool(m.any())

    # ---------- 估值 ----------

    def equity(self, prices: dict[str, float]) -> float:
        pos_val = sum(q * prices.get(s, 0.0) for s, q in self.positions.items())
        return self.cash + pos_val

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {s: 0.0 for s in self.positions}
        return {s: q * prices.get(s, 0.0) / eq for s, q in self.positions.items()}

    def mark(self, day: str, prices: dict[str, float], note: str) -> float:
        eq = self.equity(prices)
        pos_val = eq - self.cash
        df = self._read("equity.parquet")
        row = pd.DataFrame([{"day": day, "equity": round(eq, 4),
                             "cash": round(self.cash, 4),
                             "positions_value": round(pos_val, 4), "note": note}])
        if df is not None and len(df):
            df = df[df["day"] != day]                 # upsert：settled 覆盖 intraday
            df = pd.concat([df, row], ignore_index=True).sort_values("day")
        else:
            df = row
        df.to_parquet(self.dir / "equity.parquet", index=False)
        return eq

    def equity_series(self) -> pd.Series:
        df = self._read("equity.parquet")
        if df is None or not len(df):
            return pd.Series(dtype=float)
        s = pd.Series(df["equity"].to_numpy(),
                      index=pd.to_datetime(df["day"], utc=True))
        return s.sort_index()

    def trades_df(self) -> pd.DataFrame:
        t = self._read("trades.parquet")
        return t if t is not None else pd.DataFrame(columns=TRADE_COLS)
