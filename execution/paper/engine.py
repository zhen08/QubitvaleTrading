"""Paper 每日引擎 v2 —— 结算补账 + 当日实时调仓（review 加固版）。

时序语义与研究引擎一致：D-1 收盘决策 → D 日生效。每次运行（幂等）：
  1) SETTLE：补齐已结算未入账的 UTC 日。当日调仓若从未完成（runs 注册表判定，R2），
     以该日开盘价 ±slip_floor 补执行（mode=catchup），并回放该时点的事件门与
     risk_flags 历史（R6）；以收盘价 mark（settled 覆盖 intraday）。
  2) LIVE：今日若未完成调仓，用 Bitget 实时 bid/ask（缺则 last±slip_floor，R3）成交。
守卫（R6）：
  - 信号必须来自 D-1 决策 bar；Vision 未到则用 Bitget 日线补尾部（跨源差已验 <0.1%）；
    仍缺 → 跳过 live 交易并记 P1 事故，绝不静默沿用旧信号；
  - risk_flags 过期（TTL）→ 视为"状态未知"，禁止加仓并记 P2；
  - 所有事故落盘 store/ops/incidents.parquet（P0 统计的依据）。
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import pandas as pd

from data import storeio
from execution.paper.ledger import Ledger
from intel.event_gate import entries_blocked
from intel.news_scorer import load_flags_asof, load_risk_flags
from ops import incident_log
from strategies.donchian_ensemble import (load_spot_daily, refresh_signals,
                                          targets_for_day)

log = logging.getLogger("qvt.paper")

CCXT_SYMBOL = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT", "SOLUSDT": "SOL/USDT"}
NOMINAL_EXEC_UTC = pd.Timedelta(minutes=10)   # catchup 回放风控时假定的名义执行时刻 00:10


@contextmanager
def _exclusive_lock(path):
    """整个 run_daily 生命周期的进程级独占锁（第二轮 review P2：原子写只防半文件，
    不防两个重叠任务的丢失更新）。flock 随进程退出自动释放，无陈锁问题。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", encoding="utf-8")
    try:
        import fcntl
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            yield False
            return
        f.write(str(os.getpid()))
        f.flush()
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
    except ImportError:  # 非 POSIX 平台：无锁降级并警告
        log.warning("fcntl unavailable — running WITHOUT concurrency lock")
        try:
            yield True
        finally:
            f.close()


def _missing_decision_bars(dfs: dict[str, pd.DataFrame], decision_day: pd.Timestamp) -> list[str]:
    """P1 修复：逐资产验证 D-1 决策 bar 存在且 OHLC 非空。返回缺失的 symbol 列表。"""
    missing = []
    for sym, df in dfs.items():
        if decision_day not in df.index:
            missing.append(sym)
            continue
        row = df.loc[decision_day]
        if pd.isna(row.get("close")) or pd.isna(row.get("open")):
            missing.append(sym)
    return missing


# ---------------- 行情 ----------------

def _live_quotes(symbols: list[str], slip_floor_bps: float) -> dict[str, dict]:
    """{sym: {mark, buy, sell}}：买用 ask、卖用 bid；缺盘口则 last±slip_floor（R3）。"""
    from data.collectors.bitget_live import exchange  # lazy: 需要网络

    ex = exchange()
    out = {}
    slip = slip_floor_bps / 1e4
    for s in symbols:
        t = ex.fetch_ticker(CCXT_SYMBOL[s])
        last = float(t["last"])
        out[s] = {
            "mark": last,
            "buy": float(t["ask"]) if t.get("ask") else last * (1 + slip),
            "sell": float(t["bid"]) if t.get("bid") else last * (1 - slip),
        }
    return out


def _bitget_tail_fill(dfs: dict[str, pd.DataFrame], need_through: pd.Timestamp) -> tuple[dict, list[str]]:
    """Vision 日线未覆盖到 need_through 时，用 Bitget 已结算日线补尾部（仅内存，
    不落库——Vision 到货后 settle 用官方数据）。返回 (dfs, 补过的说明)。"""
    from data.collectors.bitget_live import fetch_recent_ohlcv

    notes = []
    for sym, df in dfs.items():
        last = df.index.max()
        if last >= need_through:
            continue
        bg = fetch_recent_ohlcv(CCXT_SYMBOL[sym], "1d", limit=10)
        bg = bg.set_index(pd.to_datetime(bg["ts"], utc=True).dt.normalize())
        bg = bg[(bg.index > last) & (bg.index <= need_through)]
        if not len(bg):
            continue
        add = pd.DataFrame({
            "ts": bg.index, "open": bg["open"], "high": bg["high"], "low": bg["low"],
            "close": bg["close"], "volume": bg["volume"],
            "quote_volume": (bg["volume"] * bg["close"]),
        }, index=bg.index)
        dfs[sym] = pd.concat([df, add]).sort_index()
        notes.append(f"{sym}+{len(add)}bar(bitget tail)")
    return dfs, notes


# ---------------- 风控（live 与 catchup 共用） ----------------

def _apply_risk_rules(targets: dict[str, float], current_w: dict[str, float],
                      settings: dict, now: pd.Timestamp, flags: dict | None,
                      flags_unknown: bool) -> tuple[dict, list[str]]:
    """事件门 + 新闻旗（只限制加仓，永不阻止减仓）。flags_unknown=True 时按
    "状态未知"保守处理：一律禁加仓（R6）。"""
    notes: list[str] = []
    cfg = settings.get("intel", {})
    adj = dict(targets)

    blocked, why = entries_blocked(now, settings)
    if blocked:
        for s in adj:
            adj[s] = min(adj[s], current_w.get(s, 0.0))
        notes.append(f"event_gate: no new entries ({why})")

    if flags_unknown:
        for s in adj:
            adj[s] = min(adj[s], current_w.get(s, 0.0))
        notes.append("risk_flags unknown/stale: no adds (conservative)")
        return adj, notes

    spec = (flags or {}).get("asset_neg_severity", {})
    market = int((flags or {}).get("market_neg_severity", 0))
    no_add = int(cfg.get("risk_flag_no_add_severity", 4))
    halve = int(cfg.get("risk_flag_halve_severity", 5))
    for s in adj:
        sev = int(spec.get(s.replace("USDT", ""), 0))
        if sev >= halve:
            adj[s] = min(adj[s], current_w.get(s, 0.0)) * 0.5
            notes.append(f"risk_flag sev{sev}: halve {s}")
        elif (sev >= no_add or market >= halve) and adj[s] > current_w.get(s, 0.0):
            adj[s] = current_w.get(s, 0.0)
            notes.append(f"risk_flag spec{sev}/mkt{market}: no add {s}")
    return adj, notes


# ---------------- 调仓（卖先买后 + 现金护栏在账本层） ----------------

def _rebalance(led: Ledger, targets: dict[str, float], quotes: dict[str, dict],
               *, day: str, ts: pd.Timestamp, mode: str, reason: str,
               settings: dict) -> list[dict]:
    pcfg = settings["paper"]
    thr = float(pcfg.get("rebalance_threshold", 0.02))
    fee = float(pcfg.get("fee_bps_side", 10.0))
    min_usd = float(pcfg.get("min_trade_usdt", 10.0))
    marks = {s: q["mark"] for s, q in quotes.items()}
    eq = led.equity(marks)
    cur_w = led.weights(marks)

    plan = []
    for sym, tw in targets.items():
        cw = cur_w.get(sym, 0.0)
        if abs(tw - cw) <= thr:
            continue
        plan.append((sym, tw, "sell" if tw < cw else "buy"))
    plan.sort(key=lambda x: 0 if x[2] == "sell" else 1)      # R3: 先卖释放现金再买

    trades = []
    for sym, tw, side in plan:
        px = quotes[sym][side]
        row = led.execute(ts=ts, day=day, symbol=sym, target_qty=tw * eq / px,
                          price=px, fee_bps=fee, mode=mode, reason=reason,
                          min_trade_usdt=min_usd)
        if row:
            trades.append(row)
    return trades


# ---------------- 主流程 ----------------

def run_daily(settings: dict, now: pd.Timestamp | None = None) -> dict:
    """公共入口：独占锁内执行；已有运行在进行时优雅跳过（不视为事故）。"""
    store = storeio.store_dir(settings)
    with _exclusive_lock(store / "paper" / ".run.lock") as acquired:
        if not acquired:
            log.warning("another run_daily holds the lock — skipping this invocation")
            return {"skipped": "another run in progress",
                    "date": str((now or pd.Timestamp.now(tz='UTC')).date())}
        return _run_daily_locked(settings, now)


def _run_daily_locked(settings: dict, now: pd.Timestamp | None = None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    today = now.normalize()
    store = storeio.store_dir(settings)
    pcfg = settings["paper"]
    slip_floor = float(pcfg.get("slip_floor_bps", 1.0))

    summary: dict = {"date": str(today.date()), "settled": [], "live_trades": [],
                     "notes": [], "incidents": []}

    def incident(level: str, kind: str, detail: str, day: str | None = None) -> None:
        incident_log.record(store, level, kind, detail, day)
        summary["incidents"].append(f"[{level}] {kind}: {detail}")

    # 数据 + 信号（R6：live 决策 bar 缺失时用 Bitget 尾部补齐）
    dfs = {s: load_spot_daily(store, s) for s in settings["symbols"]}
    decision_day = today - pd.Timedelta(days=1)
    try:
        dfs, tail_notes = _bitget_tail_fill(dfs, decision_day)
        summary["notes"] += tail_notes
    except Exception as exc:  # noqa: BLE001
        incident("P2", "bitget_tail_fail", str(exc))
    weights = refresh_signals(settings, dfs)

    led = Ledger.load_or_init(store, float(pcfg["initial_capital_usdt"]),
                              str(pcfg["start_date"]))
    start = pd.Timestamp(led.start_date, tz="UTC")

    # ---------- 1) SETTLE ----------
    from_day = (pd.Timestamp(led.last_settled, tz="UTC") + pd.Timedelta(days=1)) \
        if led.last_settled else start
    for day in pd.date_range(from_day, today - pd.Timedelta(days=1), freq="D", tz="UTC"):
        dstr = str(day.date())
        bars = {}
        ok = all(day in dfs[s].index for s in dfs) and all(
            not pd.isna(dfs[s].loc[day, ["open", "close"]]).any() for s in dfs)
        if not ok:
            incident("P2", "settle_deferred", f"{dstr}: daily bar missing/NaN", dstr)
            break
        for s, df in dfs.items():
            bars[s] = {"open": float(df.loc[day, "open"]), "close": float(df.loc[day, "close"])}

        if not led.run_completed(dstr):
            miss = _missing_decision_bars(dfs, day - pd.Timedelta(days=1))
            targets = None if miss else targets_for_day(weights, day, strict=True)
            if targets is None:
                incident("P1", "catchup_signal_missing",
                         f"{dstr}: D-1 bar missing for {miss or 'combined index'} — "
                         "rebalance skipped (positions carried)", dstr)
                led.record_run(dstr, "skipped", 0, note=f"missing={miss}")
            else:
                # R6: catchup 也回放该时点的事件门与历史 risk_flags
                asof = day + NOMINAL_EXEC_UTC
                hist_flags = load_flags_asof(settings, asof)
                slip = slip_floor / 1e4
                quotes = {s: {"mark": b["open"], "buy": b["open"] * (1 + slip),
                              "sell": b["open"] * (1 - slip)} for s, b in bars.items()}
                cur_w = led.weights({s: q["mark"] for s, q in quotes.items()})
                # 历史无归档旗（归档功能启用前的日期）→ 只回放事件门，flags 跳过并注记；
                # 不追溯性地禁加仓（那会改写历史行为而非复现它）。
                targets, notes = _apply_risk_rules(targets, cur_w, settings, asof,
                                                   hist_flags, flags_unknown=False)
                if hist_flags is None:
                    notes.append("flags history unavailable for this day (gate-only replay)")
                summary["notes"] += [f"{dstr}: {n}" for n in notes]
                done = _rebalance(led, targets, quotes, day=dstr, ts=asof,
                                  mode="catchup", reason="missed-live backfill @open±slip",
                                  settings=settings)
                led.record_run(dstr, "catchup", len(done),
                               note="flags_hist" if hist_flags else "flags_unavailable")
                if done:
                    incident("P2", "catchup_fill", f"{dstr}: {len(done)} fill(s)", dstr)
        led.mark(dstr, {s: b["close"] for s, b in bars.items()}, note="settled")
        led.last_settled = dstr
        summary["settled"].append(dstr)

    # ---------- 2) LIVE（今日） ----------
    dstr = str(today.date())
    if today >= start and not led.run_completed(dstr):
        miss = _missing_decision_bars(dfs, today - pd.Timedelta(days=1))
        targets = None if miss else targets_for_day(weights, today, strict=True)
        if targets is None:
            incident("P1", "signal_missing_live",
                     f"{dstr}: D-1 decision bar unavailable for "
                     f"{miss or 'combined index'} — live trading skipped "
                     "(no per-asset zeroing)", dstr)
        else:
            try:
                quotes = _live_quotes(settings["symbols"], slip_floor)
            except Exception as exc:  # noqa: BLE001
                incident("P1", "live_prices_unavailable", str(exc), dstr)
                quotes = None
            if quotes:
                flags = load_risk_flags(settings)
                if flags.get("stale"):
                    incident("P2", "risk_flags_stale",
                             f"age={flags.get('age_hours')}h > TTL, no adds", dstr)
                marks = {s: q["mark"] for s, q in quotes.items()}
                cur_w = led.weights(marks)
                targets, notes = _apply_risk_rules(targets, cur_w, settings, now,
                                                   flags, flags_unknown=bool(flags.get("stale")))
                summary["notes"] += notes
                trades = _rebalance(led, targets, quotes, day=dstr, ts=now, mode="live",
                                    reason="daily rebalance", settings=settings)
                led.record_run(dstr, "live", len(trades))
                summary["live_trades"] = trades
                drift = {}
                for sym, df in dfs.items():
                    prev = df.loc[:today - pd.Timedelta(days=1)]
                    if len(prev):
                        ref = float(prev["close"].iloc[-1])
                        drift[sym] = round(1e4 * (marks[sym] - ref) / ref, 1)
                summary["exec_drift_bps_vs_prev_close"] = drift
                led.mark(dstr, marks, note="intraday")

    led.save()
    eqs = led.equity_series()
    if len(eqs):
        summary["equity"] = round(float(eqs.iloc[-1]), 2)
        summary["equity_ret_pct_since_start"] = round(
            100 * (float(eqs.iloc[-1]) / led.initial_capital - 1), 3)
    summary["positions"] = {k: round(v, 8) for k, v in led.positions.items()}
    summary["cash"] = round(led.cash, 2)
    return summary
