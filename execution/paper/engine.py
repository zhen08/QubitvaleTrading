"""Paper 每日引擎 v3 —— 多账本（2026-07-13）：每个策略独立账本，共享数据/行情/风控。

时序语义与研究引擎一致：D-1 收盘决策 → D 日生效。每次运行（幂等，全程独占锁）：
  对每本账（settings.paper.books × strategies.registry）：
  1) SETTLE：补齐已结算未入账的 UTC 日（runs 注册表判定；catchup 以开盘价 ±slip_floor
     补执行并回放当日事件门与历史风险旗；收盘价 mark 覆盖 intraday）。
  2) LIVE：今日若未完成调仓，用 Bitget 实时 bid/ask（缺则 last±slip_floor）成交。
共享守卫（R6/P1）：D-1 决策 bar 逐资产验证（Vision 未到用 Bitget 尾部补，仍缺则该账本
当日跳过 + 具名 P1，缺数据绝不当清仓信号）；risk_flags TTL 过期 → 全账本禁加仓；
所有事故落盘 ops/incidents.parquet（kind 前缀含账本名）。
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
from strategies.base import load_spot_daily, persist_signals, targets_for_day
from strategies.registry import get_strategy

log = logging.getLogger("qvt.paper")

CCXT_SYMBOL = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT", "SOLUSDT": "SOL/USDT"}
NOMINAL_EXEC_UTC = pd.Timedelta(minutes=10)   # catchup 回放风控时假定的名义执行时刻 00:10


@contextmanager
def _exclusive_lock(path):
    """整个 run_daily 生命周期的进程级独占锁（覆盖所有账本）。"""
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
    """P1：逐资产验证 D-1 决策 bar 存在且 OHLC 非空。返回缺失的 symbol 列表。"""
    missing = []
    for sym, df in dfs.items():
        if decision_day not in df.index:
            missing.append(sym)
            continue
        row = df.loc[decision_day]
        if pd.isna(row.get("close")) or pd.isna(row.get("open")):
            missing.append(sym)
    return missing


def _live_quotes(symbols: list[str], slip_floor_bps: float) -> dict[str, dict]:
    """{sym: {mark, buy, sell}}：买用 ask、卖用 bid；缺盘口则 last±slip_floor。"""
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
    """Vision 日线未覆盖到 need_through 时，用 Bitget 已结算日线补尾部（仅内存）。"""
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


def _apply_risk_rules(targets: dict[str, float], current_w: dict[str, float],
                      settings: dict, now: pd.Timestamp, flags: dict | None,
                      flags_unknown: bool) -> tuple[dict, list[str]]:
    """事件门 + 新闻旗（只限制加仓，永不阻止减仓）。flags_unknown → 保守禁加仓。"""
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
    plan.sort(key=lambda x: 0 if x[2] == "sell" else 1)      # 先卖释放现金再买

    trades = []
    for sym, tw, side in plan:
        px = quotes[sym][side]
        row = led.execute(ts=ts, day=day, symbol=sym, target_qty=tw * eq / px,
                          price=px, fee_bps=fee, mode=mode, reason=reason,
                          min_trade_usdt=min_usd)
        if row:
            trades.append(row)
    return trades


# ---------------- 单账本处理 ----------------

def _run_book(book: str, bcfg: dict, weights: pd.DataFrame,
              dfs: dict[str, pd.DataFrame], settings: dict, store,
              now: pd.Timestamp, today: pd.Timestamp,
              quotes: dict | None, flags: dict,
              incident) -> dict:
    pcfg = settings["paper"]
    slip_floor = float(pcfg.get("slip_floor_bps", 1.0))
    led = Ledger.load_or_init(store, float(bcfg["initial_capital_usdt"]),
                              str(bcfg["start_date"]), book=book)
    bs: dict = {"settled": [], "live_trades": [], "notes": []}
    start = pd.Timestamp(led.start_date, tz="UTC")

    # ---------- 1) SETTLE ----------
    from_day = (pd.Timestamp(led.last_settled, tz="UTC") + pd.Timedelta(days=1)) \
        if led.last_settled else start
    for day in pd.date_range(from_day, today - pd.Timedelta(days=1), freq="D", tz="UTC"):
        dstr = str(day.date())
        ok = all(day in dfs[s].index for s in dfs) and all(
            not pd.isna(dfs[s].loc[day, ["open", "close"]]).any() for s in dfs)
        if not ok:
            incident("P2", f"{book}:settle_deferred", f"{dstr}: daily bar missing/NaN", dstr)
            break
        bars = {s: {"open": float(df.loc[day, "open"]), "close": float(df.loc[day, "close"])}
                for s, df in dfs.items()}

        if not led.run_completed(dstr):
            miss = _missing_decision_bars(dfs, day - pd.Timedelta(days=1))
            targets = None if miss else targets_for_day(weights, day, strict=True)
            if targets is None:
                incident("P1", f"{book}:catchup_signal_missing",
                         f"{dstr}: D-1 bar missing for {miss or 'combined index'} — "
                         "rebalance skipped (positions carried)", dstr)
                led.record_run(dstr, "skipped", 0, note=f"missing={miss}")
            else:
                asof = day + NOMINAL_EXEC_UTC
                hist_flags = load_flags_asof(settings, asof)
                slip = slip_floor / 1e4
                q = {s: {"mark": b["open"], "buy": b["open"] * (1 + slip),
                         "sell": b["open"] * (1 - slip)} for s, b in bars.items()}
                cur_w = led.weights({s: v["mark"] for s, v in q.items()})
                targets, notes = _apply_risk_rules(targets, cur_w, settings, asof,
                                                   hist_flags, flags_unknown=False)
                if hist_flags is None:
                    notes.append("flags history unavailable (gate-only replay)")
                bs["notes"] += [f"{dstr}: {n}" for n in notes]
                done = _rebalance(led, targets, q, day=dstr, ts=asof, mode="catchup",
                                  reason="missed-live backfill @open±slip",
                                  settings=settings)
                led.record_run(dstr, "catchup", len(done),
                               note="flags_hist" if hist_flags else "flags_unavailable")
                if done:
                    incident("P2", f"{book}:catchup_fill", f"{dstr}: {len(done)} fill(s)", dstr)
        led.mark(dstr, {s: b["close"] for s, b in bars.items()}, note="settled")
        led.last_settled = dstr
        bs["settled"].append(dstr)

    # ---------- 2) LIVE（今日） ----------
    dstr = str(today.date())
    if today >= start and not led.run_completed(dstr) and quotes:
        miss = _missing_decision_bars(dfs, today - pd.Timedelta(days=1))
        targets = None if miss else targets_for_day(weights, today, strict=True)
        if targets is None:
            incident("P1", f"{book}:signal_missing_live",
                     f"{dstr}: D-1 decision bar unavailable for "
                     f"{miss or 'combined index'} — live trading skipped", dstr)
        else:
            marks = {s: v["mark"] for s, v in quotes.items()}
            cur_w = led.weights(marks)
            if flags.get("stale"):
                incident("P2", f"{book}:risk_flags_stale",
                         f"age={flags.get('age_hours')}h > TTL, no adds", dstr)
            targets, notes = _apply_risk_rules(targets, cur_w, settings, now, flags,
                                               flags_unknown=bool(flags.get("stale")))
            bs["notes"] += notes
            trades = _rebalance(led, targets, quotes, day=dstr, ts=now, mode="live",
                                reason="daily rebalance", settings=settings)
            led.record_run(dstr, "live", len(trades))
            bs["live_trades"] = trades
            led.mark(dstr, marks, note="intraday")

    led.save()
    eqs = led.equity_series()
    if len(eqs):
        bs["equity"] = round(float(eqs.iloc[-1]), 2)
        bs["ret_pct_since_start"] = round(
            100 * (float(eqs.iloc[-1]) / led.initial_capital - 1), 3)
    bs["positions"] = {k: round(v, 8) for k, v in led.positions.items()}
    bs["cash"] = round(led.cash, 2)
    if "equity" in bs:
        bs["pnl"] = round(bs["equity"] - led.initial_capital, 2)
    tdf = led.trades_df()
    bs["fees"] = round(float(tdf["fee"].sum()), 2) if len(tdf) else 0.0
    return bs


# ---------------- 主流程 ----------------

def run_daily(settings: dict, now: pd.Timestamp | None = None) -> dict:
    """公共入口：独占锁内执行全部账本；已有运行在进行时优雅跳过。"""
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
    books: dict = pcfg["books"]

    summary: dict = {"date": str(today.date()), "books": {}, "notes": [], "incidents": []}

    def incident(level: str, kind: str, detail: str, day: str | None = None) -> None:
        incident_log.record(store, level, kind, detail, day)
        summary["incidents"].append(f"[{level}] {kind}: {detail}")

    # 共享：数据 + Bitget 尾部补齐（一次，所有账本共用）
    dfs = {s: load_spot_daily(store, s) for s in settings["symbols"]}
    try:
        dfs, tail_notes = _bitget_tail_fill(dfs, today - pd.Timedelta(days=1))
        summary["notes"] += tail_notes
    except Exception as exc:  # noqa: BLE001
        incident("P2", "bitget_tail_fail", str(exc))

    # 共享：实时行情（一次）与当前风险旗
    try:
        quotes = _live_quotes(settings["symbols"],
                              float(pcfg.get("slip_floor_bps", 1.0)))
    except Exception as exc:  # noqa: BLE001
        incident("P1", "live_prices_unavailable", str(exc))
        quotes = None
    flags = load_risk_flags(settings)

    # 逐账本：信号 → settle → live
    for book, bcfg in books.items():
        try:
            strat = get_strategy(book)
            weights = strat.compute_weights(dfs)
            persist_signals(store, book, weights)
            summary["books"][book] = _run_book(book, bcfg, weights, dfs, settings,
                                               store, now, today, quotes, flags,
                                               incident)
        except Exception as exc:  # noqa: BLE001 — 单账本故障不拖垮其他账本
            incident("P1", f"{book}:book_failed", repr(exc))
            summary["books"][book] = {"error": repr(exc)}
    return summary
