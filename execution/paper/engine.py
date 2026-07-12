"""Paper 每日引擎：结算补账 + 当日实时调仓。

时序语义与研究引擎严格一致：D-1 收盘决策 → D 日生效。
每次运行做两件事（幂等，可任意时间任意频率重跑）：
  1) SETTLE：把所有已结算而未入账的 UTC 日补齐——若当日调仓从未实时执行过，
     以该日 **开盘价** 补执行（mode=catchup，计入"错过实时执行"运维指标）；
     以该日收盘价 mark 权益（note=settled，覆盖 intraday 标记）。
  2) LIVE：今日（UTC）若尚未调仓，用 Bitget 实时价执行（mode=live），
     记录 vs 昨收的执行漂移；事件门与 risk_flags 只作用于 live 加仓。
"""
from __future__ import annotations

import logging

import pandas as pd

from data import storeio
from execution.paper.ledger import Ledger
from intel.event_gate import entries_blocked
from intel.news_scorer import load_risk_flags
from strategies.donchian_ensemble import load_spot_daily, refresh_signals, targets_for_day

log = logging.getLogger("qvt.paper")

CCXT_SYMBOL = {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT", "SOLUSDT": "SOL/USDT"}


def _live_prices(symbols: list[str]) -> dict[str, float]:
    from data.collectors.bitget_live import exchange  # lazy: 需要网络

    ex = exchange()
    out = {}
    for s in symbols:
        t = ex.fetch_ticker(CCXT_SYMBOL[s])
        out[s] = float(t["last"])
    return out


def _apply_risk_rules(targets: dict[str, float], current_w: dict[str, float],
                      settings: dict, now: pd.Timestamp, live: bool) -> tuple[dict, list[str]]:
    """事件门 + 新闻旗（只限制加仓，永不阻止减仓）。返回 (调整后目标, 说明)。"""
    if not live:
        return targets, []
    notes: list[str] = []
    cfg = settings.get("intel", {})
    adj = dict(targets)

    blocked, why = entries_blocked(now, settings)
    if blocked:
        for s in adj:
            cur = current_w.get(s, 0.0)
            if adj[s] > cur:
                adj[s] = cur
        notes.append(f"event_gate: no new entries ({why})")

    flags = load_risk_flags(settings)
    spec = flags.get("asset_neg_severity", {})
    market = int(flags.get("market_neg_severity", 0))
    no_add = int(cfg.get("risk_flag_no_add_severity", 4))
    halve = int(cfg.get("risk_flag_halve_severity", 5))
    for s in adj:
        sev = int(spec.get(s.replace("USDT", ""), 0))
        if sev >= halve:                      # 资产特定的存在级风险 → 减半
            adj[s] = min(adj[s], current_w.get(s, 0.0)) * 0.5
            notes.append(f"risk_flag sev{sev}: halve {s}")
        elif (sev >= no_add or market >= halve) and adj[s] > current_w.get(s, 0.0):
            adj[s] = current_w.get(s, 0.0)    # 特定 sev4+ 或市场级 sev5+ → 禁加仓
            notes.append(f"risk_flag spec{sev}/mkt{market}: no add {s}")
    return adj, notes


def _rebalance(led: Ledger, targets: dict[str, float], prices: dict[str, float],
               *, day: str, ts: pd.Timestamp, mode: str, reason: str,
               settings: dict) -> list[dict]:
    pcfg = settings["paper"]
    thr = float(pcfg.get("rebalance_threshold", 0.02))
    fee = float(pcfg.get("fee_bps_side", 10.0))
    min_usd = float(pcfg.get("min_trade_usdt", 10.0))
    eq = led.equity(prices)
    cur_w = led.weights(prices)
    trades = []
    for sym, tw in targets.items():
        cw = cur_w.get(sym, 0.0)
        if abs(tw - cw) <= thr:
            continue
        target_qty = tw * eq / prices[sym]
        row = led.execute(ts=ts, day=day, symbol=sym, target_qty=target_qty,
                          price=prices[sym], fee_bps=fee, mode=mode,
                          reason=reason, min_trade_usdt=min_usd)
        if row:
            trades.append(row)
    return trades


def run_daily(settings: dict, now: pd.Timestamp | None = None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    today = now.normalize()
    store = storeio.store_dir(settings)
    pcfg = settings["paper"]

    weights = refresh_signals(settings)
    dfs = {s: load_spot_daily(store, s) for s in settings["symbols"]}
    led = Ledger.load_or_init(store, float(pcfg["initial_capital_usdt"]),
                              str(pcfg["start_date"]))

    summary: dict = {"date": str(today.date()), "settled": [], "live_trades": [],
                     "notes": [], "incidents": []}

    # ---------- 1) SETTLE ----------
    start = pd.Timestamp(led.start_date, tz="UTC")
    from_day = (pd.Timestamp(led.last_settled, tz="UTC") + pd.Timedelta(days=1)) \
        if led.last_settled else start
    for day in pd.date_range(from_day, today - pd.Timedelta(days=1), freq="D", tz="UTC"):
        dstr = str(day.date())
        opens, closes, ok = {}, {}, True
        for sym, df in dfs.items():
            if day not in df.index:
                ok = False
                break
            opens[sym] = float(df.loc[day, "open"])
            closes[sym] = float(df.loc[day, "close"])
        if not ok:
            summary["incidents"].append(f"{dstr}: daily bar missing, settle deferred")
            break
        if not led.rebalanced_on(dstr):
            targets = targets_for_day(weights, day)
            done = _rebalance(led, targets, opens, day=dstr, ts=day, mode="catchup",
                              reason="missed-live backfill @open", settings=settings)
            if done:
                summary["incidents"].append(f"{dstr}: {len(done)} catchup fill(s)")
        led.mark(dstr, closes, note="settled")
        led.last_settled = dstr
        summary["settled"].append(dstr)

    # ---------- 2) LIVE（今日） ----------
    dstr = str(today.date())
    if today >= start and not led.rebalanced_on(dstr, mode="live") \
            and not led.rebalanced_on(dstr, mode="catchup"):
        try:
            prices = _live_prices(settings["symbols"])
        except Exception as exc:  # noqa: BLE001 — 无实时价则明日 catchup 兜底
            summary["incidents"].append(f"live prices unavailable: {exc}")
            prices = None
        if prices:
            targets = targets_for_day(weights, today)
            cur_w = led.weights(prices)
            targets, notes = _apply_risk_rules(targets, cur_w, settings, now, live=True)
            summary["notes"] += notes
            trades = _rebalance(led, targets, prices, day=dstr, ts=now, mode="live",
                                reason="daily rebalance", settings=settings)
            summary["live_trades"] = trades
            # 执行漂移：live 价 vs 昨收（研究假设的成交参考）
            drift = {}
            for sym, df in dfs.items():
                prev = df.loc[:today - pd.Timedelta(days=1)]
                if len(prev):
                    ref = float(prev["close"].iloc[-1])
                    drift[sym] = round(1e4 * (prices[sym] - ref) / ref, 1)
            summary["exec_drift_bps_vs_prev_close"] = drift
            led.mark(dstr, prices, note="intraday")

    led.save()
    eqs = led.equity_series()
    if len(eqs):
        summary["equity"] = round(float(eqs.iloc[-1]), 2)
        summary["equity_ret_pct_since_start"] = round(
            100 * (float(eqs.iloc[-1]) / led.initial_capital - 1), 3)
    summary["positions"] = {k: round(v, 8) for k, v in led.positions.items()}
    summary["cash"] = round(led.cash, 2)
    return summary
