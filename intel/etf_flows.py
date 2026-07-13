"""ETF-flow risk gate: sustained US spot ETF net outflows restrict ADDS only.

Same risk-off-only philosophy as the news/event overlay — it can block or halve
*increasing* a position, never blocks reducing/exiting. Only BTC and ETH have US
spot ETFs, so SOL is never gated. The trailing net flow is computed as-of the
decision day from data/store/etf/flows.parquet, so catchup replays the regime
that actually held on a past day (the parquet is itself the dated history).

Data source: data.collectors.etf_flows (CoinGlass). No data / no API key -> empty
gate (fail-open): a supplementary signal must not halt the strategy.
"""
from __future__ import annotations

import logging

import pandas as pd

from data import storeio

log = logging.getLogger("qvt.etf")

# trading symbol -> ETF asset key. Only BTC/ETH are GATED (mature, deep ETFs whose
# flow magnitudes fit the $M thresholds). SOL flows are collected + shown for info
# but not gated (nascent, small-scale ETF — thresholds would never fire meaningfully).
ASSETS_WITH_ETF = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
DISPLAY_ASSETS = ("BTC", "ETH", "SOL")


def load_flows(settings: dict) -> pd.DataFrame | None:
    df = storeio.read_parquet_if_exists(storeio.etf_flows_path(storeio.store_dir(settings)))
    if df is None or df.empty:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _cfg(settings: dict) -> dict:
    c = settings.get("intel", {})
    return {
        "look": int(c.get("etf_lookback_days", 5)),
        "min_days": int(c.get("etf_min_days", 3)),
        "no_add": float(c.get("etf_outflow_no_add_usd_m", 300.0)),
        "trim": float(c.get("etf_outflow_trim_usd_m", 1000.0)),
    }


def etf_gate_asof(settings: dict, day: pd.Timestamp, flows: pd.DataFrame | None = None) -> dict:
    """Per-symbol gate as-of `day` (UTC midnight).

    Returns {symbol: {action, net_usd_m, days, days_out}} where action is
    'halve' (trailing net outflow >= trim), 'block' (>= no_add), or None. Assets
    with fewer than min_days of data in the window are omitted (no gate).
    """
    cfg = _cfg(settings)
    df = flows if flows is not None else load_flows(settings)
    out: dict[str, dict] = {}
    if df is None:
        return out
    day = day.normalize()
    for sym, asset in ASSETS_WITH_ETF.items():
        a = df[(df["asset"] == asset) & (df["date"] <= day)].sort_values("date").tail(cfg["look"])
        if len(a) < cfg["min_days"]:
            continue
        net = float(a["net_flow_usd_m"].sum())
        days_out = int((a["net_flow_usd_m"] < 0).sum())
        action = None
        if -net >= cfg["trim"]:
            action = "halve"
        elif -net >= cfg["no_add"]:
            action = "block"
        out[sym] = {"action": action, "net_usd_m": round(net, 1),
                    "days": len(a), "days_out": days_out}
    return out


def etf_oneline(settings: dict) -> str:
    """Compact one-liner for the daily summary, or '' if no data."""
    df = load_flows(settings)
    if df is None:
        return ""
    now = pd.Timestamp.now(tz="UTC").normalize()
    gate = etf_gate_asof(settings, now, df)
    parts = []
    for sym, g in gate.items():
        tag = {"halve": " TRIM", "block": " no-add"}.get(g["action"], "")
        parts.append(f"{ASSETS_WITH_ETF[sym]} ${g['net_usd_m']:+,.0f}M({g['days_out']}/{g['days']}){tag}")
    return ("ETF flow " + f"{_cfg(settings)['look']}d: " + " · ".join(parts)) if parts else ""


def etf_summary(settings: dict) -> str:
    """Full text for the /etf command."""
    df = load_flows(settings)
    if df is None:
        return "QVT ETF flows: no data yet (set COINGLASS_API_KEY and run the daily job)."
    now = pd.Timestamp.now(tz="UTC").normalize()
    cfg = _cfg(settings)
    gate = etf_gate_asof(settings, now, df)
    sym_of = {a: s for s, a in ASSETS_WITH_ETF.items()}
    lines = [f"QVT ETF flows (US spot, {cfg['look']}d trailing net)"]
    state = {"halve": "TRIM (halve adds)", "block": "no adds", None: "ok"}
    for asset in DISPLAY_ASSETS:
        a = df[df["asset"] == asset].sort_values("date").tail(cfg["look"])
        if a.empty:
            continue
        if asset in sym_of:                                   # gated asset
            tag = state.get(gate.get(sym_of[asset], {}).get("action"), "ok")
        else:                                                 # SOL: info only
            tag = "info only (no gate)"
        lines.append(f"\n{asset}: net ${a['net_flow_usd_m'].sum():+,.0f}M "
                     f"({int((a['net_flow_usd_m'] < 0).sum())}/{len(a)} days out) -> {tag}")
        for r in a.itertuples():
            lines.append(f"  {r.date.date()} ${r.net_flow_usd_m:+,.1f}M")
    if len(lines) == 1:
        return "QVT ETF flows: data present but no rows in the trailing window."
    lines.append(f"\nthresholds (BTC/ETH): no-add >= ${cfg['no_add']:,.0f}M out, "
                 f"halve >= ${cfg['trim']:,.0f}M out")
    return "\n".join(lines)
