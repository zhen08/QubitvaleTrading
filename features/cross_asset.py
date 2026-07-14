"""Deterministic feature table for the cross-asset DL research program (plan §6).

One row per (decision_ts, symbol). Row for crypto bar-date D carries
`decision_ts = D+1 00:00 UTC` — the wall-clock moment the daily job decides,
using crypto bars ≤ D (the repo's D-1 convention) and any external record with
`available_at <= decision_ts`.

Causality rules implemented here:
  - Cross-asset session features are computed on the *session grid* first and
    as-of joined afterwards, so weekend/holiday rows carry the last session's
    feature values flagged stale — a forward-filled close never creates a new
    return or volatility observation (§5.4).
  - BTC/SPY rolling correlation and beta consume the crypto close of the same
    calendar date, which is only final at session_date+1 00:00 UTC, so those
    features carry that later `available_at`, not the 21:30 session stamp.
  - The VIX percentile is an expanding (past-only) rank.
  - Labels (§7.2) look strictly forward from D and live in `label_*` columns;
    `label_minz5` is stored σ20-normalized without a floor — a training-fold
    floor rescales it by `sigma20d/max(sigma20d, floor)` downstream (the same
    denominator applies to every horizon, so the min is preserved).

No scaling happens here: means/stds/quantiles are fit per training fold in
research/dl/dataset.py. All numeric features are unitless ratios or logs.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data import storeio
from data.collectors.cross_asset_daily import load_daily
from research.costs import daily_funding

log = logging.getLogger("qvt.features")

SCHEMA_VERSION = "xa-dl-v1"

CRYPTO_FEATURES = [
    "c_ret1_n", "c_ret5_n", "c_ret21_n", "c_ret63_n", "c_ret126_n", "c_ret252_n",
    "c_vol5", "c_vol20", "c_vol22", "c_vol60",
    "c_range", "c_volz20", "c_dch_high55", "c_dch_low20", "c_ma20", "c_ma100",
    "c_fund_d", "c_fund_m21",
]
EQUITY_FEATURES = [
    "x_spy_ret1", "x_spy_ret5", "x_spy_ret20", "x_spy_vol20",
    "x_spy_dd20", "x_spy_dd60", "x_spy_ma20", "x_spy_ma100",
    "x_qqq_spy_spread20",
    "x_corr20_btc_spy", "x_corr60_btc_spy", "x_beta60_btc_spy",
    "x_spy_closed", "x_spy_days_since",
]
VIX_FEATURES = ["x_vix_log", "x_vix_chg1", "x_vix_chg5", "x_vix_pctile",
                "x_vix_ma20", "x_stress", "x_vix_days_since"]
GLD_FEATURES = ["x_gld_ret1", "x_gld_ret5", "x_gld_ret20", "x_gld_vol20",
                "x_gld_dd20", "x_gld_dd60", "x_gld_ma20", "x_gld_ma100",
                "x_gld_closed", "x_gld_days_since"]
MASKS = ["m_fund", "m_equity", "m_vix", "m_gld"]
LABELS = ["label_logvol5", "label_minz5"]
META = ["decision_ts", "bar_date", "symbol", "c_sigma20d",
        "prov_spy_session", "prov_vix_session", "prov_gld_session"]

ALL_FEATURES = CRYPTO_FEATURES + EQUITY_FEATURES + VIX_FEATURES + GLD_FEATURES

STALE_LIMIT_DAYS = 5.5   # > this since last session close => data invalid (mask 0)
TAIL_Z = -2.0            # §7.2 tail-event threshold (in σ20·√h units)


def schema_hash() -> str:
    payload = json.dumps({"version": SCHEMA_VERSION, "features": ALL_FEATURES,
                          "masks": MASKS, "labels": LABELS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def features_path(store: Path) -> Path:
    return store / "features" / "dl_cross_asset.parquet"


# ---------------- crypto leg ----------------

def _crypto_frame(df: pd.DataFrame, funding: pd.Series | None) -> pd.DataFrame:
    """Per-symbol crypto features + labels, indexed by bar_date D (uses bars <= D;
    labels use bars D+1..D+5)."""
    px = df.set_index(pd.to_datetime(df["ts"], utc=True).dt.normalize()).sort_index()
    close = px["close"].astype(float)
    lr = np.log(close).diff()
    sig20d = lr.rolling(20).std(ddof=1)

    out = pd.DataFrame(index=px.index)
    for h, name in [(1, "c_ret1_n"), (5, "c_ret5_n"), (21, "c_ret21_n"),
                    (63, "c_ret63_n"), (126, "c_ret126_n"), (252, "c_ret252_n")]:
        out[name] = lr.rolling(h).sum() / (sig20d * np.sqrt(h))
    for w, name in [(5, "c_vol5"), (20, "c_vol20"), (22, "c_vol22"), (60, "c_vol60")]:
        out[name] = lr.rolling(w).std(ddof=1) * np.sqrt(365)
    out["c_range"] = (px["high"].astype(float) - px["low"].astype(float)) / close
    vol = px["volume"].astype(float)
    out["c_volz20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std(ddof=1)
    out["c_dch_high55"] = close / close.rolling(55).max() - 1
    out["c_dch_low20"] = close / close.rolling(20).min() - 1
    out["c_ma20"] = close / close.rolling(20).mean() - 1
    out["c_ma100"] = close / close.rolling(100).mean() - 1
    if funding is not None and len(funding):
        f = funding.reindex(out.index)
        out["c_fund_d"] = f
        out["c_fund_m21"] = f.rolling(21, min_periods=10).mean()
        out["m_fund"] = f.notna().astype(float)
        out[["c_fund_d", "c_fund_m21"]] = out[["c_fund_d", "c_fund_m21"]].fillna(0.0)
    else:
        out["c_fund_d"] = 0.0
        out["c_fund_m21"] = 0.0
        out["m_fund"] = 0.0
    out["c_sigma20d"] = sig20d

    # ---- labels (strictly forward) ----
    logc = np.log(close)
    # realized daily vol over D+1..D+5 (uncentered RMS of forward returns)
    fwd_sq = pd.concat([lr.shift(-h).pow(2) for h in range(1, 6)], axis=1)
    out["label_logvol5"] = np.log(np.sqrt(fwd_sq.mean(axis=1, skipna=False)))
    fwd_z = pd.concat(
        [(logc.shift(-h) - logc) / (sig20d * np.sqrt(h)) for h in range(1, 6)], axis=1)
    out["label_minz5"] = fwd_z.min(axis=1, skipna=False)
    return out


# ---------------- cross-asset session grids ----------------

def _etf_session_features(xa: pd.DataFrame, prefix: str) -> pd.DataFrame:
    close = xa.set_index("session_date")["close"].astype(float)
    lr = np.log(close).diff()
    out = pd.DataFrame(index=close.index)
    out[f"{prefix}_ret1"] = lr
    out[f"{prefix}_ret5"] = lr.rolling(5).sum()
    out[f"{prefix}_ret20"] = lr.rolling(20).sum()
    out[f"{prefix}_vol20"] = lr.rolling(20).std(ddof=1) * np.sqrt(252)
    out[f"{prefix}_dd20"] = close / close.rolling(20).max() - 1
    out[f"{prefix}_dd60"] = close / close.rolling(60).max() - 1
    out[f"{prefix}_ma20"] = close / close.rolling(20).mean() - 1
    out[f"{prefix}_ma100"] = close / close.rolling(100).mean() - 1
    out["available_at"] = xa.set_index("session_date")["available_at"]
    out["bar_end"] = xa.set_index("session_date")["bar_end"]
    return out.reset_index()


def _vix_session_features(xa: pd.DataFrame) -> pd.DataFrame:
    close = xa.set_index("session_date")["close"].astype(float)
    loglvl = np.log(close)
    out = pd.DataFrame(index=close.index)
    out["x_vix_log"] = loglvl
    out["x_vix_chg1"] = loglvl.diff()
    out["x_vix_chg5"] = loglvl.diff(5)
    out["x_vix_pctile"] = close.expanding(min_periods=60).rank(pct=True)
    out["x_vix_ma20"] = close / close.rolling(20).mean() - 1
    out["available_at"] = xa.set_index("session_date")["available_at"]
    out["bar_end"] = xa.set_index("session_date")["bar_end"]
    return out.reset_index()


def _asof(dest: pd.DataFrame, session: pd.DataFrame, cols: list[str],
          days_col: str | None, prov_col: str) -> pd.DataFrame:
    """As-of join session features onto decision rows by available_at."""
    s = session.copy()
    # Parquet roundtrips can yield us-resolution timestamps; merge_asof requires
    # identical dtypes on both keys.
    s["available_at"] = pd.to_datetime(s["available_at"], utc=True).astype("datetime64[ns, UTC]")
    dest = dest.copy()
    dest["decision_ts"] = pd.to_datetime(dest["decision_ts"], utc=True).astype("datetime64[ns, UTC]")
    s = s.sort_values("available_at")
    j = pd.merge_asof(dest.sort_values("decision_ts"), s,
                      left_on="decision_ts", right_on="available_at",
                      direction="backward")
    j[prov_col] = j["session_date"]
    if days_col is not None:
        j[days_col] = (j["decision_ts"] - j["bar_end"]).dt.total_seconds() / 86400.0
    return j[["decision_ts", prov_col] + cols + ([days_col] if days_col else [])]


def _empty_session(cols: list[str]) -> pd.DataFrame:
    base = {"session_date": pd.Series(dtype="datetime64[ns]"),
            "available_at": pd.Series(dtype="datetime64[ns, UTC]"),
            "bar_end": pd.Series(dtype="datetime64[ns, UTC]")}
    return pd.DataFrame({**base, **{c: pd.Series(dtype="float64") for c in cols}})


def _session_or_empty(store: Path, symbol: str, builder, cols: list[str]) -> pd.DataFrame:
    """A missing store degrades to an empty session grid -> masked zeros downstream,
    never a crash and never fabricated observations."""
    try:
        return builder(load_daily(store, symbol))
    except FileNotFoundError:
        log.warning("cross-asset store missing for %s — group will be masked", symbol)
        return _empty_session(cols)


def build_feature_table(store: Path, symbols: list[str]) -> tuple[pd.DataFrame, dict]:
    etf_cols = lambda p: [f"{p}_ret1", f"{p}_ret5", f"{p}_ret20", f"{p}_vol20",  # noqa: E731
                          f"{p}_dd20", f"{p}_dd60", f"{p}_ma20", f"{p}_ma100"]
    spy = _session_or_empty(store, "SPY",
                            lambda d: _etf_session_features(d, "x_spy"), etf_cols("x_spy"))
    qqq = _session_or_empty(store, "QQQ",
                            lambda d: _etf_session_features(d, "x_qqq"), etf_cols("x_qqq"))
    gld = _session_or_empty(store, "GLD",
                            lambda d: _etf_session_features(d, "x_gld"), etf_cols("x_gld"))
    vix = _session_or_empty(store, "VIX", _vix_session_features,
                            ["x_vix_log", "x_vix_chg1", "x_vix_chg5",
                             "x_vix_pctile", "x_vix_ma20"])

    # QQQ/SPY relative-strength spread on the common session grid.
    spread = pd.merge(spy[["session_date", "x_spy_ret20", "available_at", "bar_end"]],
                      qqq[["session_date", "x_qqq_ret20"]], on="session_date")
    spread["x_qqq_spy_spread20"] = spread["x_qqq_ret20"] - spread["x_spy_ret20"]

    # Equity stress interaction on the SPY/VIX common grid (both final by 21:30).
    stress = pd.merge(spy[["session_date", "x_spy_ret1"]],
                      vix[["session_date", "x_vix_pctile", "available_at", "bar_end"]],
                      on="session_date")
    stress["x_stress"] = (-stress["x_spy_ret1"]).clip(lower=0) * stress["x_vix_pctile"]

    # BTC/SPY correlation & beta: same-calendar-date crypto close is only final at
    # session_date+1 00:00 UTC — stamp that availability, not the session's 21:30.
    btc_px = pd.read_parquet(storeio.klines_path(store, "spot", "BTCUSDT", "1d"))
    btc_lr = (np.log(btc_px.set_index(
        pd.to_datetime(btc_px["ts"], utc=True).dt.normalize())["close"].astype(float))
        .diff())
    btc_lr.index = btc_lr.index.tz_localize(None)
    corr = pd.merge(spy[["session_date", "x_spy_ret1"]],
                    btc_lr.rename("btc_ret1"), left_on="session_date",
                    right_index=True, how="inner").set_index("session_date")
    corr["x_corr20_btc_spy"] = corr["btc_ret1"].rolling(20).corr(corr["x_spy_ret1"])
    corr["x_corr60_btc_spy"] = corr["btc_ret1"].rolling(60).corr(corr["x_spy_ret1"])
    corr["x_beta60_btc_spy"] = (corr["btc_ret1"].rolling(60).cov(corr["x_spy_ret1"])
                                / corr["x_spy_ret1"].rolling(60).var())
    corr = corr.reset_index()
    corr["available_at"] = (corr["session_date"].dt.tz_localize("UTC")
                            + pd.Timedelta(days=1))
    corr["bar_end"] = corr["available_at"]

    frames = []
    for sym in symbols:
        px = pd.read_parquet(storeio.klines_path(store, "spot", sym, "1d"))
        fpath = storeio.funding_um_path(store, sym)
        fund = daily_funding(pd.read_parquet(fpath)) if fpath.exists() else None
        if fund is not None:
            fund.index = fund.index.tz_localize(None)
        cf = _crypto_frame(px, fund)
        cf.index = cf.index.tz_localize(None)
        cf = cf.reset_index().rename(columns={"ts": "bar_date", "index": "bar_date"})
        cf["symbol"] = sym
        cf["decision_ts"] = (cf["bar_date"].dt.tz_localize("UTC") + pd.Timedelta(days=1))
        frames.append(cf)
    base = pd.concat(frames, ignore_index=True)

    dest = base[["decision_ts"]].drop_duplicates().sort_values("decision_ts")
    spy_cols = ["x_spy_ret1", "x_spy_ret5", "x_spy_ret20", "x_spy_vol20",
                "x_spy_dd20", "x_spy_dd60", "x_spy_ma20", "x_spy_ma100"]
    gld_cols = ["x_gld_ret1", "x_gld_ret5", "x_gld_ret20", "x_gld_vol20",
                "x_gld_dd20", "x_gld_dd60", "x_gld_ma20", "x_gld_ma100"]
    vix_cols = ["x_vix_log", "x_vix_chg1", "x_vix_chg5", "x_vix_pctile", "x_vix_ma20"]
    joined = dest
    for session, cols, days_col, prov in [
        (spy, spy_cols, "x_spy_days_since", "prov_spy_session"),
        (gld, gld_cols, "x_gld_days_since", "prov_gld_session"),
        (vix, vix_cols, "x_vix_days_since", "prov_vix_session"),
        (spread, ["x_qqq_spy_spread20"], None, "_prov_spread"),
        (stress, ["x_stress"], None, "_prov_stress"),
        (corr, ["x_corr20_btc_spy", "x_corr60_btc_spy", "x_beta60_btc_spy"],
         None, "_prov_corr"),
    ]:
        part = _asof(dest, session, cols, days_col, prov)
        joined = joined.merge(part, on="decision_ts", how="left")

    joined["x_spy_closed"] = (joined["x_spy_days_since"] > 1.0).astype(float)
    joined["x_gld_closed"] = (joined["x_gld_days_since"] > 1.0).astype(float)
    joined["m_equity"] = ((joined["x_spy_days_since"] <= STALE_LIMIT_DAYS)
                          & joined["x_spy_ret1"].notna()).astype(float)
    joined["m_vix"] = ((joined["x_vix_days_since"] <= STALE_LIMIT_DAYS)
                       & joined["x_vix_log"].notna()).astype(float)
    joined["m_gld"] = ((joined["x_gld_days_since"] <= STALE_LIMIT_DAYS)
                       & joined["x_gld_ret1"].notna()).astype(float)
    joined = joined.drop(columns=[c for c in joined.columns if c.startswith("_prov")])

    table = base.merge(joined, on="decision_ts", how="left")
    # Rolling-warmup NaNs inside otherwise-valid windows would become silently
    # meaningful zeros below — mask the group instead (the crypto 252-day warmup
    # already excludes these rows in practice; this keeps the invariant exact).
    table.loc[table["x_corr60_btc_spy"].isna(), "m_equity"] = 0.0
    # Masked groups contribute zeros, never silently meaningful values (§4.2).
    for m, cols in [("m_equity", EQUITY_FEATURES), ("m_vix", VIX_FEATURES),
                    ("m_gld", GLD_FEATURES)]:
        feat = [c for c in cols if not c.endswith(("_closed", "_days_since"))]
        table.loc[table[m] == 0, feat] = 0.0
        table[feat] = table[feat].fillna(0.0)
    table = table[[*META, *ALL_FEATURES, *MASKS, *LABELS]].copy()

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "schema_hash": schema_hash(),
        "built_at": str(pd.Timestamp.now(tz="UTC")),
        "symbols": symbols,
        "rows": int(len(table)),
        "decision_range": [str(table["decision_ts"].min()),
                           str(table["decision_ts"].max())],
        "columns": {"features": ALL_FEATURES, "masks": MASKS, "labels": LABELS},
    }
    return table, manifest


def persist(store: Path, table: pd.DataFrame, manifest: dict) -> Path:
    path = features_path(store)
    storeio.write_parquet(table, path)
    path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return path
