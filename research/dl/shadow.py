"""Daily shadow-multiplier logging for the prospective E2 test (no paper book).

Logs, per (decision_ts, symbol): the frozen E2 ensemble's forecast and
multiplier, the B1/B2 reference multipliers computed under the identical
mapping, and the base donchian weight the book actually used — enough to
reconstruct overlay-vs-base returns later without any retroactive choice.

Append-only with first-write-wins per (decision_ts, symbol, variant): a
re-run of the daily job never rewrites an already-logged decision.
Failures raise; the caller (run_paper_daily step 3.5) records P3 and moves on.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data import storeio
from features.cross_asset import build_feature_table
from research.dl.baselines import HARBaseline, _har_design
from research.dl.overlay import risk_multiplier
from strategies.donchian_tcn_risk_overlay import (compute_risk_multiplier,
                                                  load_manifest)

log = logging.getLogger("qvt.dl.shadow")


def shadow_path(store: Path) -> Path:
    return store / "dl_shadow" / "multipliers.parquet"


def _base_weight_for(store: Path, decision_ts: pd.Timestamp, symbol: str) -> float:
    """Donchian book weight for the same decision: signals are keyed by the
    D-1 bar date, the feature table by wall-clock decision (bar date + 1d)."""
    df = storeio.read_parquet_if_exists(store / "signals" / "donchian_ensemble.parquet")
    if df is None or symbol not in df.columns:
        return float("nan")
    day = pd.Timestamp(decision_ts) - pd.Timedelta(days=1)
    hit = df[pd.to_datetime(df["decision_date"], utc=True) == day]
    return float(hit[symbol].iloc[-1]) if len(hit) else float("nan")


def run_shadow(settings: dict) -> dict:
    store = storeio.store_dir(settings)
    cfg = settings.get("dl_shadow") or {}
    model_id = cfg.get("model_id")
    if not model_id:
        raise RuntimeError("dl_shadow.model_id not configured")

    table, manifest = build_feature_table(store, settings["symbols"])
    man = load_manifest(store, model_id)
    if pd.Timestamp.now(tz="UTC") > pd.Timestamp(man["retrain_due"]):
        log.warning("shadow model %s past retrain_due (%s) — protocol refreeze pending",
                    model_id, man["retrain_due"])

    # E2 (verifies artifact identity + cross-asset status inside; fail-closed)
    e2 = compute_risk_multiplier(store, model_id, table)
    logged_at = str(pd.Timestamp.now(tz="UTC"))

    # B1/B2 references on the same latest rows, same mapping, frozen parameters
    latest = (table.sort_values("decision_ts").groupby("symbol").tail(1)
              .set_index("symbol").loc[e2["symbol"]])
    har_sigma = np.exp(np.column_stack(
        [np.ones(len(latest)), _har_design(latest)]) @ np.asarray(man["har_coef"]))
    rows = [e2.assign(variant=man["variant"])]
    for variant, sigma in [("B1", latest["c_sigma20d"].to_numpy()), ("B2", har_sigma)]:
        rows.append(pd.DataFrame({
            "symbol": e2["symbol"].to_numpy(),
            "decision_ts": e2["decision_ts"].to_numpy(),
            "sigma_hat": sigma,
            "p_tail_raw": np.nan, "p_tail_cal": np.nan,
            "multiplier": risk_multiplier(sigma, man["sigma_ref"]),
            "checkpoint_hash": man["checkpoint_hash"], "variant": variant,
        }))
    out = pd.concat(rows, ignore_index=True)
    out["logged_at"] = logged_at
    out["model_id"] = model_id
    out["schema_hash"] = manifest["schema_hash"]

    out["base_weight"] = [
        _base_weight_for(store, ts, sym)
        for ts, sym in zip(out["decision_ts"], out["symbol"])]
    out["shadow_weight"] = (out["base_weight"] * out["multiplier"]).clip(0.0, 1.0)

    path = shadow_path(store)
    existing = storeio.read_parquet_if_exists(path)
    if existing is not None and len(existing):
        merged = pd.concat([existing, out], ignore_index=True)
        merged = (merged.drop_duplicates(subset=["decision_ts", "symbol", "variant"],
                                         keep="first")
                        .sort_values(["decision_ts", "symbol", "variant"])
                        .reset_index(drop=True))
    else:
        merged = out
    storeio.write_parquet(merged, path)
    n_new = len(merged) - (len(existing) if existing is not None else 0)
    summary = {"model_id": model_id, "rows_new": int(n_new),
               "decision_ts": str(e2["decision_ts"].iloc[0]),
               "multipliers": {r["symbol"]: round(float(r["multiplier"]), 3)
                               for _, r in e2.iterrows()}}
    log.info("shadow log: %s", json.dumps(summary))
    return summary
