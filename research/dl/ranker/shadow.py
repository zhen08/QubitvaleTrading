"""Daily L/S shadow logging for the Track 3-LS prospective test.

Per decision date: verify the frozen artifacts, predict `p` for the latest
usable universe date, apply the frozen L/S rule (hysteresis state reconstructed
from the previous log rows), and append intended target weights for variant
"LS" plus the "R1LS" momentum reference. First-write-wins per
(decision_ts, symbol, variant): re-runs never rewrite a logged decision.

New listings enter the panel only via the monthly full `backfill_universe`
rerun; the daily tail updater only advances symbols already on disk — the
110-bar history rule keeps brand-new listings out of the universe far longer
than that lag anyway.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data import storeio
from research.dl.ranker import data as D
from research.dl.ranker import portfolio as P
from research.dl.ranker.freeze import verify_artifacts, model_dir
from research.dl.ranker.model import GRURanker
from research.dl.ranker.walkforward import _score_frame

log = logging.getLogger("qvt.ranker.shadow")


def ls_shadow_path(store: Path) -> Path:
    return store / "dl_shadow" / "ranker_ls.parquet"


def _held_from_log(log_df: pd.DataFrame | None, variant: str) -> tuple[set, set]:
    if log_df is None or not len(log_df):
        return set(), set()
    v = log_df[log_df["variant"] == variant]
    if not len(v):
        return set(), set()
    last_ts = v["decision_ts"].max()
    last = v[v["decision_ts"] == last_ts]
    longs = set(last.loc[last["weight"] > 0, "symbol"])
    shorts = set(last.loc[last["weight"] < 0, "symbol"])
    return longs, shorts


def run_ls_shadow(settings: dict) -> dict:
    import torch
    store = storeio.store_dir(settings)
    model_id = (settings.get("dl_shadow") or {}).get("ranker_ls_model_id")
    if not model_id:
        raise RuntimeError("dl_shadow.ranker_ls_model_id not configured")
    man = verify_artifacts(store, model_id)
    if pd.Timestamp.now(tz="UTC") > pd.Timestamp(man["retrain_due"]):
        log.warning("LS shadow model %s past retrain_due (%s)",
                    model_id, man["retrain_due"])

    rd = D.load_ranker_data(store)
    d_last = rd.dates.max()
    floor = float(man["sigma_floor"])
    z = D.normalized_returns(rd, floor)
    zrank = D.cross_rank(z, rd.member)
    seq = D.build_sequences(rd, z, zrank, pd.DatetimeIndex([d_last]),
                            require_label=False)
    if seq is None:
        raise RuntimeError(f"no member sequences on {d_last}")
    X, _, meta = seq
    mean = torch.tensor(man["standardize_mean"])
    std = torch.tensor(man["standardize_std"])
    X, _ = D.standardize(X, (mean, std))

    probs = []
    for s in man["seeds"]:
        model = GRURanker(int(man["width"]))
        state = torch.load(model_dir(store, model_id) / f"seed_{s}.pt",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            probs.append(torch.sigmoid(model(X)).numpy())
    p = np.mean(probs, axis=0)
    score = _score_frame(meta, p, rd.member)

    shortable = set(man["shortable_um"])
    existing = storeio.read_parquet_if_exists(ls_shadow_path(store))
    rows = []
    for variant, sc_row, neutral in [
        ("LS", score.loc[d_last], 0.5),
        ("R1LS", rd.mom21.loc[d_last], 0.0),
    ]:
        hl, hs = _held_from_log(existing, variant)
        target = P.ls_targets(sc_row, rd.member.loc[d_last], shortable, hl, hs,
                              neutral=neutral)
        ranks = sc_row.where(rd.member.loc[d_last]).rank(ascending=False,
                                                         method="first")
        # log every held/target name plus flat exits (weight 0 for names dropped)
        for s in set(target) | hl | hs:
            rows.append({"decision_ts": d_last, "symbol": s, "variant": variant,
                         "score": float(sc_row.get(s, np.nan)),
                         "rank": float(ranks.get(s, np.nan)),
                         "weight": float(target.get(s, 0.0))})
    out = pd.DataFrame(rows)
    out["model_id"] = model_id
    out["checkpoint_hash"] = man["checkpoint_hash"]
    out["logged_at"] = str(pd.Timestamp.now(tz="UTC"))

    if existing is not None and len(existing):
        merged = pd.concat([existing, out], ignore_index=True)
        merged = (merged.drop_duplicates(subset=["decision_ts", "symbol", "variant"],
                                         keep="first")
                        .sort_values(["decision_ts", "variant", "symbol"])
                        .reset_index(drop=True))
    else:
        merged = out
    storeio.write_parquet(merged, ls_shadow_path(store))
    n_new = len(merged) - (len(existing) if existing is not None else 0)
    ls_rows = out[out["variant"] == "LS"]
    summary = {"model_id": model_id, "decision_ts": str(d_last),
               "rows_new": int(n_new),
               "longs": sorted(ls_rows.loc[ls_rows["weight"] > 0, "symbol"]),
               "shorts": sorted(ls_rows.loc[ls_rows["weight"] < 0, "symbol"])}
    log.info("LS shadow: %s", summary)
    return summary
