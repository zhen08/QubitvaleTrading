"""Freeze the Track 3-LS prospective shadow model
(research/reports/track3_ls_shadow_preregistration.md).

Mechanical transform of the frozen ranker protocol: expanding train ending 182
days ago, last 182 days as validation (early stop + width selection by
validation net Sharpe of the L/S rule, ties → smaller width), 5-day embargo,
5-seed ensemble of the selected width persisted with full identity hashes.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from data.collectors.universe import list_um_symbols
from research import metrics
from research.dl.ranker import data as D
from research.dl.ranker import portfolio as P
from research.dl.ranker.model import SEEDS, WIDTHS
from research.dl.ranker.walkforward import (EMBARGO_DAYS, VAL_DAYS,
                                            _score_frame, _train_all_widths)

log = logging.getLogger("qvt.ranker.freeze")

BOOK = "ranker_ls_shadow"
RETRAIN_DAYS = 182
SCHEMA = {"version": "ranker-v1", "seq_len": D.SEQ_LEN, "channels": 2,
          "adv_floor": D.ADV_FLOOR_USD, "top_rank": D.TOP_RANK,
          "min_history": D.MIN_HISTORY_BARS, "ls_slot": P.LS_SLOT_WEIGHT}


def schema_hash() -> str:
    return hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest()[:16]


def model_dir(store: Path, model_id: str) -> Path:
    return store / "models" / BOOK / model_id


def checkpoint_hash(trained: list) -> str:
    h = hashlib.sha256()
    for t in trained:
        for k in sorted(t.state_dict):
            h.update(k.encode())
            h.update(t.state_dict[k].numpy().tobytes())
    return h.hexdigest()[:16]


def freeze_ls_shadow(store: Path, seeds=SEEDS, widths=WIDTHS,
                     model_id: str | None = None,
                     rd: D.RankerData | None = None) -> dict:
    import torch
    rd = rd or D.load_ranker_data(store)
    shortable = list_um_symbols()
    t_end = rd.dates.max()
    train_end = t_end - pd.Timedelta(days=VAL_DAYS)
    emb = pd.Timedelta(days=EMBARGO_DAYS)
    train_dates = rd.dates[rd.dates <= train_end - emb]
    val_dates = rd.dates[(rd.dates > train_end) & (rd.dates <= t_end - emb)]
    if len(train_dates) < 400 or len(val_dates) < 60:
        raise RuntimeError("insufficient data to freeze the LS shadow model")

    floor = D.fold_floor(rd, train_dates)
    z = D.normalized_returns(rd, floor)
    zrank = D.cross_rank(z, rd.member)
    X_tr, y_tr, _ = D.build_sequences(rd, z, zrank, train_dates, require_label=True)
    X_va, y_va, meta_va = D.build_sequences(rd, z, zrank, val_dates, require_label=False)
    X_tr, stats = D.standardize(X_tr)
    X_va, _ = D.standardize(X_va, stats)

    trained_by_width = _train_all_widths(X_tr, y_tr, X_va, y_va, widths, seeds)
    from research.dl.ranker.model import predict_ensemble
    val_sharpes = {}
    for w in widths:
        p_va = predict_ensemble(trained_by_width[w], X_va)
        score_va = _score_frame(meta_va, p_va, rd.member)
        sim = P.simulate_ls(val_dates[val_dates.isin(score_va.index)], score_va,
                            rd.member, rd.ret_next, rd.cost_rate, shortable)
        val_sharpes[w] = metrics.sharpe(sim.net)
    sel = max(sorted(widths), key=lambda w: (round(val_sharpes[w], 6), -w))
    trained = trained_by_width[sel]

    model_id = model_id or f"ls_shadow_{t_end.date()}"
    d = model_dir(store, model_id)
    d.mkdir(parents=True, exist_ok=True)
    for t in trained:
        torch.save(t.state_dict, d / f"seed_{t.seed}.pt")
    mean, std = stats
    stats_arr = np.concatenate([mean.numpy(), std.numpy()]).astype(np.float64)
    manifest = {
        "book": BOOK, "model_id": model_id, "family": "track3_ls",
        "purpose": ("prospective L/S shadow — NOT a paper book; registered in "
                    "research/reports/track3_ls_shadow_preregistration.md"),
        "schema_hash": schema_hash(),
        "width": sel, "seeds": [t.seed for t in trained],
        "checkpoint_hash": checkpoint_hash(trained),
        "standardize_mean": [float(v) for v in mean.numpy()],
        "standardize_std": [float(v) for v in std.numpy()],
        "stats_hash": hashlib.sha256(stats_arr.tobytes()).hexdigest()[:16],
        "sigma_floor": floor,
        "val_sharpe_by_width": {str(w): round(v, 3) for w, v in val_sharpes.items()},
        "shortable_um": sorted(shortable),
        "frozen_at": str(pd.Timestamp.now(tz="UTC")),
        "train_end": str(train_end), "val_end": str(t_end),
        "retrain_due": str(t_end + pd.Timedelta(days=RETRAIN_DAYS)),
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("frozen %s: width=%d ckpt=%s val_sharpes=%s retrain_due=%s",
             model_id, sel, manifest["checkpoint_hash"],
             manifest["val_sharpe_by_width"], manifest["retrain_due"][:10])
    return manifest


class ArtifactMismatch(RuntimeError):
    """Identity mismatch — fail closed, never run an unverified checkpoint."""


def load_manifest(store: Path, model_id: str) -> dict:
    p = model_dir(store, model_id) / "manifest.json"
    if not p.exists():
        raise ArtifactMismatch(f"missing manifest: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def verify_artifacts(store: Path, model_id: str) -> dict:
    import torch
    man = load_manifest(store, model_id)
    if man.get("schema_hash") != schema_hash():
        raise ArtifactMismatch("ranker schema drift")
    h = hashlib.sha256()
    for seed in man["seeds"]:
        p = model_dir(store, model_id) / f"seed_{seed}.pt"
        if not p.exists():
            raise ArtifactMismatch(f"missing checkpoint {p}")
        state = torch.load(p, map_location="cpu", weights_only=True)
        for k in sorted(state):
            h.update(k.encode())
            h.update(state[k].numpy().tobytes())
    if h.hexdigest()[:16] != man["checkpoint_hash"]:
        raise ArtifactMismatch("checkpoint hash mismatch")
    stats_arr = np.concatenate([np.asarray(man["standardize_mean"], dtype=np.float64),
                                np.asarray(man["standardize_std"], dtype=np.float64)])
    if hashlib.sha256(stats_arr.tobytes()).hexdigest()[:16] != man["stats_hash"]:
        raise ArtifactMismatch("standardize-stats hash mismatch")
    return man
