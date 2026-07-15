"""Freeze a shadow-deployment artifact set for the prospective E2 test.

Protocol (identical to the walk-forward fold construction, §10.1): expanding
training window ending 182 days ago, the last 182 days as validation (early
stopping + tail calibration), 5-day label embargo at the boundary. No test
window — the test is the FUTURE. Retrain due 182 days after freezing; each
refreeze appends to the trial ledger.

Artifacts follow the strategies/donchian_tcn_risk_overlay contract exactly,
so the daily shadow step reuses its fail-closed identity verification.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research.dl.baselines import HARBaseline
from research.dl.dataset import build_fold, tail_floor
from research.dl.overlay import fit_tail_calibrator, sigma_ref_from_train
from research.dl.train import SEEDS, checkpoint_hash, predict_ensemble, train_fold
from strategies.donchian_tcn_risk_overlay import BOOK, model_dir

log = logging.getLogger("qvt.dl.freeze")

VAL_DAYS = 182
RETRAIN_DAYS = 182


def freeze_shadow(store: Path, table: pd.DataFrame, variant: str = "E2",
                  seeds=SEEDS, model_id: str | None = None) -> dict:
    """Train the frozen protocol on all available data and persist artifacts."""
    import torch
    from sklearn.isotonic import IsotonicRegression

    t_end = table["decision_ts"].max()
    train_end = t_end - pd.Timedelta(days=VAL_DAYS)
    fd = build_fold(table, variant, train_end, t_end, t_end, require_test=False)
    if fd is None:
        raise RuntimeError("insufficient data to freeze the shadow model")

    trained, diag = train_fold(fd.tensors, seeds)
    n_seq = fd.tensors["train"][0].shape[2]
    n_ctx = fd.tensors["train"][1].shape[1]

    seq_va, ctx_va, _, yt_va, _ = fd.tensors["val"]
    _, p_va = predict_ensemble(trained, seq_va, ctx_va, n_seq, n_ctx)
    calibrate, n_pos = fit_tail_calibrator(p_va, yt_va.numpy())
    calibration = None
    if calibrate is not None:
        # persist the isotonic map as breakpoints (np.interp reconstructs it)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_va[~np.isnan(yt_va.numpy())], yt_va.numpy()[~np.isnan(yt_va.numpy())])
        calibration = {"x": [float(v) for v in iso.X_thresholds_],
                       "y": [float(v) for v in iso.y_thresholds_]}

    har = HARBaseline().fit(fd.train)

    model_id = model_id or f"{variant.lower()}_shadow_{t_end.date()}"
    d = model_dir(store, model_id)
    d.mkdir(parents=True, exist_ok=True)
    for ts in trained:
        torch.save(ts.state_dict, d / f"seed_{ts.seed}.pt")

    scaler_mean = fd.scaler.mean.to_numpy()
    scaler_std = fd.scaler.std.to_numpy()
    from features.cross_asset import schema_hash
    manifest = {
        "book": BOOK, "model_id": model_id, "variant": variant,
        "purpose": ("prospective shadow test — NOT a paper book; registered in "
                    "research/reports/dl_trial_ledger.md"),
        "schema_hash": schema_hash(),
        "seeds": [ts.seed for ts in trained],
        "checkpoint_hash": checkpoint_hash(trained),
        "scaler_mean": [float(v) for v in scaler_mean],
        "scaler_std": [float(v) for v in scaler_std],
        "scaler_hash": hashlib.sha256(
            np.concatenate([scaler_mean, scaler_std]).tobytes()).hexdigest()[:16],
        "sigma_ref": sigma_ref_from_train(fd.train),
        "sigma_floor": tail_floor(fd.train),
        "tail_gate_active": calibrate is not None,
        "val_tail_positives": n_pos,
        "calibration": calibration,
        "har_coef": [float(v) for v in har.coef_],   # B2 reference forecaster
        "frozen_at": str(pd.Timestamp.now(tz="UTC")),
        "train_end": str(train_end), "val_end": str(t_end),
        "retrain_due": str(t_end + pd.Timedelta(days=RETRAIN_DAYS)),
        "train_diag": diag,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("frozen %s: ckpt=%s val_pos=%d retrain_due=%s",
             model_id, manifest["checkpoint_hash"], n_pos, manifest["retrain_due"])
    return manifest
