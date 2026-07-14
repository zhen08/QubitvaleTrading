"""donchian_tcn_risk_overlay — Donchian ensemble × TCN risk multiplier (plan §12).

**NOT REGISTERED. DO NOT ADD TO config/settings.yaml → paper.books OR
strategies/registry.py UNTIL THE §11.1 RESEARCH GATE PASSES.** This module
exists so the artifact-identity and fail-closed machinery is testable before
any promotion decision; registering it early would put an ungated model in
the multi-book selection space (registry header discipline).

Artifact contract (data/store/models/donchian_tcn_risk_overlay/<model_id>/):
  manifest.json   schema_hash, scaler mean/std + hash, sigma_ref, sigma_floor,
                  seeds, checkpoint_hash, variant, thresholds, frozen dates
  seed_<n>.pt     one state_dict per registered seed

Fail-closed semantics (plan §12.3/§17):
  - any artifact-identity mismatch raises ArtifactMismatch (P1): the caller
    must BLOCK exposure increases and may still reduce; never load an
    unverified checkpoint, never fall back to a different model;
  - stale/failed cross-asset data likewise degrades to OverlayUnavailable —
    the base Donchian book is unaffected (multiplier is simply not applied to
    *increases*), and missing data is never a liquidation signal.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from features.cross_asset import schema_hash

log = logging.getLogger("qvt.signal")

BOOK = "donchian_tcn_risk_overlay"


class ArtifactMismatch(RuntimeError):
    """Checkpoint/scaler/schema identity mismatch — P1, block adds."""


class OverlayUnavailable(RuntimeError):
    """Inference inputs unavailable/stale — block adds, allow reductions."""


def model_dir(store: Path, model_id: str) -> Path:
    return store / "models" / BOOK / model_id


def load_manifest(store: Path, model_id: str) -> dict:
    p = model_dir(store, model_id) / "manifest.json"
    if not p.exists():
        raise ArtifactMismatch(f"missing manifest: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def verify_artifacts(store: Path, model_id: str) -> dict:
    """Recompute every identity in the manifest; raise ArtifactMismatch on any drift."""
    import torch
    man = load_manifest(store, model_id)
    if man.get("schema_hash") != schema_hash():
        raise ArtifactMismatch(
            f"feature schema drift: manifest {man.get('schema_hash')} "
            f"!= code {schema_hash()}")
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
    scaler = np.concatenate([np.asarray(man["scaler_mean"], dtype=float),
                             np.asarray(man["scaler_std"], dtype=float)])
    if hashlib.sha256(scaler.tobytes()).hexdigest()[:16] != man["scaler_hash"]:
        raise ArtifactMismatch("scaler hash mismatch")
    return man


def cross_asset_status_ok(store: Path) -> bool:
    p = store / "cross_asset" / "status.json"
    if not p.exists():
        return False
    try:
        return bool(json.loads(p.read_text(encoding="utf-8")).get("ok"))
    except (ValueError, OSError):
        return False


def compute_risk_multiplier(store: Path, model_id: str,
                            feature_table: pd.DataFrame) -> pd.DataFrame:
    """Latest-row risk multiplier per symbol from the frozen ensemble.

    Raises OverlayUnavailable when the cross-asset QC status is not OK — the
    caller must treat the risk state as unknown (block adds, allow reductions).
    """
    import torch

    from research.dl.dataset import CTX_FEATURES, GROUP_FEATURES, GROUP_MASK, SEQ_FEATURES
    from research.dl.models import LOOKBACK, CrossAssetTCN
    from research.dl.overlay import risk_multiplier
    from features.cross_asset import ALL_FEATURES, MASKS

    man = verify_artifacts(store, model_id)
    if not cross_asset_status_ok(store):
        raise OverlayUnavailable("cross-asset status not OK — risk state unknown")

    mean = pd.Series(man["scaler_mean"], index=ALL_FEATURES)
    std = pd.Series(man["scaler_std"], index=ALL_FEATURES)
    rows = []
    for sym, g in feature_table.groupby("symbol"):
        g = g.sort_values("decision_ts").tail(LOOKBACK)
        if len(g) < LOOKBACK:
            raise OverlayUnavailable(f"{sym}: only {len(g)} rows for {LOOKBACK}-day lookback")
        scaled = ((g[ALL_FEATURES] - mean) / std).clip(-8, 8).fillna(0.0)
        for group, cols in GROUP_FEATURES.items():
            m = (g[GROUP_MASK[group]] == 0).to_numpy()
            scaled.loc[m, cols] = 0.0
        scaled[MASKS] = g[MASKS].to_numpy()
        seq = torch.tensor(scaled[SEQ_FEATURES].to_numpy(dtype=np.float32)).unsqueeze(0)
        ctx = torch.tensor(scaled[CTX_FEATURES].to_numpy(dtype=np.float32))[-1:].clone()

        vols, probs = [], []
        for seed in man["seeds"]:
            model = CrossAssetTCN(seq.shape[2], ctx.shape[1])
            state = torch.load(model_dir(store, model_id) / f"seed_{seed}.pt",
                               map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                pv, tl = model(seq, ctx)
            vols.append(float(pv))
            probs.append(float(torch.sigmoid(tl)))
        sigma_hat = float(np.exp(np.mean(vols)))
        p_tail = float(np.mean(probs))
        mult = float(risk_multiplier(np.array([sigma_hat]), man["sigma_ref"],
                                     np.array([p_tail]) if man.get("tail_gate_active")
                                     else None)[0])
        rows.append({"symbol": sym, "decision_ts": g["decision_ts"].iloc[-1],
                     "sigma_hat": sigma_hat, "p_tail": p_tail,
                     "multiplier": mult, "checkpoint_hash": man["checkpoint_hash"]})
    return pd.DataFrame(rows)
