"""Artifact-identity fail-closed tests (plan §15.3). Offline."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from features.cross_asset import ALL_FEATURES, schema_hash
from research.dl.models import CrossAssetTCN
from strategies.donchian_tcn_risk_overlay import (ArtifactMismatch,
                                                  OverlayUnavailable,
                                                  cross_asset_status_ok,
                                                  model_dir, verify_artifacts)

MODEL_ID = "testmodel"
SEEDS = [17, 29]


def _write_artifacts(store, tamper: str | None = None):
    d = model_dir(store, MODEL_ID)
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = CrossAssetTCN(18, 39)
        state = model.state_dict()
        torch.save(state, d / f"seed_{seed}.pt")
        for k in sorted(state):
            h.update(k.encode())
            h.update(state[k].numpy().tobytes())
    mean = np.zeros(len(ALL_FEATURES))
    std = np.ones(len(ALL_FEATURES))
    man = {
        "schema_hash": schema_hash() if tamper != "schema" else "deadbeef00000000",
        "checkpoint_hash": h.hexdigest()[:16] if tamper != "ckpt" else "0" * 16,
        "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
        "scaler_hash": hashlib.sha256(
            np.concatenate([mean, std]).tobytes()).hexdigest()[:16]
        if tamper != "scaler" else "1" * 16,
        "seeds": SEEDS, "sigma_ref": 0.02, "sigma_floor": 1e-4,
        "tail_gate_active": True,
    }
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")


def test_valid_artifacts_verify(tmp_path):
    _write_artifacts(tmp_path)
    man = verify_artifacts(tmp_path, MODEL_ID)
    assert man["seeds"] == SEEDS


@pytest.mark.parametrize("tamper", ["schema", "ckpt", "scaler"])
def test_identity_mismatch_fails_closed(tmp_path, tamper):
    _write_artifacts(tmp_path, tamper=tamper)
    with pytest.raises(ArtifactMismatch):
        verify_artifacts(tmp_path, MODEL_ID)


def test_missing_checkpoint_fails_closed(tmp_path):
    _write_artifacts(tmp_path)
    (model_dir(tmp_path, MODEL_ID) / "seed_17.pt").unlink()
    with pytest.raises(ArtifactMismatch):
        verify_artifacts(tmp_path, MODEL_ID)


def test_tampered_weights_fail_closed(tmp_path):
    _write_artifacts(tmp_path)
    p = model_dir(tmp_path, MODEL_ID) / "seed_17.pt"
    state = torch.load(p, weights_only=True)
    k = sorted(state)[0]
    state[k] = state[k] + 1.0
    torch.save(state, p)
    with pytest.raises(ArtifactMismatch):
        verify_artifacts(tmp_path, MODEL_ID)


def test_missing_manifest_fails_closed(tmp_path):
    with pytest.raises(ArtifactMismatch):
        verify_artifacts(tmp_path, MODEL_ID)


def test_status_gate(tmp_path):
    assert not cross_asset_status_ok(tmp_path)          # missing -> not ok
    p = tmp_path / "cross_asset" / "status.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"ok": False}))
    assert not cross_asset_status_ok(tmp_path)
    p.write_text(json.dumps({"ok": True}))
    assert cross_asset_status_ok(tmp_path)
    p.write_text("{corrupt")
    assert not cross_asset_status_ok(tmp_path)          # corrupt -> fail closed
