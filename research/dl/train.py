"""Deterministic CPU training of the fixed TCN, 5-seed ensemble (plan §7.4/§10.2).

Loss: huber(pred_logvol, label_logvol) + class-weighted BCE(tail_logit, tail_label)
      + weight decay (AdamW). Not optimized on portfolio Sharpe (§7.4).
All hyperparameters below are ex-ante fixed; changing any creates a new trial.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from research.dl.models import CrossAssetTCN

log = logging.getLogger("qvt.dl.train")

SEEDS = (17, 29, 43, 71, 101)
MAX_EPOCHS = 100
PATIENCE = 10
BATCH = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
N_THREADS = 4


def _determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(N_THREADS)


def _loss(pred_vol, tail_logit, y_vol, y_tail, pos_weight):
    huber = nn.functional.huber_loss(pred_vol, y_vol)
    bce = nn.functional.binary_cross_entropy_with_logits(
        tail_logit, y_tail, pos_weight=pos_weight)
    return huber + bce


@dataclass
class TrainedSeed:
    seed: int
    state_dict: dict
    best_epoch: int
    val_loss: float


def train_seed(seed: int, tensors: dict) -> TrainedSeed:
    _determinism(seed)
    seq_tr, ctx_tr, yv_tr, yt_tr, _ = tensors["train"]
    seq_va, ctx_va, yv_va, yt_va, _ = tensors["val"]
    # rows with NaN labels never appear in train/val (dataset guarantees)
    n_pos = float(yt_tr.sum())
    pos_weight = torch.tensor((len(yt_tr) - n_pos) / max(n_pos, 1.0))
    va_ok = ~torch.isnan(yv_va)

    model = CrossAssetTCN(seq_tr.shape[2], ctx_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n = len(seq_tr)
    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            pv, tl = model(seq_tr[idx], ctx_tr[idx])
            loss = _loss(pv, tl, yv_tr[idx], yt_tr[idx], pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv, tl = model(seq_va, ctx_va)
            vloss = float(_loss(pv[va_ok], tl[va_ok], yv_va[va_ok],
                                yt_va[va_ok], pos_weight))
        if vloss < best_loss - 1e-5:
            best_loss, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return TrainedSeed(seed=seed, state_dict=best_state,
                       best_epoch=best_epoch, val_loss=best_loss)


def predict_ensemble(seeds: list[TrainedSeed], seq, ctx,
                     n_seq: int, n_ctx: int) -> tuple[np.ndarray, np.ndarray]:
    """Arithmetic-mean ensemble (§10.2): mean log-vol; mean tail probability."""
    vols, probs = [], []
    for ts in seeds:
        model = CrossAssetTCN(n_seq, n_ctx)
        model.load_state_dict(ts.state_dict)
        model.eval()
        with torch.no_grad():
            pv, tl = model(seq, ctx)
        vols.append(pv.numpy())
        probs.append(torch.sigmoid(tl).numpy())
    return np.mean(vols, axis=0), np.mean(probs, axis=0)


def checkpoint_hash(seeds: list[TrainedSeed]) -> str:
    h = hashlib.sha256()
    for ts in seeds:
        for k in sorted(ts.state_dict):
            h.update(k.encode())
            h.update(ts.state_dict[k].numpy().tobytes())
    return h.hexdigest()[:16]


def train_fold(tensors: dict, seeds=SEEDS) -> tuple[list[TrainedSeed], dict]:
    trained = [train_seed(s, tensors) for s in seeds]
    diag = {
        "seed_val_losses": {t.seed: round(t.val_loss, 5) for t in trained},
        "seed_best_epochs": {t.seed: t.best_epoch for t in trained},
        "val_loss_dispersion": round(float(np.std([t.val_loss for t in trained])), 5),
        "checkpoint_hash": checkpoint_hash(trained),
    }
    return trained, diag
