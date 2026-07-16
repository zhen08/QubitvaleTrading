"""GRU ranker + deterministic training (§3 of the preregistration).

One shared 1-layer GRU, widths from {5, 10, 20}; sigmoid head on the last
hidden state; BCE loss; AdamW lr 1e-3 wd 1e-4, batch 512, max 100 epochs,
patience 10 on validation loss, grad clip 1.0; seeds 17/29/43/71/101,
arithmetic-mean ensemble, CPU-deterministic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("qvt.ranker.model")

WIDTHS = (5, 10, 20)
SEEDS = (17, 29, 43, 71, 101)
MAX_EPOCHS = 100
PATIENCE = 10
BATCH = 512
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
N_THREADS = 4


class GRURanker(nn.Module):
    def __init__(self, hidden: int, n_features: int = 2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)      # logits


def _determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(N_THREADS)


@dataclass
class TrainedRanker:
    seed: int
    width: int
    state_dict: dict
    val_loss: float
    best_epoch: int


def train_seed(seed: int, width: int, X_tr, y_tr, X_va, y_va,
               max_epochs: int | None = None) -> TrainedRanker:
    _determinism(seed)
    model = GRURanker(width)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.functional.binary_cross_entropy_with_logits
    va_ok = torch.isfinite(y_va)
    n = len(X_tr)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    for epoch in range(max_epochs if max_epochs is not None else MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = bce(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(bce(model(X_va[va_ok]), y_va[va_ok]))
        if vloss < best - 1e-5:
            best, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return TrainedRanker(seed=seed, width=width, state_dict=best_state,
                         val_loss=best, best_epoch=best_epoch)


def predict_ensemble(trained: list[TrainedRanker], X) -> np.ndarray:
    probs = []
    for tr in trained:
        model = GRURanker(tr.width)
        model.load_state_dict(tr.state_dict)
        model.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 4096):
                out.append(torch.sigmoid(model(X[i:i + 4096])))
            probs.append(torch.cat(out).numpy())
    return np.mean(probs, axis=0)
