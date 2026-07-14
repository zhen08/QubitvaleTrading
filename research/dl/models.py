"""Fixed initial architecture (plan §7.3). Changing anything here creates a new
trial family and must be added to the trial ledger.

Crypto sequence branch: causal TCN — 3 residual blocks, 8 filters, kernel 3,
dilations 1/2/4, dropout 0.10, lookback 90.
Cross-asset context branch: 2-layer MLP (16, 8), dropout 0.10, input = latest
cross-asset state + availability masks.
Fusion: concat both embeddings -> shared linear -> two heads
(log-vol regression; tail-probability logit).
"""
from __future__ import annotations

import torch
import torch.nn as nn

LOOKBACK = 90
TCN_CHANNELS = 8
KERNEL = 3
DILATIONS = (1, 2, 4)
DROPOUT = 0.10
CTX_WIDTHS = (16, 8)


class CausalConv1d(nn.Conv1d):
    """Left-padded conv: output[t] depends only on inputs <= t."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__(in_ch, out_ch, kernel, dilation=dilation)
        self._pad = (kernel - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(nn.functional.pad(x, (self._pad, 0)))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, KERNEL, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, KERNEL, dilation)
        self.drop = nn.Dropout(DROPOUT)
        self.act = nn.ReLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.conv1(x)))
        h = self.drop(self.act(self.conv2(h)))
        return self.act(h + self.skip(x))


class CrossAssetTCN(nn.Module):
    """Two-branch multi-task risk forecaster.

    forward(seq [B, LOOKBACK, n_seq], ctx [B, n_ctx]) ->
        (pred_logvol [B], tail_logit [B])
    """

    def __init__(self, n_seq_features: int, n_ctx_features: int):
        super().__init__()
        blocks, in_ch = [], n_seq_features
        for d in DILATIONS:
            blocks.append(ResidualBlock(in_ch, TCN_CHANNELS, d))
            in_ch = TCN_CHANNELS
        self.tcn = nn.Sequential(*blocks)
        w1, w2 = CTX_WIDTHS
        self.ctx = nn.Sequential(
            nn.Linear(n_ctx_features, w1), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(w1, w2), nn.ReLU(), nn.Dropout(DROPOUT))
        fused = TCN_CHANNELS + w2
        self.fuse = nn.Sequential(nn.Linear(fused, fused), nn.ReLU())
        self.head_vol = nn.Linear(fused, 1)
        self.head_tail = nn.Linear(fused, 1)

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor):
        h_seq = self.tcn(seq.transpose(1, 2))[:, :, -1]   # causal: last step only
        h = self.fuse(torch.cat([h_seq, self.ctx(ctx)], dim=1))
        return self.head_vol(h).squeeze(-1), self.head_tail(h).squeeze(-1)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
