"""Fold dataset builder: ablation masks, fold-local scaling, sequences, embargo.

Ablation (plan §4.2): E-variants zero out feature groups *by construction* and
zero the matching availability mask, so an ablated model cannot distinguish
"feature removed" from "data invalid" — exactly the semantics at inference time.

Scaling (plan §6.1): means/stds are fit on training rows only, and for group
features only on rows where the group's mask is 1; masked entries stay exactly
zero after scaling.

Tail label (plan §7.2): `label_minz5` was stored σ20-normalized without a floor;
the training fold's 5th percentile of positive σ20 rescales it here
(`minz * σ20/max(σ20, floor)`) before thresholding at −2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from features.cross_asset import (ALL_FEATURES, CRYPTO_FEATURES, EQUITY_FEATURES,
                                  GLD_FEATURES, MASKS, VIX_FEATURES)
from research.dl.models import LOOKBACK

TAIL_Z = -2.0

# E-family registration (§4.2). Groups listed here are *kept*.
VARIANTS: dict[str, dict] = {
    "E1": {"equity": False, "vix": False, "gld": False},
    "E2": {"equity": True, "vix": False, "gld": False},
    "E3": {"equity": True, "vix": True, "gld": False},
    "E4": {"equity": True, "vix": True, "gld": True},
}
GROUP_FEATURES = {"equity": EQUITY_FEATURES, "vix": VIX_FEATURES, "gld": GLD_FEATURES}
GROUP_MASK = {"equity": "m_equity", "vix": "m_vix", "gld": "m_gld"}

SEQ_FEATURES = CRYPTO_FEATURES                     # per-day crypto sequence branch
CTX_FEATURES = EQUITY_FEATURES + VIX_FEATURES + GLD_FEATURES + MASKS  # context branch


@dataclass
class Scaler:
    """Per-feature mean/std fit on training rows (mask-aware). Deterministic."""
    mean: pd.Series
    std: pd.Series

    def transform(self, df: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
        out = (df - self.mean) / self.std
        out = out.clip(-8, 8).fillna(0.0)
        # re-zero masked group entries after scaling
        for group, cols in GROUP_FEATURES.items():
            m = masks[GROUP_MASK[group]] == 0
            out.loc[m, [c for c in cols if c in out.columns]] = 0.0
        return out

    def hash(self) -> str:
        import hashlib
        payload = np.concatenate([self.mean.to_numpy(), self.std.to_numpy()])
        return hashlib.sha256(payload.tobytes()).hexdigest()[:16]


def fit_scaler(train: pd.DataFrame) -> Scaler:
    cols = ALL_FEATURES
    mean = pd.Series(0.0, index=cols)
    std = pd.Series(1.0, index=cols)
    grouped = {c: g for g, cs in GROUP_FEATURES.items() for c in cs}
    for c in cols:
        rows = train
        if c in grouped:
            rows = train[train[GROUP_MASK[grouped[c]]] == 1]
        v = rows[c].dropna()
        if len(v) > 20:
            mean[c] = float(v.mean())
            s = float(v.std(ddof=1))
            std[c] = s if s > 1e-9 else 1.0
    return Scaler(mean=mean, std=std)


def apply_variant(table: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Zero out ablated groups and their masks (unavailable by construction)."""
    cfg = VARIANTS[variant]
    out = table.copy()
    for group, keep in cfg.items():
        if not keep:
            out[GROUP_FEATURES[group]] = 0.0
            out[GROUP_MASK[group]] = 0.0
    return out


def tail_floor(train: pd.DataFrame) -> float:
    pos = train["c_sigma20d"][train["c_sigma20d"] > 0]
    return float(pos.quantile(0.05)) if len(pos) else 1e-4


def tail_label(df: pd.DataFrame, floor: float) -> pd.Series:
    scale = df["c_sigma20d"] / df["c_sigma20d"].clip(lower=floor)
    return ((df["label_minz5"] * scale) <= TAIL_Z).astype(float)


@dataclass
class FoldData:
    variant: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scaler: Scaler
    sigma_floor: float
    # tensors: dict split -> (seq [N,L,F], ctx [N,C], y_vol [N], y_tail [N], meta index)
    tensors: dict = field(default_factory=dict)


def _sequences(hist: pd.DataFrame, scaled: pd.DataFrame, floor: float,
               allow_nan_labels: bool = False):
    """Per-symbol windows of SEQ_FEATURES + latest CTX row + labels.

    Positional (integer) alignment throughout: decision_ts is a non-unique index
    (all symbols share each timestamp), so label-based alignment is unsafe.
    """
    import torch
    h = hist.reset_index()
    s = scaled.reset_index(drop=True)
    seqs, ctxs, y_vol, y_tail, meta = [], [], [], [], []
    for sym in sorted(h["symbol"].unique()):
        rows = h.index[h["symbol"] == sym]
        order = h.loc[rows].sort_values("decision_ts").index
        X = s.loc[order, SEQ_FEATURES].to_numpy(dtype=np.float32)
        C = s.loc[order, CTX_FEATURES].to_numpy(dtype=np.float32)
        graw = h.loc[order]
        yv = graw["label_logvol5"].to_numpy(dtype=np.float32)
        yt = tail_label(graw, floor).to_numpy(dtype=np.float32)
        ok = ~np.isnan(X).any(axis=1)
        ts = graw["decision_ts"].to_numpy()
        for i in range(LOOKBACK - 1, len(order)):
            if not ok[i - LOOKBACK + 1:i + 1].all():
                continue
            if np.isnan(yv[i]) and not allow_nan_labels:
                continue
            seqs.append(X[i - LOOKBACK + 1:i + 1])
            ctxs.append(C[i])
            y_vol.append(yv[i])
            y_tail.append(yt[i])
            meta.append((ts[i], sym))
    if not seqs:
        return None
    return (torch.from_numpy(np.stack(seqs)), torch.from_numpy(np.stack(ctxs)),
            torch.tensor(np.asarray(y_vol)), torch.tensor(np.asarray(y_tail)),
            pd.MultiIndex.from_tuples(meta, names=["decision_ts", "symbol"]))


def build_fold(table: pd.DataFrame, variant: str,
               train_end: pd.Timestamp, val_end: pd.Timestamp,
               test_end: pd.Timestamp, embargo_days: int = 5,
               require_test: bool = True) -> FoldData | None:
    """Expanding-window fold with a label-horizon embargo at every boundary.

    A row's label consumes bars up to decision_ts+5d, so training rows must end
    `embargo_days` before the validation window opens (and likewise val/test).
    require_test=False is the freeze path (train+val only, deploy forward).
    """
    t = apply_variant(table, variant).set_index("decision_ts")
    emb = pd.Timedelta(days=embargo_days)
    train = t[t.index <= train_end - emb]
    val = t[(t.index > train_end) & (t.index <= val_end - emb)]
    test = t[(t.index > val_end) & (t.index <= test_end)]
    if len(train) < 500 or len(val) < 100 or (require_test and len(test) < 50):
        return None
    # sequences may reach back across the boundary for *inputs* (normal
    # information set); labels are what the embargo protects.
    scaler = fit_scaler(train)
    floor = tail_floor(train)
    fd = FoldData(variant=variant, train=train, val=val, test=test,
                  scaler=scaler, sigma_floor=floor)
    for name, split_end, split in [("train", train_end - emb, train),
                                   ("val", val_end - emb, val),
                                   ("test", test_end, test)]:
        if name == "test" and not require_test and not len(split):
            fd.tensors["test"] = None
            continue
        hist = t[t.index <= split_end]            # inputs may include earlier bars
        scaled = scaler.transform(hist[ALL_FEATURES], hist[MASKS])
        scaled[MASKS] = hist[MASKS].to_numpy()    # masks enter the context branch unscaled
        want = split.index
        # test rows near the data tail have undefined labels but still need
        # predictions for the economic overlay
        tensors = _sequences(hist, scaled, floor, allow_nan_labels=(name == "test"))
        if tensors is None:
            return None
        seq, ctx, yv, yt, meta = tensors
        keep = meta.get_level_values("decision_ts").isin(want)
        fd.tensors[name] = (seq[keep], ctx[keep], yv[keep], yt[keep], meta[keep])
    return fd
