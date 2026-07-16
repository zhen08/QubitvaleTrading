"""Ranker data layer: wide matrices, membership, labels, fold sequences (§1-§3).

All state is (date × symbol) wide DataFrames built once from the universe
panel and the per-symbol 1d parquets; folds slice them by date. Causality:
every quantity at date D uses bars ≤ D; labels look exactly one day forward.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data import storeio
from data.collectors.universe import universe_panel_path

log = logging.getLogger("qvt.ranker.data")

DATA_START = "2020-07-01"        # §2
ADV_FLOOR_USD = 5e6              # §1.2
TOP_RANK = 50                    # §1.3
MIN_HISTORY_BARS = 110           # §1.4
MIN_MEMBERS = 30                 # §1 (usable date)
SEQ_LEN = 90                     # §3
LABEL_HORIZON = 1                # §3 (next-day vs universe median)

# fee/slippage per §4: repo spot cost model constants (SPOT_TAKER equivalents)
FEE_BPS = 10.0
SLIP_FLOOR_BPS = 1.0
IMPACT_Y = 1.0
NOTIONAL_USD = 10_000.0


@dataclass
class RankerData:
    ret: pd.DataFrame          # daily log return at D (date × symbol)
    sigma20: pd.DataFrame      # trailing 20d std of ret, at D
    member: pd.DataFrame       # §1 membership at D (bool)
    ret_next: pd.DataFrame     # simple return D -> D+1 (evaluation stream)
    cost_rate: pd.DataFrame    # one-way cost per unit turnover, decided at D
    mom21: pd.DataFrame        # trailing 21d log return (R1 score)
    label: pd.DataFrame        # 1.0 if next-day return > member median, NaN w/o label
    dates: pd.DatetimeIndex    # usable dates (>= MIN_MEMBERS members)


def load_ranker_data(store: Path) -> RankerData:
    panel = pd.read_parquet(universe_panel_path(store))
    panel = panel[panel["date"] >= pd.Timestamp(DATA_START, tz="UTC")]
    symbols = sorted(panel["symbol"].unique())

    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    adv30 = panel.pivot(index="date", columns="symbol", values="adv30").sort_index()
    dvol = panel.pivot(index="date", columns="symbol", values="dollar_vol").sort_index()

    # bar counts need pre-panel history (panel rows start ~bar 30): count real
    # bars per symbol from the store once, aligned to panel dates.
    counts = {}
    for sym in symbols:
        df = storeio.read_parquet_if_exists(storeio.klines_path(store, "spot", sym, "1d"))
        if df is None:
            continue
        ts = pd.to_datetime(df["ts"], utc=True).dt.normalize()
        counts[sym] = pd.Series(np.arange(1, len(ts) + 1), index=ts)
    bars = pd.DataFrame(counts).reindex(close.index).ffill()

    lret = np.log(close).diff()
    sigma20 = lret.rolling(20).std(ddof=1)
    mom21 = lret.rolling(21).sum()

    rank = adv30.rank(axis=1, ascending=False, method="first")
    member = ((adv30 >= ADV_FLOOR_USD) & (rank <= TOP_RANK)
              & (bars >= MIN_HISTORY_BARS) & close.notna())

    n_members = member.sum(axis=1)
    usable = n_members[n_members >= MIN_MEMBERS].index
    member.loc[~member.index.isin(usable)] = False

    ret_next = (close.shift(-1) / close - 1.0)

    # §3 label: next-day return above the point-in-time member median
    masked_next = ret_next.where(member)
    med = masked_next.median(axis=1)
    label = (masked_next.gt(med, axis=0)).astype(float).where(masked_next.notna())
    label = label.where(member)

    # one-way cost per unit turnover, decided at D (causal inputs)
    slip_bps = (IMPACT_Y * sigma20 * np.sqrt(NOTIONAL_USD / dvol.replace(0, np.nan))
                * 1e4).clip(lower=SLIP_FLOOR_BPS)
    cost_rate = (FEE_BPS + slip_bps.fillna(SLIP_FLOOR_BPS)) / 1e4

    log.info("ranker data: %d dates (%d usable), %d symbols, member median %d",
             len(close), len(usable), len(symbols), int(n_members.reindex(usable).median()))
    return RankerData(ret=lret, sigma20=sigma20, member=member, ret_next=ret_next,
                      cost_rate=cost_rate, mom21=mom21, label=label,
                      dates=pd.DatetimeIndex(usable))


def normalized_returns(rd: RankerData, sigma_floor: float) -> pd.DataFrame:
    """Channel 1: r / max(σ20, fold floor)."""
    return rd.ret / rd.sigma20.clip(lower=sigma_floor)


def cross_rank(z: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """Channel 2: per-day pct rank of channel 1 among that day's members."""
    return z.where(member).rank(axis=1, pct=True)


def fold_floor(rd: RankerData, train_dates: pd.DatetimeIndex) -> float:
    s = rd.sigma20.loc[rd.sigma20.index.isin(train_dates)]
    pos = s.to_numpy().ravel()
    pos = pos[np.isfinite(pos) & (pos > 0)]
    return float(np.quantile(pos, 0.05)) if len(pos) else 1e-4


def build_sequences(rd: RankerData, z: pd.DataFrame, zrank: pd.DataFrame,
                    dates: pd.DatetimeIndex, require_label: bool = True):
    """Tensor rows for member cells on `dates`: (X [N, SEQ_LEN, 2], y [N], meta).

    Sequence windows may reach back before `dates` (normal information set);
    a row is dropped if its 90-day window has any NaN in channel 1.
    Fully vectorized: per-cell pandas access here once cost ~40 min per fold.
    """
    import torch
    all_dates = z.index
    Z = z.to_numpy(dtype=np.float32)
    R = np.nan_to_num(zrank.to_numpy(dtype=np.float32), nan=0.5)
    M = rd.member.reindex(index=all_dates, columns=z.columns).fillna(False).to_numpy(bool)
    Y = rd.label.reindex(index=all_dates, columns=z.columns).to_numpy(np.float32)
    # valid 90-day channel-1 window ending at i, per column
    finite = np.isfinite(Z)
    csum = np.cumsum(finite, axis=0)
    full = np.zeros_like(finite)
    full[SEQ_LEN - 1:] = (csum[SEQ_LEN - 1:]
                          - np.vstack([np.zeros((1, Z.shape[1]), dtype=int),
                                       csum[:-SEQ_LEN]])) == SEQ_LEN

    date_mask = all_dates.isin(dates)
    ok = M & full & date_mask[:, None]
    if require_label:
        ok &= np.isfinite(Y)
    ii, jj = np.where(ok)
    if not len(ii):
        return None
    # gather windows: sliding_window_view over axis 0, then fancy-index rows
    zw = np.lib.stride_tricks.sliding_window_view(Z, SEQ_LEN, axis=0)  # [T-89, C, 90]
    rw = np.lib.stride_tricks.sliding_window_view(R, SEQ_LEN, axis=0)
    sel = ii - (SEQ_LEN - 1)
    X = np.stack([zw[sel, jj], rw[sel, jj]], axis=2).astype(np.float32)  # [N, 90, 2]
    y = Y[ii, jj]
    meta = pd.MultiIndex.from_arrays(
        [all_dates[ii], z.columns.to_numpy()[jj]], names=["date", "symbol"])
    return torch.from_numpy(X.copy()), torch.tensor(y), meta


def standardize(X, stats=None):
    """Standardize both channels with training-fold statistics (§3)."""
    import torch
    if stats is None:
        mean = X.mean(dim=(0, 1))
        std = X.std(dim=(0, 1)).clamp(min=1e-6)
        stats = (mean, std)
    mean, std = stats
    return ((X - mean) / std).clamp(-8, 8), stats
