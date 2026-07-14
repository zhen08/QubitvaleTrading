"""Deterministic baseline forecasters B1-B3 (plan §4.2).

All three feed the identical §7.5 multiplier mapping. B2 (HAR-RV) is per-fold
pooled OLS with fixed 1/5/22-day lags — no hyperparameter search, matching the
pooled (shared-across-symbols) structure of the TCN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-6
VIX_GATE_QUANTILE = 0.90


def _har_design(df: pd.DataFrame) -> np.ndarray:
    """log daily-RV proxies at 1/5/22 days. |r1| = |c_ret1_n| * sigma20d."""
    r1 = (df["c_ret1_n"] * df["c_sigma20d"]).abs()
    rv5 = df["c_vol5"] / np.sqrt(365)
    rv22 = df["c_vol22"] / np.sqrt(365)
    X = np.column_stack([np.log(r1 + EPS), np.log(rv5 + EPS), np.log(rv22 + EPS)])
    return X


class HARBaseline:
    """B2: pooled log-HAR forecast of label_logvol5, OLS fit on the training fold."""

    def fit(self, train: pd.DataFrame) -> "HARBaseline":
        rows = train.dropna(subset=["label_logvol5", "c_vol5", "c_vol22", "c_sigma20d"])
        X = _har_design(rows)
        y = rows["label_logvol5"].to_numpy()
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        A = np.column_stack([np.ones(ok.sum()), X[ok]])
        self.coef_, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        return self

    def predict_sigma(self, df: pd.DataFrame) -> np.ndarray:
        """sigma_hat in daily units."""
        X = _har_design(df)
        logvol = np.column_stack([np.ones(len(X)), X]) @ self.coef_
        return np.exp(logvol)


def sigma20_daily(df: pd.DataFrame) -> np.ndarray:
    """B1/B3 vol forecast: trailing σ20 in daily units."""
    return df["c_sigma20d"].to_numpy()


def vix_gate(train: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    """B3 deterministic stress gate: 0.5 when VIX log-level exceeds the
    training-fold 90th percentile (rows with m_vix=0 keep gate 1 — unknown
    state must not force exits, §5.5)."""
    ref = train.loc[train["m_vix"] == 1, "x_vix_log"]
    if len(ref) < 100:
        return np.ones(len(df))
    thr = float(ref.quantile(VIX_GATE_QUANTILE))
    gate = np.ones(len(df))
    hot = (df["m_vix"].to_numpy() == 1) & (df["x_vix_log"].to_numpy() > thr)
    gate[hot] = 0.5
    return gate
