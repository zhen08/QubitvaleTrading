"""Shared §7.5 risk-multiplier mapping + validation-fold tail calibration.

The SAME mapping serves B1/B2/B3 and E1-E4 so every comparison isolates the
forecaster, not the decision rule:

    vol_component  = clip(sigma_ref / sigma_hat, 0, 1)
    risk_multiplier = vol_component * tail_gate

sigma_ref  = training-fold median of realized 5-day vol (daily units), fixed
             before the test window opens.
tail_gate  on calibrated tail probability: 1.0 below 0.25; 0.5 in [0.25, 0.5);
             0.0 at/above 0.5. Variants without a tail head use gate = 1.0.
Fold fallback (§7.2): fewer than 10 positive tail events in validation ->
tail head reported "insufficient events" and gate forced to 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_VAL_POSITIVES = 10
GATE_HALF_P = 0.25
GATE_ZERO_P = 0.50


def sigma_ref_from_train(train: pd.DataFrame) -> float:
    """Training-fold median realized 5-day vol (daily units, from the label)."""
    rv = np.exp(train["label_logvol5"].dropna())
    return float(rv.median())


def fit_tail_calibrator(p_val: np.ndarray, y_val: np.ndarray):
    """Isotonic calibration on the validation fold (choice registered ex ante).

    Returns (calibrate_fn, n_positives). With < MIN_VAL_POSITIVES positive
    events the calibrator is None and the tail gate must fall back to 1.0.
    """
    ok = ~np.isnan(y_val)
    p_val, y_val = p_val[ok], y_val[ok]
    n_pos = int(y_val.sum())
    if n_pos < MIN_VAL_POSITIVES:
        return None, n_pos
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, y_val)
    return iso.predict, n_pos


def tail_gate(p_calibrated: np.ndarray | None) -> np.ndarray | float:
    if p_calibrated is None:
        return 1.0
    gate = np.ones_like(p_calibrated)
    gate[p_calibrated >= GATE_HALF_P] = 0.5
    gate[p_calibrated >= GATE_ZERO_P] = 0.0
    return gate


def risk_multiplier(sigma_hat: np.ndarray, sigma_ref: float,
                    p_tail_calibrated: np.ndarray | None = None) -> np.ndarray:
    vol_component = np.clip(sigma_ref / np.maximum(sigma_hat, 1e-8), 0.0, 1.0)
    return vol_component * tail_gate(p_tail_calibrated)
