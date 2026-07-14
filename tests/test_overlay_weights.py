"""Overlay-weight invariants (plan §15.2/§7.1/§7.5). Offline."""
import numpy as np
import pandas as pd

from research.dl.overlay import (fit_tail_calibrator, risk_multiplier,
                                 sigma_ref_from_train, tail_gate)
from research.dl.evaluation import apply_threshold


def test_multiplier_bounded_0_1():
    sigma_hat = np.array([0.001, 0.01, 0.05, 1.0, 1e-12])
    m = risk_multiplier(sigma_hat, sigma_ref=0.02)
    assert (m >= 0).all() and (m <= 1).all()
    assert m[0] == 1.0            # calm forecast -> full weight
    assert m[3] < 0.05            # extreme forecast -> heavy cut


def test_overlay_never_exceeds_base():
    base = pd.Series([0.0, 0.25, 0.5, 1.0])
    mult = pd.Series([1.0, 0.5, 1.0, 0.7])
    final = (base * mult).clip(0.0, 1.0)
    assert (final <= base + 1e-12).all()          # §7.1: reduce-only
    assert final.iloc[0] == 0.0                   # flat base stays flat


def test_tail_gate_levels():
    p = np.array([0.0, 0.24, 0.25, 0.49, 0.5, 0.9])
    g = tail_gate(p)
    assert list(g) == [1.0, 1.0, 0.5, 0.5, 0.0, 0.0]


def test_tail_gate_fallback_without_calibrator():
    assert tail_gate(None) == 1.0


def test_calibrator_fallback_on_scarce_positives():
    p = np.linspace(0, 1, 50)
    y = np.zeros(50)
    y[:3] = 1.0                                   # only 3 positives < 10
    fn, n_pos = fit_tail_calibrator(p, y)
    assert fn is None and n_pos == 3


def test_calibrator_fits_with_enough_positives():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)       # well-calibrated by design
    fn, n_pos = fit_tail_calibrator(p, y)
    assert fn is not None and n_pos >= 10
    out = fn(np.array([0.1, 0.9]))
    assert out[1] > out[0]


def test_sigma_ref_is_train_median():
    train = pd.DataFrame({"label_logvol5": np.log([0.01, 0.02, 0.03])})
    assert abs(sigma_ref_from_train(train) - 0.02) < 1e-12


def test_rebalance_threshold_hysteresis():
    target = pd.Series([0.0, 0.01, 0.015, 0.5, 0.505, 0.49, 0.3])
    held = apply_threshold(target, 0.02)
    # small wiggles never trade; big moves do
    assert list(held) == [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.3]
