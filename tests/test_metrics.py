"""Metrics: Sharpe/DD 基本量、DSR 行为、PBO 行为。"""
import numpy as np
import pandas as pd

from research import metrics


def test_max_drawdown_known():
    r = pd.Series([0.10, -0.50, 0.20])
    assert abs(metrics.max_drawdown(r) + 0.50) < 1e-12


def test_sharpe_annualization():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.001, 0.01, 2000))
    sr_pp = metrics.sharpe_per_period(r)
    assert abs(metrics.sharpe(r, "1d") - sr_pp * np.sqrt(365)) < 1e-9


def test_expected_max_sharpe_grows_with_trials():
    v = 0.01
    e2 = metrics.expected_max_sharpe(2, v)
    e100 = metrics.expected_max_sharpe(100, v)
    assert 0 < e2 < e100


def test_dsr_noise_vs_signal():
    rng = np.random.default_rng(42)
    noise = pd.Series(rng.normal(0.0, 0.01, 1500))
    strong = pd.Series(rng.normal(0.002, 0.01, 1500))
    dsr_noise = metrics.deflated_sharpe(noise, n_trials=32, sr_variance=0.005)
    dsr_strong = metrics.deflated_sharpe(strong, n_trials=1, sr_variance=0.0)
    assert dsr_noise < 0.9
    assert dsr_strong > 0.99


def test_pbo_noise_near_half_dominant_low():
    rng = np.random.default_rng(0)
    noise = pd.DataFrame(rng.normal(0, 0.01, (300, 8)))
    pbo_noise = metrics.pbo_cscv(noise, n_blocks=10)
    assert 0.2 <= pbo_noise <= 0.8

    dominant = noise.copy()
    dominant[0] = rng.normal(0.008, 0.01, 300)  # 一列真实占优
    pbo_dom = metrics.pbo_cscv(dominant, n_blocks=10)
    assert pbo_dom < pbo_noise
    assert pbo_dom < 0.35
