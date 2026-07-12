"""黄金测试：生产信号 ≡ 研究引擎语义（同数据同权重、决策日 shift 正确）。"""
import numpy as np
import pandas as pd

from research.strategies import GRIDS, donchian
from strategies.donchian_ensemble import SYMBOL_WEIGHT, compute_weights, targets_for_day


def _df(n=400, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))), index=idx)
    return pd.DataFrame({"ts": idx, "close": close.to_numpy(),
                         "open": close.to_numpy(), "high": close.to_numpy() * 1.01,
                         "low": close.to_numpy() * 0.99, "quote_volume": 1e9})


def test_weights_equal_research_ensemble():
    dfs = {"BTCUSDT": _df(seed=3), "ETHUSDT": _df(seed=4), "SOLUSDT": _df(seed=5)}
    weights = compute_weights(dfs)
    for sym, df in dfs.items():
        expected = pd.concat(
            [donchian(df, **p) for p in GRIDS["donchian"]], axis=1
        ).mean(axis=1) * SYMBOL_WEIGHT
        expected.index = expected.index.normalize()
        pd.testing.assert_series_equal(weights[sym], expected.fillna(0.0),
                                       check_names=False)


def test_weights_are_quantized_and_bounded():
    dfs = {"BTCUSDT": _df(seed=6)}
    w = compute_weights(dfs)["BTCUSDT"]
    allowed = {round(k / 4 * SYMBOL_WEIGHT, 10) for k in range(5)}
    assert set(round(v, 10) for v in w.unique()).issubset(allowed)
    assert w.max() <= SYMBOL_WEIGHT + 1e-12


def test_targets_for_day_uses_previous_decision():
    dfs = {"BTCUSDT": _df(seed=7)}
    w = compute_weights(dfs)
    day = w.index[-1] + pd.Timedelta(days=1)           # 明天生效 = 今天(最后决策日)的值
    t = targets_for_day(w, day)
    assert t["BTCUSDT"] == float(w["BTCUSDT"].iloc[-1])
    t2 = targets_for_day(w, w.index[-1])               # 最后决策日当天 → 用前一天
    assert t2["BTCUSDT"] == float(w["BTCUSDT"].iloc[-2])
