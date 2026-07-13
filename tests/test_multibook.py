"""多账本支架（2026-07-13）：注册表、tsmom 集成一致性、账本命名空间隔离。"""
import numpy as np
import pandas as pd

from execution.paper.ledger import Ledger
from research.strategies import GRIDS, tsmom
from strategies import tsmom_ensemble
from strategies.registry import STRATEGIES, get_strategy

TS = pd.Timestamp("2026-07-13T00:10:00Z")


def _df(n=400, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.03, n))), index=idx)
    return pd.DataFrame({"ts": idx, "close": close.to_numpy(), "open": close.to_numpy(),
                         "high": close.to_numpy() * 1.01, "low": close.to_numpy() * 0.99,
                         "quote_volume": 1e9})


def test_registry_resolves_both_books():
    assert set(STRATEGIES) == {"donchian_ensemble", "tsmom_ensemble"}
    for name in STRATEGIES:
        assert callable(get_strategy(name).compute_weights)


def test_tsmom_weights_match_research_and_bounded():
    dfs = {"BTCUSDT": _df(seed=9), "ETHUSDT": _df(seed=10)}
    w = tsmom_ensemble.compute_weights(dfs)
    for sym, df in dfs.items():
        expected = pd.concat(
            [tsmom(df, **p, long_short=False, max_lev=1.0) for p in GRIDS["tsmom"]],
            axis=1).mean(axis=1) * tsmom_ensemble.SYMBOL_WEIGHT
        expected.index = expected.index.normalize()
        pd.testing.assert_series_equal(w[sym], expected, check_names=False)
    valid = w.stack().dropna()
    assert (valid >= -1e-12).all() and (valid <= 1 / 3 + 1e-9).all()   # 长平、≤1/3


def test_tsmom_missing_dates_stay_nan():
    idx = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    dfs = {"BTCUSDT": _df(300), "SOLUSDT": _df(299)}                    # SOL 少一天
    dfs["SOLUSDT"]["ts"] = idx[:-1]
    w = tsmom_ensemble.compute_weights(dfs)
    assert pd.isna(w.iloc[-1]["SOLUSDT"])


def test_ledger_books_are_isolated(tmp_path):
    a = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-12", book="donchian_ensemble")
    b = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-13", book="tsmom_ensemble")
    a.execute(ts=TS, day="2026-07-13", symbol="ETHUSDT", target_qty=1.0,
              price=1_800.0, fee_bps=10.0, mode="live", reason="t")
    a.record_run("2026-07-13", "live", 1)
    b2 = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-13", book="tsmom_ensemble")
    assert b2.positions == {} and b2.cash == 10_000.0                  # B 账本不受影响
    assert not b2.run_completed("2026-07-13")
    a2 = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-12", book="donchian_ensemble")
    assert a2.positions == {"ETHUSDT": 1.0}
