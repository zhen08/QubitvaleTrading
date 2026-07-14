"""No-lookahead tests for the DL feature table (plan §15.2). Offline, synthetic store."""
import numpy as np
import pandas as pd
import pytest

from data import storeio
from data.collectors.cross_asset_daily import canonical_frame
from features.cross_asset import ALL_FEATURES, LABELS, build_feature_table

RNG = np.random.default_rng(7)
N_DAYS = 420


def _write_crypto(store, symbol, closes):
    ts = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
    df = pd.DataFrame({
        "ts": ts, "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": 1000.0, "quote_volume": 1e6,
        "taker_buy_base": 0.0, "taker_buy_quote": 0.0, "trades": 10,
        "symbol": symbol, "market": "spot", "timeframe": "1d",
    })
    storeio.write_parquet(df, storeio.klines_path(store, "spot", symbol, "1d"))


def _write_xa(store, symbol, closes, dates):
    raw = pd.DataFrame({"session_date": dates, "open": closes,
                        "high": closes * 1.01, "low": closes * 0.99,
                        "close": closes, "volume": 1e6})
    now = pd.Timestamp("2030-01-01", tz="UTC")
    df = canonical_frame(symbol, raw, "test", now=now)
    storeio.write_parquet(df, store / "cross_asset"
                          / ("index" if symbol == "VIX" else "market")
                          / symbol / "1d.parquet")


@pytest.fixture()
def store(tmp_path):
    closes = 100 * np.exp(np.cumsum(RNG.normal(0, 0.03, N_DAYS)))
    _write_crypto(tmp_path, "BTCUSDT", closes)
    sessions = pd.bdate_range("2024-01-01", periods=300)
    for sym in ("SPY", "QQQ", "GLD", "VIX"):
        xc = 20 + 400 * RNG.random() if sym != "VIX" else 18.0
        _write_xa(tmp_path, sym, xc * np.exp(np.cumsum(RNG.normal(0, 0.01, 300))),
                  sessions)
    return tmp_path


def _build(store):
    table, _ = build_feature_table(store, ["BTCUSDT"])
    return table.set_index("decision_ts")


def test_future_crypto_bars_do_not_change_features(store):
    base = _build(store)
    # mutate the last 30 crypto bars and rebuild
    df = pd.read_parquet(storeio.klines_path(store, "spot", "BTCUSDT", "1d"))
    df.loc[df.index[-30:], ["open", "high", "low", "close"]] *= 3.0
    storeio.write_parquet(df, storeio.klines_path(store, "spot", "BTCUSDT", "1d"))
    mutated = _build(store)

    cutoff = pd.to_datetime(df["ts"].iloc[-30], utc=True)  # first mutated bar date
    safe = base.index < cutoff - pd.Timedelta(days=5)      # beyond label horizon
    pd.testing.assert_frame_equal(base.loc[safe, ALL_FEATURES],
                                  mutated.loc[safe, ALL_FEATURES])
    # labels inside the horizon MUST differ (they look forward)
    horizon = (base.index >= cutoff - pd.Timedelta(days=5)) & (base.index < cutoff)
    assert not base.loc[horizon, LABELS].equals(mutated.loc[horizon, LABELS])


def test_future_us_sessions_do_not_change_past_rows(store):
    base = _build(store)
    path = store / "cross_asset" / "market" / "SPY" / "1d.parquet"
    df = pd.read_parquet(path)
    df.loc[df.index[-40:], "close"] *= 2.0
    storeio.write_parquet(df, path)
    mutated = _build(store)
    cutoff = df["session_date"].iloc[-40].tz_localize("UTC")
    safe = base.index <= cutoff  # decision at/before first mutated session's date
    pd.testing.assert_frame_equal(base.loc[safe, ALL_FEATURES],
                                  mutated.loc[safe, ALL_FEATURES])


def test_weekend_rows_reuse_last_session_without_new_returns(store):
    t = _build(store)
    # Saturday 00:00 UTC decision sees Friday's close fresh; Sunday and Monday
    # decisions reuse the same Friday value with growing staleness.
    sun = t[t.index.dayofweek == 6]
    assert len(sun)
    for d in sun.index[5:10]:
        sat = d - pd.Timedelta(days=1)
        mon = d + pd.Timedelta(days=1)
        if sat in t.index and mon in t.index:
            assert t.loc[d, "x_spy_ret1"] == t.loc[sat, "x_spy_ret1"]
            assert t.loc[mon, "x_spy_ret1"] == t.loc[sat, "x_spy_ret1"]
            assert t.loc[d, "x_spy_days_since"] > t.loc[sat, "x_spy_days_since"]
            assert t.loc[mon, "x_spy_days_since"] > t.loc[d, "x_spy_days_since"]
            assert t.loc[d, "x_spy_closed"] == 1.0


def test_missing_group_is_masked_zero_not_silent_value(store):
    (store / "cross_asset" / "market" / "GLD" / "1d.parquet").unlink()
    t = _build(store)
    assert (t["m_gld"] == 0).all()
    gld_feats = [c for c in ALL_FEATURES
                 if c.startswith("x_gld") and not c.endswith(("_closed", "_days_since"))]
    assert (t[gld_feats] == 0.0).all().all()


def test_missing_gld_file_raises_cleanly_or_masks(store):
    """load_daily raises FileNotFoundError; builder must not fabricate GLD data."""
    # covered by test above via unlink; here assert other groups unaffected
    (store / "cross_asset" / "market" / "GLD" / "1d.parquet").unlink()
    t = _build(store)
    assert (t["m_equity"].iloc[-50:] == 1).all()
