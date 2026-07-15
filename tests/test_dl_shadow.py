"""Shadow-logging tests: freeze -> daily log -> idempotency -> E2 ablation. Offline."""
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from data import storeio
from data.collectors.cross_asset_daily import canonical_frame
from features.cross_asset import build_feature_table
import research.dl.train as T
from research.dl.freeze import freeze_shadow
from research.dl.shadow import run_shadow, shadow_path
from strategies.donchian_tcn_risk_overlay import compute_risk_multiplier

RNG = np.random.default_rng(11)
N_DAYS = 900


def _write_crypto(store, symbol, closes):
    ts = pd.date_range("2023-01-01", periods=len(closes), freq="D", tz="UTC")
    df = pd.DataFrame({
        "ts": ts, "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": 1000.0, "quote_volume": 1e6,
        "taker_buy_base": 0.0, "taker_buy_quote": 0.0, "trades": 10,
        "symbol": symbol, "market": "spot", "timeframe": "1d",
    })
    storeio.write_parquet(df, storeio.klines_path(store, "spot", symbol, "1d"))


def _write_xa(store, symbol, seed):
    g = np.random.default_rng(seed)
    sessions = pd.bdate_range("2023-01-01", periods=640)
    closes = (18.0 if symbol == "VIX" else 300.0) * np.exp(
        np.cumsum(g.normal(0, 0.01, len(sessions))))
    raw = pd.DataFrame({"session_date": sessions, "open": closes,
                        "high": closes * 1.01, "low": closes * 0.99,
                        "close": closes, "volume": 1e6})
    df = canonical_frame(symbol, raw, "test", now=pd.Timestamp("2030-01-01", tz="UTC"))
    storeio.write_parquet(df, store / "cross_asset"
                          / ("index" if symbol == "VIX" else "market")
                          / symbol / "1d.parquet")


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_EPOCHS", 2)
    _write_crypto(tmp_path, "BTCUSDT",
                  100 * np.exp(np.cumsum(RNG.normal(0, 0.03, N_DAYS))))
    for i, sym in enumerate(("SPY", "QQQ", "GLD", "VIX")):
        _write_xa(tmp_path, sym, seed=i)
    # cross-asset status OK + a donchian signals table for base weights
    p = tmp_path / "cross_asset" / "status.json"
    p.write_text(json.dumps({"ok": True}))
    sig = pd.DataFrame({"decision_date": pd.date_range("2023-01-01", periods=N_DAYS,
                                                       freq="D", tz="UTC"),
                        "BTCUSDT": 0.25})
    storeio.write_parquet(sig, tmp_path / "signals" / "donchian_ensemble.parquet")
    return tmp_path


def _settings(store):
    return {"store_dir": str(store), "symbols": ["BTCUSDT"],
            "dl_shadow": {"model_id": "e2_test"}}


def _freeze(store):
    table, _ = build_feature_table(store, ["BTCUSDT"])
    return freeze_shadow(store, table, variant="E2", seeds=(17, 29),
                         model_id="e2_test")


def test_freeze_and_shadow_roundtrip(store, monkeypatch):
    monkeypatch.setattr(storeio, "store_dir", lambda s: store)
    man = _freeze(store)
    assert man["variant"] == "E2" and man["har_coef"] and len(man["har_coef"]) == 4
    s1 = run_shadow(_settings(store))
    log = pd.read_parquet(shadow_path(store))
    assert set(log["variant"]) == {"E2", "B1", "B2"}
    assert ((log["multiplier"] >= 0) & (log["multiplier"] <= 1)).all()
    assert (log["shadow_weight"] <= log["base_weight"] + 1e-12).all()  # reduce-only
    # idempotent: re-run logs nothing new for the same decision
    s2 = run_shadow(_settings(store))
    assert s2["rows_new"] == 0


def test_e2_ignores_vix_by_construction(store, monkeypatch):
    monkeypatch.setattr(storeio, "store_dir", lambda s: store)
    _freeze(store)
    table1, _ = build_feature_table(store, ["BTCUSDT"])
    out1 = compute_risk_multiplier(store, "e2_test", table1)
    # scramble VIX history and rebuild — E2 output must be bit-identical
    vix_path = store / "cross_asset" / "index" / "VIX" / "1d.parquet"
    vix = pd.read_parquet(vix_path)
    vix[["open", "high", "low", "close"]] *= 5.0
    storeio.write_parquet(vix, vix_path)
    table2, _ = build_feature_table(store, ["BTCUSDT"])
    out2 = compute_risk_multiplier(store, "e2_test", table2)
    assert out1["multiplier"].tolist() == out2["multiplier"].tolist()
    assert out1["sigma_hat"].tolist() == out2["sigma_hat"].tolist()
    # …but scrambling SPY (a kept group) must change the inputs. A constant
    # multiple would be invisible (all features are returns/ratios), so inject
    # per-row noise that changes the return path itself.
    spy_path = store / "cross_asset" / "market" / "SPY" / "1d.parquet"
    spy = pd.read_parquet(spy_path)
    noise = 1.0 + np.random.default_rng(3).normal(0, 0.05, len(spy))
    for col in ("open", "high", "low", "close"):
        spy[col] = spy[col] * noise
    storeio.write_parquet(spy, spy_path)
    table3, _ = build_feature_table(store, ["BTCUSDT"])
    out3 = compute_risk_multiplier(store, "e2_test", table3)
    assert out3["sigma_hat"].tolist() != out1["sigma_hat"].tolist()