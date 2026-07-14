"""QC-gate tests for the cross-asset store (plan §15.1/§5.5). Offline."""
import pandas as pd

from data.collectors.cross_asset_daily import canonical_frame, qc_frame

NOW = pd.Timestamp("2026-07-14 00:10", tz="UTC")


def _frame(dates, **kw):
    raw = pd.DataFrame({
        "session_date": pd.to_datetime(dates),
        "open": kw.get("open", 100.0), "high": kw.get("high", 101.0),
        "low": kw.get("low", 99.0), "close": kw.get("close", 100.5),
        "volume": kw.get("volume", 1e6),
    })
    return canonical_frame("SPY", raw, "test", now=NOW)


def test_clean_frame_passes():
    dates = pd.bdate_range("2026-06-01", "2026-07-10")
    assert qc_frame(_frame(dates), is_etf=True) == []


def test_duplicate_session_flagged():
    df = _frame(["2026-07-09", "2026-07-10"])
    dup = pd.concat([df, df.tail(1)], ignore_index=True)
    assert any("duplicate" in i for i in qc_frame(dup, is_etf=True))


def test_ohlc_violation_flagged():
    df = _frame(["2026-07-10"], low=100.9, high=100.1)  # low > open? construct violation
    df.loc[0, ["open", "high", "low", "close"]] = [100, 99, 101, 100]  # high < low
    assert any("OHLC" in i for i in qc_frame(df, is_etf=True))


def test_ohlc_not_applied_to_index():
    df = _frame(["2026-07-10"])
    df.loc[0, ["open", "high", "low", "close"]] = [100, 99, 101, 100]
    assert qc_frame(df, is_etf=False) == []  # structure-only for VIX


def test_availability_before_bar_end_flagged():
    df = _frame(["2026-07-10"])
    df.loc[0, "available_at"] = df.loc[0, "bar_end"] - pd.Timedelta(hours=1)
    assert any("available_at earlier" in i for i in qc_frame(df, is_etf=True))


def test_holiday_gap_ok_but_long_gap_flagged():
    # Thanksgiving-style 4-day gap: fine. 10-day hole: flagged.
    ok = _frame(["2025-11-26", "2025-12-01"])
    assert not any("gap" in i for i in qc_frame(ok, is_etf=True))
    bad = _frame(["2026-06-01", "2026-06-15"])
    assert any("gap" in i for i in qc_frame(bad, is_etf=True))


def test_negative_price_flagged():
    df = _frame(["2026-07-10"])
    df.loc[0, "close"] = -1.0
    assert any("non-positive" in i for i in qc_frame(df, is_etf=True))
