"""Timing/no-lookahead tests for the cross-asset store (plan §15.1). Offline."""
from datetime import date, time

import pandas as pd
import pytest

from data.collectors.cross_asset_daily import (AVAILABILITY_FLOOR_UTC,
                                               canonical_frame, upsert_daily)


def _raw(dates, close=100.0):
    return pd.DataFrame({
        "session_date": pd.to_datetime(dates),
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 1e6,
    })


NOW = pd.Timestamp("2026-07-14 00:10", tz="UTC")


def test_available_at_floor_is_2130_utc():
    df = canonical_frame("SPY", _raw(["2026-07-10"]), "test", now=NOW)
    assert df["available_at"].iloc[0] == pd.Timestamp("2026-07-10 21:30", tz="UTC")


def test_incomplete_session_dropped():
    """A bar for a session whose availability floor hasn't passed must not be stored."""
    now = pd.Timestamp("2026-07-13 18:00", tz="UTC")  # before Monday's 21:30 floor
    df = canonical_frame("SPY", _raw(["2026-07-10", "2026-07-13"]), "test", now=now)
    assert list(df["session_date"].dt.date) == [date(2026, 7, 10)]


def test_dst_summer_and_winter_bar_end():
    """EDT close = 20:00 UTC, EST close = 21:00 UTC; both < 21:30 floor."""
    df = canonical_frame("SPY", _raw(["2026-01-15", "2026-06-15"]), "test", now=NOW)
    ends = df.set_index(df["session_date"].dt.date)["bar_end"]
    assert ends[date(2026, 1, 15)] == pd.Timestamp("2026-01-15 21:00", tz="UTC")
    assert ends[date(2026, 6, 15)] == pd.Timestamp("2026-06-15 20:00", tz="UTC")
    assert (df["available_at"] >= df["bar_end"]).all()


def test_vix_bar_end_1615_et():
    df = canonical_frame("VIX", _raw(["2026-06-15"]), "test",
                         close_et=time(16, 15), now=NOW)
    assert df["bar_end"].iloc[0] == pd.Timestamp("2026-06-15 20:15", tz="UTC")


def test_asof_join_excludes_us_close_after_decision():
    """The D-1 crypto decision at 00:00 UTC must see Friday's close on Sat/Sun/Mon,
    and must NOT see a close whose available_at is after the decision timestamp."""
    df = canonical_frame("SPY", _raw(["2026-07-09", "2026-07-10"]), "test", now=NOW)
    decision_fri_open = pd.Timestamp("2026-07-10 00:00", tz="UTC")   # Fri 00:00 UTC
    decision_sat = pd.Timestamp("2026-07-11 00:00", tz="UTC")
    vis_fri = df[df["available_at"] <= decision_fri_open]
    vis_sat = df[df["available_at"] <= decision_sat]
    assert list(vis_fri["session_date"].dt.date) == [date(2026, 7, 9)]
    assert list(vis_sat["session_date"].dt.date) == [date(2026, 7, 9), date(2026, 7, 10)]


def test_upsert_first_write_wins(tmp_path):
    first = canonical_frame("SPY", _raw(["2026-07-10"], close=100.0), "test", now=NOW)
    upsert_daily(tmp_path, first)
    revised = canonical_frame("SPY", _raw(["2026-07-10"], close=999.0), "test",
                              now=NOW + pd.Timedelta(hours=1))
    merged = upsert_daily(tmp_path, revised)
    assert len(merged) == 1
    assert merged["close"].iloc[0] == 100.0          # original preserved


def test_upsert_appends_new_sessions(tmp_path):
    upsert_daily(tmp_path, canonical_frame("SPY", _raw(["2026-07-09"]), "test", now=NOW))
    merged = upsert_daily(
        tmp_path, canonical_frame("SPY", _raw(["2026-07-09", "2026-07-10"]), "test", now=NOW))
    assert list(merged["session_date"].dt.date) == [date(2026, 7, 9), date(2026, 7, 10)]


def test_weekend_carry_has_no_synthetic_session():
    """The store never contains weekend rows; forward-fill state is the feature
    layer's job and must be masked there, not materialized as bars here."""
    df = canonical_frame("SPY", _raw(["2026-07-10", "2026-07-13"]), "test", now=NOW)
    assert df["session_date"].dt.dayofweek.isin([5, 6]).sum() == 0
