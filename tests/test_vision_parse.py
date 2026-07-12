"""Unit tests for Binance Vision parsing helpers (no network)."""
import io
import zipfile

import pandas as pd

from data.collectors import binance_vision as bv
from data.collectors.common import normalize_epoch_series, read_zipped_csv


def _zip_bytes(csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", csv_text)
    return buf.getvalue()


# --- timestamp unit auto-detection ---

def test_normalize_epoch_units():
    t = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    sec, ms, us = int(t.timestamp()), int(t.timestamp() * 1e3), int(t.timestamp() * 1e6)
    for v in (sec, ms, us):
        out = normalize_epoch_series(pd.Series([v, v + 1]))
        assert out.iloc[0] == t, f"unit detect failed for {v}"


# --- header sniffing ---

def test_read_zipped_csv_headerless_and_header():
    headerless = _zip_bytes("1569888000000,8000,8100,7900,8050,10,1569891599999,80500,100,5,40000,0\n")
    df1 = read_zipped_csv(headerless)
    assert df1.shape == (1, 12) and df1.iloc[0, 0] == 1569888000000

    with_header = _zip_bytes(
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,tb_base,tb_quote,ignore\n"
        "1569888000000,8000,8100,7900,8050,10,1569891599999,80500,100,5,40000,0\n"
    )
    df2 = read_zipped_csv(with_header)
    assert df2.shape == (1, 12) and int(df2.iloc[0, 0]) == 1569888000000


def test_parse_klines_microsecond_spot_2025():
    t0 = int(pd.Timestamp("2025-02-01", tz="UTC").timestamp() * 1e6)  # microseconds
    csv = f"{t0},100000,101000,99000,100500,12.5,{t0 + 3599999999},1256000,999,6.2,623000,0\n"
    df = bv.parse_klines(_zip_bytes(csv))
    assert df.loc[0, "ts"] == pd.Timestamp("2025-02-01", tz="UTC")
    assert df.loc[0, "close"] == 100500.0
    assert df.loc[0, "trades"] == 999


# --- funding parsing: header, headerless, column-order heuristic ---

def test_parse_funding_with_header():
    csv = (
        "calc_time,funding_interval_hours,last_funding_rate\n"
        "1735689600000,8,0.0001\n1735718400000,8,-0.00005\n"
    )
    df = bv.parse_funding(_zip_bytes(csv))
    assert list(df["funding_rate"]) == [0.0001, -0.00005]
    assert df.loc[0, "interval_hours"] == 8


def test_parse_funding_headerless_swapped_columns():
    # rate column first, interval second — heuristic must still pick |x|<=0.05 as rate
    csv = "1735689600000,0.0002,8\n1735718400000,-0.0001,8\n"
    df = bv.parse_funding(_zip_bytes(csv))
    assert list(df["funding_rate"]) == [0.0002, -0.0001]


# --- month range helpers ---

def test_month_range():
    assert bv.month_range("2025-11", "2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]


def test_last_complete_month():
    assert bv.last_complete_month(pd.Timestamp("2026-07-12", tz="UTC")) == "2026-06"
    assert bv.last_complete_month(pd.Timestamp("2026-01-01", tz="UTC")) == "2025-12"


# --- full-missing-day repair ---

def test_repair_missing_days(monkeypatch):
    ts = pd.date_range("2022-02-24", "2022-03-02", freq="D", tz="UTC")
    df = pd.DataFrame({"ts": ts, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                       "volume": 1.0, "quote_volume": 1.0, "trades": 1,
                       "taker_buy_base": 0.5, "taker_buy_quote": 0.5})
    df = df[~df["ts"].isin(pd.to_datetime(["2022-02-26", "2022-02-27"], utc=True))]

    def fake_day(market, symbol, tf, ymd):
        t = pd.Timestamp(ymd, tz="UTC")
        return pd.DataFrame({"ts": [t], "open": [9.0], "high": [9.0], "low": [9.0],
                             "close": [9.0], "volume": [0.0], "quote_volume": [0.0],
                             "trades": [0], "taker_buy_base": [0.0], "taker_buy_quote": [0.0]})

    monkeypatch.setattr(bv, "fetch_kline_day", fake_day)
    fixed, n = bv.repair_missing_days(df, "um", "SOLUSDT", "1d", workers=2)
    assert n == 2 and len(fixed) == 7
    assert fixed["ts"].is_monotonic_increasing
