"""Unit tests for QC structural checks (no network)."""
import pandas as pd

from research.qc_report import QCResult, check_klines_structure

QC_CFG = {"max_missing_pct": 0.2, "max_dup": 0, "max_ohlc_violations": 0}


def _mk(ts_list, o=100.0, h=101.0, l=99.0, c=100.5, v=1.0):
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(ts_list, utc=True),
            "open": o, "high": h, "low": l, "close": c, "volume": v,
        }
    )


def test_clean_series_passes():
    ts = pd.date_range("2026-01-01", periods=1000, freq="h", tz="UTC")
    res = QCResult()
    check_klines_structure(_mk(ts), "1h", QC_CFG, "t/clean", res)
    assert res.gate_passed


def test_gap_detected():
    ts = pd.date_range("2026-01-01", periods=1000, freq="h", tz="UTC").delete([100, 101, 102, 103, 104])
    res = QCResult()
    check_klines_structure(_mk(ts), "1h", QC_CFG, "t/gap", res)  # 0.5% missing > 0.2%
    assert not res.gate_passed


def test_duplicate_detected():
    ts = list(pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"))
    ts.append(ts[3])
    res = QCResult()
    check_klines_structure(_mk(ts), "1h", QC_CFG, "t/dup", res)
    assert not res.gate_passed


def test_ohlc_violation_detected():
    ts = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    df = _mk(ts)
    df.loc[4, "low"] = 100.6  # low above both open and close
    res = QCResult()
    check_klines_structure(df, "1h", QC_CFG, "t/ohlc", res)
    assert not res.gate_passed
