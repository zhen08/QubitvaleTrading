"""ETF-flow risk gate: regime detection (as-of), engine integration (adds-only),
and the Farside HTML fallback parser."""
import pandas as pd

from data import storeio
from data.collectors.etf_flows import _num, parse_farside
from execution.paper.engine import _apply_risk_rules
from intel.etf_flows import etf_gate_asof

# A clear timestamp: outside every config/calendar.yaml event window (CPI 07-14, FOMC 07-29),
# so the event gate never interferes with these tests.
CLEAR = pd.Timestamp("2026-09-15T00:10:00Z")


def _settings(tmp_path) -> dict:
    return {
        "store_dir": str(tmp_path),
        "intel": {"etf_lookback_days": 5, "etf_min_days": 3,
                  "etf_outflow_no_add_usd_m": 300, "etf_outflow_trim_usd_m": 1000},
    }


def _write_flows(tmp_path, records: list[tuple]) -> None:
    """records: (asset, 'YYYY-MM-DD', net_flow_usd_m)."""
    df = pd.DataFrame([{"asset": a, "date": pd.Timestamp(d, tz="UTC"),
                        "net_flow_usd_m": v, "price_usd": 0.0, "fetched_at": "x"}
                       for a, d, v in records])
    storeio.write_parquet(df, storeio.etf_flows_path(storeio.store_dir({"store_dir": str(tmp_path)})))


def test_no_data_means_empty_gate(tmp_path):
    assert etf_gate_asof(_settings(tmp_path), CLEAR) == {}


def test_sustained_outflow_blocks_and_severe_halves(tmp_path):
    _write_flows(tmp_path, [
        ("BTC", "2026-09-11", -100), ("BTC", "2026-09-12", -80),
        ("BTC", "2026-09-13", -90),  ("BTC", "2026-09-14", -60),   # 5d net -330 -> block
        ("BTC", "2026-09-10", 0),
        ("ETH", "2026-09-11", -300), ("ETH", "2026-09-12", -300),
        ("ETH", "2026-09-13", -300), ("ETH", "2026-09-14", -200),  # 5d net -1100 -> halve
        ("ETH", "2026-09-10", 0),
    ])
    g = etf_gate_asof(_settings(tmp_path), CLEAR)
    assert g["BTCUSDT"]["action"] == "block" and g["BTCUSDT"]["net_usd_m"] == -330.0
    assert g["ETHUSDT"]["action"] == "halve"


def test_inflows_do_not_gate(tmp_path):
    _write_flows(tmp_path, [("BTC", f"2026-09-1{i}", 200) for i in range(5)])
    assert etf_gate_asof(_settings(tmp_path), CLEAR)["BTCUSDT"]["action"] is None


def test_insufficient_days_omitted(tmp_path):
    _write_flows(tmp_path, [("BTC", "2026-09-13", -500), ("BTC", "2026-09-14", -500)])  # only 2 < min 3
    assert "BTCUSDT" not in etf_gate_asof(_settings(tmp_path), CLEAR)


def test_sol_never_gated(tmp_path):
    _write_flows(tmp_path, [("SOL", f"2026-09-1{i}", -999) for i in range(5)])
    assert etf_gate_asof(_settings(tmp_path), CLEAR) == {}


def test_asof_only_uses_past_days(tmp_path):
    # Heavy outflows land AFTER the as-of day -> must not affect the earlier decision.
    _write_flows(tmp_path, [
        ("BTC", "2026-09-05", 50), ("BTC", "2026-09-06", 50), ("BTC", "2026-09-07", 50),
        ("BTC", "2026-09-13", -900), ("BTC", "2026-09-14", -900),
    ])
    early = etf_gate_asof(_settings(tmp_path), pd.Timestamp("2026-09-07T00:10:00Z"))
    assert early["BTCUSDT"]["action"] is None            # only the +150 inflows are visible
    late = etf_gate_asof(_settings(tmp_path), CLEAR)
    assert late["BTCUSDT"]["action"] == "halve"          # now the outflows count


def test_gate_restricts_adds_only_never_exits(tmp_path):
    _write_flows(tmp_path, [
        ("BTC", "2026-09-11", -200), ("BTC", "2026-09-12", -200),
        ("BTC", "2026-09-13", -200), ("BTC", "2026-09-14", -200),   # -800 -> block (>=300, <1000)
    ])
    s = _settings(tmp_path)
    benign = {"asset_neg_severity": {}, "market_neg_severity": 0}

    # want to ADD BTC (target 0.3 > current 0.1) -> capped to current
    adj, notes = _apply_risk_rules({"BTCUSDT": 0.3}, {"BTCUSDT": 0.1}, s, CLEAR, benign, False)
    assert adj["BTCUSDT"] == 0.1 and any("no add BTCUSDT" in n for n in notes)

    # want to REDUCE BTC (target 0.05 < current 0.2) -> exit always allowed, untouched
    adj2, _ = _apply_risk_rules({"BTCUSDT": 0.05}, {"BTCUSDT": 0.2}, s, CLEAR, benign, False)
    assert adj2["BTCUSDT"] == 0.05


def test_halve_cuts_position(tmp_path):
    _write_flows(tmp_path, [
        ("ETH", "2026-09-11", -300), ("ETH", "2026-09-12", -300),
        ("ETH", "2026-09-13", -300), ("ETH", "2026-09-14", -300),   # -1200 -> halve
    ])
    s = _settings(tmp_path)
    benign = {"asset_neg_severity": {}, "market_neg_severity": 0}
    adj, notes = _apply_risk_rules({"ETHUSDT": 0.3}, {"ETHUSDT": 0.2}, s, CLEAR, benign, False)
    assert abs(adj["ETHUSDT"] - 0.1) < 1e-9 and any("halve ETHUSDT" in n for n in notes)


# ---- Farside HTML fallback parser ----

def test_num_parsing():
    assert _num("1,234.5") == 1234.5
    assert _num("(45.6)") == -45.6
    assert _num("-") == 0.0 and _num("") == 0.0
    assert _num("(1,000.0)") == -1000.0
    assert _num("abc") is None


# Mirrors real Farside layout: blank Date header, per-ticker columns, blank-labeled
# Total as the LAST column, a 'Fee' row, and Total/Average/Maximum/Minimum summary rows.
FARSIDE_HTML = """
<html><body>
<table>
<tr><td></td><td>IBIT</td><td>FBTC</td><td>GBTC</td><td></td></tr>
<tr><td>Fee</td><td>0.25%</td><td>0.25%</td><td>1.50%</td><td></td></tr>
<tr><td>05 Jan 2025</td><td>100.0</td><td>50.0</td><td>(30.0)</td><td>120.0</td></tr>
<tr><td>06 Jan 2025</td><td>-</td><td>1,234.5</td><td>(1,000.0)</td><td>234.5</td></tr>
<tr><td>07 Jan 2025</td><td>10</td><td>10</td><td>10</td><td>(45.6)</td></tr>
<tr><td>Total</td><td>110</td><td>1294.5</td><td>(1020)</td><td>308.9</td></tr>
<tr><td>Average</td><td>36.7</td><td>431.5</td><td>(340)</td><td>103.0</td></tr>
<tr><td>Minimum</td><td>0</td><td>10</td><td>(1000)</td><td>(45.6)</td></tr>
</table>
</body></html>
"""


def test_parse_farside_table():
    rows = parse_farside(FARSIDE_HTML, "BTC", fetched_at="t")
    # 3 data rows; blank header, Fee, and Total/Average/Minimum summary rows all skipped
    assert [r["date"].strftime("%Y-%m-%d") for r in rows] == ["2025-01-05", "2025-01-06", "2025-01-07"]
    assert [r["net_flow_usd_m"] for r in rows] == [120.0, 234.5, -45.6]   # Total column (last cell)
    assert all(r["asset"] == "BTC" and r["source"] == "farside" for r in rows)


def test_parse_farside_no_table_is_empty():
    assert parse_farside("<html><body>Just a moment... (Cloudflare)</body></html>", "SOL") == []
