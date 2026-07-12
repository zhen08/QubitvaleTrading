"""Paper 账本会计正确性与幂等基元。"""
import pandas as pd

from execution.paper.ledger import Ledger

TS = pd.Timestamp("2026-07-12T00:10:00Z")


def _led(tmp_path):
    return Ledger.load_or_init(tmp_path, initial_capital=10_000.0,
                               start_date="2026-07-12")


def test_buy_sell_accounting(tmp_path):
    led = _led(tmp_path)
    led.execute(ts=TS, day="2026-07-12", symbol="BTCUSDT", target_qty=0.1,
                price=50_000.0, fee_bps=10.0, mode="live", reason="t")
    assert abs(led.cash - (10_000 - 5_000 - 5.0)) < 1e-9          # 买入含费
    assert abs(led.equity({"BTCUSDT": 50_000.0}) - 9_995.0) < 1e-9
    led.execute(ts=TS, day="2026-07-13", symbol="BTCUSDT", target_qty=0.0,
                price=51_000.0, fee_bps=10.0, mode="live", reason="t")
    assert abs(led.cash - (4_995.0 + 5_100 - 5.1)) < 1e-9          # 卖出扣费
    assert led.positions == {}


def test_min_trade_threshold_skips_dust(tmp_path):
    led = _led(tmp_path)
    row = led.execute(ts=TS, day="2026-07-12", symbol="BTCUSDT", target_qty=0.0001,
                      price=50_000.0, fee_bps=10.0, mode="live", reason="t",
                      min_trade_usdt=10.0)
    assert row is None and led.cash == 10_000.0


def test_rebalanced_on_and_mark_upsert(tmp_path):
    led = _led(tmp_path)
    assert not led.rebalanced_on("2026-07-12")
    led.execute(ts=TS, day="2026-07-12", symbol="ETHUSDT", target_qty=1.0,
                price=1_800.0, fee_bps=10.0, mode="live", reason="t")
    assert led.rebalanced_on("2026-07-12")
    assert led.rebalanced_on("2026-07-12", mode="live")
    assert not led.rebalanced_on("2026-07-12", mode="catchup")

    led.mark("2026-07-12", {"ETHUSDT": 1_800.0}, note="intraday")
    led.mark("2026-07-12", {"ETHUSDT": 1_900.0}, note="settled")   # 覆盖
    s = led.equity_series()
    assert len(s) == 1
    assert abs(float(s.iloc[0]) - led.equity({"ETHUSDT": 1_900.0})) < 1e-6


def test_state_persistence_roundtrip(tmp_path):
    led = _led(tmp_path)
    led.execute(ts=TS, day="2026-07-12", symbol="SOLUSDT", target_qty=10.0,
                price=77.0, fee_bps=10.0, mode="live", reason="t")
    led.save()
    led2 = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-12")
    assert abs(led2.cash - led.cash) < 1e-9
    assert led2.positions == {"SOLUSDT": 10.0}
