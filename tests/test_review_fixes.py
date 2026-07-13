"""Review（2026-07-13）修复的回归测试——每条对应 review 的一个反例。"""
import json

import numpy as np
import pandas as pd

from execution.paper.engine import _rebalance
from execution.paper.ledger import Ledger
from research.qc_report import QCResult, check_klines_structure
from strategies.donchian_ensemble import compute_weights, targets_for_day

QC_CFG = {"max_missing_pct": 0.2, "max_dup": 0, "max_ohlc_violations": 0}
TS = pd.Timestamp("2026-07-12T00:10:00Z")
SETTINGS = {"paper": {"rebalance_threshold": 0.02, "fee_bps_side": 10.0,
                      "min_trade_usdt": 10.0}}


def _mk(ts, o=100.0, h=101.0, l=99.0, c=100.5, v=1.0):
    return pd.DataFrame({"ts": pd.to_datetime(ts, utc=True),
                         "open": o, "high": h, "low": l, "close": c, "volume": v})


# ---------- R4: QC 反例（review 提供的三个都必须 FAIL） ----------

def test_qc_nan_row_fails():
    ts = pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC")
    df = _mk(ts)
    df.loc[50, ["open", "high", "low", "close", "volume"]] = np.nan
    res = QCResult()
    check_klines_structure(df, "1h", QC_CFG, "t/nan", res)
    assert not res.gate_passed


def test_qc_30min_offset_fails():
    ts = pd.date_range("2026-01-01 00:30", periods=200, freq="h", tz="UTC")  # 全部错位 30 分钟
    res = QCResult()
    check_klines_structure(_mk(ts), "1h", QC_CFG, "t/offset", res)
    assert not res.gate_passed


def test_qc_extra_offgrid_records_fail_and_missing_nonnegative():
    ts = list(pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"))
    ts.append(pd.Timestamp("2026-01-03 05:17", tz="UTC"))  # 网格外多余记录
    res = QCResult()
    check_klines_structure(_mk(sorted(ts)), "1h", QC_CFG, "t/extra", res)
    assert not res.gate_passed
    detail = res.checks[0].detail
    assert "missing=-" not in detail          # missing 不允许为负
    assert "misaligned=1" in detail


# ---------- R2: 崩溃恢复（账本状态从事件日志重放，快照失真自动纠正） ----------

def test_ledger_recovers_from_stale_snapshot(tmp_path):
    led = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-12")
    led.execute(ts=TS, day="2026-07-12", symbol="ETHUSDT", target_qty=0.5,
                price=1_800.0, fee_bps=10.0, mode="live", reason="t")
    # 模拟"成交已落盘、快照保存前崩溃"：把 state.json 回写成成交前的旧快照
    stale = {"initial_capital": 10_000.0, "start_date": "2026-07-12",
             "last_settled": None, "cash": 10_000.0, "positions": {}}
    (tmp_path / "paper" / "state.json").write_text(json.dumps(stale), encoding="utf-8")

    led2 = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-12")
    assert led2.positions == {"ETHUSDT": 0.5}                  # 重放恢复，而非空仓
    assert abs(led2.cash - (10_000 - 900 - 0.9)) < 1e-9


def test_run_registry_not_inferred_from_trades(tmp_path):
    led = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-12")
    led.execute(ts=TS, day="2026-07-12", symbol="ETHUSDT", target_qty=0.1,
                price=1_800.0, fee_bps=10.0, mode="live", reason="t")
    assert led.rebalanced_on("2026-07-12")                     # 有成交
    assert not led.run_completed("2026-07-12")                 # 但调仓未注册完成 → 重跑会补差额
    led.record_run("2026-07-12", "live", 1)
    assert led.run_completed("2026-07-12", "live")


# ---------- R3: 现金护栏 + 满仓不透支 ----------

def test_cash_cap_prevents_negative_cash(tmp_path):
    led = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-12")
    led.execute(ts=TS, day="2026-07-12", symbol="BTCUSDT", target_qty=1.0,  # 想买 $50k
                price=50_000.0, fee_bps=10.0, mode="live", reason="t")
    assert led.cash >= -1e-6
    assert abs(led.cash) < 1e-4                                 # 全部现金用尽但不为负
    assert led.positions["BTCUSDT"] < 0.2001                    # 被封顶在 ~10000/1.001/50000


def test_full_allocation_rebalance_no_negative_cash(tmp_path):
    """Review 反例：三币各 1/3 满仓 → 原实现 cash=-10；现在必须 ≥0。"""
    led = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-12")
    quotes = {s: {"mark": 100.0, "buy": 100.01, "sell": 99.99}
              for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")}
    targets = {s: 1.0 / 3.0 for s in quotes}
    trades = _rebalance(led, targets, quotes, day="2026-07-12", ts=TS,
                        mode="live", reason="t", settings=SETTINGS)
    assert led.cash >= -1e-6
    assert len(trades) == 3
    eq = led.equity({s: 100.0 for s in quotes})
    assert 9_970 < eq <= 10_000                                 # 只损失费用+滑点


# ---------- R6: 信号严格时效（不再静默沿用旧决策） ----------

def test_targets_for_day_strict_returns_none_when_stale():
    idx = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 200, 300), index=idx)
    dfs = {"BTCUSDT": pd.DataFrame({"ts": idx, "close": close.to_numpy(),
                                    "open": close.to_numpy(), "high": close.to_numpy(),
                                    "low": close.to_numpy(), "quote_volume": 1e9})}
    w = compute_weights(dfs)
    future = w.index[-1] + pd.Timedelta(days=5)                 # D-1 决策 bar 缺失
    assert targets_for_day(w, future, strict=True) is None
    assert targets_for_day(w, future, strict=False) is not None  # 仅离线分析允许回退


# ---------- 第二轮 review：P1 逐资产时效 / 并发锁 / settled-only ----------

def _price_df(idx):
    c = np.linspace(100, 200, len(idx))
    return pd.DataFrame({"ts": idx, "close": c, "open": c, "high": c, "low": c,
                         "quote_volume": 1e9})


def test_one_asset_missing_decision_bar_invalidates_whole_day():
    """Review 反例：SOL 缺 D-1 bar 时，原实现给 SOL 填 0（= 清仓信号）。"""
    idx_full = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    idx_short = idx_full[:-1]                       # SOL 少最后一天
    dfs = {"BTCUSDT": _price_df(idx_full), "ETHUSDT": _price_df(idx_full),
           "SOLUSDT": _price_df(idx_short)}
    w = compute_weights(dfs)
    day = idx_full[-1] + pd.Timedelta(days=1)       # 决策日 = idx_full[-1]，SOL 缺
    assert pd.isna(w.loc[idx_full[-1], "SOLUSDT"])  # 缺数据保留 NaN，不再填 0
    assert targets_for_day(w, day, strict=True) is None   # 整日拒绝


def test_missing_decision_bars_helper_names_symbols():
    from execution.paper.engine import _missing_decision_bars
    idx = pd.date_range("2026-07-01", periods=10, freq="D", tz="UTC")
    dfs = {"BTCUSDT": _price_df(idx).set_index(idx),
           "SOLUSDT": _price_df(idx[:-1]).set_index(idx[:-1])}
    assert _missing_decision_bars(dfs, idx[-1]) == ["SOLUSDT"]
    assert _missing_decision_bars(dfs, idx[-2]) == []


def test_exclusive_lock_blocks_second_holder(tmp_path):
    from execution.paper.engine import _exclusive_lock
    lock = tmp_path / ".run.lock"
    with _exclusive_lock(lock) as a:
        assert a is True
        with _exclusive_lock(lock) as b:            # 重叠运行 → 拿不到锁
            assert b is False
    with _exclusive_lock(lock) as c:                # 释放后可再获取
        assert c is True


def test_equity_series_settled_only_excludes_intraday(tmp_path):
    led = Ledger.load_or_init(tmp_path, 10_000.0, "2026-07-12")
    led.mark("2026-07-12", {}, note="settled")
    led.mark("2026-07-13", {}, note="intraday")
    assert len(led.equity_series()) == 2
    s = led.equity_series(settled_only=True)
    assert len(s) == 1 and str(s.index[0].date()) == "2026-07-12"


def test_bootstrap_band_deterministic_and_widens():
    from ops.tracking import bootstrap_band
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0005, 0.016, 1400)
    b1 = bootstrap_band(rets, 7)
    b2 = bootstrap_band(rets, 7)
    assert b1 == b2                                  # 固定种子可复现
    b42 = bootstrap_band(rets, 42)
    assert (b42["p97_5"] - b42["p2_5"]) > (b1["p97_5"] - b1["p2_5"])  # 带随视界变宽
