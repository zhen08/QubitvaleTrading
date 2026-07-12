"""Engine correctness: no-lookahead shift, cost charging, funding sign."""
import numpy as np
import pandas as pd

from research.costs import CostModel
from research.engine import run_backtest

CM = CostModel(fee_bps_side=10.0, slip_floor_bps=0.0, notional_usd=0.0)


def _df(closes):
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({"ts": idx, "close": closes, "quote_volume": 1e9})


def test_lookahead_cheat_is_neutralized():
    """作弊信号 pos[t]=sign(ret[t]) 若无 shift 会稳赚；引擎 shift 后在交替行情中稳亏。"""
    closes, p = [100.0], 100.0
    for i in range(200):
        p = p * (1.10 if i % 2 == 0 else 1 / 1.10)
        closes.append(p)
    df = _df(closes)
    ret = df["close"].pct_change()
    cheat = pd.Series(np.sign(ret.to_numpy()), index=pd.to_datetime(df["ts"], utc=True)).fillna(0)
    res = run_backtest(df, cheat, CM)
    assert res.gross.sum() < -1.0  # shift 使其始终踩反


def test_costs_charged_on_turnover():
    df = _df([100.0] * 10)  # 价格不动 → 毛收益 0
    pos = pd.Series(0.0, index=pd.to_datetime(df["ts"], utc=True))
    pos.iloc[2] = 1.0  # 第 2 根收盘开仓
    pos.iloc[5:] = 0.0
    pos.iloc[3:5] = 1.0
    res = run_backtest(df, pos, CM)
    assert abs(res.gross.sum()) < 1e-12
    assert abs(res.cost.sum() - 2 * 10 / 1e4) < 1e-9  # 开+平各 10bps
    assert abs(res.net.sum() + 2 * 10 / 1e4) < 1e-9


def test_funding_sign_long_pays_positive():
    df = _df([100.0] * 5)
    idx = pd.DatetimeIndex(pd.to_datetime(df["ts"], utc=True))
    pos = pd.Series(1.0, index=idx)
    funding = pd.Series(0.001, index=idx.normalize())
    res = run_backtest(df, pos, CostModel(fee_bps_side=0.0, slip_floor_bps=0.0), funding)
    held_days = int((res.pos != 0).sum())
    assert abs(res.funding_pnl.sum() + 0.001 * held_days) < 1e-12  # 多头支付
