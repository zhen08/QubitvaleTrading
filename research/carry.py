"""Funding-rate carry historical simulation (delta-neutral: spot long + perp short).

模型（保守简化，均已披露）：
  资金各半：0.5 仓位买现货、0.5 做空永续（1x，名义 = 0.5×资本/腿）。
  日收益（对总资本）= 0.5 × 当日资金费合计 × 是否在场（空头在 funding>0 时收费）。
  Delta 中性 → 价格项抵消；基差漂移与再平衡为二阶项，未建模（低杠杆下影响小）。
  出入场成本 = 0.5×(现货边 + 合约边)（bps，对总资本），每次状态切换收一次。
  开关变体：7 日滚动年化资金费 > enter_apr 进场，< exit_apr 离场（t 收盘决定，t+1 生效）。
风险提示（BIS WP1087）：carry 是崩盘/强平风险补偿；本模拟不含极端行情下的
平仓滑点尖峰与保证金链条，实际收益应打折看待。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research.costs import daily_funding
from research.metrics import ann_vol, max_drawdown, sharpe


@dataclass
class CarryResult:
    name: str
    net: pd.Series
    time_in_market: float
    n_roundtrips: int
    net_apr_pct: float
    sharpe: float
    max_dd_pct: float
    avg_funding_apr_pct: float


def carry_sim(
    funding_df: pd.DataFrame,
    enter_apr: float | None = None,
    exit_apr: float | None = None,
    spot_side_bps: float = 11.0,   # 10bps 费 + 1bp 滑点
    um_side_bps: float = 7.0,      # 6bps 费 + 1bp 滑点
    name: str = "carry",
) -> CarryResult:
    fd = daily_funding(funding_df)                      # 每日资金费合计（8h×3）
    ann = (fd.rolling(7).mean() * 365).fillna(0.0)      # 7 日滚动年化

    if enter_apr is None:
        in_pos = pd.Series(1.0, index=fd.index)
    else:
        state, states = 0.0, []
        for v in ann:                                    # 决策于 t 收盘
            if state == 0.0 and v > enter_apr:
                state = 1.0
            elif state == 1.0 and v < exit_apr:
                state = 0.0
            states.append(state)
        in_pos = pd.Series(states, index=fd.index)

    held = in_pos.shift(1).fillna(0.0)                  # t+1 生效
    switches = held.diff().abs().fillna(held.iloc[0] if len(held) else 0.0)
    cost_per_switch = 0.5 * (spot_side_bps + um_side_bps) / 1e4
    net = 0.5 * fd * held - switches * cost_per_switch

    return CarryResult(
        name=name,
        net=net,
        time_in_market=float(held.mean()),
        n_roundtrips=int(switches.sum() // 2),
        net_apr_pct=round(100 * float(net.mean() * 365), 2),
        sharpe=round(sharpe(net, "1d"), 2),
        max_dd_pct=round(100 * max_drawdown(net), 2),
        avg_funding_apr_pct=round(100 * float(fd.mean() * 365), 2),
    )


def run_carry_suite(funding_df: pd.DataFrame) -> list[CarryResult]:
    return [
        carry_sim(funding_df, name="always_on"),
        carry_sim(funding_df, enter_apr=0.05, exit_apr=0.00, name="filter_5/0"),
        carry_sim(funding_df, enter_apr=0.10, exit_apr=0.02, name="filter_10/2"),
    ]
