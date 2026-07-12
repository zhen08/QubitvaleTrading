"""Vectorized daily-bar backtest engine with built-in no-lookahead protection.

约定（全引擎唯一的时序规则，测试覆盖）：
  策略函数返回 pos_decided[t] = 在第 t 根 K 线**收盘时**决定的目标仓位。
  引擎内部执行 pos_held = pos_decided.shift(1)：仓位从下一根 K 线起生效。
  成本按 |Δpos_held| × (费率+滑点) 在换仓当根收取；资金费按持仓收付。

为什么不用 vectorbt：日线 × ≤50 变体的网格，纯 pandas 即毫秒级，且这段逻辑
必须完全可审计——依赖越少越好。1h 大网格扫描时再引入加速器。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research.costs import CostModel, rolling_inputs


@dataclass
class BacktestResult:
    net: pd.Series          # 净收益（每根）
    gross: pd.Series        # 毛收益
    pos: pd.Series          # 实际持仓（已 shift）
    turnover: pd.Series     # |Δpos|
    cost: pd.Series         # 交易成本
    funding_pnl: pd.Series  # 资金费损益（现货为 0）

    @property
    def ann_turnover(self) -> float:
        return float(self.turnover.mean() * 365)


def run_backtest(
    df: pd.DataFrame,
    pos_decided: pd.Series,
    cost_model: CostModel,
    funding_by_day: pd.Series | None = None,
) -> BacktestResult:
    """df: klines（含 close/quote_volume，index 为 ts 或含 ts 列）。"""
    if "ts" in df.columns:
        df = df.set_index(pd.to_datetime(df["ts"], utc=True))
    close = df["close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    pos_decided = pos_decided.reindex(close.index).fillna(0.0).clip(-3, 3)
    pos_held = pos_decided.shift(1).fillna(0.0)          # ← 防前视的唯一入口
    turnover = pos_held.diff().abs()
    turnover.iloc[0] = abs(float(pos_held.iloc[0]))

    daily_vol, dollar_volume = rolling_inputs(df)
    cost = turnover * cost_model.cost_rate(daily_vol, dollar_volume)

    gross = pos_held * ret

    if funding_by_day is not None:
        f = funding_by_day.reindex(close.index.normalize()).fillna(0.0)
        f.index = close.index
        funding_pnl = -pos_held * f                       # 多头付正费率
    else:
        funding_pnl = pd.Series(0.0, index=close.index)

    net = gross - cost + funding_pnl
    return BacktestResult(net=net, gross=gross, pos=pos_held,
                          turnover=turnover, cost=cost, funding_pnl=funding_pnl)
