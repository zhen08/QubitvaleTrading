"""Cost model: Bitget fees + √-impact slippage + funding.

费率现实（2026-07，Bitget 官方）：
  现货   maker/taker 均 0.10%（BGB 抵扣 0.08%）→ 现货每边取 10bps（保守）
  USDT-M taker 0.06% / maker 0.02%
滑点：平方根冲击定律 I = Y·σ_d·√(Q/V_d)（Donier & Bonart 2015 验证 BTC 适用），
对 $10k 级订单在主流币上 ≈ 0.1–1bp，另设半点差下限 floor_bps。
资金费：多头在 funding>0 时支付。回测用 Binance 资金费史作为 Bitget 代理（两所
费率高度相关；实盘前用 funding_bitget 数据复核，见 Phase 2）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    fee_bps_side: float          # 每边手续费（bps）
    slip_floor_bps: float = 1.0  # 滑点下限（半点差近似）
    notional_usd: float = 10_000.0
    impact_y: float = 1.0        # √冲击系数 Y
    name: str = ""

    def slippage_bps(self, daily_vol: pd.Series, dollar_volume: pd.Series) -> pd.Series:
        """Per-bar one-way slippage in bps, from rolling daily vol & dollar volume."""
        dv = dollar_volume.replace(0, np.nan)
        impact = self.impact_y * daily_vol * np.sqrt(self.notional_usd / dv) * 1e4
        return impact.fillna(self.slip_floor_bps).clip(lower=self.slip_floor_bps)

    def cost_rate(self, daily_vol: pd.Series, dollar_volume: pd.Series) -> pd.Series:
        """One-way total cost per unit turnover (fraction, not bps)."""
        return (self.fee_bps_side + self.slippage_bps(daily_vol, dollar_volume)) / 1e4


# 预置模型（Phase 1 主结果用 taker；maker 作敏感性）
SPOT_TAKER = CostModel(fee_bps_side=10.0, name="spot_taker10")
UM_TAKER = CostModel(fee_bps_side=6.0, name="um_taker6")
UM_MAKER = CostModel(fee_bps_side=2.0, name="um_maker2")


def rolling_inputs(df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """(daily_vol, dollar_volume) rolling inputs for slippage, aligned to df index."""
    ret = df["close"].pct_change()
    daily_vol = ret.rolling(window).std().fillna(0.02)
    dollar_volume = df["quote_volume"].rolling(window).median()
    dollar_volume = dollar_volume.fillna(dollar_volume.median())
    return daily_vol, dollar_volume


def daily_funding(funding: pd.DataFrame) -> pd.Series:
    """8h 资金费聚合为按日合计（UTC 日），index = 当日 00:00 UTC。"""
    ts = pd.to_datetime(funding["ts"], utc=True)
    s = pd.Series(funding["funding_rate"].to_numpy(), index=ts)
    return s.groupby(s.index.normalize()).sum()
