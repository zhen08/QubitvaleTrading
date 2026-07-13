"""第二本 paper 账本：TSMOM 12 参数集成 × 3 币等权（现货长平，σ₂₀ vol targeting）。

预注册（2026-07-13，账本启用时固定）：
  - 网格 = research.strategies.GRIDS["tsmom"]（ex-ante 12 变体：lookback {30,60,90,180}
    × target_vol {10%,15%,20%}），固定参数 long_short=False、max_lev=1.0（现货部署形态，
    与 Phase 1 spot 配置一致）；
  - 信号代码与研究引擎共享 research.strategies.tsmom（黄金一致性同 donchian）；
  - Phase 1 组合口径参考：Sharpe 0.65 / 年化波动 9.8% / MaxDD −12.2% / DSR(N=4)=0.824
    ——同为**未经统计认证的研究候选**，本账本属探索性验证；
  - gate 与 donchian 账本相同（≥6 周、95% 带内、TE<2%、零 P0）；
  - **Phase 3 选择纪律（ex-ante）**：若两本账都达标，按各半仓部署两个策略，
    不做"选赢家"（避免引入 N=2 的又一层选择偏差）；只有一本达标则只试点那一本。
"""
from __future__ import annotations

import pandas as pd

from research.strategies import GRIDS, tsmom
from strategies.base import load_spot_daily, targets_for_day  # noqa: F401 (re-export)

SYMBOL_WEIGHT = 1.0 / 3.0
PARAMS = GRIDS["tsmom"]                                   # 单一事实来源（ex-ante 网格）
FIXED = {"long_short": False, "max_lev": 1.0}             # 现货长平部署形态


def symbol_weight_series(df: pd.DataFrame) -> pd.Series:
    """单币目标权重：12 个 vol-targeted TSMOM 变体的均值 × 1/3（连续值 ∈ [0, 1/3]）。"""
    variants = pd.concat(
        {f"lb{p['lookback']}_tv{int(p['target_vol']*100)}": tsmom(df, **p, **FIXED)
         for p in PARAMS}, axis=1
    )
    return variants.mean(axis=1) * SYMBOL_WEIGHT


def compute_weights(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """全历史目标权重表；缺数据日保留 NaN（与 base 的时效纪律配套）。"""
    w = pd.DataFrame({sym: symbol_weight_series(df) for sym, df in dfs.items()})
    w.index = w.index.normalize()
    return w
