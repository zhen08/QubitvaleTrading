"""账本一：donchian 4 参数集成 × 3 币等权（现货长平）。

Phase 1 修订判定：研究候选 PASS / 统计认证 FAIL（组合 DSR(N=4)=0.868、DSR(N=32)=0.750，
对账记录见 research/reports/dsr_reconciliation_2026-07-14.md）
——本账本属**探索性 paper 验证**。信号逻辑复用 research.strategies.donchian
（研究与生产同一份代码，tests/test_signals_consistency.py 黄金测试锁死）。

语义：weights.loc[D] = 在 D 收盘决定的目标权重，自 D+1 生效（与研究引擎 shift(1) 一致）。
每币权重 = mean(4 变体 ∈ {0,1}) × 1/3。缺数据日保留 NaN（P1 纪律，见 strategies/base.py）。
Phase 3 选择纪律（ex-ante，与 tsmom 账本共同约定）：两本都达标 → 各半仓部署，不选赢家。
"""
from __future__ import annotations

import logging

import pandas as pd

from data import storeio
from research.strategies import GRIDS, donchian
from strategies.base import (load_spot_daily, persist_signals,  # noqa: F401 (re-export)
                             targets_for_day)

log = logging.getLogger("qvt.signal")

SYMBOL_WEIGHT = 1.0 / 3.0
PARAMS = GRIDS["donchian"]          # 单一事实来源（ex-ante 网格）


def symbol_weight_series(df: pd.DataFrame) -> pd.Series:
    """单币目标权重时间序列（决策日索引）。"""
    variants = pd.concat(
        {f"n{p['n_entry']}": donchian(df, **p) for p in PARAMS}, axis=1
    )
    return variants.mean(axis=1) * SYMBOL_WEIGHT


def compute_weights(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """全历史目标权重表：index=决策日（UTC 00:00），columns=symbols。
    外连接产生的 NaN **保留**（缺数据 ≠ 空仓信号）。"""
    w = pd.DataFrame({sym: symbol_weight_series(df) for sym, df in dfs.items()})
    w.index = w.index.normalize()
    return w


def refresh_signals(settings: dict, dfs: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """（兼容入口）重算并落库本策略信号；多账本引擎请直接用
    compute_weights + base.persist_signals。"""
    store = storeio.store_dir(settings)
    if dfs is None:
        dfs = {sym: load_spot_daily(store, sym) for sym in settings["symbols"]}
    weights = compute_weights(dfs)
    persist_signals(store, "donchian_ensemble", weights)
    return weights
