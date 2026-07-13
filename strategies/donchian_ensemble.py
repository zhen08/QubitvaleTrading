"""生产信号服务：donchian 4 参数集成 × 3 币等权（现货长平）。

Phase 1 结论的部署形态（phase1_report_2026-07-12）。信号逻辑 **直接复用**
research.strategies.donchian —— 研究与生产同一份代码，是跟踪误差有意义的前提；
tests/test_signals_consistency.py 用黄金测试锁死这一点。

语义：weights.loc[D] = 在 D 收盘决定的目标权重，**自 D+1 日起生效**（与研究引擎
shift(1) 完全一致）。每币权重 = mean(4 个 donchian 变体 ∈ {0,1}) × 1/3
→ 单币 ∈ {0, 1/12, 1/6, 1/4, 1/3}，组合总仓位 ∈ [0, 1]。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from data import storeio
from research.strategies import GRIDS, donchian

log = logging.getLogger("qvt.signal")

SYMBOL_WEIGHT = 1.0 / 3.0
PARAMS = GRIDS["donchian"]          # 单一事实来源（ex-ante 网格）


def load_spot_daily(store: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(storeio.klines_path(store, "spot", symbol, "1d"))
    return df.set_index(pd.to_datetime(df["ts"], utc=True)).sort_index()


def symbol_weight_series(df: pd.DataFrame) -> pd.Series:
    """单币目标权重时间序列（决策日索引）。"""
    variants = pd.concat(
        {f"n{p['n_entry']}": donchian(df, **p) for p in PARAMS}, axis=1
    )
    return variants.mean(axis=1) * SYMBOL_WEIGHT


def compute_weights(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """全历史目标权重表：index=决策日（UTC 00:00），columns=symbols。"""
    w = pd.DataFrame({sym: symbol_weight_series(df) for sym, df in dfs.items()})
    w.index = w.index.normalize()
    return w.fillna(0.0)


def refresh_signals(settings: dict, dfs: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """重算并落库信号（确定性，可整表覆盖）。返回权重表。
    dfs 可由调用方注入（如 engine 用 Bitget 尾部 bar 补齐 D-1 后传入）。"""
    store = storeio.store_dir(settings)
    if dfs is None:
        dfs = {sym: load_spot_daily(store, sym) for sym in settings["symbols"]}
    weights = compute_weights(dfs)

    out_dir = store / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    weights.reset_index(names="decision_date").to_parquet(
        out_dir / "donchian_ensemble.parquet", index=False)

    latest = weights.iloc[-1]
    payload = {
        "strategy": "donchian_ensemble",
        "decision_date": str(weights.index[-1].date()),
        "effective_from": str((weights.index[-1] + pd.Timedelta(days=1)).date()),
        "target_weights": {k: round(float(v), 6) for k, v in latest.items()},
        "gross_exposure": round(float(latest.sum()), 6),
        "generated_at": str(pd.Timestamp.now(tz="UTC")),
    }
    with open(out_dir / "latest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log.info("signals: decision %s -> weights %s", payload["decision_date"],
             payload["target_weights"])
    return weights


def targets_for_day(weights: pd.DataFrame, day: pd.Timestamp,
                    strict: bool = True) -> dict[str, float] | None:
    """day（UTC 日）应持有的目标权重 = 前一日（D-1）决策。

    R6 修复：strict=True（默认）时 D-1 决策缺失返回 **None**（调用方必须把
    "无新鲜信号"当作事故处理、跳过交易），不再静默沿用更早的决策。
    strict=False 仅供离线分析。"""
    decision_day = day.normalize() - pd.Timedelta(days=1)
    if decision_day in weights.index:
        return {k: float(v) for k, v in weights.loc[decision_day].items()}
    if strict:
        return None
    earlier = weights.loc[:decision_day]
    if len(earlier):
        return {k: float(v) for k, v in earlier.iloc[-1].items()}
    return {k: 0.0 for k in weights.columns}
