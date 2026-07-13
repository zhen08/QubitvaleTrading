"""策略共用基元（多账本支架，2026-07-13）。

每个策略模块的统一接口：`compute_weights(dfs) -> DataFrame`
  index = 决策日（UTC 00:00），columns = symbols，值 = 目标权重；
  **缺数据日保留 NaN**（NaN=不知道 ≠ 0=空仓，P1 修复的语义约定）。
本模块提供数据加载、信号落库、目标时效检查——所有账本共用同一套时序纪律。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from data import storeio

log = logging.getLogger("qvt.signal")


def load_spot_daily(store: Path, symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(storeio.klines_path(store, "spot", symbol, "1d"))
    return df.set_index(pd.to_datetime(df["ts"], utc=True)).sort_index()


def persist_signals(store: Path, name: str, weights: pd.DataFrame) -> None:
    """落库信号（确定性，可整表覆盖）：signals/{name}.parquet + {name}.latest.json"""
    out_dir = store / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = weights.copy()
    df.index.name = "decision_date"
    tmp = out_dir / f"{name}.parquet.tmp"
    df.reset_index().to_parquet(tmp, index=False)
    os.replace(tmp, out_dir / f"{name}.parquet")

    latest = weights.iloc[-1]
    payload = {
        "strategy": name,
        "decision_date": str(weights.index[-1].date()),
        "effective_from": str((weights.index[-1] + pd.Timedelta(days=1)).date()),
        "target_weights": {k: (None if pd.isna(v) else round(float(v), 6))
                           for k, v in latest.items()},
        "gross_exposure": (None if latest.isna().any()
                           else round(float(latest.sum()), 6)),
        "generated_at": str(pd.Timestamp.now(tz="UTC")),
    }
    jtmp = out_dir / f"{name}.latest.json.tmp"
    jtmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(jtmp, out_dir / f"{name}.latest.json")
    log.info("signals[%s]: decision %s -> %s", name, payload["decision_date"],
             payload["target_weights"])


def targets_for_day(weights: pd.DataFrame, day: pd.Timestamp,
                    strict: bool = True) -> dict[str, float] | None:
    """day（UTC 日）应持有的目标权重 = D-1 决策。strict（生产默认）下
    D-1 行缺失 **或任一资产为 NaN** → None（整日拒绝，绝不静默回退/填零）。"""
    decision_day = day.normalize() - pd.Timedelta(days=1)
    if decision_day in weights.index:
        row = weights.loc[decision_day]
        if row.isna().any():
            if strict:
                return None
            row = row.fillna(0.0)
        return {k: float(v) for k, v in row.items()}
    if strict:
        return None
    earlier = weights.loc[:decision_day]
    if len(earlier):
        return {k: float(v) for k, v in earlier.iloc[-1].fillna(0.0).items()}
    return {k: 0.0 for k in weights.columns}
