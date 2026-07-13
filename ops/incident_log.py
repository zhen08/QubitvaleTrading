"""运维事件持久化（R6）：incidents 落盘，"零 P0"放行条件才可计算。

级别：P0 = 资损/账本失真类；P1 = 功能失败（数据缺失中止、实时价不可得、任务崩溃）；
P2 = 降级运行（catchup 补账、risk_flags 过期、新闻步骤失败）；P3 = 信息。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger("qvt.incident")

COLS = ["ts", "day", "level", "kind", "detail"]


def _path(store: Path) -> Path:
    d = store / "ops"
    d.mkdir(parents=True, exist_ok=True)
    return d / "incidents.parquet"


def record(store: Path, level: str, kind: str, detail: str, day: str | None = None) -> None:
    assert level in ("P0", "P1", "P2", "P3")
    p = _path(store)
    row = pd.DataFrame([{"ts": str(pd.Timestamp.now(tz="UTC")),
                         "day": day or str(pd.Timestamp.now(tz="UTC").date()),
                         "level": level, "kind": kind, "detail": detail[:500]}])
    df = pd.read_parquet(p) if p.exists() else None
    df = pd.concat([df, row], ignore_index=True) if df is not None else row
    tmp = p.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)
    (log.error if level in ("P0", "P1") else log.warning)("[%s] %s: %s", level, kind, detail)


def load(store: Path) -> pd.DataFrame:
    p = _path(store)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=COLS)


def counts_since(store: Path, start_day: str) -> dict[str, int]:
    df = load(store)
    if not len(df):
        return {"P0": 0, "P1": 0, "P2": 0, "P3": 0}
    df = df[df["day"] >= start_day]
    return {lv: int((df["level"] == lv).sum()) for lv in ("P0", "P1", "P2", "P3")}
