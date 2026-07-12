"""Store layout + parquet/manifest IO.

Layout (all under data/store/):
    market/{spot|um}/{SYMBOL}/{tf}.parquet     K 线（含 symbol/market/timeframe 列）
    funding_um/{SYMBOL}.parquet                Binance USDT-M 资金费率史
    funding_bitget/{SYMBOL}.parquet            Bitget 资金费率史（CCXT）
    news/rss.parquet · news/gdelt.parquet      新闻
    live/                                      Bitget 快照 JSON
    manifest.json                              回填进度（增量续传用）
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.collectors.common import REPO_ROOT


def store_dir(settings: dict) -> Path:
    d = REPO_ROOT / settings.get("store_dir", "data/store")
    d.mkdir(parents=True, exist_ok=True)
    return d


def klines_path(store: Path, market: str, symbol: str, tf: str) -> Path:
    return store / "market" / market / symbol / f"{tf}.parquet"


def funding_um_path(store: Path, symbol: str) -> Path:
    return store / "funding_um" / f"{symbol}.parquet"


def funding_bitget_path(store: Path, symbol_slug: str) -> Path:
    return store / "funding_bitget" / f"{symbol_slug}.parquet"


def news_path(store: Path, name: str) -> Path:
    return store / "news" / f"{name}.parquet"


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet_if_exists(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def merge_on_ts(existing: pd.DataFrame | None, new: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    if existing is not None and len(existing):
        new = pd.concat([existing, new], ignore_index=True)
    new = (
        new.drop_duplicates(subset=[ts_col], keep="last")
        .sort_values(ts_col)
        .reset_index(drop=True)
    )
    return new


# ---------------- manifest ----------------

def manifest_path(store: Path) -> Path:
    return store / "manifest.json"


def load_manifest(store: Path) -> dict:
    p = manifest_path(store)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(store: Path, manifest: dict) -> None:
    with open(manifest_path(store), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False, default=str)
