"""Baseline strategy signal functions.

每个函数返回 pos_decided[t]（在 t 收盘决定的目标仓位）；shift 由引擎统一执行。
网格在 GRIDS 中 **ex-ante 固定**（预注册精神：先定网格后看结果，DSR 的 N 由此而来）。
σ₂₀ vol-targeting 直接沿用 forecasting-method v2 的波动率框架。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _close(df: pd.DataFrame) -> pd.Series:
    s = df["close"].astype(float)
    if "ts" in df.columns:
        s.index = pd.to_datetime(df["ts"], utc=True)
    return s


def vol_target_scale(close: pd.Series, target_ann_vol: float, max_lev: float,
                     window: int = 20) -> pd.Series:
    """σ₂₀ 目标波动率缩放：scale = min(max_lev, target/realized_ann)。"""
    realized = close.pct_change().rolling(window).std() * math.sqrt(365)
    scale = (target_ann_vol / realized).clip(upper=max_lev)
    return scale.fillna(0.0)


# ---------------- families ----------------

def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=_close(df).index)


def sma_cross(df: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    c = _close(df)
    pos = (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(float)
    return pos


def donchian(df: pd.DataFrame, n_entry: int, n_exit: int) -> pd.Series:
    """长平通道突破：收盘破 n_entry 高点开多，破 n_exit 低点平仓。"""
    c = _close(df)
    upper = c.rolling(n_entry).max().shift(1)
    lower = c.rolling(n_exit).min().shift(1)
    pos = pd.Series(np.nan, index=c.index)
    pos[c > upper] = 1.0
    pos[c < lower] = 0.0
    return pos.ffill().fillna(0.0)


def tsmom(df: pd.DataFrame, lookback: int, target_vol: float,
          max_lev: float = 1.0, long_short: bool = False) -> pd.Series:
    """时间序列动量 + σ₂₀ vol targeting（Grobys 2025：vol 管理是动量存活的前提）。"""
    c = _close(df)
    mom = c.pct_change(lookback)
    raw = np.sign(mom) if long_short else (mom > 0).astype(float)
    pos = pd.Series(raw, index=c.index) * vol_target_scale(c, target_vol, max_lev)
    return pos.fillna(0.0)


def rsi_meanrev(df: pd.DataFrame, n: int, buy_below: float, exit_above: float) -> pd.Series:
    """RSI 均值回归（对照组：预期费后失败）。"""
    c = _close(df)
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50.0)
    pos = pd.Series(np.nan, index=c.index)
    pos[rsi < buy_below] = 1.0
    pos[rsi > exit_above] = 0.0
    return pos.ffill().fillna(0.0)


# ---------------- ex-ante 固定的参数网格 ----------------

GRIDS: dict[str, list[dict]] = {
    "sma_cross": [
        {"fast": f, "slow": s}
        for f in (20, 50, 100) for s in (100, 200, 300) if f < s
    ],                                                            # 8
    "donchian": [
        {"n_entry": n, "n_exit": max(5, n // 2)} for n in (20, 40, 55, 100)
    ],                                                            # 4
    "tsmom": [
        {"lookback": lb, "target_vol": tv}
        for lb in (30, 60, 90, 180) for tv in (0.10, 0.15, 0.20)
    ],                                                            # 12
    "rsi_meanrev": [
        {"n": n, "buy_below": b, "exit_above": e}
        for n in (7, 14) for b in (25, 30) for e in (55, 70)
    ],                                                            # 8
}

FAMILY_FUNCS = {
    "sma_cross": sma_cross,
    "donchian": donchian,
    "tsmom": tsmom,
    "rsi_meanrev": rsi_meanrev,
}

TOTAL_TRIALS_PER_SYMBOL = sum(len(v) for v in GRIDS.values())  # 32（不含 B&H 基准）
