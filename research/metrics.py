"""Performance metrics: Sharpe, drawdown, Deflated Sharpe Ratio, PBO (CSCV).

DSR — Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"：
  E[max SR_N] = √V[SR]·((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))，γ = Euler–Mascheroni
  DSR = Φ( (SR − E[maxSR])·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) )
  全部使用"每期"（非年化）SR；γ₄ 为普通峰度（正态=3）。
PBO — Bailey, Borwein, López de Prado, Zhu (2017)，CSCV：把 T×N 净收益矩阵切成
  S 块，遍历 C(S, S/2) 组合：训练半选最优列，在测试半查其相对排名 ω；
  PBO = P(ω < 0.5)。
无 scipy 依赖：正态分布用 statistics.NormalDist。
"""
from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

_N01 = NormalDist()
EULER_GAMMA = 0.5772156649015329

ANN = {"1d": 365, "4h": 6 * 365, "1h": 24 * 365}  # crypto 7×24


def sharpe(returns: pd.Series, timeframe: str = "1d") -> float:
    r = returns.dropna()
    if len(r) < 3 or float(r.std(ddof=1)) == 0.0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(ANN[timeframe]))


def sharpe_per_period(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 3 or float(r.std(ddof=1)) == 0.0:
        return 0.0
    return float(r.mean() / r.std(ddof=1))


def cagr(returns: pd.Series, timeframe: str = "1d") -> float:
    r = returns.dropna()
    if not len(r):
        return 0.0
    total = float((1 + r).prod())
    if total <= 0:
        return -1.0
    years = len(r) / ANN[timeframe]
    return total ** (1 / max(years, 1e-9)) - 1


def max_drawdown(returns: pd.Series) -> float:
    r = returns.dropna()
    if not len(r):
        return 0.0
    equity = (1 + r).cumprod()
    return float((equity / equity.cummax() - 1).min())


def ann_vol(returns: pd.Series, timeframe: str = "1d") -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * math.sqrt(ANN[timeframe])) if len(r) > 2 else 0.0


# ---------------- Deflated Sharpe Ratio ----------------

def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] under N independent trials with per-period SR variance across trials."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    z1 = _N01.inv_cdf(1 - 1 / n_trials)
    z2 = _N01.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(returns: pd.Series, n_trials: int, sr_variance: float) -> float:
    """DSR ∈ [0,1]：校正多重试验/偏度/峰度后，真实 SR>0 的置信度。"""
    r = returns.dropna()
    if len(r) < 20:
        return 0.0
    sr = sharpe_per_period(r)
    t = len(r)
    skew = float(r.skew())
    kurt = float(r.kurt()) + 3.0  # pandas 给的是超额峰度
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr**2))
    z = (sr - sr0) * math.sqrt(t - 1) / denom
    return float(_N01.cdf(z))


def trials_sr_variance(returns_matrix: pd.DataFrame) -> float:
    """跨试验（列）的每期 SR 方差，作为 E[maxSR] 的输入。"""
    srs = [sharpe_per_period(returns_matrix[c]) for c in returns_matrix.columns]
    return float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0


# ---------------- PBO via CSCV ----------------

def pbo_cscv(returns_matrix: pd.DataFrame, n_blocks: int = 10) -> float:
    """Probability of Backtest Overfitting。returns_matrix: T×N 各试验净收益。"""
    M = returns_matrix.dropna(how="all").fillna(0.0).to_numpy()
    t, n = M.shape
    if n < 2 or t < n_blocks * 5:
        return float("nan")
    blocks = np.array_split(np.arange(t), n_blocks)

    def _sr(x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=0)
        sd = x.std(axis=0, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(sd > 0, mu / sd, 0.0)
        return out

    below = 0
    total = 0
    for comb in combinations(range(n_blocks), n_blocks // 2):
        tr_idx = np.concatenate([blocks[i] for i in comb])
        te_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in comb])
        sr_tr = _sr(M[tr_idx])
        sr_te = _sr(M[te_idx])
        best = int(np.argmax(sr_tr))
        # 相对排名 ω ∈ (0,1)：best 列在测试集里的名次
        rank = float((sr_te < sr_te[best]).sum() + 0.5 * (sr_te == sr_te[best]).sum())
        omega = rank / n
        below += 1 if omega < 0.5 else 0
        total += 1
    return below / total if total else float("nan")


def summary(returns: pd.Series, timeframe: str = "1d") -> dict:
    return {
        "sharpe": round(sharpe(returns, timeframe), 2),
        "cagr_pct": round(100 * cagr(returns, timeframe), 1),
        "ann_vol_pct": round(100 * ann_vol(returns, timeframe), 1),
        "max_dd_pct": round(100 * max_drawdown(returns), 1),
        "n_bars": int(returns.dropna().shape[0]),
    }
