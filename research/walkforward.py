"""Walk-forward evaluation harness.

协议（ex-ante 固定）：
  训练窗 730 根日线（~2 年）→ 按训练期净每期 Sharpe 选出该家族最优参数
  → 在随后 182 根（~6 个月）样本外持有该参数 → 滚动前进 182 根。
  若训练期最优净 Sharpe ≤ 0：该折**空仓**（真实操作者不会部署亏损策略；已在报告披露）。
实现要点：先对网格内每个变体在全样本上算好净收益列（参数化策略无路径依赖，
逐折选择只是按列取段），再做选择——快且无泄漏（指标 warmup 用折前历史属正常信息集）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from research.costs import CostModel
from research.engine import run_backtest
from research.metrics import sharpe_per_period
from research.strategies import FAMILY_FUNCS, GRIDS


@dataclass
class WFResult:
    family: str
    oos_net: pd.Series                      # 拼接后的样本外净收益
    folds: list[dict] = field(default_factory=list)
    variant_returns: pd.DataFrame | None = None  # 全样本 T×N（供 PBO / SR 方差）

    @property
    def deployed_frac(self) -> float:
        d = [f for f in self.folds if f["deployed"]]
        return len(d) / len(self.folds) if self.folds else 0.0


def build_variant_returns(
    df: pd.DataFrame,
    family: str,
    cost_model: CostModel,
    funding_by_day: pd.Series | None = None,
    fixed_kwargs: dict | None = None,
) -> pd.DataFrame:
    """全样本上每个网格变体的净收益列，列名 = 参数串。"""
    func = FAMILY_FUNCS[family]
    cols = {}
    for params in GRIDS[family]:
        pos = func(df, **params, **(fixed_kwargs or {}))
        res = run_backtest(df, pos, cost_model, funding_by_day)
        label = ",".join(f"{k}={v}" for k, v in params.items())
        cols[label] = res.net
    return pd.DataFrame(cols)


def walk_forward(
    variant_returns: pd.DataFrame,
    family: str,
    train_bars: int = 730,
    test_bars: int = 182,
    min_test_bars: int = 30,
) -> WFResult:
    T = len(variant_returns)
    oos_parts: list[pd.Series] = []
    folds: list[dict] = []
    start = train_bars
    while start < T:
        end = min(start + test_bars, T)
        if end - start < min_test_bars:
            break
        train = variant_returns.iloc[start - train_bars:start]
        test = variant_returns.iloc[start:end]
        train_sr = {c: sharpe_per_period(train[c]) for c in variant_returns.columns}
        best = max(train_sr, key=train_sr.get)
        deployed = train_sr[best] > 0
        oos = test[best] if deployed else pd.Series(0.0, index=test.index)
        oos_parts.append(oos)
        folds.append(
            {
                "fold_start": str(test.index[0].date()),
                "fold_end": str(test.index[-1].date()),
                "chosen": best if deployed else "(flat)",
                "train_sr_pp": round(train_sr[best], 4),
                "oos_sr_pp": round(sharpe_per_period(oos), 4),
                "deployed": bool(deployed),
            }
        )
        start = end
    oos_net = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    return WFResult(family=family, oos_net=oos_net, folds=folds,
                    variant_returns=variant_returns)
