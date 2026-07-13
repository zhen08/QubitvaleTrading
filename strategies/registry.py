"""策略注册表：账本名 → 信号模块（统一接口 compute_weights(dfs)）。

新策略上线流程（纪律，不只是代码）：
  1) 在 research/ 走完 walk-forward + DSR/PBO，写入 phase1 报告（ex-ante 网格）；
  2) 新建 strategies/<name>.py，文件头写明预注册条款（网格、固定参数、gate）；
  3) 注册到 STRATEGIES + settings.paper.books 配置资金与起始日；
  4) freeze_baseline 冻结该账本期望带。
注意：每加一本账本都在扩大多重检验空间——Phase 3 的选择纪律见各策略文件头。
"""
from __future__ import annotations

from strategies import donchian_ensemble, tsmom_ensemble

STRATEGIES = {
    "donchian_ensemble": donchian_ensemble,
    "tsmom_ensemble": tsmom_ensemble,
}


def get_strategy(name: str):
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy '{name}' — registered: {sorted(STRATEGIES)}")
    return STRATEGIES[name]
