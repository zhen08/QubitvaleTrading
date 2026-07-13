"""冻结 Phase 2 期望带基准（自动化正式启动时执行一次）。

Usage: python -m scripts.freeze_baseline [--force]
把 Phase 1 组合口径的 OOS 日收益序列与 μ/σ 固化到 data/store/paper/baseline.*，
此后 paper_review 的期望带不再随数据更新漂移（第二轮 review 修正）。
setup_mac.sh 会在安装时自动调用（已存在则跳过）。
"""
from __future__ import annotations

import argparse

from data.collectors.common import load_settings, setup_logging
from ops.tracking import freeze_baseline


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="覆盖已冻结的基准（需说明理由并留档）")
    args = ap.parse_args()
    try:
        meta = freeze_baseline(load_settings(), force=args.force)
        print(f"baseline frozen: {meta['n_days']} days {meta['window']} "
              f"mu_d={meta['mu_d']:.5f} sd_d={meta['sd_d']:.5f}")
    except FileExistsError as exc:
        print(f"skip: {exc}")


if __name__ == "__main__":
    main()
