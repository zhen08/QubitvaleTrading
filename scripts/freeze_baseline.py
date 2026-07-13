"""冻结各账本的期望带基准（自动化启动时执行；已冻结的账本自动跳过）。

Usage: python -m scripts.freeze_baseline [--force] [--book NAME]
"""
from __future__ import annotations

import argparse

from data.collectors.common import load_settings, setup_logging
from ops.tracking import freeze_baseline


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="覆盖已冻结基准（需留档说明理由）")
    ap.add_argument("--book", default=None, help="只处理指定账本")
    args = ap.parse_args()
    settings = load_settings()
    books = settings["paper"]["books"]
    targets = [args.book] if args.book else list(books)
    for book in targets:
        try:
            meta = freeze_baseline(settings, book, force=args.force)
            print(f"[{book}] frozen: {meta['n_days']} days {meta['window']} "
                  f"mu_d={meta['mu_d']:.5f} sd_d={meta['sd_d']:.5f}")
        except FileExistsError as exc:
            print(f"[{book}] skip: {exc}")


if __name__ == "__main__":
    main()
