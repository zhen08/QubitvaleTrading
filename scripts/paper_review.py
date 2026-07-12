"""生成 paper 周度复盘报告（paper vs 模型回放 vs Phase1 期望带）。

Usage: python -m scripts.paper_review
"""
from __future__ import annotations

from data.collectors.common import load_settings, setup_logging
from ops.tracking import build_review, write_review


def main() -> None:
    setup_logging()
    settings = load_settings()
    report, stats = build_review(settings)
    path = write_review(settings)
    print(report)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
