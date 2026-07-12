"""Run the full Phase 1 research suite and write the report.

Usage: python -m scripts.run_phase1
Exit code 0 无论门槛是否通过（门槛结果写在报告里；这是研究结论，不是错误）。
"""
from __future__ import annotations

from data.collectors.common import setup_logging
from research.phase1 import run_all, write_report


def main() -> None:
    setup_logging()
    df, folds, carry, portfolio = run_all()
    path, gate = write_report(df, folds, carry, portfolio)
    print(f"\nreport -> {path}\ngate: {'PASS' if gate else 'FAIL (no family qualifies)'}")


if __name__ == "__main__":
    main()
