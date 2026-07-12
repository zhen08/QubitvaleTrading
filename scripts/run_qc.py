"""Run Phase 0 data-quality checks; exit code 1 if gate fails.

Usage: python -m scripts.run_qc
"""
from __future__ import annotations

import sys

from data.collectors.common import load_settings, setup_logging
from research import qc_report


def main() -> None:
    setup_logging()
    settings = load_settings()
    report, passed = qc_report.run(settings)
    print("\n" + report)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
