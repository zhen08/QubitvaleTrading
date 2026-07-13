"""Run the full Phase 1 research suite and write the report.

Usage: python -m scripts.run_phase1
Exit code 0 whether or not the gate passes (the gate result is written in the report; it is a research conclusion, not an error).
"""
from __future__ import annotations

from data.collectors.common import setup_logging
from research.phase1 import run_all, write_report


def main() -> None:
    setup_logging()
    df, folds, carry, portfolio = run_all()
    path, certified = write_report(df, folds, carry, portfolio)
    print(f"\nreport -> {path}")
    print(f"statistical certification: {'PASS' if certified else 'FAIL — research candidate only'}"
          " (see the report's gate section for the two-tier verdict)")


if __name__ == "__main__":
    main()
