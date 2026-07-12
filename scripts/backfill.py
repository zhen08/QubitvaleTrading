"""Full/incremental historical backfill from Binance Vision.

Usage (from repo root):
    python -m scripts.backfill                       # everything in settings.yaml
    python -m scripts.backfill --symbols BTCUSDT --markets spot --timeframes 1d
    python -m scripts.backfill --no-funding
Re-runs are incremental (manifest + existing parquet aware).
"""
from __future__ import annotations

import argparse

from data.collectors import binance_vision
from data.collectors.common import load_settings, setup_logging


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--markets", nargs="*", default=None, choices=["spot", "um"])
    ap.add_argument("--timeframes", nargs="*", default=None)
    ap.add_argument("--no-funding", action="store_true")
    args = ap.parse_args()

    settings = load_settings()
    results = binance_vision.backfill_all(
        settings,
        symbols=args.symbols,
        markets=args.markets,
        timeframes=args.timeframes,
        do_funding=not args.no_funding,
    )
    total_rows = sum(r.rows for r in results)
    print(f"\nBackfill done: {len(results)} series, {total_rows:,} rows total.")


if __name__ == "__main__":
    main()
