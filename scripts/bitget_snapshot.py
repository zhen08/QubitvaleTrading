"""Bitget live snapshot + funding history backfill (public endpoints).

Usage: python -m scripts.bitget_snapshot
"""
from __future__ import annotations

import json

from data.collectors import bitget_live
from data.collectors.common import load_settings, setup_logging


def main() -> None:
    setup_logging()
    settings = load_settings()
    snap = bitget_live.snapshot(settings)
    counts = bitget_live.backfill_funding_history(settings)
    print(json.dumps(snap, indent=1, ensure_ascii=False))
    print("bitget funding rows:", counts)


if __name__ == "__main__":
    main()
