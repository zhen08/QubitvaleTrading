"""Track 3 universe backfill: all eligible Vision spot USDT pairs, 1d bars.

Usage: python -m scripts.backfill_universe [--limit N] [--panel-only]

Resumable: reuses the Vision manifest (pre-listing 404s recorded, incremental
tail). Rerunning is cheap and doubles as the universe's incremental updater.
--panel-only skips downloading and just rebuilds the point-in-time panel.
"""
from __future__ import annotations

import argparse
import logging

from data import storeio
from data.collectors import binance_vision
from data.collectors.common import load_settings, setup_logging
from data.collectors.universe import (build_universe_panel, eligible_usdt_bases,
                                      list_spot_symbols, universe_coverage)

log = logging.getLogger("qvt.universe")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap symbol count (testing)")
    ap.add_argument("--panel-only", action="store_true")
    args = ap.parse_args()

    settings = load_settings()
    store = storeio.store_dir(settings)

    symbols = eligible_usdt_bases(list_spot_symbols())
    log.info("eligible USDT symbols after fixed rules: %d", len(symbols))
    if args.limit:
        symbols = symbols[:args.limit]

    if not args.panel_only:
        done = 0
        for sym in symbols:
            try:
                binance_vision.backfill_all(settings, symbols=[sym], markets=["spot"],
                                            timeframes=["1d"], do_funding=False)
            except Exception as exc:  # noqa: BLE001 — one bad symbol must not kill the sweep
                log.warning("%s backfill failed (continuing): %s", sym, exc)
            done += 1
            if done % 25 == 0:
                log.info("universe backfill progress: %d/%d", done, len(symbols))

    panel = build_universe_panel(store, symbols)
    cov = universe_coverage(panel)
    print(f"panel rows: {len(panel):,}  symbols: {panel['symbol'].nunique()}  "
          f"dates: {panel['date'].nunique()}")
    print(cov.to_string())


if __name__ == "__main__":
    main()
