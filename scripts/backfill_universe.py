"""Track 3 universe backfill: all eligible Vision spot USDT pairs, 1d bars.

Usage: python -m scripts.backfill_universe [--limit N] [--panel-only]
       python -m scripts.backfill_universe --symbol KLAYUSDT   (internal: one symbol)

Resumable: reuses the Vision manifest (pre-listing 404s recorded, incremental
tail). Rerunning is cheap and doubles as the universe's incremental updater.
--panel-only skips downloading and just rebuilds the point-in-time panel.

Each symbol runs in its own subprocess with a hard timeout: a single stalled
TLS connection (observed with KLAYUSDT on 2026-07-15: an unmoved socket
send-queue that never trips the requests timeout) must cost one symbol,
not the whole sweep. Timed-out symbols are reported at the end.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from data import storeio
from data.collectors import binance_vision
from data.collectors.common import load_settings, setup_logging
from data.collectors.universe import (build_universe_panel, eligible_usdt_bases,
                                      list_spot_symbols, universe_coverage)

log = logging.getLogger("qvt.universe")

PER_SYMBOL_TIMEOUT_S = 600


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap symbol count (testing)")
    ap.add_argument("--panel-only", action="store_true")
    ap.add_argument("--symbol", default=None, help="internal: backfill one symbol")
    args = ap.parse_args()

    settings = load_settings()
    store = storeio.store_dir(settings)

    if args.symbol:                      # child mode: one symbol, then exit
        binance_vision.backfill_all(settings, symbols=[args.symbol],
                                    markets=["spot"], timeframes=["1d"],
                                    do_funding=False)
        return

    symbols = eligible_usdt_bases(list_spot_symbols())
    log.info("eligible USDT symbols after fixed rules: %d", len(symbols))
    if args.limit:
        symbols = symbols[:args.limit]

    failed: list[str] = []
    if not args.panel_only:
        for done, sym in enumerate(symbols, 1):
            try:
                subprocess.run([sys.executable, "-m", "scripts.backfill_universe",
                                "--symbol", sym],
                               timeout=PER_SYMBOL_TIMEOUT_S, check=True,
                               capture_output=True)
            except subprocess.TimeoutExpired:
                failed.append(sym)
                log.warning("%s timed out after %ds (skipped)", sym, PER_SYMBOL_TIMEOUT_S)
            except Exception as exc:  # noqa: BLE001 — one bad symbol must not kill the sweep
                failed.append(sym)
                log.warning("%s backfill failed (continuing): %s", sym, exc)
            if done % 25 == 0:
                log.info("universe backfill progress: %d/%d", done, len(symbols))

    panel = build_universe_panel(store, symbols)
    cov = universe_coverage(panel)
    print(f"panel rows: {len(panel):,}  symbols: {panel['symbol'].nunique()}  "
          f"dates: {panel['date'].nunique()}")
    if failed:
        print(f"failed/timed-out symbols ({len(failed)}): {failed} — rerun to retry")
    print(cov.to_string())


if __name__ == "__main__":
    main()
