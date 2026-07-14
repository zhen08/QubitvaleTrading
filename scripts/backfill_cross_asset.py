"""Backfill SPY/QQQ/GLD/VIX daily history for the cross-asset DL research program.

Usage: python -m scripts.backfill_cross_asset [--start 2018-09-01]

Start default 2018-09-01 gives the 90-day sequence lookback plus indicator warmup
before the crypto history begins (2019-01). Idempotent: first-write-wins upsert.
"""
from __future__ import annotations

import argparse
from datetime import date

from data import storeio
from data.collectors.common import load_settings, setup_logging
from data.collectors.cross_asset_daily import (CboeVixProvider, ETF_SYMBOLS,
                                               YahooDailyProvider, qc_frame,
                                               upsert_daily)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-09-01")
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.today()

    settings = load_settings()
    store = storeio.store_dir(settings)

    yahoo = YahooDailyProvider()
    for sym in ETF_SYMBOLS:
        df = yahoo.fetch_history(sym, start, end)
        merged = upsert_daily(store, df)
        issues = qc_frame(merged, is_etf=True)
        print(f"{sym}: {len(merged)} sessions "
              f"({merged['session_date'].iloc[0].date()} → {merged['session_date'].iloc[-1].date()})"
              + (f"  QC ISSUES: {issues}" if issues else "  QC ok"))

    vix = CboeVixProvider().fetch_history("VIX", start, end)
    merged = upsert_daily(store, vix)
    issues = qc_frame(merged, is_etf=False)
    print(f"VIX: {len(merged)} sessions "
          f"({merged['session_date'].iloc[0].date()} → {merged['session_date'].iloc[-1].date()})"
          + (f"  QC ISSUES: {issues}" if issues else "  QC ok"))


if __name__ == "__main__":
    main()
