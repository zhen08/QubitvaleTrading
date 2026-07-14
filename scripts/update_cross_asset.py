"""Daily incremental cross-asset update + QC + freshness status.

Usage: python -m scripts.update_cross_asset

Fetches the recent tail (30 calendar days) for SPY/QQQ/GLD/VIX, upserts
(first-write-wins), runs the structural + cross-source QC gate, and writes
data/store/cross_asset/status.json for the inference-time fail-closed check
(plan §5.5/§12.3): `ok=false` must block exposure increases, never force exits.

Exit code 1 when QC fails or data is unexpectedly stale — mirroring scripts.run_qc.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta

import pandas as pd

from data import storeio
from data.collectors.common import load_settings, setup_logging
from data.collectors.cross_asset_daily import (CboeVixProvider, ETF_SYMBOLS,
                                               YahooDailyProvider, freshness,
                                               load_daily, qc_cross_source,
                                               qc_frame, upsert_daily)


def main() -> int:
    setup_logging()
    settings = load_settings()
    store = storeio.store_dir(settings)
    start = date.today() - timedelta(days=30)
    end = date.today()

    issues: list[str] = []
    yahoo = YahooDailyProvider()
    for sym in ETF_SYMBOLS:
        try:
            upsert_daily(store, yahoo.fetch_history(sym, start, end))
        except Exception as exc:  # noqa: BLE001 — degrade to staleness, never crash the job
            issues.append(f"{sym}: fetch failed ({exc})")
    try:
        upsert_daily(store, CboeVixProvider().fetch_history("VIX", start, end))
    except Exception as exc:  # noqa: BLE001
        issues.append(f"VIX: fetch failed ({exc})")

    for sym in (*ETF_SYMBOLS, "VIX"):
        try:
            issues += qc_frame(load_daily(store, sym), is_etf=sym != "VIX")
        except FileNotFoundError:
            issues.append(f"{sym}: store missing")
    issues += qc_cross_source(store)

    status = freshness(store)
    status["qc_issues"] = issues
    status["ok"] = bool(status["ok"] and not issues)
    path = store / "cross_asset" / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)

    for sym, info in status["symbols"].items():
        print(f"{sym:4s} {info.get('status'):7s} last={info.get('last_session')}")
    for issue in issues:
        print(f"QC: {issue}")
    print(f"cross-asset status: {'OK' if status['ok'] else 'NOT OK (fail-closed: block adds)'}")
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
