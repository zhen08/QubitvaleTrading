"""CoinGlass US spot ETF net-flow collector (BTC, ETH) — free "Hobbyist" tier.

Requires COINGLASS_API_KEY in .env. Best-effort and non-fatal (like the GDELT
collector): a missing key or a failed request just skips the update, leaving the
last-known flows in place so the risk gate keeps using the most recent data.

Store: data/store/etf/flows.parquet
  cols [asset, date, net_flow_usd_m, price_usd, fetched_at]  (one row per asset/day)

Docs: https://open-api-v4.coinglass.com/api/etf/{bitcoin|ethereum}/flow-history
      header CG-API-KEY; each row = {timestamp(ms), flow_usd (daily net USD), price_usd, ...}.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

from data import storeio
from data.collectors.common import http_session, load_env

log = logging.getLogger("qvt.etf")

BASE = "https://open-api-v4.coinglass.com/api/etf/{path}/flow-history"
ASSETS = {"BTC": "bitcoin", "ETH": "ethereum"}   # etf asset -> CoinGlass path segment


def _fetch(session, path: str, key: str) -> list[dict]:
    r = session.get(BASE.format(path=path),
                    headers={"CG-API-KEY": key, "accept": "application/json"},
                    timeout=30)
    r.raise_for_status()
    j = r.json()
    data = j.get("data") if isinstance(j, dict) else j
    return data or []


def collect(settings: dict) -> int:
    """Fetch BTC+ETH ETF flow history, upsert by (asset, date). Returns new-row count."""
    load_env()
    key = os.environ.get("COINGLASS_API_KEY", "")
    if not key:
        log.info("etf_flows: no COINGLASS_API_KEY set — skipping (gate stays inactive)")
        return 0

    session = http_session()
    fetched_at = str(pd.Timestamp.now(tz="UTC"))
    rows: list[dict] = []
    for asset, path in ASSETS.items():
        try:
            recs = _fetch(session, path, key)
        except Exception as exc:  # noqa: BLE001 — one bad asset/request must not kill the run
            log.warning("etf_flows %s failed (non-fatal): %s", asset, exc)
            continue
        for rec in recs:
            ts, flow = rec.get("timestamp"), rec.get("flow_usd")
            if ts is None or flow is None:
                continue
            rows.append({
                "asset": asset,
                "date": pd.to_datetime(int(ts), unit="ms", utc=True).normalize(),
                "net_flow_usd_m": round(float(flow) / 1e6, 3),
                "price_usd": rec.get("price_usd"),
                "fetched_at": fetched_at,
            })
        log.info("etf_flows %s: %d rows", asset, sum(1 for r in rows if r["asset"] == asset))

    if not rows:
        return 0
    new = pd.DataFrame(rows)
    store = storeio.store_dir(settings)
    path = storeio.etf_flows_path(store)
    existing = storeio.read_parquet_if_exists(path)
    if existing is not None and len(existing):
        existing["date"] = pd.to_datetime(existing["date"], utc=True)
        n_before = len(existing)
        combined = pd.concat([existing, new], ignore_index=True)
        # newest fetch wins for a given (asset, date) — flows can be revised post-close
        combined = combined.drop_duplicates(subset=["asset", "date"], keep="last")
        n_new = len(combined) - n_before
    else:
        combined = new
        n_new = len(new)
    combined = combined.sort_values(["asset", "date"]).reset_index(drop=True)
    storeio.write_parquet(combined, path)
    return int(max(0, n_new))
