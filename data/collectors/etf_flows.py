"""US spot ETF net-flow collector (BTC, ETH, SOL).

Primary source: CoinGlass free "Hobbyist" API (BTC + ETH; needs COINGLASS_API_KEY).
Fallback source: Farside Investors HTML tables (BTC + ETH + SOL; free, no key). The
fallback fills any asset the primary didn't cover — so SOL (absent from the CoinGlass
free tier) comes from Farside, and if CoinGlass is unavailable Farside supplies all three.

Both sources are best-effort and non-fatal (like the GDELT collector): a missing key,
a Cloudflare 403, or a layout change just skips that source, leaving last-known flows
in place. NOTE: Farside is Cloudflare-protected and typically 403s from a headless/server
IP — it works from a normal browser network but may be blocked on a VPS/cloud box.

Store: data/store/etf/flows.parquet
  cols [asset, date, net_flow_usd_m, price_usd, source, fetched_at]  (one row per asset/day)
"""
from __future__ import annotations

import logging
import os
from html.parser import HTMLParser

import pandas as pd

from data import storeio
from data.collectors.common import http_session, load_env

log = logging.getLogger("qvt.etf")

TRACKED = ("BTC", "ETH", "SOL")
_BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0")

# ---- CoinGlass (primary) ----
CG_BASE = "https://open-api-v4.coinglass.com/api/etf/{path}/flow-history"
CG_ASSETS = {"BTC": "bitcoin", "ETH": "ethereum"}   # free tier: BTC + ETH only

# ---- Farside (fallback) ----
FARSIDE = {"BTC": "https://farside.co.uk/btc/",
           "ETH": "https://farside.co.uk/eth/",
           "SOL": "https://farside.co.uk/sol/"}


def _collect_coinglass(settings: dict, key: str) -> list[dict]:
    session = http_session()
    fetched_at = str(pd.Timestamp.now(tz="UTC"))
    rows: list[dict] = []
    for asset, path in CG_ASSETS.items():
        try:
            r = session.get(CG_BASE.format(path=path),
                            headers={"CG-API-KEY": key, "accept": "application/json"},
                            timeout=30)
            r.raise_for_status()
            j = r.json()
            recs = (j.get("data") if isinstance(j, dict) else j) or []
        except Exception as exc:  # noqa: BLE001
            log.warning("coinglass %s failed (non-fatal): %s", asset, exc)
            continue
        for rec in recs:
            ts, flow = rec.get("timestamp"), rec.get("flow_usd")
            if ts is None or flow is None:
                continue
            rows.append({"asset": asset,
                         "date": pd.to_datetime(int(ts), unit="ms", utc=True).normalize(),
                         "net_flow_usd_m": round(float(flow) / 1e6, 3),
                         "price_usd": rec.get("price_usd"),
                         "source": "coinglass", "fetched_at": fetched_at})
    return rows


class _TableRows(HTMLParser):
    """Flatten every <tr> on the page into a list of cell-text lists."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _num(s: str) -> float | None:
    """Parse a Farside cell: '1,234.5' -> 1234.5, '(45.6)' -> -45.6, '-'/'' -> 0.0."""
    s = s.strip().replace(",", "").replace("US$", "").replace("$", "")
    if s in ("", "-", "–", "—", "n/a", "N/A"):
        return 0.0
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").strip("+")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def parse_farside(html: str, asset: str, fetched_at: str = "") -> list[dict]:
    """Parse a Farside ETF-flow page: pull (Date, Total) from the flow table.

    Robust to changing per-ticker columns — it locates the header row containing
    both 'Date' and 'Total' and reads only those two. Summary rows (Total/Average/
    Maximum/Minimum) have a non-date first cell and are skipped by the date parse.
    """
    p = _TableRows()
    p.feed(html)
    header_i = None
    for i, r in enumerate(p.rows):
        low = [c.strip().lower() for c in r]
        if "date" in low and "total" in low:
            header_i, header = i, low
            break
    if header_i is None:
        return []
    di, ti = header.index("date"), header.index("total")
    out: list[dict] = []
    for r in p.rows[header_i + 1:]:
        if len(r) <= max(di, ti):
            continue
        d = pd.to_datetime(r[di].strip(), format="%d %b %Y", errors="coerce", utc=True)
        if pd.isna(d):
            continue
        v = _num(r[ti])
        if v is None:
            continue
        out.append({"asset": asset, "date": d.normalize(), "net_flow_usd_m": round(v, 3),
                    "price_usd": None, "source": "farside", "fetched_at": fetched_at})
    return out


def _collect_farside(session, assets: tuple[str, ...]) -> list[dict]:
    fetched_at = str(pd.Timestamp.now(tz="UTC"))
    rows: list[dict] = []
    for asset in assets:
        url = FARSIDE.get(asset)
        if not url:
            continue
        try:
            r = session.get(url, headers={"User-Agent": _BROWSER_UA,
                                          "Accept": "text/html,application/xhtml+xml"},
                            timeout=30)
            r.raise_for_status()
            got = parse_farside(r.text, asset, fetched_at)
        except Exception as exc:  # noqa: BLE001 — Cloudflare 403 / layout change is non-fatal
            log.warning("farside %s failed (non-fatal, often Cloudflare 403 on server IPs): %s",
                        asset, exc)
            continue
        rows.extend(got)
        log.info("farside %s: %d rows", asset, len(got))
    return rows


def collect(settings: dict) -> int:
    """CoinGlass primary + Farside fallback (fills missing assets, incl. SOL).

    Upserts by (asset, date). Returns the number of newly-added rows.
    """
    load_env()
    session = http_session()
    key = os.environ.get("COINGLASS_API_KEY", "")

    rows: list[dict] = _collect_coinglass(settings, key) if key else []
    if not key:
        log.info("etf_flows: no COINGLASS_API_KEY — trying Farside fallback only")
    covered = {r["asset"] for r in rows}
    missing = tuple(a for a in TRACKED if a not in covered)
    if missing:
        rows += _collect_farside(session, missing)      # Farside supplies SOL and any CG gaps

    if not rows:
        log.info("etf_flows: no data from any source (gate stays idle)")
        return 0
    new = pd.DataFrame(rows)
    store = storeio.store_dir(settings)
    path = storeio.etf_flows_path(store)
    existing = storeio.read_parquet_if_exists(path)
    if existing is not None and len(existing):
        existing["date"] = pd.to_datetime(existing["date"], utc=True)
        n_before = len(existing)
        combined = pd.concat([existing, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["asset", "date"], keep="last")  # newest fetch wins
        n_new = len(combined) - n_before
    else:
        combined = new
        n_new = len(new)
    combined = combined.sort_values(["asset", "date"]).reset_index(drop=True)
    storeio.write_parquet(combined, path)
    log.info("etf_flows: wrote %d rows (%d new); sources=%s",
             len(combined), max(0, n_new), sorted(new["source"].unique()))
    return int(max(0, n_new))
