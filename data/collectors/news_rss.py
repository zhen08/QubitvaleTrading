"""RSS news collector — stdlib parser (xml.etree), no feedparser dependency.

Feeds are RSS 2.0 (CoinDesk / Cointelegraph / The Block / 吴说 Substack)。
产出 append-only 去重表：data/store/news/rss.parquet
列: [source, title, link, published_utc, summary, fetched_at]
"""
from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import pandas as pd

from data import storeio
from data.collectors.common import http_get

log = logging.getLogger("qvt.rss")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean(text: str | None, limit: int = 1000) -> str:
    if not text:
        return ""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()[:limit]


def _parse_pubdate(text: str | None) -> pd.Timestamp | None:
    if not text:
        return None
    try:  # RFC 822 (standard RSS)
        dt = parsedate_to_datetime(text.strip())
        ts = pd.Timestamp(dt)
    except (TypeError, ValueError):
        try:  # ISO fallback
            ts = pd.Timestamp(text.strip())
        except ValueError:
            return None
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


def parse_rss(content: bytes, source: str) -> list[dict]:
    root = ET.fromstring(content)
    items = root.findall(".//item")  # RSS 2.0
    rows = []
    fetched_at = str(pd.Timestamp.now(tz="UTC"))
    for it in items:
        def _text(tag: str) -> str | None:
            el = it.find(tag)
            return el.text if el is not None else None

        link = (_text("link") or "").strip()
        title = _clean(_text("title"), 300)
        if not link or not title:
            continue
        rows.append(
            {
                "source": source,
                "title": title,
                "link": link,
                "link_hash": hashlib.sha1(link.encode()).hexdigest(),
                "published_utc": _parse_pubdate(_text("pubDate")),
                "summary": _clean(_text("description")),
                "fetched_at": fetched_at,
            }
        )
    return rows


def collect(settings: dict) -> int:
    """Fetch all feeds, append new (deduped by link_hash). Returns count of new rows."""
    feeds: dict[str, str] = settings["news"]["rss_feeds"]
    all_rows: list[dict] = []
    for source, url in feeds.items():
        try:
            r = http_get(url, timeout=30)
            rows = parse_rss(r.content, source)
            all_rows.extend(rows)
            log.info("rss %-14s %d items", source, len(rows))
        except Exception as exc:  # noqa: BLE001 — one bad feed must not kill the run
            log.warning("rss %s failed: %s", source, exc)

    if not all_rows:
        return 0
    new = pd.DataFrame(all_rows)
    store = storeio.store_dir(settings)
    path = storeio.news_path(store, "rss")
    existing = storeio.read_parquet_if_exists(path)
    if existing is not None and len(existing):
        new = new[~new["link_hash"].isin(set(existing["link_hash"]))]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined = combined.sort_values("published_utc").reset_index(drop=True)
    storeio.write_parquet(combined, path)
    return len(new)
