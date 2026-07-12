"""GDELT DOC 2.0 collector — 免费全球新闻全文检索（15 分钟更新）。

产出 append-only 去重表：data/store/news/gdelt.parquet
列: [title, url, url_hash, domain, language, source_country, seen_utc, fetched_at]
"""
from __future__ import annotations

import hashlib
import logging

import pandas as pd

from data import storeio
from data.collectors.common import http_get

log = logging.getLogger("qvt.gdelt")

API = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch(settings: dict) -> pd.DataFrame:
    cfg = settings["news"]
    params = {
        "query": cfg["gdelt_query"],
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(cfg.get("gdelt_maxrecords", 100)),
        "timespan": cfg.get("gdelt_timespan", "24h"),
        "sort": "DateDesc",
    }
    # GDELT 免费接口慢且限流狠（429 常见）：单次尝试、长超时、失败非致命。
    # 采集是周期性任务（Phase 2 为 15–30 分钟一轮），错过一轮无损。
    try:
        r = http_get(API, params=params, timeout=90, retries=1)
    except Exception as exc:  # noqa: BLE001
        log.warning("gdelt fetch failed (throttled/slow is normal): %s", exc)
        return pd.DataFrame()
    try:
        arts = r.json().get("articles", [])
    except ValueError:  # GDELT occasionally returns plain-text errors
        log.warning("gdelt returned non-JSON (len=%d)", len(r.content))
        return pd.DataFrame()
    rows = []
    fetched_at = str(pd.Timestamp.now(tz="UTC"))
    for a in arts:
        url = a.get("url", "")
        if not url:
            continue
        rows.append(
            {
                "title": (a.get("title") or "")[:300],
                "url": url,
                "url_hash": hashlib.sha1(url.encode()).hexdigest(),
                "domain": a.get("domain", ""),
                "language": a.get("language", ""),
                "source_country": a.get("sourcecountry", ""),
                "seen_utc": pd.to_datetime(a.get("seendate"), format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce"),
                "fetched_at": fetched_at,
            }
        )
    return pd.DataFrame(rows)


def collect(settings: dict) -> int:
    new = fetch(settings)
    if new.empty:
        return 0
    store = storeio.store_dir(settings)
    path = storeio.news_path(store, "gdelt")
    existing = storeio.read_parquet_if_exists(path)
    if existing is not None and len(existing):
        new = new[~new["url_hash"].isin(set(existing["url_hash"]))]
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined = combined.sort_values("seen_utc").reset_index(drop=True)
    storeio.write_parquet(combined, path)
    return len(new)
