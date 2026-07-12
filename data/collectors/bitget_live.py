"""Bitget live collectors via CCXT (public endpoints, no API keys needed).

角色分工（报告 §4.1）：Binance Vision 提供深度历史；Bitget 是实盘执行所与实时源。
Phase 0 提供：行情/资金费/OI 快照、近期 K 线拉取、Bitget 资金费率历史落库。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import ccxt
import pandas as pd

from data import storeio
from data.collectors.common import REPO_ROOT

log = logging.getLogger("qvt.bitget")

_EX: ccxt.bitget | None = None


def exchange(retries: int = 3) -> ccxt.bitget:
    global _EX
    if _EX is None:
        ex = ccxt.bitget({"enableRateLimit": True, "timeout": 60_000})
        # 某些环境（如带 TLS 检查代理的沙箱）通过 REQUESTS_CA_BUNDLE 指定自定义 CA。
        # ccxt 传参为 `verify=self.verify and self.validateServerSsl`（truthy 字符串会被
        # 折叠成 True），因此把 CA 路径放到 validateServerSsl 才能真正生效。
        # 本地环境未设置该变量时此段为 no-op。
        import os
        _bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
        if _bundle:
            ex.validateServerSsl = _bundle
        for attempt in range(1, retries + 1):
            try:
                ex.load_markets()
                break
            except ccxt.NetworkError as exc:  # transient hiccups are common
                if attempt == retries:
                    raise
                log.warning("load_markets attempt %d failed (%s), retrying…", attempt, exc)
                import time as _time
                _time.sleep(3 * attempt)
        _EX = ex
    return _EX


def _slug(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTCUSDT_PERP'; 'BTC/USDT' -> 'BTCUSDT'."""
    base = symbol.split("/")[0]
    quote = symbol.split("/")[1].split(":")[0]
    return f"{base}{quote}_PERP" if ":" in symbol else f"{base}{quote}"


def snapshot(settings: dict) -> dict:
    """One-shot market snapshot -> data/store/live/{latest.json, snapshot_*.json}."""
    ex = exchange()
    cfg = settings["bitget"]
    out: dict = {"fetched_at": str(pd.Timestamp.now(tz="UTC")), "spot": {}, "swap": {}}

    for s in cfg["spot_symbols"]:
        t = ex.fetch_ticker(s)
        out["spot"][s] = {
            "last": t.get("last"), "bid": t.get("bid"), "ask": t.get("ask"),
            "quote_volume_24h": t.get("quoteVolume"), "ts": t.get("datetime"),
        }
    for s in cfg["swap_symbols"]:
        t = ex.fetch_ticker(s)
        row = {
            "last": t.get("last"), "bid": t.get("bid"), "ask": t.get("ask"),
            "quote_volume_24h": t.get("quoteVolume"), "ts": t.get("datetime"),
        }
        try:
            fr = ex.fetch_funding_rate(s)
            row["funding_rate"] = fr.get("fundingRate")
            row["next_funding_time"] = fr.get("fundingDatetime")
        except Exception as exc:  # noqa: BLE001 — snapshot fields are best-effort
            log.warning("funding rate %s: %s", s, exc)
        try:
            oi = ex.fetch_open_interest(s)
            row["open_interest"] = oi.get("openInterestAmount") or oi.get("openInterestValue")
        except Exception as exc:  # noqa: BLE001
            log.warning("open interest %s: %s", s, exc)
        out["swap"][s] = row

    store = storeio.store_dir(settings)
    live_dir: Path = store / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    for name in (f"snapshot_{stamp}.json", "latest.json"):
        with open(live_dir / name, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    return out


def fetch_recent_ohlcv(symbol: str, timeframe: str = "1d", limit: int = 40) -> pd.DataFrame:
    """Recent candles as DataFrame [ts, open, high, low, close, volume] (UTC)."""
    ex = exchange()
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def backfill_funding_history(settings: dict) -> dict[str, int]:
    """Bitget funding history (8h) -> data/store/funding_bitget/{slug}.parquet, merged."""
    ex = exchange()
    cfg = settings["bitget"]
    store = storeio.store_dir(settings)
    pages = int(cfg.get("funding_history_pages", 5))
    counts: dict[str, int] = {}

    for s in cfg["swap_symbols"]:
        records: list[dict] = []
        # Bitget v2 history-fund-rate 用 pageNo/pageSize 分页（pageNo=1 最新，往后翻更旧）
        for page_no in range(1, pages + 1):
            try:
                page = ex.fetch_funding_rate_history(s, limit=100, params={"pageNo": page_no})
            except Exception as exc:  # noqa: BLE001 — best-effort pagination
                log.warning("funding history %s p%d: %s", s, page_no, exc)
                break
            if not page:
                break
            records.extend(page)
            if len(page) < 100:
                break
        if not records:
            counts[_slug(s)] = 0
            continue
        df = pd.DataFrame(
            {
                "ts": pd.to_datetime([r["timestamp"] for r in records], unit="ms", utc=True),
                "funding_rate": [r.get("fundingRate") for r in records],
            }
        ).dropna()
        path = storeio.funding_bitget_path(store, _slug(s))
        merged = storeio.merge_on_ts(storeio.read_parquet_if_exists(path), df)
        merged["symbol"] = _slug(s)
        storeio.write_parquet(merged, path)
        counts[_slug(s)] = len(merged)
        log.info("bitget funding %-14s rows=%d", _slug(s), len(merged))
    return counts
