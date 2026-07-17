"""Track 3 data foundation: point-in-time spot universe from Binance Vision (plan §9.1).

Survivorship-bias control: symbols are enumerated from the Vision S3 bucket
listing, which retains full history for delisted pairs; the universe on any
historical date is derived from trailing dollar volume computed out of the
archive itself, never from a current listings snapshot.

Eligibility rules are FIXED ex ante (changing them = new preregistered rule set):
  - quote asset USDT;
  - base is not a stablecoin / fiat / commodity-backed token
    (any base containing "USD", plus the explicit list below);
  - base is not a leveraged token (UP/DOWN/BULL/BEAR suffixes);
  - base is not a wrapped duplicate of an asset that trades directly.
Market-cap constraints (plan §9.1) are deferred: no free point-in-time
market-cap history is wired yet; the first universe uses executable
30-day dollar volume only, and this limitation must be stated in any
Track 3 report.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from data import storeio
from data.collectors.common import http_get

log = logging.getLogger("qvt.universe")

S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINE_PREFIX = "data/spot/monthly/klines/"

# Fixed exclusion lists (ex ante; documented in the module docstring).
STABLE_FIAT_BASES = {
    "DAI", "UST", "USTC", "PAX", "PAXG", "EUR", "GBP", "AUD", "TRY", "BRL",
    "RUB", "UAH", "NGN", "ZAR", "BIDR", "IDRT", "VAI", "AEUR", "EURI", "XUSD",
}
WRAPPED_BASES = {"WBTC", "WETH", "WBETH", "WSOL", "STETH", "CBETH", "BETH", "WNXM"}
LEVERAGED_RE = re.compile(r".*(UP|DOWN|BULL|BEAR)$")


def list_spot_symbols() -> list[str]:
    """All spot symbols ever archived on Binance Vision (paginated S3 listing)."""
    symbols: list[str] = []
    marker = ""
    while True:
        r = http_get(S3_LIST, params={"prefix": KLINE_PREFIX, "delimiter": "/",
                                      "marker": marker}, timeout=60)
        text = r.text
        page = re.findall(rf"<Prefix>{re.escape(KLINE_PREFIX)}([^<]+)/</Prefix>", text)
        symbols.extend(page)
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", text)
        if not (m and "<IsTruncated>true</IsTruncated>" in text):
            break
        marker = m.group(1)
    log.info("vision listing: %d spot symbols (all quotes, incl. delisted)", len(symbols))
    return symbols


def eligible_usdt_bases(symbols: list[str]) -> list[str]:
    """Apply the fixed §9.1 rules; returns eligible SYMBOLS (…USDT)."""
    out = []
    for sym in symbols:
        if not sym.endswith("USDT") or sym == "USDT":
            continue
        base = sym[:-4]
        if "USD" in base:                       # USDC/TUSD/FDUSD/USDE/SUSD/…
            continue
        if base in STABLE_FIAT_BASES or base in WRAPPED_BASES:
            continue
        if LEVERAGED_RE.match(base):
            continue
        out.append(sym)
    return sorted(out)


def universe_panel_path(store: Path) -> Path:
    return store / "universe" / "panel.parquet"


def build_universe_panel(store: Path, symbols: list[str],
                         adv_window: int = 30, top_n: int = 50) -> pd.DataFrame:
    """Point-in-time daily panel: (date, symbol, close, dollar_vol, adv30, rank).

    `adv30[D]` uses quote volume through bar D — known at the D+1 00:00 UTC
    decision, matching the repo's D-1 convention. Ranks are per-date over
    symbols with a full trailing window (min_periods=adv_window), so a
    just-listed asset cannot enter the universe early.
    """
    frames = []
    for sym in symbols:
        path = storeio.klines_path(store, "spot", sym, "1d")
        df = storeio.read_parquet_if_exists(path)
        if df is None or len(df) < adv_window:
            continue
        ts = pd.to_datetime(df["ts"], utc=True).dt.normalize()
        part = pd.DataFrame({
            "date": ts, "symbol": sym,
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "dollar_vol": pd.to_numeric(df["quote_volume"], errors="coerce"),
        })
        part["adv30"] = part["dollar_vol"].rolling(adv_window,
                                                   min_periods=adv_window).mean()
        frames.append(part)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.dropna(subset=["adv30"])
    panel["rank"] = panel.groupby("date")["adv30"].rank(ascending=False, method="first")
    panel["in_universe"] = panel["rank"] <= top_n
    panel = panel.sort_values(["date", "rank"]).reset_index(drop=True)
    storeio.write_parquet(panel, universe_panel_path(store))
    return panel


def universe_coverage(panel: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Per-year diagnostics: how many assets had a full 30d window, and whether
    the ≥30-asset Track 3 prerequisite holds."""
    g = panel.groupby(panel["date"].dt.year)
    out = pd.DataFrame({
        "assets_with_adv30_median": g.apply(
            lambda d: int(d.groupby("date")["symbol"].count().median()),
            include_groups=False),
    })
    out["track3_prereq_30plus"] = out["assets_with_adv30_median"] >= 30
    return out


UM_KLINE_PREFIX = "data/futures/um/monthly/klines/"


def list_um_symbols() -> set[str]:
    """All USDT-M perp symbols ever archived on Vision (short-leg eligibility)."""
    symbols: set[str] = set()
    marker = ""
    while True:
        r = http_get(S3_LIST, params={"prefix": UM_KLINE_PREFIX, "delimiter": "/",
                                      "marker": marker}, timeout=60)
        text = r.text
        symbols.update(re.findall(
            rf"<Prefix>{re.escape(UM_KLINE_PREFIX)}([^<]+)/</Prefix>", text))
        m = re.search(r"<NextMarker>([^<]+)</NextMarker>", text)
        if not (m and "<IsTruncated>true</IsTruncated>" in text):
            break
        marker = m.group(1)
    log.info("vision UM listing: %d perp symbols", len(symbols))
    return symbols


def update_universe_tail(store: Path, symbols: list[str], workers: int = 16) -> int:
    """Lean daily incremental: fetch only missing T-1 daily files per symbol
    (no monthly refetch, no checksums-per-month sweep). Returns symbols updated."""
    from concurrent.futures import ThreadPoolExecutor

    from data.collectors import binance_vision as bv
    from data.collectors.common import utc_today

    yesterday = utc_today() - pd.Timedelta(days=1)

    def _one(sym: str) -> int:
        path = storeio.klines_path(store, "spot", sym, "1d")
        existing = storeio.read_parquet_if_exists(path)
        if existing is None or not len(existing):
            return 0
        last = pd.Timestamp(existing["ts"].max()).floor("D")
        days = pd.date_range(last + pd.Timedelta(days=1), yesterday, freq="D", tz="UTC")
        if not len(days) or len(days) > 21:      # long gaps belong to the full backfill
            return 0
        frames = []
        for day in days:
            try:
                df = bv.fetch_kline_day("spot", sym, "1d", day.strftime("%Y-%m-%d"))
            except Exception:  # noqa: BLE001 — tail file not published yet / transient
                df = None
            if df is not None and len(df):
                frames.append(df)
        if not frames:
            return 0
        merged = storeio.merge_on_ts(existing, pd.concat(frames, ignore_index=True))
        merged["symbol"], merged["market"], merged["timeframe"] = sym, "spot", "1d"
        storeio.write_parquet(merged, path)
        return 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, symbols))
    n = sum(results)
    log.info("universe tail update: %d/%d symbols advanced", n, len(symbols))
    return n
