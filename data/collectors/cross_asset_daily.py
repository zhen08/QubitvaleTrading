"""Cross-asset daily bars: SPY / QQQ / GLD (ETFs) + VIX (index).

Implements Stage 1 of research/deep_learning_cross_asset_implementation_plan.md (§5).

Providers (behind one protocol so a keyed official-close source can be swapped in):
  - CboeVixProvider   VIX daily OHLC from the official Cboe CSV (free, no key).
  - YahooDailyProvider SPY/QQQ/GLD OHLCV via the Yahoo chart API through curl_cffi
    (consolidated closes/volume; unofficial endpoint — a keyed Tiingo/Alpaca-SIP
    adapter is the intended long-term production source, same interface).
  - StooqDailyProvider optional QC alternative (JS-walled on some networks).
  - fred_close()      FRED fredgraph CSV closes (SP500 / NASDAQ100 / VIXCLS) for
    second-source QC. Index proxies: QC compares *returns*, not levels.

`available_at` semantics (plan §5.4): the U.S. close and the final VIX print are
disseminated minutes after 16:00/16:15 ET and EOD files can be revised during the
evening, so a completed session's value is stamped available no earlier than
21:30 UTC on the session date (covers both EST and EDT closes with margin). Rows
ingested before their floor are dropped (incomplete session). First write wins on
(symbol, session_date) so point-in-time history is never silently rewritten.

Store (plan §5.3):
  data/store/cross_asset/market/{SPY,QQQ,GLD}/1d.parquet
  data/store/cross_asset/index/VIX/1d.parquet
"""
from __future__ import annotations

import io
import logging
from datetime import date, time, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from data import storeio
from data.collectors.common import http_get

log = logging.getLogger("qvt.xasset")

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

# Completed-session availability floor: never stamp a same-day close available
# before this UTC wall time (16:00 ET close = 20:00/21:00 UTC; VIX final ≈16:15 ET).
AVAILABILITY_FLOOR_UTC = time(21, 30)

ETF_SYMBOLS = ("SPY", "QQQ", "GLD")
INDEX_SYMBOLS = ("VIX",)

COLUMNS = ["symbol", "session_date", "bar_start", "bar_end", "available_at",
           "open", "high", "low", "close", "volume", "is_market_open",
           "source", "ingested_at"]

CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# FRED second-source series for return-level QC (index proxies for the ETFs).
FRED_QC_SERIES = {"SPY": "SP500", "QQQ": "NASDAQ100", "VIX": "VIXCLS"}


def cross_asset_path(store: Path, symbol: str) -> Path:
    kind = "index" if symbol in INDEX_SYMBOLS else "market"
    return store / "cross_asset" / kind / symbol / "1d.parquet"


def _floor_utc(session_date: pd.Series) -> pd.Series:
    """21:30 UTC on each session date, tz-aware."""
    return pd.to_datetime(session_date).dt.tz_localize(UTC) + pd.Timedelta(
        hours=AVAILABILITY_FLOOR_UTC.hour, minutes=AVAILABILITY_FLOOR_UTC.minute)


def _session_bounds(session_date: pd.Series, close_et: time) -> tuple[pd.Series, pd.Series]:
    """Regular-session start/end in UTC from NY wall clock (DST-aware).

    Early-close days keep the regular 16:00 ET label — the availability floor
    (21:30 UTC) still upper-bounds true availability, which is what correctness
    of the as-of join depends on.
    """
    d = pd.to_datetime(session_date)
    start = d.dt.tz_localize(NY) + pd.Timedelta(hours=9, minutes=30)
    end = d.dt.tz_localize(NY) + pd.Timedelta(hours=close_et.hour, minutes=close_et.minute)
    return start.dt.tz_convert(UTC), end.dt.tz_convert(UTC)


def canonical_frame(symbol: str, df: pd.DataFrame, source: str,
                    close_et: time = time(16, 0),
                    now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Normalize a provider frame (session_date + OHLC[+volume]) to the storage contract.

    Drops sessions whose availability floor is still in the future (incomplete or
    same-evening bars we cannot yet trust as final).
    """
    now = now or pd.Timestamp.now(tz="UTC")
    out = df.copy()
    out["session_date"] = pd.to_datetime(out["session_date"]).dt.normalize()
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.drop_duplicates(subset=["session_date"], keep="last").sort_values("session_date")
    bar_start, bar_end = _session_bounds(out["session_date"], close_et)
    floor = _floor_utc(out["session_date"])
    out["symbol"] = symbol
    out["bar_start"] = bar_start
    out["bar_end"] = bar_end
    out["available_at"] = floor
    out["is_market_open"] = True
    out["source"] = source
    out["ingested_at"] = now
    if "volume" not in out:
        out["volume"] = pd.NA
    out = out[out["available_at"] <= now]          # completed & disseminated only
    return out[COLUMNS].reset_index(drop=True)


class CrossAssetDailyProvider(Protocol):
    def fetch_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Canonical-contract frame for [start, end]."""
        ...


# ---------------- Cboe VIX (official) ----------------

class CboeVixProvider:
    source = "cboe"

    def fetch_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        assert symbol == "VIX"
        r = http_get(CBOE_VIX_URL, timeout=60)
        raw = pd.read_csv(io.BytesIO(r.content))
        raw.columns = [c.strip().lower() for c in raw.columns]
        raw = raw.rename(columns={"date": "session_date"})
        raw["session_date"] = pd.to_datetime(raw["session_date"])
        m = (raw["session_date"].dt.date >= start) & (raw["session_date"].dt.date <= end)
        # VIX is computed into the 16:15 ET settlement window.
        return canonical_frame("VIX", raw.loc[m], self.source, close_et=time(16, 15))


# ---------------- Yahoo chart API (consolidated OHLCV) ----------------

class YahooDailyProvider:
    source = "yahoo"
    BASE = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

    def fetch_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        from curl_cffi import requests as cr  # TLS-fingerprint gate, like Farside
        p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
        p2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=2)).timestamp())
        r = cr.get(self.BASE.format(sym=symbol),
                   params={"period1": p1, "period2": p2, "interval": "1d",
                           "events": "div,split"},
                   impersonate="chrome", timeout=30)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        quote = res["indicators"]["quote"][0]
        raw = pd.DataFrame({
            # Yahoo stamps bars with the session *open* epoch in exchange time.
            "session_date": pd.to_datetime(res["timestamp"], unit="s", utc=True)
                              .tz_convert(NY).normalize().tz_localize(None),
            "open": quote["open"], "high": quote["high"],
            "low": quote["low"], "close": quote["close"], "volume": quote["volume"],
        })
        m = (raw["session_date"].dt.date >= start) & (raw["session_date"].dt.date <= end)
        return canonical_frame(symbol, raw.loc[m], self.source)


# ---------------- Stooq (optional QC alternative; JS-walled on some networks) ----------------

class StooqDailyProvider:
    source = "stooq"
    BASE = "https://stooq.com/q/d/l/"

    def fetch_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        from curl_cffi import requests as cr
        r = cr.get(self.BASE, params={"s": f"{symbol.lower()}.us", "i": "d",
                                      "d1": start.strftime("%Y%m%d"),
                                      "d2": end.strftime("%Y%m%d")},
                   impersonate="chrome", timeout=30)
        r.raise_for_status()
        if r.text.lstrip().startswith("<"):
            raise RuntimeError("stooq returned an HTML challenge page, not CSV")
        raw = pd.read_csv(io.StringIO(r.text))
        raw.columns = [c.strip().lower() for c in raw.columns]
        raw = raw.rename(columns={"date": "session_date"})
        return canonical_frame(symbol, raw, self.source)


def fred_close(series_id: str, start: date) -> pd.Series:
    """FRED close-only series (QC second source). Index = naive session date."""
    r = http_get(FRED_CSV, params={"id": series_id, "cosd": str(start)}, timeout=60)
    raw = pd.read_csv(io.BytesIO(r.content), na_values=["."])
    raw.columns = ["session_date", "close"]
    raw["session_date"] = pd.to_datetime(raw["session_date"])
    return raw.dropna().set_index("session_date")["close"].astype(float)


# ---------------- storage ----------------

def upsert_daily(store: Path, df: pd.DataFrame) -> pd.DataFrame:
    """First-write-wins upsert per (symbol, session_date); atomic parquet write.

    Point-in-time discipline: a session already on disk keeps its original values
    and `available_at`/`ingested_at`. Provider revisions therefore never silently
    rewrite history — cross-source QC is where discrepancies must surface.
    """
    if df.empty:
        return df
    assert df["symbol"].nunique() == 1
    path = cross_asset_path(store, df["symbol"].iloc[0])
    existing = storeio.read_parquet_if_exists(path)
    if existing is not None and len(existing):
        merged = pd.concat([existing, df], ignore_index=True)
        merged = (merged.drop_duplicates(subset=["session_date"], keep="first")
                        .sort_values("session_date").reset_index(drop=True))
    else:
        merged = df.sort_values("session_date").reset_index(drop=True)
    storeio.write_parquet(merged, path)
    return merged


def load_daily(store: Path, symbol: str) -> pd.DataFrame:
    df = storeio.read_parquet_if_exists(cross_asset_path(store, symbol))
    if df is None:
        raise FileNotFoundError(f"no cross-asset store for {symbol}; run scripts.backfill_cross_asset")
    return df.sort_values("session_date").reset_index(drop=True)


# ---------------- QC gate (plan §5.5) ----------------

def qc_frame(df: pd.DataFrame, is_etf: bool) -> list[str]:
    """Structural checks on one symbol's stored frame. Returns list of issues."""
    issues: list[str] = []
    sym = df["symbol"].iloc[0] if len(df) else "?"
    if df["session_date"].duplicated().any():
        issues.append(f"{sym}: duplicate session_date rows")
    if not df["bar_end"].is_monotonic_increasing:
        issues.append(f"{sym}: bar_end not monotonic")
    if not df["available_at"].is_monotonic_increasing:
        issues.append(f"{sym}: available_at not monotonic")
    if (df["available_at"] < df["bar_end"]).any():
        issues.append(f"{sym}: available_at earlier than bar_end")
    if is_etf:
        bad = ~((df["low"] <= df[["open", "close"]].min(axis=1))
                & (df[["open", "close"]].max(axis=1) <= df["high"]))
        if bad.any():
            issues.append(f"{sym}: {int(bad.sum())} rows violate OHLC ordering")
        if (pd.to_numeric(df["volume"], errors="coerce").fillna(0) < 0).any():
            issues.append(f"{sym}: negative volume")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        issues.append(f"{sym}: non-positive prices")
    # Session-gap sanity: >7 calendar days between consecutive sessions is beyond
    # any US holiday cluster and means missing data, not a holiday.
    gaps = df["session_date"].diff().dt.days.dropna()
    if (gaps > 7).any():
        worst = df["session_date"][gaps[gaps > 7].index]
        issues.append(f"{sym}: unexplained session gaps at {list(worst.dt.date)[:5]}")
    return issues


def qc_cross_source(store: Path, lookback_sessions: int = 40,
                    ret_tol: float = 0.006, level_tol: float = 0.005) -> list[str]:
    """Recent-window agreement with FRED (returns for ETF-vs-index, levels for VIX).

    ETF vs index proxy differs by dividends on ex-days and tracking error, so the
    comparison is on daily log returns with a 60 bp tolerance and flags only when
    more than 2 of the recent sessions breach.
    """
    import numpy as np
    issues: list[str] = []
    for sym, series_id in FRED_QC_SERIES.items():
        try:
            df = load_daily(store, sym)
        except FileNotFoundError:
            continue
        tail = df.tail(lookback_sessions).set_index("session_date")
        try:
            alt = fred_close(series_id, tail.index[0].date() - timedelta(days=7))
        except Exception as exc:  # noqa: BLE001 — QC source outage is non-fatal
            issues.append(f"{sym}: QC second source {series_id} unavailable ({exc})")
            continue
        joined = pd.concat({"ours": tail["close"], "alt": alt}, axis=1).dropna()
        if len(joined) < 10:
            issues.append(f"{sym}: only {len(joined)} overlapping QC sessions with {series_id}")
            continue
        if sym == "VIX":
            rel = (joined["ours"] / joined["alt"] - 1).abs()
            n_bad = int((rel > level_tol).sum())
            if n_bad:
                issues.append(f"VIX: {n_bad} sessions deviate >{level_tol:.1%} from {series_id}")
        else:
            ours_r = np.log(joined["ours"]).diff().dropna()
            alt_r = np.log(joined["alt"]).diff().dropna()
            diff = (ours_r - alt_r).abs().dropna()
            n_bad = int((diff > ret_tol).sum())
            if n_bad > 2:
                issues.append(f"{sym}: {n_bad} daily returns deviate >{ret_tol:.2%} from {series_id}")
    return issues


def freshness(store: Path, now: pd.Timestamp | None = None,
              max_stale_sessions: int = 1) -> dict:
    """Per-symbol freshness for the daily inference gate.

    A symbol is `stale` when more than `max_stale_sessions` *expected* NYSE-family
    sessions have closed since its last stored session (weekends/holidays are not
    staleness). Uses exchange_calendars XNYS.
    """
    import exchange_calendars as xcals
    now = now or pd.Timestamp.now(tz="UTC")
    cal = xcals.get_calendar("XNYS", start=str((now - pd.Timedelta(days=40)).date()))
    sessions = cal.sessions  # tz-naive session dates
    # Sessions whose availability floor has passed:
    closed = [s for s in sessions
              if pd.Timestamp(s, tz="UTC")
              + pd.Timedelta(hours=AVAILABILITY_FLOOR_UTC.hour,
                             minutes=AVAILABILITY_FLOOR_UTC.minute) <= now]
    out: dict[str, dict] = {"as_of": str(now), "symbols": {}}
    for sym in (*ETF_SYMBOLS, *INDEX_SYMBOLS):
        try:
            df = load_daily(store, sym)
        except FileNotFoundError:
            out["symbols"][sym] = {"status": "missing"}
            continue
        last = df["session_date"].iloc[-1]
        missed = [s for s in closed if pd.Timestamp(s) > last]
        status = "ok" if len(missed) <= max_stale_sessions else "stale"
        out["symbols"][sym] = {
            "status": status,
            "last_session": str(last.date()),
            "missed_closed_sessions": [str(pd.Timestamp(s).date()) for s in missed],
            "days_since_last_session": int((now.tz_localize(None) - last).days),
        }
    out["ok"] = all(v.get("status") == "ok" for v in out["symbols"].values())
    return out
