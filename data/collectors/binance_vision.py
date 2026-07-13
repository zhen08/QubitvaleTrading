"""Binance Vision (data.binance.vision) historical backfill.

免费官方批量数据：现货 + USDT-M 合约 K 线（月度 zip + 当月日度 zip 补尾）与资金费率史。
设计要点：
- 上市前月份返回 404 → 记录到 manifest 并跳过，不视为错误；
- 时间戳单位自适应（2025-01 起现货为微秒，见 common.normalize_epoch_series）；
- 有/无表头的 CSV 都能解析（read_zipped_csv 自动嗅探）；
- 增量续传：已有 parquet 只补新月份/新日度文件；
- 本容器 api.binance.com 被 451 地理屏蔽，因此"当日未结算尾部"不从 REST 拉，
  日度文件覆盖到 T-1（Vision 约 T+1 发布），实时尾部由 Bitget 采集器负责。
"""
from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd

from data.collectors.common import http_get, normalize_epoch_series, read_zipped_csv, utc_today
from data import storeio

log = logging.getLogger("qvt.vision")

BASE = "https://data.binance.vision/data"

# R4: 校验官方 SHA-256（Binance 声明归档可能被修订，每个 zip 均有 .CHECKSUM 文件）。
# backfill_all 会按 settings.verify_checksums 覆盖此开关。
VERIFY_CHECKSUMS = True


def _fetch_verified(url: str):
    """下载 zip 并核对官方 CHECKSUM；不匹配则整体重下一次（覆盖归档修订窗口），
    仍不匹配 → 抛错（宁缺毋错）。404 → None。"""
    r = http_get(url, ok404=True)
    if r is None or not VERIFY_CHECKSUMS:
        return r
    for attempt in (1, 2):
        rc = http_get(url + ".CHECKSUM", ok404=True)
        if rc is None:                       # 个别文件无 CHECKSUM，放行并记录
            log.debug("no CHECKSUM for %s", url)
            return r
        expected = rc.text.strip().split()[0].lower()
        actual = hashlib.sha256(r.content).hexdigest()
        if actual == expected:
            return r
        if attempt == 1:
            log.warning("CHECKSUM mismatch, re-downloading: %s", url)
            r = http_get(url, ok404=True)
            if r is None:
                return None
    raise RuntimeError(f"CHECKSUM mismatch after retry: {url}")

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
KLINE_NUM_COLS = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]


# ---------------- URL builders ----------------

def _market_seg(market: str) -> str:
    return "spot" if market == "spot" else "futures/um"


def kline_month_url(market: str, symbol: str, tf: str, ym: str) -> str:
    return f"{BASE}/{_market_seg(market)}/monthly/klines/{symbol}/{tf}/{symbol}-{tf}-{ym}.zip"


def kline_day_url(market: str, symbol: str, tf: str, ymd: str) -> str:
    return f"{BASE}/{_market_seg(market)}/daily/klines/{symbol}/{tf}/{symbol}-{tf}-{ymd}.zip"


def funding_month_url(symbol: str, ym: str) -> str:
    return f"{BASE}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"


# ---------------- date helpers ----------------

def month_range(start_ym: str, end_ym: str) -> list[str]:
    """Inclusive list of YYYY-MM strings."""
    out = []
    cur = pd.Period(start_ym, freq="M")
    end = pd.Period(end_ym, freq="M")
    while cur <= end:
        out.append(str(cur))
        cur += 1
    return out


def last_complete_month(today: pd.Timestamp | None = None) -> str:
    t = today if today is not None else utc_today()
    return str(pd.Period(t, freq="M") - 1)


# ---------------- parsers ----------------

def parse_klines(content: bytes) -> pd.DataFrame:
    df = read_zipped_csv(content)
    df = df.iloc[:, :12].copy()
    df.columns = KLINE_COLS
    out = pd.DataFrame({"ts": normalize_epoch_series(df["open_time"])})
    for c in KLINE_NUM_COLS:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    out["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return out


def parse_funding(content: bytes) -> pd.DataFrame:
    """Funding CSV: header 通常为 calc_time,funding_interval_hours,last_funding_rate，
    但旧文件可能无表头/列序不同 → 用取值范围启发式区分 rate（|x|≤0.05）与 interval（小时数）。"""
    raw = read_zipped_csv(content)
    if raw.shape[1] < 2:
        raise ValueError("funding csv has <2 columns")

    cols = [str(c).lower() for c in raw.columns]
    ts_col = rate_col = interval_col = None
    if any("time" in c for c in cols):  # header present
        for i, c in enumerate(cols):
            if "time" in c and ts_col is None:
                ts_col = raw.columns[i]
            elif "rate" in c:
                rate_col = raw.columns[i]
            elif "interval" in c:
                interval_col = raw.columns[i]
    if ts_col is None:  # headerless: col0 = epoch; distinguish the rest by magnitude
        ts_col = raw.columns[0]
        rest = list(raw.columns[1:3])
        meds = {c: float(pd.to_numeric(raw[c], errors="coerce").abs().median()) for c in rest}
        rate_col = min(meds, key=meds.get)
        if len(rest) > 1:
            interval_col = max(meds, key=meds.get)

    out = pd.DataFrame({"ts": normalize_epoch_series(raw[ts_col])})
    out["funding_rate"] = pd.to_numeric(raw[rate_col], errors="coerce")
    out["interval_hours"] = (
        pd.to_numeric(raw[interval_col], errors="coerce") if interval_col is not None else pd.NA
    )
    out = out.dropna(subset=["ts", "funding_rate"]).sort_values("ts").reset_index(drop=True)
    return out


# ---------------- fetchers ----------------

def fetch_kline_month(market: str, symbol: str, tf: str, ym: str) -> pd.DataFrame | None:
    r = _fetch_verified(kline_month_url(market, symbol, tf, ym))
    return None if r is None else parse_klines(r.content)


def fetch_kline_day(market: str, symbol: str, tf: str, ymd: str) -> pd.DataFrame | None:
    r = _fetch_verified(kline_day_url(market, symbol, tf, ymd))
    return None if r is None else parse_klines(r.content)


def fetch_funding_month(symbol: str, ym: str) -> pd.DataFrame | None:
    r = _fetch_verified(funding_month_url(symbol, ym))
    return None if r is None else parse_funding(r.content)


# ---------------- backfill orchestration ----------------

@dataclass
class SeriesResult:
    key: str
    rows: int = 0
    first_ts: str | None = None
    last_ts: str | None = None
    months_404: list[str] = field(default_factory=list)
    new_months: int = 0
    new_days: int = 0
    repaired_days: int = 0


def repair_missing_days(
    df: pd.DataFrame, market: str, symbol: str, tf: str, workers: int = 8,
) -> tuple[pd.DataFrame, int]:
    """月度 zip 偶有整日空洞（例如 um/SOLUSDT 2022-02-26..28、04-01..02），
    但日度 zip 完整 → 找出完全缺失的 UTC 日，用日度文件补。只修整日缺口；
    日内零星缺根（交易所维护）是数据源真实状态，不修。"""
    if not len(df):
        return df, 0
    ts = pd.to_datetime(df["ts"], utc=True)
    days_present = set(ts.dt.normalize())
    all_days = pd.date_range(ts.min().normalize(), ts.max().normalize(), freq="D", tz="UTC")
    missing = [d for d in all_days if d not in days_present]
    if not missing:
        return df, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        frames = list(
            ex.map(lambda d: fetch_kline_day(market, symbol, tf, d.strftime("%Y-%m-%d")), missing)
        )
    got = [f for f in frames if f is not None and len(f)]
    log.info("repair %s/%s/%s: %d missing day(s), %d recovered from daily files",
             market, symbol, tf, len(missing), len(got))
    if got:
        df = pd.concat([df, *got], ignore_index=True)
        df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
    return df, len(got)


def _series_key(market: str, symbol: str, tf: str) -> str:
    return f"{market}/{symbol}/{tf}"


def backfill_klines_series(
    store, manifest: dict, market: str, symbol: str, tf: str,
    start_ym: str, workers: int = 12,
) -> SeriesResult:
    key = _series_key(market, symbol, tf)
    res = SeriesResult(key=key)
    path = storeio.klines_path(store, market, symbol, tf)
    existing = storeio.read_parquet_if_exists(path)
    m = manifest.setdefault(key, {})
    months_404_known = set(m.get("months_404", []))

    # months to fetch: from start (or month after last stored bar) through last complete month
    if existing is not None and len(existing):
        last_ts = pd.Timestamp(existing["ts"].max())
        fetch_from = str(pd.Period(last_ts, freq="M"))  # refetch last partial month
    else:
        fetch_from = start_ym
    months = [ym for ym in month_range(fetch_from, last_complete_month()) if ym not in months_404_known]

    frames: list[pd.DataFrame] = []
    if months:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda ym: (ym, fetch_kline_month(market, symbol, tf, ym)), months))
        for ym, df in results:
            if df is None:
                res.months_404.append(ym)
            else:
                frames.append(df)
                res.new_months += 1
    # record pre-listing 404s (only those BEFORE first successful month, to avoid masking real gaps)
    ok_months = sorted(ym for ym, df in results if df is not None) if months else []
    if ok_months:
        first_ok = ok_months[0]
        pre_listing = [ym for ym in res.months_404 if ym < first_ok]
        m["months_404"] = sorted(months_404_known | set(pre_listing))
        mid_gaps = [ym for ym in res.months_404 if ym > first_ok]
        if mid_gaps:
            log.warning("%s: monthly files missing mid-series: %s", key, mid_gaps)

    # daily tail: from day after last monthly bar through UTC yesterday (files are ~T+1)
    base = storeio.merge_on_ts(existing, pd.concat(frames, ignore_index=True)) if frames else (
        existing if existing is not None else pd.DataFrame(columns=["ts"])
    )
    if len(base):
        tail_start = (pd.Timestamp(base["ts"].max()).floor("D") + pd.Timedelta(days=1))
        yesterday = utc_today() - pd.Timedelta(days=1)
        days = pd.date_range(tail_start, yesterday, freq="D", tz="UTC")
        if len(days):
            with ThreadPoolExecutor(max_workers=workers) as ex:
                dresults = list(ex.map(
                    lambda d: fetch_kline_day(market, symbol, tf, d.strftime("%Y-%m-%d")), days
                ))
            dframes = [df for df in dresults if df is not None]
            res.new_days = len(dframes)
            if dframes:
                base = storeio.merge_on_ts(base, pd.concat(dframes, ignore_index=True))

    if len(base):
        base, res.repaired_days = repair_missing_days(base, market, symbol, tf, workers)
        base = base.copy()
        base["symbol"], base["market"], base["timeframe"] = symbol, market, tf
        storeio.write_parquet(base, path)
        res.rows = len(base)
        res.first_ts, res.last_ts = str(base["ts"].min()), str(base["ts"].max())
        m.update(rows=res.rows, first_ts=res.first_ts, last_ts=res.last_ts,
                 updated_at=str(pd.Timestamp.now(tz="UTC")))
    return res


def backfill_funding_series(store, manifest: dict, symbol: str, start_ym: str, workers: int = 8) -> SeriesResult:
    key = f"funding_um/{symbol}"
    res = SeriesResult(key=key)
    path = storeio.funding_um_path(store, symbol)
    existing = storeio.read_parquet_if_exists(path)
    m = manifest.setdefault(key, {})
    months_404_known = set(m.get("months_404", []))

    if existing is not None and len(existing):
        fetch_from = str(pd.Period(pd.Timestamp(existing["ts"].max()), freq="M"))
    else:
        fetch_from = start_ym
    months = [ym for ym in month_range(fetch_from, last_complete_month()) if ym not in months_404_known]

    frames = []
    results = []
    if months:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(lambda ym: (ym, fetch_funding_month(symbol, ym)), months))
        for ym, df in results:
            if df is None:
                res.months_404.append(ym)
            else:
                frames.append(df)
                res.new_months += 1
        ok_months = sorted(ym for ym, df in results if df is not None)
        if ok_months:
            pre = [ym for ym in res.months_404 if ym < ok_months[0]]
            m["months_404"] = sorted(months_404_known | set(pre))

    base = storeio.merge_on_ts(existing, pd.concat(frames, ignore_index=True)) if frames else existing
    if base is not None and len(base):
        base = base.copy()
        base["symbol"] = symbol
        storeio.write_parquet(base, path)
        res.rows = len(base)
        res.first_ts, res.last_ts = str(base["ts"].min()), str(base["ts"].max())
        m.update(rows=res.rows, first_ts=res.first_ts, last_ts=res.last_ts,
                 updated_at=str(pd.Timestamp.now(tz="UTC")))
    return res


def backfill_all(settings: dict, symbols=None, markets=None, timeframes=None, do_funding=True) -> list[SeriesResult]:
    global VERIFY_CHECKSUMS
    VERIFY_CHECKSUMS = bool(settings.get("verify_checksums", True))
    store = storeio.store_dir(settings)
    manifest = storeio.load_manifest(store)
    workers = int(settings.get("download_workers", 12))
    out: list[SeriesResult] = []
    for market in (markets or settings["markets"]):
        for symbol in (symbols or settings["symbols"]):
            for tf in (timeframes or settings["timeframes"]):
                r = backfill_klines_series(store, manifest, market, symbol, tf,
                                           settings["start_month"], workers)
                log.info("klines %-22s rows=%-7d +%dmo +%dd  404=%d",
                         r.key, r.rows, r.new_months, r.new_days, len(r.months_404))
                out.append(r)
                storeio.save_manifest(store, manifest)
    if do_funding:
        for symbol in (symbols or settings["symbols"]):
            r = backfill_funding_series(store, manifest, symbol,
                                        settings.get("funding_start_month", "2019-09"), workers)
            log.info("funding %-20s rows=%-6d +%dmo 404=%d", r.key, r.rows, r.new_months, len(r.months_404))
            out.append(r)
            storeio.save_manifest(store, manifest)
    return out
