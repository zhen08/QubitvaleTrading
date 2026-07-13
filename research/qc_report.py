"""Phase 0 data quality checks + report.

Gates (config/settings.yaml → qc, from research report §6.6):
  1) Structure: each K-line series gap rate ≤ max_missing_pct; duplicate timestamps = 0; OHLC violations = 0;
  2) Freshness: the last daily bar is ≤ freshness_max_age_days days old (Vision daily files publish T+1);
  3) Cross-source: spot daily close over the last cross_window_days settled days,
     vs CoinGecko (alignment rule: CG's D+1 00:00 UTC snapshot price ↔ our D-day close)
     vs Coinbase (same-day bucket close compared directly), deviation on any day < cross_max_diff_pct%;
  4) Bitget consistency (informational + gate): Bitget's most recent settled close vs our spot close < 0.5%.
Funding rates get a sanity check (magnitude/cadence), warning-only with no gate.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from data import storeio
from data.collectors.common import http_get, utc_today

log = logging.getLogger("qvt.qc")

TF_DELTA = {"1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    gate: bool = True  # False = informational only


@dataclass
class QCResult:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str, gate: bool = True) -> None:
        self.checks.append(Check(name, passed, detail, gate))
        (log.info if passed else log.warning)("%s %s — %s", "PASS" if passed else "FAIL", name, detail)

    @property
    def gate_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.gate)


# ---------------- structural checks ----------------

def check_klines_structure(df: pd.DataFrame, tf: str, qc_cfg: dict, key: str, res: QCResult) -> None:
    """R4 hardening: NaN rows, timestamp-grid misalignment, and off-grid extra records
    all count as violations; missing is computed against the aligned grid and cannot be negative."""
    delta = TF_DELTA[tf]
    ts = pd.to_datetime(df["ts"], utc=True)

    # 1) NaN rows (any OHLCV missing)
    ohlcv = ["open", "high", "low", "close", "volume"]
    nan_rows = int(df[ohlcv].isna().any(axis=1).sum())

    # 2) Grid alignment: timestamps must land on the UTC grid (1d=00:00, 1h=on the hour, 4h=00/04/08…)
    #    Compare via floor (unit-independent — pandas 3.0 changed the default datetime precision from ns to us)
    floor_freq = {"1h": "h", "4h": "4h", "1d": "D"}[tf]
    aligned_mask = ts == ts.dt.floor(floor_freq)
    misaligned = int((~aligned_mask).sum())
    aligned = ts[aligned_mask]

    # 3) Gaps (computed only on the aligned grid, max(0,·) guards against negatives)
    dups = int(ts.duplicated().sum())
    if len(aligned):
        first, last = aligned.min(), aligned.max()
        expected = int((last - first) / delta) + 1
        missing = max(0, expected - aligned.nunique())
        missing_pct = 100.0 * missing / expected if expected else 0.0
    else:
        first = last = pd.NaT
        missing, missing_pct = len(df), 100.0

    # 4) OHLC consistency (judged only on non-NaN rows; NaN rows already counted as violations)
    valid = df[ohlcv].notna().all(axis=1)
    d = df[valid]
    viol = int(
        (
            (d["low"] > d[["open", "close"]].min(axis=1) + 1e-12)
            | (d["high"] < d[["open", "close"]].max(axis=1) - 1e-12)
            | (d["low"] > d["high"])
            | (d["volume"] < 0)
        ).sum()
    )
    ok = (
        missing_pct <= float(qc_cfg["max_missing_pct"])
        and dups <= int(qc_cfg["max_dup"])
        and viol <= int(qc_cfg["max_ohlc_violations"])
        and nan_rows == 0
        and misaligned == 0
    )
    res.add(
        f"structure {key}",
        ok,
        f"rows={len(df)} span={first.date() if first is not pd.NaT else '?'}→"
        f"{last.date() if last is not pd.NaT else '?'} missing={missing}({missing_pct:.3f}%) "
        f"dup={dups} ohlc_viol={viol} nan_rows={nan_rows} misaligned={misaligned}",
    )


def check_freshness(df: pd.DataFrame, qc_cfg: dict, key: str, res: QCResult) -> None:
    age_days = (utc_today() - pd.to_datetime(df["ts"], utc=True).max().normalize()).days
    ok = age_days <= int(qc_cfg["freshness_max_age_days"])
    res.add(f"freshness {key}", ok, f"last bar {df['ts'].max()} (age {age_days}d)")


# ---------------- cross-source checks ----------------

def _our_daily_closes(store: Path, symbol: str, market: str = "spot") -> pd.Series:
    df = pd.read_parquet(storeio.klines_path(store, market, symbol, "1d"))
    ts = pd.to_datetime(df["ts"], utc=True)
    s = pd.Series(df["close"].to_numpy(), index=ts.dt.normalize())
    return s[~s.index.duplicated(keep="last")]


def fetch_coingecko_daily(coin_id: str, days: int = 40, retries: int = 4) -> pd.Series:
    """CG market_chart daily: points stamped at 00:00 UTC (last point = live).
    Returns Series indexed by snapshot timestamp (00:00 UTC of day D)."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    for attempt in range(1, retries + 1):
        try:
            r = http_get(url, params={"vs_currency": "usd", "days": days, "interval": "daily"}, timeout=30)
            prices = r.json()["prices"]
            break
        except Exception:  # noqa: BLE001 — free tier 429s; back off hard
            if attempt == retries:
                raise
            time.sleep(15 * attempt)
    idx = pd.to_datetime([p[0] for p in prices], unit="ms", utc=True)
    s = pd.Series([p[1] for p in prices], index=idx)
    return s[s.index == s.index.normalize()]  # keep exact-midnight snapshots only


def fetch_coinbase_daily(product: str, days: int = 40) -> pd.Series:
    """Coinbase Exchange daily candles: [t, low, high, open, close, vol], bucket start UTC."""
    end = utc_today()
    start = end - pd.Timedelta(days=days + 2)
    r = http_get(
        f"https://api.exchange.coinbase.com/products/{product}/candles",
        params={"granularity": 86400, "start": start.isoformat(), "end": end.isoformat()},
        timeout=30,
    )
    rows = r.json()
    idx = pd.to_datetime([row[0] for row in rows], unit="s", utc=True)
    return pd.Series([row[4] for row in rows], index=idx).sort_index()


def _diff_stats(ours: pd.Series, theirs: pd.Series) -> tuple[pd.Series, float, float]:
    joined = pd.concat([ours.rename("ours"), theirs.rename("theirs")], axis=1).dropna()
    diff_pct = (100.0 * (joined["ours"] - joined["theirs"]).abs() / joined["theirs"]).sort_index()
    return diff_pct, float(diff_pct.max()), float(diff_pct.median())


def check_cross_source(store: Path, settings: dict, res: QCResult) -> None:
    qc_cfg = settings["qc"]
    window = int(qc_cfg["cross_window_days"])
    limit = float(qc_cfg["cross_max_diff_pct"])
    yesterday = utc_today() - pd.Timedelta(days=1)

    for symbol, m in settings["cross_source"].items():
        ours_all = _our_daily_closes(store, symbol)
        ours = ours_all[(ours_all.index <= yesterday)].tail(window)

        # CoinGecko: snapshot at 00:00 of D+1  ↔  our close of D
        cg = fetch_coingecko_daily(m["coingecko"], days=window + 8)
        cg_as_close = pd.Series(cg.to_numpy(), index=cg.index - pd.Timedelta(days=1))
        d1, mx1, md1 = _diff_stats(ours, cg_as_close)
        res.add(
            f"cross coingecko {symbol}",
            bool(len(d1) >= window * 0.8 and mx1 < limit),
            f"days={len(d1)} median={md1:.3f}% max={mx1:.3f}%",
        )
        time.sleep(3)  # CG free-tier rate limit

        # Coinbase: same-day bucket close
        cb = fetch_coinbase_daily(m["coinbase"], days=window + 8)
        d2, mx2, md2 = _diff_stats(ours, cb)
        res.add(
            f"cross coinbase  {symbol}",
            bool(len(d2) >= window * 0.8 and mx2 < limit),
            f"days={len(d2)} median={md2:.3f}% max={mx2:.3f}%",
        )


def check_bitget_consistency(store: Path, settings: dict, res: QCResult) -> None:
    from data.collectors.bitget_live import fetch_recent_ohlcv  # lazy: needs network+ccxt

    yesterday = utc_today() - pd.Timedelta(days=1)
    for sym_ccxt, symbol in zip(
        settings["bitget"]["spot_symbols"], settings["symbols"], strict=True
    ):
        try:
            bg = fetch_recent_ohlcv(sym_ccxt, "1d", limit=10)
        except Exception as exc:  # noqa: BLE001
            res.add(f"cross bitget {symbol}", False, f"fetch failed: {exc}", gate=False)
            continue
        bg = bg.set_index(pd.to_datetime(bg["ts"], utc=True).dt.normalize())["close"]
        ours = _our_daily_closes(store, symbol)
        common = [d for d in bg.index if d in ours.index and d <= yesterday]
        if not common:
            res.add(f"cross bitget {symbol}", False, "no overlapping settled day", gate=False)
            continue
        d = max(common)
        diff = 100.0 * abs(ours[d] - bg[d]) / bg[d]
        res.add(f"cross bitget {symbol}", bool(diff < 0.5), f"{d.date()} diff={diff:.3f}%")


# ---------------- funding sanity (warn-only) ----------------

def check_funding(store: Path, settings: dict, res: QCResult) -> None:
    for symbol in settings["symbols"]:
        path = storeio.funding_um_path(store, symbol)
        if not path.exists():
            res.add(f"funding {symbol}", False, "missing file", gate=False)
            continue
        df = pd.read_parquet(path)
        ts = pd.to_datetime(df["ts"], utc=True).sort_values()
        med_gap_h = float(ts.diff().dropna().median() / pd.Timedelta(hours=1))
        extreme = int((df["funding_rate"].abs() > 0.03).sum())
        res.add(
            f"funding {symbol}",
            bool(0.5 <= med_gap_h <= 12.5 and extreme == 0),
            f"rows={len(df)} span={ts.min().date()}→{ts.max().date()} median_gap={med_gap_h:.1f}h |rate|>3%: {extreme}",
            gate=False,
        )


# ---------------- runner ----------------

def run(settings: dict) -> tuple[str, bool]:
    store = storeio.store_dir(settings)
    res = QCResult()
    for market in settings["markets"]:
        for symbol in settings["symbols"]:
            for tf in settings["timeframes"]:
                key = f"{market}/{symbol}/{tf}"
                path = storeio.klines_path(store, market, symbol, tf)
                if not path.exists():
                    res.add(f"structure {key}", False, "missing parquet")
                    continue
                df = pd.read_parquet(path)
                check_klines_structure(df, tf, settings["qc"], key, res)
                if tf == "1d":
                    check_freshness(df, settings["qc"], key, res)
    check_cross_source(store, settings, res)
    check_bitget_consistency(store, settings, res)
    check_funding(store, settings, res)

    lines = [
        f"# Phase 0 Data QC Report — {utc_today().date()}",
        "",
        f"Conclusion: **{'PASS ✅' if res.gate_passed else 'FAIL ❌'}** (PASS only if all gate items pass; gate=no is informational)",
        "",
        "| Check | Result | Gate | Details |",
        "|---|---|---|---|",
    ]
    for c in res.checks:
        lines.append(f"| {c.name} | {'✅' if c.passed else '❌'} | {'yes' if c.gate else 'no'} | {c.detail} |")
    report = "\n".join(lines) + "\n"

    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qc_report_{utc_today().date()}.md"
    out_path.write_text(report, encoding="utf-8")
    log.info("QC report -> %s (gate=%s)", out_path, res.gate_passed)
    return report, res.gate_passed
