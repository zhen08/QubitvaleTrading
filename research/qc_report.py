"""Phase 0 data quality checks + report.

门槛（config/settings.yaml → qc，源自调研报告 §6.6）：
  1) 结构：每条 K 线序列缺口率 ≤ max_missing_pct；重复时间戳 = 0；OHLC 违例 = 0；
  2) 新鲜度：日线最后一根距今 ≤ freshness_max_age_days 天（Vision 日度文件 T+1 发布）；
  3) 跨源：最近 cross_window_days 个已结算日的现货日线收盘，
     vs CoinGecko（对齐规则：CG 在 D+1 00:00 UTC 的快照价 ↔ 我们 D 日收盘）
     vs Coinbase（同日桶收盘直接对比），任意一天偏差 < cross_max_diff_pct%；
  4) Bitget 一致性（信息项+门槛）：Bitget 最近已结算日收盘 vs 本库现货收盘 < 0.5%。
资金费率做 sanity 检查（幅度/节奏），只警告不设门槛。
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
    delta = TF_DELTA[tf]
    ts = pd.to_datetime(df["ts"], utc=True)
    dups = int(ts.duplicated().sum())
    first, last = ts.min(), ts.max()
    expected = int((last - first) / delta) + 1
    missing = expected - ts.nunique()
    missing_pct = 100.0 * missing / expected if expected else 0.0
    viol = int(
        (
            (df["low"] > df[["open", "close"]].min(axis=1) + 1e-12)
            | (df["high"] < df[["open", "close"]].max(axis=1) - 1e-12)
            | (df["low"] > df["high"])
            | (df["volume"] < 0)
        ).sum()
    )
    ok = (
        missing_pct <= float(qc_cfg["max_missing_pct"])
        and dups <= int(qc_cfg["max_dup"])
        and viol <= int(qc_cfg["max_ohlc_violations"])
    )
    res.add(
        f"structure {key}",
        ok,
        f"rows={len(df)} span={first.date()}→{last.date()} missing={missing}({missing_pct:.3f}%) dup={dups} ohlc_viol={viol}",
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
        f"# Phase 0 数据质检报告 — {utc_today().date()}",
        "",
        f"结论：**{'PASS ✅' if res.gate_passed else 'FAIL ❌'}**（门槛项全部通过才算 PASS；gate=否 为信息项）",
        "",
        "| 检查 | 结果 | 门槛 | 详情 |",
        "|---|---|---|---|",
    ]
    for c in res.checks:
        lines.append(f"| {c.name} | {'✅' if c.passed else '❌'} | {'是' if c.gate else '否'} | {c.detail} |")
    report = "\n".join(lines) + "\n"

    out_dir = Path(__file__).resolve().parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qc_report_{utc_today().date()}.md"
    out_path.write_text(report, encoding="utf-8")
    log.info("QC report -> %s (gate=%s)", out_path, res.gate_passed)
    return report, res.gate_passed
