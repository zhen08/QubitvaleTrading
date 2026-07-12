"""Shared helpers: settings, HTTP with retries, timestamp normalization, zipped-CSV parsing.

All timestamps are normalized to tz-aware UTC. Epoch unit is auto-detected because
Binance Vision switched spot kline timestamps from milliseconds to microseconds on
2025-01-01, and other sources use seconds or milliseconds.
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml

log = logging.getLogger("qvt")

REPO_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(level: int = logging.INFO) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%H:%M:%S",
        )


def load_settings(path: str | Path | None = None) -> dict:
    p = Path(path) if path else REPO_ROOT / "config" / "settings.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env(path: str | Path | None = None) -> None:
    """把 REPO_ROOT/.env 的 KEY=VALUE 注入 os.environ（不覆盖已有值）。无依赖版 dotenv。"""
    import os

    p = Path(path) if path else REPO_ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_SESSION: requests.Session | None = None


def http_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "QubitvaleTrading/0.1 (personal research)"})
        adapter = requests.adapters.HTTPAdapter(pool_connections=24, pool_maxsize=24)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _SESSION = s
    return _SESSION


def http_get(
    url: str,
    *,
    params: dict | None = None,
    retries: int = 3,
    timeout: int = 60,
    ok404: bool = False,
    backoff: float = 2.0,
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Response | None:
    """GET with retries. Returns None on 404 when ok404=True (e.g. pre-listing months)."""
    s = http_session()
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = s.get(url, params=params, timeout=timeout)
            if r.status_code == 404 and ok404:
                return None
            if r.status_code in retry_statuses and attempt < retries:
                time.sleep(backoff**attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:  # includes connection errors
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 404 and ok404:
                return None
            last_exc = exc
            if attempt < retries:
                time.sleep(backoff**attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last_exc


def normalize_epoch_series(s: pd.Series) -> pd.Series:
    """Epoch integers of unknown unit (s/ms/us/ns) -> tz-aware UTC datetimes."""
    v = pd.to_numeric(s, errors="coerce")
    med = float(v.dropna().abs().median())
    if med > 1e17:
        unit = "ns"
    elif med > 1e14:
        unit = "us"
    elif med > 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(v, unit=unit, utc=True)


def read_zipped_csv(content: bytes) -> pd.DataFrame:
    """Read the single CSV inside a Binance Vision zip. Header row auto-detected
    (spot files historically have no header; newer futures files do)."""
    zf = zipfile.ZipFile(io.BytesIO(content))
    name = zf.namelist()[0]
    raw = zf.read(name)
    first_field = raw.split(b"\n", 1)[0].split(b",")[0].strip()
    has_header = not first_field.replace(b".", b"").replace(b"-", b"").isdigit()
    return pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)


def utc_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").normalize()
