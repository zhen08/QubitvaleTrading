"""Daily incremental update: new Vision monthly/daily files + Bitget funding + news.

Usage: python -m scripts.update_data
(等价于 backfill 增量模式 + bitget 资金费 + RSS/GDELT 一轮采集)
"""
from __future__ import annotations

from data.collectors import binance_vision, bitget_live, gdelt, news_rss
from data.collectors.common import load_settings, setup_logging


def main() -> None:
    setup_logging()
    settings = load_settings()
    binance_vision.backfill_all(settings)
    bitget_live.backfill_funding_history(settings)
    # Cross-asset dailies (SPY/QQQ/GLD/VIX) — best-effort, never blocks the crypto update.
    try:
        from scripts.update_cross_asset import main as update_cross_asset
        xa = "ok" if update_cross_asset() == 0 else "NOT OK (see cross_asset/status.json)"
    except Exception as exc:  # noqa: BLE001
        xa = f"failed ({exc})"
    n_rss = news_rss.collect(settings)
    n_gdelt = gdelt.collect(settings)
    print(f"update done. news: rss +{n_rss}, gdelt +{n_gdelt}; cross-asset: {xa}")


if __name__ == "__main__":
    main()
