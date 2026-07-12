"""One round of news collection (RSS + GDELT).

Usage: python -m scripts.collect_news
"""
from __future__ import annotations

from data.collectors import gdelt, news_rss
from data.collectors.common import load_settings, setup_logging


def main() -> None:
    setup_logging()
    settings = load_settings()
    n_rss = news_rss.collect(settings)
    n_gdelt = gdelt.collect(settings)
    print(f"news collected: rss +{n_rss} new, gdelt +{n_gdelt} new")


if __name__ == "__main__":
    main()
