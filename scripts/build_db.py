"""(Re)build DuckDB views over the parquet store.

DuckDB 视图存的是字面路径，换机器（容器 ↔ Mac）后运行本脚本重建即可：
    python -m scripts.build_db
生成 data/store/quant.duckdb，视图：klines / funding_um / funding_bitget / news_rss / news_gdelt
"""
from __future__ import annotations

import duckdb

from data import storeio
from data.collectors.common import load_settings, setup_logging


VIEWS = {
    "klines": "market/*/*/*.parquet",
    "funding_um": "funding_um/*.parquet",
    "funding_bitget": "funding_bitget/*.parquet",
    "news_rss": "news/rss.parquet",
    "news_gdelt": "news/gdelt.parquet",
}


def main() -> None:
    setup_logging()
    settings = load_settings()
    store = storeio.store_dir(settings)
    db_path = store / "quant.duckdb"
    con = duckdb.connect(str(db_path))
    for name, glob in VIEWS.items():
        pattern = str(store / glob)
        try:
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{pattern}')")
            n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            print(f"view {name:16s} rows={n:,}")
        except duckdb.Error as exc:
            print(f"view {name:16s} skipped ({exc})")
    con.close()
    print(f"duckdb -> {db_path}")


if __name__ == "__main__":
    main()
