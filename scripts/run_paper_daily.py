"""Phase 2 每日一键任务：数据增量 → 新闻/打分 → 信号 → paper 调仓 → 通知。

Usage: python -m scripts.run_paper_daily [--no-news]
幂等：任意时间、任意次数重跑都安全；错过的天数会以开盘价 catchup 补账（计入运维指标）。
建议 Mac cron（UTC 00:10 = 北京 08:10）：
  10 8 * * * cd ~/Dev/QubitvaleTrading && /usr/bin/python3 -m scripts.run_paper_daily >> logs/paper.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import logging

from data.collectors import binance_vision, gdelt, news_rss
from data.collectors.common import load_settings, setup_logging
from execution.paper.engine import run_daily
from intel.news_scorer import refresh_risk_flags
from ops import telegram

log = logging.getLogger("qvt.daily")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-news", action="store_true")
    args = ap.parse_args()
    settings = load_settings()

    # 1) 数据增量（失败则中止——不能盲跑）
    binance_vision.backfill_all(settings, do_funding=False)

    # 2) 新闻 + 风险旗（尽力而为，失败不阻断）
    if not args.no_news:
        try:
            news_rss.collect(settings)
            gdelt.collect(settings)
            refresh_risk_flags(settings)
        except Exception as exc:  # noqa: BLE001
            log.warning("news/scorer step failed (non-fatal): %s", exc)

    # 3) 信号 + paper 调仓
    summary = run_daily(settings)
    print(json.dumps(summary, indent=1, ensure_ascii=False))

    # 4) 通知
    pos = ", ".join(f"{k}:{v:.5f}" for k, v in summary.get("positions", {}).items()) or "空仓"
    msg = (f"*QVT paper* {summary['date']}\n"
           f"equity: ${summary.get('equity', '?')} "
           f"({summary.get('equity_ret_pct_since_start', 0)}% since start)\n"
           f"positions: {pos}\n"
           f"live trades: {len(summary.get('live_trades', []))}, "
           f"incidents: {len(summary.get('incidents', []))}")
    if summary.get("notes"):
        msg += "\nnotes: " + "; ".join(summary["notes"])
    telegram.send(msg)


if __name__ == "__main__":
    main()
