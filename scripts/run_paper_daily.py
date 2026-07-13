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

from data import storeio
from data.collectors import binance_vision, etf_flows, gdelt, news_rss
from data.collectors.common import load_settings, setup_logging
from execution.paper.engine import run_daily
from intel.etf_flows import etf_oneline
from intel.news_scorer import refresh_risk_flags
from ops import incident_log, telegram

log = logging.getLogger("qvt.daily")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-news", action="store_true")
    args = ap.parse_args()
    settings = load_settings()
    store = storeio.store_dir(settings)

    # 1) 数据增量（失败 → P1 事故并中止：不能盲跑；engine 的 Bitget 尾部机制
    #    只兜"Vision 未发布"的正常时滞，不兜采集器整体故障）
    try:
        binance_vision.backfill_all(settings, do_funding=False)
    except Exception as exc:  # noqa: BLE001
        incident_log.record(store, "P1", "backfill_failed", str(exc))
        raise SystemExit(1)

    # 2) 新闻 + 风险旗（失败 → P2 事故；flags TTL 会让引擎保守禁加仓，不会静默沿用）
    if not args.no_news:
        try:
            news_rss.collect(settings)
            gdelt.collect(settings)
            refresh_risk_flags(settings)
        except Exception as exc:  # noqa: BLE001
            incident_log.record(store, "P2", "news_step_failed", str(exc))
        # ETF flows (BTC/ETH). Best-effort & non-fatal — missing key/data just leaves the gate idle.
        try:
            etf_flows.collect(settings)
        except Exception as exc:  # noqa: BLE001
            incident_log.record(store, "P3", "etf_flows_failed", str(exc))

    # 3) 信号 + 各账本调仓
    summary = run_daily(settings)
    print(json.dumps(summary, indent=1, ensure_ascii=False))

    # 4) 通知（多账本汇总）
    lines = [f"*QVT paper* {summary['date']}"]
    for book, bs in summary.get("books", {}).items():
        if "error" in bs:
            lines.append(f"{book}: ERROR {bs['error'][:80]}")
            continue
        pos = ", ".join(f"{k}:{v:.5f}" for k, v in bs.get("positions", {}).items()) or "flat"
        lines.append(f"{book}: ${bs.get('equity', '?')} "
                     f"({bs.get('ret_pct_since_start', 0)}%) | "
                     f"PnL ${bs.get('pnl', 0):+,.2f} | fees ${bs.get('fees', 0):.2f} | "
                     f"{pos} | live {len(bs.get('live_trades', []))}")
        for n in bs.get("notes", [])[:3]:
            lines.append(f"  · {n}")
    etf_line = etf_oneline(settings)
    if etf_line:
        lines.append(etf_line)
    lines.append(f"incidents: {len(summary.get('incidents', []))}")
    telegram.send("\n".join(lines))


if __name__ == "__main__":
    main()
