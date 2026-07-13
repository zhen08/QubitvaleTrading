"""查看各 paper 账本状态：仓位、权益、最新信号、风险旗。

Usage: python -m scripts.paper_status
"""
from __future__ import annotations

import json

from data import storeio
from data.collectors.common import load_settings, setup_logging
from execution.paper.ledger import Ledger
from intel.news_scorer import load_risk_flags


def main() -> None:
    setup_logging()
    settings = load_settings()
    store = storeio.store_dir(settings)

    for book, bcfg in settings["paper"]["books"].items():
        led = Ledger.load_or_init(store, float(bcfg["initial_capital_usdt"]),
                                  str(bcfg["start_date"]), book=book)
        print(f"== {book} ==")
        print(f"cash: ${led.cash:,.2f}  positions: {led.positions or '空仓'}")
        eq = led.equity_series()
        if len(eq):
            print(f"last equity: ${eq.iloc[-1]:,.2f} @ {eq.index[-1].date()} "
                  f"({100*(eq.iloc[-1]/led.initial_capital-1):+.2f}% since {bcfg['start_date']})")
        print(f"last_settled: {led.last_settled}")
        sig = store / "signals" / f"{book}.latest.json"
        if sig.exists():
            s = json.loads(sig.read_text(encoding="utf-8"))
            print(f"signal(decision {s['decision_date']}): {s['target_weights']}")
        t = led.trades_df()
        if len(t):
            r = t.iloc[-1]
            print(f"last trade: {r['day']} {r['side']} {r['symbol']} "
                  f"{r['qty']:.6g} @ {r['price']:.2f} [{r['mode']}]")
        print()

    flags = load_risk_flags(settings)
    print("== risk flags ==")
    print(json.dumps({k: flags.get(k) for k in
                      ("generated_at", "scorer", "asset_neg_severity",
                       "market_neg_severity", "stale")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
