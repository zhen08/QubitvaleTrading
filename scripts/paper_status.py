"""查看 paper 当前状态：仓位、权益、最新信号、风险旗。

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
    pcfg = settings["paper"]
    led = Ledger.load_or_init(store, float(pcfg["initial_capital_usdt"]),
                              str(pcfg["start_date"]))

    print("== state ==")
    print(f"cash: ${led.cash:.2f}  positions: {led.positions}")
    eq = led.equity_series()
    if len(eq):
        print(f"last equity mark: ${eq.iloc[-1]:.2f} @ {eq.index[-1].date()} "
              f"({100*(eq.iloc[-1]/led.initial_capital-1):+.2f}% since start)")
    print(f"last_settled: {led.last_settled}")

    sig = store / "signals" / "latest.json"
    if sig.exists():
        print("\n== latest signal ==")
        print(sig.read_text(encoding="utf-8"))

    flags = load_risk_flags(settings)
    print("== risk flags ==")
    print(json.dumps({k: flags.get(k) for k in ("generated_at", "scorer", "asset_neg_severity")},
                     ensure_ascii=False))

    t = led.trades_df()
    if len(t):
        print(f"\n== last trades ({min(5, len(t))}/{len(t)}) ==")
        print(t.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
