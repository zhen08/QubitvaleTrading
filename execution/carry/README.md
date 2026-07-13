# execution/carry — Phase 3+ placeholder

A dedicated executor for funding-rate carry (spot long + perpetual short, delta neutral), connecting
directly to Bitget via CCXT. Low leverage (≤2x), low-frequency rebalancing. The data is ready:
data/store/funding_um/ (Binance history) and data/store/funding_bitget/ (Bitget history, where live
trading happens).

Prerequisites: Phase 1 completes the carry historical simulation (including fees and basis), and
Phase 2 paper trading validation.
