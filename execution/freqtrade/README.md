# execution/freqtrade — Phase 2 占位

方向性策略的执行层将放在这里（Freqtrade user_data、各交易所 config、策略适配器）。

按照调研报告（Crypto/auto-trading-system-research-2026-07-12.md）§6.6：
只有当 research/ 中至少一个策略通过"净成本、walk-forward 样本外、DSR>0"门槛后，
才进入本目录的搭建（dry-run → Bitget 合约 demo → 小额实盘）。
