# execution/carry — Phase 3+ 占位

资金费率 carry（现货多 + 永续空，delta 中性）的专用执行器，CCXT 直连 Bitget。
低杠杆（≤2x）、低频调仓。数据已就绪：data/store/funding_um/（Binance 史）与
data/store/funding_bitget/（Bitget 史，实盘所在地）。

先决条件：Phase 1 完成 carry 历史模拟（含费用与基差），Phase 2 模拟盘验证。
