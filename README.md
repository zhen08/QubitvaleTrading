# QubitvaleTrading

个人加密货币量化交易系统（现货 + USDT-M 永续），按 [调研报告](../../Documents/Claude/Projects/Crypto/auto-trading-system-research-2026-07-12.md) §6 的五层架构分阶段搭建。

**当前状态（2026-07-13 review 修订）**：Phase 1 = **研究候选通过、统计认证未通过**（部署组合 DSR(N=4)=0.868 / DSR(N=32)=0.395，均 <0.95，详见 `research/reports/phase1_report_2026-07-13.md`）。Phase 2 = **探索性 paper 验证已搭建**（起始 2026-07-12），**自动化尚未部署**——在 Mac 上运行 `bash scripts/setup_mac.sh` 完成部署后，6 周 gate 计时才正式起算。本仓库当前不构成任何实盘部署依据。

> 免责声明：本仓库仅为个人研究工具，不构成投资建议。合约杠杆交易风险极高。

## Phase 0 已交付

- **历史数据回填**（Binance Vision 官方免费批量数据）：BTC/ETH/SOL × 现货/USDT-M 合约 × 1h/4h/1d，2019-01 →（月度 zip + 当月日度 zip 补尾，T-1）；USDT-M 资金费率史
- **Bitget 实时采集**（CCXT 公共端点）：行情/资金费/OI 快照、Bitget 资金费率史
- **新闻采集**：RSS（CoinDesk / Cointelegraph / The Block / 吴说）+ GDELT DOC 2.0（免费）
- **数据质检**：缺口/重复/OHLC 一致性 + 日线收盘 vs CoinGecko & Coinbase 跨源校验（门槛 <0.5%）+ Bitget 一致性
- 单元测试（解析器与 QC 逻辑，离线可跑）

## 快速开始

```bash
pip install -r requirements.txt

python -m scripts.backfill          # 全量/增量回填（重跑自动续传）
python -m scripts.bitget_snapshot   # Bitget 快照 + 资金费史
python -m scripts.collect_news      # RSS + GDELT 采一轮
python -m scripts.run_qc            # 质检，门槛不过 exit 1
python -m scripts.build_db          # 重建 DuckDB 视图（换机器后跑一次）
python -m scripts.update_data       # 日常一键增量（数据+资金费+新闻）
python -m scripts.run_phase1        # 复跑 Phase 1 研究套件（walk-forward + DSR/PBO + carry）
python -m scripts.run_paper_daily   # ★ Phase 2 每日任务：数据→新闻打分→信号→paper 调仓→通知
python -m scripts.paper_status      # 查看 paper 仓位/权益/信号/风险旗
python -m scripts.paper_review      # 生成周度复盘（paper vs 模型回放 vs Phase1 期望带）
pytest -q                           # 单元测试
```

## Phase 2 模拟盘（运行中）

**部署形态**（Phase 1 结论）：donchian 4 参数集成 × BTC/ETH/SOL 等权，现货长平，
虚拟 $10,000。信号逻辑与研究引擎共享同一份代码（`research/strategies.donchian`），
黄金测试锁死一致性。

**自动化部署（必做，Mac 终端一条命令）**：

```
bash scripts/setup_mac.sh
```

脚本会建仓库内 `.venv`（不依赖系统 python 是否有 pandas）、安装 launchd 每日任务
（本地 08:10；睡眠错过则唤醒后补跑）、**冻结期望带基准**（`scripts.freeze_baseline`，
6 周 gate 的口径自此固定，不随数据更新漂移）并立即试跑一次。每日任务幂等可任意重跑，
且整个运行持有**进程级独占锁**（重叠的手动/定时任务自动跳过，防丢失更新）：
错过的天数以当日**开盘价 ±slip_floor** 补账（mode=catchup，并回放当日事件门与
历史风险旗）；当日用 Bitget **bid/ask**（缺盘口则 last±slip_floor）成交（mode=live）。
信号有效性按**逐资产**校验：任一币缺 D-1 决策 bar → 整日不调仓并记 P1（缺数据
绝不会被当成清仓信号）。复盘的期望带用冻结 OOS 样本的 block bootstrap 经验分布，
统计只取 settled 权益。

**已知简化**（$10k 虚拟盘量级下影响可忽略，如实记录）：不模拟价格/数量步长取整与
部分成交；catchup 日的历史风险旗仅在归档存在时回放（归档自 2026-07-13 启用）。

**风控规则**（只限制加仓，永不阻止减仓）：CPI/FOMC 等排期事件前 36h 至后 1h 禁开新仓
（`config/calendar.yaml`，需人工核实维护）；新闻风险旗——资产特定负面 sev≥4 禁加仓、
sev≥5 减半，市场级(ALL)旗需 sev≥5（打分器走 **OpenRouter** 接口调 DeepSeek：`.env` 配
`OPENROUTER_API_KEY`，模型默认 `deepseek/deepseek-v4-flash` 可用 `OPENROUTER_MODEL` 覆盖；
无 key 时关键词规则兜底）；**旗过期（TTL 24h）视为状态未知 → 保守禁加仓**；
信号必须来自 D-1 决策 bar（Vision 未到用 Bitget 尾部 bar 补齐，仍缺则跳过交易并记
P1 事故——绝不静默沿用旧信号）。运维事故持久化于 `data/store/ops/incidents.parquet`
（P0-P3 分级，"零 P0"由此可审计）。**paper 状态以本机 `data/store/paper/` 为准**；
换机器运行前先同步该目录（含 signals/、intel/、ops/）。账本为事件溯源设计：
cash/持仓永远从 `trades.parquet` 重放重建，任意时点崩溃可恢复。

**Phase 2 放行门槛**：自动化稳定运行起连续 ≥6 周，paper 累计收益落在期望分布 95%
带内、跟踪误差(年化) < 2%、零 P0 事故（`paper_review` 自动核算）。因 Phase 1 统计
认证未通过，6 周达标也只支持『极小额、可全损』级别的 Phase 3 试点。

所有命令在**仓库根目录**运行。配置见 `config/settings.yaml`；密钥复制 `.env.example` 为 `.env`（Phase 0 无需任何密钥）。

## 数据存储

`data/store/`（不进 git，可随时重建）：

```
market/{spot|um}/{SYMBOL}/{1h|4h|1d}.parquet   K 线（symbol/market/timeframe 已内嵌列）
funding_um/{SYMBOL}.parquet                    Binance USDT-M 资金费率史
funding_bitget/{SYMBOL}_PERP.parquet           Bitget 资金费率史（≈166 天滚动）
news/rss.parquet · news/gdelt.parquet          新闻（按 link/url 去重，append-only）
live/latest.json                               最近一次 Bitget 快照
manifest.json                                  回填进度（增量续传）
quant.duckdb                                   视图层（build_db 重建）
```

用 DuckDB 查询示例：

```sql
SELECT market, symbol, timeframe, count(*) rows, min(ts) first, max(ts) last
FROM klines GROUP BY 1,2,3 ORDER BY 1,2,3;
```

## 设计注记

- **时间戳单位自适应**：Binance Vision 现货 K 线 2025-01 起从毫秒改为微秒，`normalize_epoch_series` 按量级自动识别（s/ms/us/ns），有单测覆盖。
- **地理屏蔽降级**：本项目的云端会话中 `api.binance.com` 返回 451（地理屏蔽），因此增量尾部完全依赖 Vision 日度文件（T+1），**实时价格一律走 Bitget**。在你自己的网络或东京 VPS 上不受影响，架构不变。
- **上市前 404**：SOL 现货 2020-08、SOL 合约 2020-09 才有数据，回填器把上市前月份记入 manifest 跳过，不视为缺口。
- **QC 跨源对齐**：CoinGecko 日频点是 D 日 00:00 UTC 快照 ≈ 我们 D-1 日收盘；Coinbase 日桶收盘与我们同日对齐。USD vs USDT 计价差异通常 <0.1%，门槛 0.5% 已包含此项。

## 路线图（详见调研报告 §6.6）

| Phase | 内容 | 放行门槛 | 状态 |
|---|---|---|---|
| 0 | 数据地基 + 仓库骨架 | QC 全过、跨源 <0.5% | ✅ 2026-07-12 |
| 1 | 研究平台（成本模型、walk-forward、DSR/PBO）+ 基线策略（趋势/TSMOM/carry 模拟） | 部署对象 DSR(N=4)与(N=32) 均≥0.95 | ⚠️ 研究候选 PASS / **统计认证 FAIL**（0.868 / 0.395） |
| 2 | 探索性模拟盘（自研 paper 引擎¹ + 信号服务 + 事件门/新闻旗） | 自动化后连续 ≥6 周达标 | 🟡 已搭建；自动化待 `setup_mac.sh` |
| 3 | 小额实盘（硬风控、无提现权限 key、杀开关） | 4–8 周与模拟一致 | ⬜ |
| 4 | 多所扩展 / 事件驱动 / 月度校准 | 持续 | ⬜ |

¹ 计划原写 freqtrade dry-run；改为自研 paper 引擎的理由：模拟盘的目的是测量
"信号→执行"相对回测的漂移，必须与研究引擎语义逐 bar 一致（同一份信号代码、同一套
成本假设），framework 自带的成交模型会把框架差异混进跟踪误差。freqtrade 仍是
Phase 3 实盘执行层的候选（届时对比自研 CCXT 执行器与 freqtrade 的取舍）。
