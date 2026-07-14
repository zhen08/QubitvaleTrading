# QubitvaleTrading

A personal crypto quantitative trading system (spot + USDT-M perpetuals), built in phases following the five-layer architecture in §6 of the [research report](../../Documents/Claude/Projects/Crypto/auto-trading-system-research-2026-07-12.md).

**Current status (revised in the 2026-07-13 review)**: Phase 1 = **research candidate PASS, statistical certification FAIL** (deployment portfolio DSR(N=4)=0.868 / DSR(N=32)=0.750, both <0.95; see `research/reports/phase1_report_2026-07-13.md`; an earlier 0.395 figure circulated in this README was a stale pre-fix value — see `research/reports/dsr_reconciliation_2026-07-14.md`). Phase 2 = **exploratory paper validation is set up** (started 2026-07-12), but **automation is not yet deployed** — the 6-week gate clock only officially starts once you run `bash scripts/setup_mac.sh` on the Mac to complete deployment. This repository currently does not constitute any basis for live deployment.

> Disclaimer: This repository is purely a personal research tool and does not constitute investment advice. Leveraged futures trading carries extremely high risk.

## Phase 0 delivered

- **Historical data backfill** (Binance Vision official free bulk data): BTC/ETH/SOL × spot/USDT-M futures × 1h/4h/1d, 2019-01 → present (monthly zips + current-month daily zips to fill the tail, T-1); USDT-M funding-rate history
- **Bitget live collection** (CCXT public endpoints): market/funding/OI snapshots, Bitget funding-rate history
- **News collection**: RSS (CoinDesk / Cointelegraph / The Block / Wu Blockchain) + GDELT DOC 2.0 (free)
- **Data QC**: gaps/duplicates/OHLC consistency + daily close vs CoinGecko & Coinbase cross-source validation (threshold <0.5%) + Bitget consistency
- Unit tests (parsers and QC logic, runnable offline)

## Quick start

```bash
pip install -r requirements.txt

python -m scripts.backfill          # full/incremental backfill (auto-resumes on rerun)
python -m scripts.bitget_snapshot   # Bitget snapshot + funding history
python -m scripts.collect_news      # one round of RSS + GDELT collection
python -m scripts.run_qc            # QC; exit 1 if a threshold fails
python -m scripts.build_db          # rebuild DuckDB views (run once after switching machines)
python -m scripts.update_data       # daily one-click incremental (data + funding + news)
python -m scripts.run_phase1        # rerun the Phase 1 research suite (walk-forward + DSR/PBO + carry)
python -m scripts.run_paper_daily   # ★ Phase 2 daily job: data → news scoring → signals → paper rebalance → notify
python -m scripts.paper_status      # view paper positions/equity/signals/risk flags
python -m scripts.paper_review      # generate the weekly review (paper vs model replay vs Phase 1 expected band)
pytest -q                           # unit tests
```

## Phase 2 paper trading (running)

**Multi-book (since 2026-07-13)**: each strategy has its own independent virtual ledger (each
$10,000, with independent fills/equity/runs registry/frozen baseline, under directory
`data/store/paper/<book>/`), sharing data, market feeds, risk controls, and the global lock.
The book list lives in `config/settings.yaml → paper.books`; strategies are registered in
`strategies/registry.py` (unified interface `compute_weights(dfs)`; signal code is always shared
with the research engine and locked down by golden tests):

| Book | Form | Phase 1 portfolio-level reference | Started |
|---|---|---|---|
| donchian_ensemble | 4-parameter ensemble × 3 coins, spot long/flat | Sharpe 0.74 / MaxDD −28% / DSR(N=4) 0.868 | 2026-07-12 |
| tsmom_ensemble | 12-parameter ensemble × 3 coins, spot long/flat + σ₂₀ vol targeting | Sharpe 0.65 / vol 9.8% / MaxDD −12% / DSR(N=4) 0.824 | 2026-07-13 |

**Phase 3 selection discipline (ex-ante, hard-coded in the strategy file headers)**: if both books
pass → deploy both strategies at half size each, **do NOT "pick the winner"** (that would introduce
yet another layer of N=2 selection bias); if only one passes, pilot only that one.
The process and discipline for adding new books are in the header of `strategies/registry.py` —
every book added enlarges the multiple-testing space.

**Automated deployment (required, one command in the Mac terminal)**:

```
bash scripts/setup_mac.sh
```

The script creates an in-repo `.venv` (not dependent on whether the system Python has pandas),
installs the launchd daily job (local 08:10; catches up after wake if a run was missed while
asleep), **freezes the expected-band baseline** (`scripts.freeze_baseline`, so the 6-week gate's
reference is fixed from here on and does not drift with data updates), and runs once immediately.
The daily job is idempotent and can be rerun arbitrarily, and the entire run holds a
**process-level exclusive lock** (overlapping manual/scheduled jobs auto-skip, preventing lost
updates): missed days are back-filled at that day's **open price ±slip_floor** (mode=catchup,
replaying that day's event gate and historical risk flags); the current day fills using Bitget
**bid/ask** (or last±slip_floor if the book is missing) (mode=live). Signal validity is checked
**per-asset**: if any coin lacks its D-1 decision bar → no rebalance for the whole day and a P1
is logged (missing data is never treated as a liquidation signal). The review's expected band uses
the block-bootstrap empirical distribution of the frozen OOS sample; statistics only use settled
equity.

**Known simplifications** (negligible at the $10k virtual-book scale, recorded faithfully): does not
model price/quantity step-size rounding or partial fills; historical risk flags on catchup days are
only replayed when the archive exists (archiving enabled since 2026-07-13).

**Risk-control rules** (only restrict adding to positions, never block reducing): no new positions
from 36h before to 1h after scheduled events like CPI/FOMC (`config/calendar.yaml`, requires manual
verification and upkeep); news risk flags — asset-specific negative sev≥4 blocks adds, sev≥5 halves,
market-level (ALL) flags require sev≥5 (the scorer calls DeepSeek via the **OpenRouter** interface:
set `OPENROUTER_API_KEY` in `.env`, model defaults to `deepseek/deepseek-v4-flash`, overridable with
`OPENROUTER_MODEL`; falls back to keyword rules when no key is set); **an expired flag (TTL 24h) is
treated as unknown state → conservatively block adds**; signals must come from the D-1 decision bar
(if Vision hasn't arrived, fill with the Bitget tail bar; if still missing, skip trading and log a
P1 incident — never silently reuse an old signal). Operational incidents are persisted to
`data/store/ops/incidents.parquet` (graded P0-P3, so "zero P0" is auditable). **The authoritative
paper state is the local `data/store/paper/`**; sync that directory (including signals/, intel/, ops/)
before running on another machine. The ledger is event-sourced by design: cash/positions are always
replayed and rebuilt from `trades.parquet`, so it can recover from a crash at any point in time.

**Phase 2 pass gate**: from stable automated operation onward, ≥6 consecutive weeks with paper
cumulative return inside the 95% expected-distribution band, tracking error (annualized) < 2%, and
zero P0 incidents (`paper_review` computes this automatically). Because Phase 1 statistical
certification did NOT pass, even meeting the 6-week gate only supports a Phase 3 pilot at the
"tiny amount, fully losable" level.

Run all commands from the **repository root**. Configuration is in `config/settings.yaml`; for
secrets, copy `.env.example` to `.env` (Phase 0 needs no keys at all).

## Data storage

`data/store/` (not in git, rebuildable at any time):

```
market/{spot|um}/{SYMBOL}/{1h|4h|1d}.parquet   K-lines (symbol/market/timeframe embedded as columns)
funding_um/{SYMBOL}.parquet                    Binance USDT-M funding-rate history
funding_bitget/{SYMBOL}_PERP.parquet           Bitget funding-rate history (≈166-day rolling)
news/rss.parquet · news/gdelt.parquet          news (deduped by link/url, append-only)
live/latest.json                               most recent Bitget snapshot
manifest.json                                  backfill progress (incremental resume)
quant.duckdb                                   view layer (rebuilt by build_db)
```

Example DuckDB query:

```sql
SELECT market, symbol, timeframe, count(*) rows, min(ts) first, max(ts) last
FROM klines GROUP BY 1,2,3 ORDER BY 1,2,3;
```

## Design notes

- **Adaptive timestamp units**: Binance Vision spot K-lines switched from milliseconds to microseconds starting 2025-01; `normalize_epoch_series` auto-detects the unit by magnitude (s/ms/us/ns), covered by unit tests.
- **Geo-block degradation**: in this project's cloud sessions, `api.binance.com` returns 451 (geo-blocked), so the incremental tail relies entirely on Vision daily files (T+1), and **real-time prices always go through Bitget**. On your own network or a Tokyo VPS this is not an issue and the architecture is unchanged.
- **Pre-listing 404**: SOL spot only has data from 2020-08 and SOL futures from 2020-09; the backfiller records pre-listing months in the manifest as skipped and does not treat them as gaps.
- **QC cross-source alignment**: the CoinGecko daily point is the D-day 00:00 UTC snapshot ≈ our D-1 daily close; the Coinbase daily-bucket close aligns with our same-day close. USD vs USDT pricing differences are usually <0.1%; the 0.5% threshold already accounts for this.

## Roadmap (see §6.6 of the research report for details)

| Phase | Content | Pass gate | Status |
|---|---|---|---|
| 0 | Data foundation + repo skeleton | All QC passes, cross-source <0.5% | ✅ 2026-07-12 |
| 1 | Research platform (cost model, walk-forward, DSR/PBO) + baseline strategies (trend/TSMOM/carry simulation) | Deployment target DSR(N=4) and (N=32) both ≥0.95 | ⚠️ Research candidate PASS / **statistical certification FAIL** (0.868 / 0.750) |
| 2 | Exploratory paper trading (in-house paper engine¹ + multi-book: donchian + tsmom + event gate/news flags) | Each book ≥6 consecutive weeks meeting the gate after automation | 🟡 Set up; automation pending `setup_mac.sh` |
| 3 | Small live trading (hard risk controls, no-withdrawal-permission key, kill switch) | 4–8 weeks consistent with paper | ⬜ |
| 4 | Multi-exchange expansion / event-driven / monthly recalibration | Ongoing | ⬜ |

¹ The plan originally called for freqtrade dry-run; the reason for switching to an in-house paper
engine: the purpose of paper trading is to measure the "signal → execution" drift relative to the
backtest, which requires bar-by-bar semantic identity with the research engine (the same signal code,
the same cost assumptions); a framework's built-in fill model would mix framework differences into the
tracking error. freqtrade remains a candidate for the Phase 3 live execution layer (at which point
we'll compare the trade-offs between the in-house CCXT executor and freqtrade).
