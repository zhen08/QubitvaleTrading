# Track 3 Preregistration — Cross-Sectional Crypto Ranker

**Registered:** 2026-07-16, before any model code for this family was written
and before any model touched `data/store/universe/panel.parquet`.
**Plan reference:** `research/deep_learning_cross_asset_implementation_plan.md` §9/§10/§11 (v2).
**Ledger:** counted in `dl_trial_ledger.md`. Every choice below is frozen; any
change after first results creates a new trial family.

## 1. Universe (point-in-time, survivorship-safe)

- Source: the Vision panel built 2026-07-15 (599/600 eligible symbols; fixed
  eligibility rules in `data/collectors/universe.py`). Known gap: KLAYUSDT
  (top-50 in 2021-2023) is absent until backfilled from another network; the
  first research run must either include it or quantify its absence.
- Membership on decision date D (information through bar D only):
  1. full 30-day ADV window (`min_periods=30`, so new listings cannot enter early);
  2. **ADV30 ≥ $5M** (the "liquid, executable" floor — the panel shows rank-30
     ADV under $3M before 2021, which fails the §9.1 spirit on count alone);
  3. rank ≤ 50 by ADV30;
  4. at least 110 daily bars of history (90-day sequence + label/indicator warmup).
- A date is usable only if ≥ 30 symbols pass. Market-cap constraints remain
  deferred and must be disclosed in every report of this family.

## 2. Data window and walk-forward

- Panel rows from **2020-07-01**; usable dates begin when rule §1 admits ≥30 names.
- Expanding walk-forward: **initial training minimum 730 days**, validation 182
  days, test 182 days, retrain every 182 days, 5-day embargo at every boundary
  (label horizon is 1 day; the repo-standard 5-day embargo is kept for uniformity).
  Deviation from §10.1's 3-year minimum is declared here ex ante: the
  cross-section supplies ~30-50 assets per date, so a 730-day initial window
  already contains an order of magnitude more rows than Track 1's three years
  of three assets, and it buys ~7 OOS folds (first test window opens 2023-01)
  instead of ~4.
- Identical fold boundaries for every variant and baseline.

## 3. Model (the E-family analog; 3 preregistered trials)

- One shared **GRU**, 1 layer, hidden width from the preregistered set
  **{5, 10, 20}** — three trials, nothing else varies.
- Input: 90-day sequence of vol-normalized daily log returns
  (`r_t / (σ20 · 1)`, per asset, trailing σ20 floored at the training fold's
  5th percentile of positive values), plus a per-day cross-sectional rank of
  the same return in [0, 1]. Two channels, standardized with training-fold
  statistics only.
- Head: sigmoid → `p = P(next-day return > point-in-time universe median)`.
- Loss: BCE. Optimizer AdamW, lr 1e-3, weight decay 1e-4, batch 512, max 100
  epochs, early stop on validation loss with patience 10, grad clip 1.0.
- Seeds 17/29/43/71/101, arithmetic-mean ensemble, CPU-deterministic
  (`torch.use_deterministic_algorithms(True)`), never seed-selected.

## 4. Portfolio rule (fixed)

- Rank universe members daily by ensemble `p`; ties broken by higher ADV30.
- Hold the **top 5**, equal-weight **20% each**, long-only spot, gross ≤ 1.0,
  no leverage, no shorts.
- Hysteresis: a new name enters only when ranked ≤ 5; a held name exits when
  ranked > 10 **or** its `p` < 0.50; an exited slot whose replacement has
  `p` < 0.50 stays in cash.
- Existing 2% rebalance threshold applies to weight changes.
- Costs: repo spot cost model (10 bps/side + √-impact slippage using each
  asset's own panel dollar volume, 1 bp floor); evaluate a 2x-cost stress case.

## 5. Baselines (must be beaten; part of the trial count)

- **R0** — equal-weight basket of the full universe (market proxy).
- **R1** — deterministic cross-sectional momentum: rank by trailing 21-day
  return, identical §4 portfolio rule. The neural ranker earns its complexity
  only by beating R1 net of costs.
- **R2** — cash (0 return): the long/cash rule must add value over not playing.

Family trial count for DSR: 3 GRU widths + R1 = 4 fitted/selected trials
(R0/R2 are parameter-free references), **plus** the program's cumulative prior
trials per the ledger.

## 6. Gate (all must pass; §11.1 structure)

- Best-width selection happens on **validation** folds only; the width chosen
  on validation is the single OOS candidate (no OOS-based selection).
- OOS net Sharpe > R0 and > R1; Ledoit-Wolf one-sided stationary-bootstrap
  test vs **R1** significant at 5%;
- DSR ≥ 0.95 at the family trial count above;
- PBO across {3 widths, R1} does not indicate unstable selection;
- positive incremental utility vs R1 in ≥ 70% of complete OOS folds; survives
  removal of the single best fold;
- OOS Sharpe > 0 under 2x costs;
- not driven by one regime slice (ex-ante slices of §10.3) or by any single
  asset contributing > 40% of gross profit;
- turnover reported; if annualized turnover × cost erases more than half the
  gross edge over R1, reject regardless of significance.
- Passing promotes an **exploratory paper book proposal only** (new registry
  entry, own capital, §12 discipline); it is not certification.

## 7. Amendments (2026-07-16, recorded during implementation, before any run)

Three details the original text under-specified, pinned before the first
walk-forward run (no results of any kind had been observed):

1. **Width selection criterion**: per fold, the width whose 5-seed ensemble
   maximizes the validation-window **net Sharpe of the §4 portfolio rule**
   (the same economics as the gate); ties go to the smaller width.
2. **R1 confidence analog**: R1's counterpart of the `p < 0.50` cash rule is
   trailing 21-day return ≤ 0 (no long without positive momentum). R0 has no
   threshold.
3. **Delisting handling**: a held asset with no bar on the next day
   contributes 0 return that day and is force-exited with normal cost;
   training rows without a next-day return (delisting eve) carry no label and
   are dropped from the loss.

## 8. Non-negotiables

- No feature, label, threshold, width set, or universe-rule change after the
  first walk-forward run; extensions are new preregistrations.
- Published long/short ranker results are not an expected-return estimate for
  this long/cash implementation (§9.3).
- Missing data / QC failure at inference: fail closed exactly as the existing
  books do (block adds, never force exits).
