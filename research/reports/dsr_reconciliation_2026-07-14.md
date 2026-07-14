# DSR Record Reconciliation — 2026-07-14

**Purpose:** Stage 0 of `research/deep_learning_cross_asset_implementation_plan.md` (§4.1) requires one
authoritative deployment-portfolio DSR record before any neural-overlay incremental value is measured.
The repository contained two conflicting DSR(N=32) values for the donchian deployment portfolio:
`README.md` said **0.395** while `phase1_report_2026-07-13.md` said **0.75** and the
`strategies/donchian_ensemble.py` header said **0.751**.

## Root cause

Both numbers came from the same code path (`research/phase1.py` deployment-portfolio block) at
different revisions:

- **0.395** was produced by the first 2026-07-13 review pass, where the N=32 cross-trial SR variance
  (`var_n32`, the `sr_variance` input to `metrics.deflated_sharpe`) was computed over the **full sample
  window including the 730-bar training warmup** of each (family, params) cross-coin portfolio.
- The second review pass (commit `1eec66b`, "the trial variance must use the same OOS time window as
  the certification object") fixed `var_n32` to drop the warmup bars, giving **0.750**. The report was
  regenerated; the README and the supersession note in `phase1_report_2026-07-12.md` were not.

The inflated pre-fix variance (0.000473 vs 0.000099 per-period SR variance across the 32 trial
portfolios) raised E[max SR] and depressed DSR from 0.750 to ≈0.392–0.395.

## Authoritative reproduction (2026-07-14, data through 2026-07-12)

Recomputed with `research.metrics.deflated_sharpe` on the walk-forward OOS index (identical
construction to `phase1.run_all`; spot legs, SPOT_TAKER costs, ensemble weights, 3-coin equal weight):

| family | OOS bars | net Sharpe | MaxDD | DSR(N=4) | DSR(N=32) |
|---|---|---|---|---|---|
| sma_cross | 1414 | 0.636 | −36.0% | 0.817 | 0.680 |
| **donchian (deployment)** | **1414** | **0.738** | **−28.4%** | **0.868** | **0.750** |
| tsmom | 1414 | 0.650 | −12.2% | 0.824 | 0.689 |
| rsi_meanrev | 1414 | 0.350 | −38.1% | 0.634 | 0.461 |

Inputs frozen with this record: `n_trials` = 4 (family selection) and 32
(`TOTAL_TRIALS_PER_SYMBOL`, portfolio-level parameter×family); `sr_variance` = 0.000052 (N=4) and
0.000099 (N=32), both computed on the post-warmup (OOS-aligned) window; common OOS window = 1,414
bars (three-coin intersection, 2022-08 → 2026-07).

## Resolution

- Authoritative record: **DSR(N=4) = 0.868, DSR(N=32) = 0.750** — the `phase1_report_2026-07-13.md`
  figures. The 0.751 in the strategy header was the same computation at 3-decimal rounding drift.
- `README.md`, `phase1_report_2026-07-12.md` supersession note, and the
  `strategies/donchian_ensemble.py` header now all state 0.750 and point here.
- **The verdict does not change**: 0.750 < 0.95, statistical certification remains FAIL; the Donchian
  book stays an uncertified exploratory paper candidate.
- This record is the frozen B0 baseline for the cross-asset deep-learning research program
  (plan §4.1/§4.2): OOS net Sharpe 0.738, MaxDD −28.4%, 1,414-bar common OOS window.
