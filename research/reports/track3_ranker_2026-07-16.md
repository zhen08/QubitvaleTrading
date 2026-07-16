# Track 3 Ranker Research Report — 2026-07-16

Protocol: `track3_ranker_preregistration.md` (frozen 2026-07-16). OOS 2023-01-26 → 2026-07-14 (1266 days).

**Data disclosure**: KLAYUSDT ABSENT from the panel — §1 requires this gap be quantified before certification-grade use.

## Economic summary (net, 2% threshold, per-asset impact costs)

| variant | Sharpe | CAGR% | MaxDD% | Sortino | ES95% | ann.turnover | 2x-cost SR |
|---|---|---|---|---|---|---|---|
| GRU (val-selected width) | -0.16 | -28.9 | -87.8 | -0.23 | -8.22 | 91.9 | -0.38684486642558136 |
| R0 equal-weight | -0.32 | -40.1 | -89.3 | -0.46 | -8.93 | 10.9 | — |
| R1 21d momentum | -0.89 | -75.1 | -99.4 | -1.45 | -11.12 | 66.8 | -1.0827614532250323 |

Per-width OOS Sharpe: {'GRU_w5': -0.3041786363235581, 'GRU_w10': -0.44151472380327966, 'GRU_w20': -0.20284898955275343}
PBO across widths + R1: 0.1626984126984127
DSR family(N=4): 0.185 · program(N=12): 0.107
LW vs R1: {'n': 1266, 'sharpe_diff': np.float64(0.03817), 'sharpe_p': 0.031, 'mean_diff_daily': 0.0021431, 'mean_p': 0.0135} · vs R0: {'n': 1266, 'sharpe_diff': np.float64(0.00831), 'sharpe_p': 0.148, 'mean_diff_daily': 0.0003485, 'mean_p': 0.131}
Folds positive vs R1: 5/6; Sharpe diff w/o best fold: 0.513
Max single-asset share of gross profit: 0.099
Gross edge vs R1 (daily): 0.0020359; incremental cost: -0.0001071

## Regime slices

- GRU: {'etf_era': 0.37973273058876666, 'post_crash_recoupling': -2.005084174024494}
- R1:  {'etf_era': -0.5943452096264324, 'post_crash_recoupling': -1.952120217276976}

## §6 gate

```json
{
  "beats_R0_sharpe": true,
  "beats_R1_sharpe": true,
  "lw_vs_R1_p<0.05": true,
  "dsr_family>=0.95": false,
  "pbo_not_unstable": true,
  "folds_positive>=70pct": true,
  "survives_best_fold_removal": true,
  "stress2x_sharpe>0": false,
  "concentration<=40pct": true,
  "turnover_rule": true,
  "ALL_PASS": false
}
```

## Fold ledger

See companion CSV. Width selection used validation net Sharpe only (amendment 1); every width's OOS series enters PBO.
## Track 3 decision — REJECT for promotion (gate: 8/10 conditions pass)

The §6 gate fails on the two conditions that matter most for an absolute-return
product: DSR(N=4) = 0.185 (the strategy's own OOS Sharpe is −0.16, so there is
no confidence its true Sharpe exceeds zero) and the 2x-cost stress Sharpe is
negative. No paper book is proposed.

What the run established, recorded honestly:

1. **Relative ranking skill is real.** The GRU beat the deterministic momentum
   ranker R1 with Ledoit-Wolf p = 0.031 (Sharpe) / 0.013 (mean differential),
   in 5 of 6 OOS folds, surviving best-fold removal (+0.51 Sharpe diff), with
   no single-asset concentration (max 9.9% of gross profit) and *lower*
   incremental cost than R1. In the 2023-2025 ETF-era slice the GRU was
   positive (+0.38) while R1 was −0.59. Selection across widths was stable
   (PBO 0.16).
2. **The preregistered product lost money anyway.** The long-only top-50-ADV
   altcoin universe was a falling market over the OOS window: equal-weight R0
   returned −40% CAGR with −89% MaxDD, and momentum R1 −75% CAGR with −99.4%
   MaxDD. Relative skill inside a collapsing asset class does not make a
   long/cash product; the p < 0.5 cash rule reduced but did not avoid the
   drawdown (GRU MaxDD −87.8%).
3. **Implications for any future family** (motivation only, not evidence):
   the informative direction is long/short or market-neutral use of the same
   cross-sectional signal, which the §9.3 long-only constraint deliberately
   excluded for the first pass. That would be a NEW preregistration with its
   own trial count, shorting-feasibility analysis (borrow/funding costs on
   perps), and this window may not be reused for selection.
4. KLAYUSDT remains absent from the panel (disclosed above); given the reject
   verdict no re-run is required, but any future family must close the gap.
