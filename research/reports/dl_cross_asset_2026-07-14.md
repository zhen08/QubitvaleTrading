# Cross-Asset DL Research Report — 2026-07-14

Feature schema `9efe4a530c66289e` · rows 7662 · OOS union 2023-03-11 → 2026-07-12 (1220 days)

## Economic summary (net of costs, 2% rebalance threshold for all variants)

| variant | Sharpe | CAGR% | MaxDD% | Sortino | ES95% | DSR(N=7) | 2x-cost SR | mult<1 frac | LW p vs B0 (SR) | LW p vs B2 (SR) | Δmean p vs B0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 (no overlay) | 1.06 | 31.4 | -28.5 | 1.31 | -3.66 | — | 1.032 | — | — | — | — |
| B1 | 1.03 | 26.9 | -28.1 | 1.27 | -3.24 | 0.966 | 0.998 | 0.366 | 0.706 | 0.646 | 0.971 |
| B2 | 1.04 | 29.2 | -28.4 | 1.28 | -3.5 | 0.968 | 1.009 | 0.227 | 0.746 | — | 0.979 |
| B3 | 1.03 | 26.6 | -27.9 | 1.26 | -3.24 | 0.965 | 0.990 | 0.377 | 0.749 | 0.722 | 0.975 |
| E1 | 0.98 | 25.1 | -27.8 | 1.2 | -3.28 | 0.959 | 0.933 | 0.316 | 0.893 | 0.860 | 0.981 |
| E2 | 1.08 | 28.6 | -25.9 | 1.33 | -3.24 | 0.973 | 1.034 | 0.329 | 0.357 | 0.253 | 0.833 |
| E3 | 1.04 | 27.2 | -27.4 | 1.27 | -3.3 | 0.967 | 0.991 | 0.334 | 0.635 | 0.534 | 0.913 |
| E4 | 1.06 | 27.3 | -25.3 | 1.29 | -3.19 | 0.969 | 1.007 | 0.305 | 0.518 | 0.442 | 0.875 |

PBO across the 7 overlay variants: 0.302

## Per-fold incremental utility vs B0 (coarse screen)

- **B1**: 3/7 folds positive; Sharpe diff w/o best fold -0.024; per-coin ΔSR {'BTCUSDT': -0.002, 'ETHUSDT': -0.029, 'SOLUSDT': -0.009}
- **B2**: 2/7 folds positive; Sharpe diff w/o best fold -0.035; per-coin ΔSR {'BTCUSDT': -0.004, 'ETHUSDT': -0.032, 'SOLUSDT': 0.001}
- **B3**: 3/7 folds positive; Sharpe diff w/o best fold -0.029; per-coin ΔSR {'BTCUSDT': -0.019, 'ETHUSDT': -0.029, 'SOLUSDT': -0.012}
- **E1**: 1/7 folds positive; Sharpe diff w/o best fold -0.066; per-coin ΔSR {'BTCUSDT': -0.044, 'ETHUSDT': -0.022, 'SOLUSDT': -0.081}
- **E2**: 5/7 folds positive; Sharpe diff w/o best fold 0.017; per-coin ΔSR {'BTCUSDT': 0.064, 'ETHUSDT': 0.013, 'SOLUSDT': 0.027}
- **E3**: 4/7 folds positive; Sharpe diff w/o best fold -0.035; per-coin ΔSR {'BTCUSDT': 0.01, 'ETHUSDT': -0.028, 'SOLUSDT': 0.009}
- **E4**: 4/7 folds positive; Sharpe diff w/o best fold -0.016; per-coin ΔSR {'BTCUSDT': 0.003, 'ETHUSDT': 0.051, 'SOLUSDT': -0.013}

## Regime slices (ex-ante calendar boundaries)

- **B1**: {'etf_era': 1.5217024787288904, 'post_crash_recoupling': -2.332700136556307}
- **B2**: {'etf_era': 1.513582618035896, 'post_crash_recoupling': -2.334388345803223}
- **B3**: {'etf_era': 1.5100507062027833, 'post_crash_recoupling': -2.3147692252188565}
- **E1**: {'etf_era': 1.466502863716025, 'post_crash_recoupling': -2.487915835244057}
- **E2**: {'etf_era': 1.5373611415370325, 'post_crash_recoupling': -2.192779867456838}
- **E3**: {'etf_era': 1.5117941381166171, 'post_crash_recoupling': -2.290582758961308}
- **E4**: {'etf_era': 1.496526230504044, 'post_crash_recoupling': -2.156555370173791}

## Predictive metrics (pooled by variant; HAR-RV = B2 reference)

| variant | n | vol MAE(log) | QLIKE | tail⁺ | PR-AUC | Brier |
|---|---|---|---|---|---|---|
| B1 | 3645 | 0.4450 | -5.7450 | 320 |  |  |
| B2 | 3645 | 0.4116 | -5.8002 | 320 |  |  |
| B3 | 3645 | 0.4450 | -5.7450 | 320 |  |  |
| E1 | 3645 | 0.4111 | -5.8118 | 320 | 0.124 | 0.0844 |
| E2 | 3645 | 0.4087 | -5.8298 | 320 | 0.140 | 0.0827 |
| E3 | 3645 | 0.4099 | -5.8333 | 320 | 0.143 | 0.0838 |
| E4 | 3645 | 0.4212 | -5.7040 | 320 | 0.148 | 0.0846 |

## §11.1 gate verdict

```json
{
  "per_variant": {
    "E1": {
      "beats_B0_sharpe": false,
      "beats_best_B": false,
      "lw_vs_B2_p<0.05": false,
      "dsr_n7>=0.95": true,
      "maxdd_improve>=15pct": false,
      "cagr_loss<=20pct": false,
      "folds_positive>=70pct": false,
      "stress2x_sharpe>0": true,
      "survives_best_fold_removal": false
    },
    "E2": {
      "beats_B0_sharpe": true,
      "beats_best_B": true,
      "lw_vs_B2_p<0.05": false,
      "dsr_n7>=0.95": true,
      "maxdd_improve>=15pct": false,
      "cagr_loss<=20pct": true,
      "folds_positive>=70pct": true,
      "stress2x_sharpe>0": true,
      "survives_best_fold_removal": true
    },
    "E3": {
      "beats_B0_sharpe": false,
      "beats_best_B": false,
      "lw_vs_B2_p<0.05": false,
      "dsr_n7>=0.95": true,
      "maxdd_improve>=15pct": false,
      "cagr_loss<=20pct": true,
      "folds_positive>=70pct": false,
      "stress2x_sharpe>0": true,
      "survives_best_fold_removal": false
    },
    "E4": {
      "beats_B0_sharpe": false,
      "beats_best_B": true,
      "lw_vs_B2_p<0.05": false,
      "dsr_n7>=0.95": true,
      "maxdd_improve>=15pct": false,
      "cagr_loss<=20pct": true,
      "folds_positive>=70pct": false,
      "stress2x_sharpe>0": true,
      "survives_best_fold_removal": false
    },
    "E3_beats_E1": true,
    "E4_beats_E3": true
  },
  "any_E_passes_all": false,
  "note": "Passing promotes an exploratory paper book only; the base strategy is uncertified (DSR(N=32)=0.750 < 0.95) and no overlay on an uncertified base yields a certified composite."
}
```

## Trial ledger

49 fold×variant trials recorded in the companion CSV. The DSR trial count for any future promotion decision must include all seven overlay variants plus any protocol change logged after this run.

## Track 1 decision (plan §16 Phase 4) — REJECT for promotion

No E-variant passes all §11.1 conditions → no paper book is created;
`donchian_tcn_risk_overlay` stays unregistered. Decided strictly by the
predefined gate; per §4.3 the B-family also fails (B1-B3 never beat B0), so no
deterministic overlay is promoted either.

What the data did show, recorded for any *future preregistered* family — not
as grounds for promoting anything from this one:

1. **Cross-asset information adds predictive value.** The preregistered
   ablation ordering held out of sample: E3 beats E1 and E4 beats E3 on
   PR-AUC (0.124 → 0.143 → 0.148), and E2/E3 beat the HAR-RV baseline on
   QLIKE (−5.830/−5.833 vs −5.800). A three-coefficient HAR remains
   embarrassingly close.
2. **Predictive value did not convert to significant economic value.** The
   §11.2 rejection rule "predictive metrics improve but net portfolio metrics
   do not" applies to E3/E4. The strongest variant, E2 (SPY + QQQ/SPY spread,
   no VIX), beat B0 (Sharpe 1.08 vs 1.06, MaxDD −25.9% vs −28.5%, 5/7 folds,
   survives best-fold removal, positive ΔSR on all three coins) but with
   Ledoit-Wolf p = 0.36 vs B0 — indistinguishable from noise, and its MaxDD
   improvement (9%) is below the 15% bar.
3. **Selection discipline.** E2 was not the preregistered primary comparison
   (E3 was). Treating E2's pattern as a finding and re-testing it would be a
   new trial family on fresh data (or a longer OOS window), counted in the
   ledger; promoting it now on this sample would be exactly the selection this
   protocol exists to prevent.
4. The overlay concept is intact but unproven: every overlay cut CAGR roughly
   in proportion to risk in a strongly trending OOS window (2023-03 → 2026-07,
   base Sharpe 1.06). A risk overlay earns its keep in regimes this window
   mostly lacked.
