# Phase 1 Research Report — 2026-07-13

**Protocol (ex-ante)**: the parameter grid is fixed before results (32 trials/market); 2-year train → 6-month out-of-sample rolling, parameters selected by net Sharpe over the training period, and the fold goes flat if the training-period best is ≤0; cost = fees + √impact slippage (spot 10bps per side + slippage, USDT-M taker 6bps / maker 2bps; $10k notional); futures include daily funding (Binance series used as a Bitget proxy). `full_best_sharpe` is the full-sample post-hoc best (overfitting upper bound, for reference only); the verdict looks only at walk-forward OOS.

## Gate verdict (2026-07-13 revision: two-tier standard, certification targets the deployment object)

**Research candidate: PASS ✅** (within-family basis: OOS Sharpe>0 and within-family DSR≥0.95; 3 rows qualify)

**Statistical certification: FAIL ❌ — not passed** (deployment object = donchian parameter ensemble × 3-coin portfolio: DSR(N=4 family selection)=0.868, DSR(N=32 portfolio-level parameter×family)=0.75, both required to be ≥0.95)

> Revision note: the original gate corrected DSR only by the within-family trial count, which underestimated the selection that actually occurred (4 families × 3 coins × 2 markets × cost basis × select-param/ensemble switch), and the certification object did not match the deployment object (PBO 0.66–0.92 is also a strong warning). **When certification does not pass, Phase 2 is positioned as exploratory paper validation and does not constitute strategy certification**; the `family_gate` column is only a weak within-family basis for screening reference.

### Research candidates (rows qualifying on the within-family basis)

| symbol | market | family | oos_sharpe | dsr | dsr_meta32 | pbo | ens_oos_sharpe | ens_oos_maxdd_pct | bh_oos_sharpe |
|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | spot | donchian | 0.85 | 0.961 | 0.569 | 0.66 | 0.77 | -37.2 | 0.52 |
| ETHUSDT | spot | donchian | 0.89 | 0.96 | 0.723 | 0.7 | 0.88 | -42.8 | 0.57 |
| SOLUSDT | spot | donchian | 1.02 | 0.976 | 0.66 | 0.92 | 0.74 | -44.1 | 0.62 |

## Main results (walk-forward out-of-sample, net of costs)

| symbol | market | family | cost | full_best_sharpe | oos_sharpe | oos_cagr_pct | oos_maxdd_pct | dsr | dsr_meta32 | pbo | ens_oos_sharpe | bh_oos_sharpe | family_gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | spot | donchian | spot_taker10 | 1.22 | 0.85 | 26.8 | -36.5 | 0.961 | 0.569 | 0.66 | 0.77 | 0.52 | True |
| BTCUSDT | spot | rsi_meanrev | spot_taker10 | 0.41 | 0.06 | -4.2 | -50.9 | 0.349 | 0.047 | 0.69 | 0.13 | 0.52 | False |
| BTCUSDT | spot | sma_cross | spot_taker10 | 0.95 | 0.55 | 16.1 | -72.0 | 0.803 | 0.299 | 0.51 | 0.66 | 0.52 | False |
| BTCUSDT | spot | tsmom | spot_taker10 | 1.28 | 0.25 | 2.0 | -20.3 | 0.291 | 0.109 | 0.16 | 0.59 | 0.52 | False |
| BTCUSDT | um | donchian | um_taker6 | 0.82 | 0.67 | 17.2 | -30.8 | 0.915 | 0.659 | 0.96 | 0.51 | 0.23 | False |
| BTCUSDT | um | donchian | um_maker2 | 0.82 | 0.68 | 17.5 | -30.4 | 0.92 | 0.664 | 0.97 | 0.52 | 0.23 | False |
| BTCUSDT | um | rsi_meanrev | um_taker6 | 0.56 | -0.1 | -9.2 | -55.2 | 0.208 | 0.11 | 0.72 | 0.08 | 0.23 | False |
| BTCUSDT | um | rsi_meanrev | um_maker2 | 0.57 | -0.09 | -9.0 | -54.9 | 0.214 | 0.111 | 0.73 | 0.09 | 0.23 | False |
| BTCUSDT | um | sma_cross | um_taker6 | 0.87 | 0.65 | 19.1 | -43.3 | 0.841 | 0.64 | 0.37 | 0.58 | 0.23 | False |
| BTCUSDT | um | sma_cross | um_maker2 | 0.87 | 0.65 | 19.2 | -43.3 | 0.842 | 0.637 | 0.37 | 0.59 | 0.23 | False |
| BTCUSDT | um | tsmom | um_taker6 | 0.85 | 0.24 | 2.5 | -28.8 | 0.407 | 0.303 | 0.44 | 0.39 | 0.23 | False |
| BTCUSDT | um | tsmom | um_maker2 | 0.9 | 0.2 | 1.8 | -28.3 | 0.36 | 0.273 | 0.39 | 0.44 | 0.23 | False |
| ETHUSDT | spot | donchian | spot_taker10 | 1.03 | 0.89 | 37.2 | -41.6 | 0.96 | 0.723 | 0.7 | 0.88 | 0.57 | True |
| ETHUSDT | spot | rsi_meanrev | spot_taker10 | 0.92 | 0.46 | 10.6 | -35.1 | 0.556 | 0.331 | 0.19 | 0.24 | 0.57 | False |
| ETHUSDT | spot | sma_cross | spot_taker10 | 0.79 | 0.78 | 33.6 | -57.2 | 0.949 | 0.628 | 0.83 | 0.6 | 0.57 | False |
| ETHUSDT | spot | tsmom | spot_taker10 | 1.24 | 0.36 | 3.3 | -21.7 | 0.401 | 0.253 | 0.08 | 0.78 | 0.57 | False |
| ETHUSDT | um | donchian | um_taker6 | 0.93 | 0.34 | 5.8 | -38.8 | 0.692 | 0.319 | 0.75 | 0.24 | -0.02 | False |
| ETHUSDT | um | donchian | um_maker2 | 0.94 | 0.35 | 6.2 | -38.6 | 0.698 | 0.324 | 0.73 | 0.25 | -0.02 | False |
| ETHUSDT | um | rsi_meanrev | um_taker6 | 1.09 | 0.53 | 14.3 | -35.3 | 0.563 | 0.481 | 0.25 | 0.05 | -0.02 | False |
| ETHUSDT | um | rsi_meanrev | um_maker2 | 1.09 | 0.54 | 14.5 | -35.3 | 0.569 | 0.481 | 0.25 | 0.06 | -0.02 | False |
| ETHUSDT | um | sma_cross | um_taker6 | 0.75 | 0.07 | -5.4 | -56.4 | 0.422 | 0.153 | 0.65 | -0.19 | -0.02 | False |
| ETHUSDT | um | sma_cross | um_maker2 | 0.75 | 0.08 | -5.3 | -56.3 | 0.424 | 0.152 | 0.65 | -0.19 | -0.02 | False |
| ETHUSDT | um | tsmom | um_taker6 | 1.06 | 0.08 | 0.2 | -24.1 | 0.171 | 0.156 | 0.22 | 0.42 | -0.02 | False |
| ETHUSDT | um | tsmom | um_maker2 | 1.1 | -0.01 | -1.4 | -32.6 | 0.124 | 0.115 | 0.21 | 0.47 | -0.02 | False |
| SOLUSDT | spot | donchian | spot_taker10 | 1.4 | 1.02 | 52.0 | -54.3 | 0.976 | 0.66 | 0.92 | 0.74 | 0.62 | True |
| SOLUSDT | spot | rsi_meanrev | spot_taker10 | 0.6 | 0.22 | -5.5 | -79.3 | 0.48 | 0.117 | 0.55 | 0.36 | 0.62 | False |
| SOLUSDT | spot | sma_cross | spot_taker10 | 1.11 | 0.55 | 15.3 | -68.7 | 0.769 | 0.289 | 0.83 | 0.66 | 0.62 | False |
| SOLUSDT | spot | tsmom | spot_taker10 | 1.25 | 0.77 | 7.5 | -16.1 | 0.8 | 0.451 | 0.27 | 0.52 | 0.62 | False |
| SOLUSDT | um | donchian | um_taker6 | 1.49 | 0.25 | 1.3 | -46.0 | 0.572 | 0.128 | 0.42 | 0.75 | 0.76 | False |
| SOLUSDT | um | donchian | um_maker2 | 1.5 | 0.26 | 1.8 | -45.8 | 0.577 | 0.135 | 0.41 | 0.76 | 0.76 | False |
| SOLUSDT | um | rsi_meanrev | um_taker6 | 0.67 | 0.48 | 10.6 | -65.0 | 0.67 | 0.242 | 0.4 | 0.62 | 0.76 | False |
| SOLUSDT | um | rsi_meanrev | um_maker2 | 0.68 | 0.48 | 11.0 | -65.0 | 0.673 | 0.25 | 0.39 | 0.62 | 0.76 | False |
| SOLUSDT | um | sma_cross | um_taker6 | 1.02 | -0.33 | -30.7 | -77.9 | 0.177 | 0.012 | 0.88 | 0.63 | 0.76 | False |
| SOLUSDT | um | sma_cross | um_maker2 | 1.03 | -0.33 | -30.7 | -77.9 | 0.178 | 0.013 | 0.88 | 0.63 | 0.76 | False |
| SOLUSDT | um | tsmom | um_taker6 | 0.97 | 0.25 | 2.4 | -23.6 | 0.284 | 0.133 | 0.18 | 0.11 | 0.76 | False |
| SOLUSDT | um | tsmom | um_maker2 | 1.0 | 0.34 | 3.7 | -22.9 | 0.344 | 0.175 | 0.19 | 0.15 | 0.76 | False |

## Cross-coin equal-weight portfolio (spot taker cost; each family = parameter ensemble × 3 coins equal-weight)

The only degree of freedom in the construction is **which family to pick** (N=4 trials; no parameter selection, no coin selection); the window is the three-coin OOS intersection. This is the number closest to the actual deployment form, and the basis where the DSR correction is least contaminated by selection.

| family | sharpe | cagr_pct | ann_vol_pct | max_dd_pct | n_bars | dsr_n4 | dsr_n32 |
|---|---|---|---|---|---|---|---|
| sma_cross | 0.64 | 18.6 | 38.2 | -36.0 | 1414 | 0.817 | 0.68 |
| donchian | 0.74 | 19.6 | 30.5 | -28.4 | 1414 | 0.868 | 0.75 |
| tsmom | 0.65 | 6.0 | 9.8 | -12.2 | 1414 | 0.824 | 0.689 |
| rsi_meanrev | 0.35 | 6.3 | 30.7 | -38.1 | 1414 | 0.634 | 0.461 |

## Funding-rate carry simulation (delta neutral, 1x, not a prediction-class gate)

> ⚠️ This simulation includes only funding income and entry/exit costs; it does **not** model close-out slippage spikes in extreme conditions, instantaneous basis blowouts, or the margin chain (BIS WP1087: carry is essentially compensation for crash risk). The Sharpe in the table substantially overstates stability; interpret the reality as 'high-single-digit APR + rare tail events'.

### BTCUSDT (Binance funding 2020→now)

| variant | net APR% | Sharpe | MaxDD% | time in market | round trips | avg funding APR% over period |
|---|---|---|---|---|---|---|
| always_on | 5.95 | 10.56 | -0.74 | 100% | 0 | 11.92 |
| filter_5/0 | 5.44 | 9.1 | -0.34 | 80% | 17 | 11.92 |
| filter_10/2 | 4.79 | 7.99 | -0.38 | 53% | 15 | 11.92 |

### ETHUSDT (Binance funding 2020→now)

| variant | net APR% | Sharpe | MaxDD% | time in market | round trips | avg funding APR% over period |
|---|---|---|---|---|---|---|
| always_on | 7.08 | 9.75 | -0.89 | 100% | 0 | 14.18 |
| filter_5/0 | 6.56 | 8.75 | -0.89 | 80% | 19 | 14.18 |
| filter_10/2 | 6.23 | 8.46 | -0.28 | 56% | 10 | 14.18 |

### SOLUSDT (Binance funding 2020→now)

| variant | net APR% | Sharpe | MaxDD% | time in market | round trips | avg funding APR% over period |
|---|---|---|---|---|---|---|
| always_on | 0.02 | 0.01 | -19.87 | 100% | 0 | 0.08 |
| filter_5/0 | 4.15 | 5.75 | -0.76 | 59% | 26 | 0.08 |
| filter_10/2 | 4.09 | 5.85 | -0.81 | 43% | 17 | 0.08 |

## Conclusions

1. **Two-tier verdict**: research candidate passes (BTCUSDT, ETHUSDT, SOLUSDT spot donchian, within-family DSR 0.96–0.98); **statistical certification does not pass** — deployment portfolio DSR(N=4)=0.868, DSR(N=32)=0.75, below the 0.95 threshold.

2. **Qualitative strength of evidence**: single-market meta-DSR(32) is only 0.57–0.72, and PBO 0.66–0.92 is a strong warning (within-family parameter rankings are unstable; the ensemble only removes parameter-selection risk, not family-level uncertainty). Supporting points: the three coins agree in direction, the parameter ensemble is robust, and it agrees with the literature (JFQA 2025 CTREND). Qualitatively an **uncertified research candidate**: real but moderate-strength evidence, worthy only of exploratory paper validation, not any basis for live deployment.

3. **Exploratory paper validation object (not a certified strategy)**: `donchian` parameter ensemble × 3 coins equal-weight (spot long/flat). Expected portfolio characteristics (net of costs, 3.9-year OOS intersection): Sharpe ≈ 0.74, CAGR ≈ 19.6%, annualized vol ≈ 30.5%, MaxDD ≈ -28.4%. The tsmom ensemble (low vol, MaxDD −12%) is observed in parallel in the paper book as a diversification candidate (this portfolio is not pre-registered; observed only, not deployed).

4. **Execution market: spot**: um is uniformly weaker than spot (funding drag + sample-period differences), and turnover is so low that maker/taker sensitivity is negligible — spot execution also naturally eliminates leverage and liquidation risk.

5. **carry**: BTC/ETH always-on net APR ~6–7% (understood after discounting for tail risk); SOL average funding ≈ 0, so a threshold switch is **required**. The carry executor stays in Phase 3 as planned.

6. **Control group behaves normally**: RSI mean-reversion is at the bottom on the portfolio basis (0.35) — the process discriminates between good and bad strategies, a sign that the methodology self-check passes.

## Appendix: fold-by-fold record of each market's best family

### BTCUSDT/spot/donchian

| fold (OOS window) | selected params | train SR/period | OOS SR/period |
|---|---|---|---|
| 2020-12-31→2021-06-30 | n_entry=40,n_exit=20 | 0.1346 | 0.0559 |
| 2021-07-01→2021-12-29 | n_entry=40,n_exit=20 | 0.1026 | 0.0546 |
| 2021-12-30→2022-06-29 | n_entry=40,n_exit=20 | 0.1103 | -0.0225 |
| 2022-06-30→2022-12-28 | n_entry=100,n_exit=50 | 0.0977 | 0.0 |
| 2022-12-29→2023-06-28 | n_entry=100,n_exit=50 | 0.0477 | -0.007 |
| 2023-06-29→2023-12-27 | n_entry=20,n_exit=10 | 0.0221 | 0.118 |
| 2023-12-28→2024-06-26 | n_entry=55,n_exit=27 | 0.035 | 0.0631 |
| 2024-06-27→2024-12-25 | n_entry=55,n_exit=27 | 0.061 | 0.1361 |
| 2024-12-26→2025-06-25 | n_entry=20,n_exit=10 | 0.0964 | 0.074 |
| 2025-06-26→2025-12-24 | n_entry=20,n_exit=10 | 0.0861 | -0.0181 |
| 2025-12-25→2026-06-24 | n_entry=20,n_exit=10 | 0.0603 | 0.011 |

### BTCUSDT/um/donchian

| fold (OOS window) | selected params | train SR/period | OOS SR/period |
|---|---|---|---|
| 2021-12-31→2022-06-30 | n_entry=40,n_exit=20 | 0.079 | -0.0245 |
| 2022-07-01→2022-12-29 | n_entry=100,n_exit=50 | 0.0794 | 0.0 |
| 2022-12-30→2023-06-29 | n_entry=100,n_exit=50 | 0.0308 | -0.0091 |
| 2023-06-30→2023-12-28 | n_entry=20,n_exit=10 | 0.0177 | 0.0977 |
| 2023-12-29→2024-06-27 | n_entry=55,n_exit=27 | 0.0279 | 0.0542 |
| 2024-06-28→2024-12-26 | n_entry=55,n_exit=27 | 0.0527 | 0.1142 |
| 2024-12-27→2025-06-26 | n_entry=20,n_exit=10 | 0.0877 | 0.0771 |
| 2025-06-27→2025-12-25 | n_entry=20,n_exit=10 | 0.0774 | -0.023 |
| 2025-12-26→2026-06-25 | n_entry=20,n_exit=10 | 0.0558 | 0.0123 |

### ETHUSDT/spot/donchian

| fold (OOS window) | selected params | train SR/period | OOS SR/period |
|---|---|---|---|
| 2020-12-31→2021-06-30 | n_entry=40,n_exit=20 | 0.0788 | 0.125 |
| 2021-07-01→2021-12-29 | n_entry=20,n_exit=10 | 0.1071 | 0.0884 |
| 2021-12-30→2022-06-29 | n_entry=20,n_exit=10 | 0.1303 | -0.0303 |
| 2022-06-30→2022-12-28 | n_entry=20,n_exit=10 | 0.1033 | -0.0041 |
| 2022-12-29→2023-06-28 | n_entry=20,n_exit=10 | 0.0807 | 0.034 |
| 2023-06-29→2023-12-27 | n_entry=20,n_exit=10 | 0.0253 | 0.0608 |
| 2023-12-28→2024-06-26 | n_entry=20,n_exit=10 | 0.0137 | 0.0353 |
| 2024-06-27→2024-12-25 | n_entry=100,n_exit=50 | 0.03 | 0.0367 |
| 2024-12-26→2025-06-25 | n_entry=55,n_exit=27 | 0.0437 | 0.0358 |
| 2025-06-26→2025-12-24 | n_entry=55,n_exit=27 | 0.0509 | 0.0946 |
| 2025-12-25→2026-06-24 | n_entry=55,n_exit=27 | 0.0584 | -0.1181 |

### ETHUSDT/um/rsi_meanrev

| fold (OOS window) | selected params | train SR/period | OOS SR/period |
|---|---|---|---|
| 2021-12-31→2022-06-30 | n=14,buy_below=25,exit_above=70 | 0.0867 | 0.0367 |
| 2022-07-01→2022-12-29 | n=14,buy_below=25,exit_above=70 | 0.0184 | 0.1373 |
| 2022-12-30→2023-06-29 | n=14,buy_below=25,exit_above=70 | 0.0583 | 0.0 |
| 2023-06-30→2023-12-28 | n=14,buy_below=25,exit_above=70 | 0.0583 | 0.027 |
| 2023-12-29→2024-06-27 | n=14,buy_below=25,exit_above=70 | 0.0598 | 0.0 |
| 2024-06-28→2024-12-26 | n=7,buy_below=25,exit_above=55 | 0.0857 | 0.0323 |
| 2024-12-27→2025-06-26 | n=7,buy_below=25,exit_above=55 | 0.0556 | 0.0391 |
| 2025-06-27→2025-12-25 | n=14,buy_below=25,exit_above=70 | 0.0616 | 0.0 |
| 2025-12-26→2026-06-25 | n=14,buy_below=25,exit_above=70 | 0.0604 | -0.0512 |

### SOLUSDT/spot/donchian

| fold (OOS window) | selected params | train SR/period | OOS SR/period |
|---|---|---|---|
| 2022-08-11→2023-02-08 | n_entry=40,n_exit=20 | 0.1357 | 0.0026 |
| 2023-02-09→2023-08-09 | n_entry=55,n_exit=27 | 0.1073 | -0.0868 |
| 2023-08-10→2024-02-07 | n_entry=20,n_exit=10 | 0.0592 | 0.1918 |
| 2024-02-08→2024-08-07 | n_entry=20,n_exit=10 | 0.0646 | 0.1078 |
| 2024-08-08→2025-02-05 | n_entry=20,n_exit=10 | 0.0829 | 0.0579 |
| 2025-02-06→2025-08-06 | n_entry=20,n_exit=10 | 0.0931 | 0.0697 |
| 2025-08-07→2026-02-04 | n_entry=20,n_exit=10 | 0.1163 | 0.0462 |
| 2026-02-05→2026-07-12 | n_entry=20,n_exit=10 | 0.0719 | -0.1235 |

### SOLUSDT/um/rsi_meanrev

| fold (OOS window) | selected params | train SR/period | OOS SR/period |
|---|---|---|---|
| 2022-09-14→2023-03-14 | n=14,buy_below=25,exit_above=70 | 0.0117 | 0.0132 |
| 2023-03-15→2023-09-12 | n=7,buy_below=25,exit_above=55 | 0.0248 | -0.0091 |
| 2023-09-13→2024-03-12 | n=14,buy_below=25,exit_above=55 | 0.0325 | 0.0 |
| 2024-03-13→2024-09-10 | n=7,buy_below=25,exit_above=70 | 0.043 | 0.0924 |
| 2024-09-11→2025-03-11 | n=7,buy_below=25,exit_above=70 | 0.0756 | 0.0007 |
| 2025-03-12→2025-09-09 | n=7,buy_below=25,exit_above=55 | 0.0639 | 0.1595 |
| 2025-09-10→2026-03-10 | n=7,buy_below=25,exit_above=55 | 0.0953 | -0.0056 |
| 2026-03-11→2026-07-12 | n=7,buy_below=25,exit_above=55 | 0.069 | -0.0171 |

