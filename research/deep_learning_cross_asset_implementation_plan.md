# Cross-Asset Deep Learning for Daily Crypto Trading — Implementation Plan

**Status:** Proposed; research only
**Date:** 2026-07-14 (v2 — revised after critical review; see §19 changelog)
**Target repository:** QubitvaleTrading
**Primary trading frequency:** Daily
**Initial deployment market:** Spot, long/flat

> This plan defines a controlled research program. It does not authorize live trading and does not treat a successful backtest or a short paper-trading period as proof of persistent alpha.

## 1. Executive Decision

Implement the work in three gated model tracks, in this order:

1. **Cross-asset TCN risk overlay for the existing Donchian ensemble.**
   Use crypto market features plus QQQ, SPY, VIX, and, last, GLD to estimate near-term volatility and tail risk. The network may only scale the existing Donchian target down; it must never create exposure that the base strategy does not already request.
2. **Cost-aware Deep Momentum Network.**
   Train a small shared LSTM/GRU to output continuous long/flat spot weights, using a net-of-cost Sharpe objective with explicit turnover regularization.
3. **Cross-sectional crypto ranker.**
   Only after a point-in-time universe of at least 30 liquid, executable spot assets exists, train a shared sequence model to rank next-day relative returns and hold a small long-only top-ranked basket.

The first production candidate is Track 1. Tracks 2 and 3 remain independent research families and must not be selected after seeing Track 1 results without counting the additional selection in the multiple-testing correction.

## 2. Why This Order

The current repository has three crypto assets (2,750 daily bars for BTC/ETH from 2019-01, 2,162 for SOL from 2020-08), a strict D-1 decision convention, a cost-aware daily backtester, two independent paper books, and only 1,414 bars in the common portfolio-level out-of-sample window. The current Donchian portfolio is an uncertified research candidate with approximately 0.74 OOS Sharpe and -28.4% maximum drawdown. The TSMOM portfolio is lower risk but also uncertified.

This creates four constraints:

- A large Transformer or a separate neural network per asset is not justified by the sample size.
- New information should first be used to reduce risk in a controlled overlay, not to replace the trading stack end to end.
- Every architecture, feature group, threshold, seed-selection rule, and strategy family expands the research trial count and must be included in DSR/PBO reporting.
- The crypto–equity relationship itself is regime-dependent, not a stable structural fact: BTC was largely decoupled from equities before 2020, tightly coupled during 2020–2022, decoupled again after the October 2025 flash crash (30-day correlation near −0.3 in late 2025), and hit record-high correlation (~0.96) by April 2026. Roughly one third of the training history predates the coupled regime. Any cross-asset result must therefore be reported per regime slice (§10.3), and a model that only works in the coupled regime must be labeled as such rather than promoted as a general risk overlay.

## 3. Objectives and Non-Goals

### 3.1 Objectives

- Test whether U.S. equity and gold market state adds stable, net-of-cost information to daily crypto risk management.
- Preserve the repository's no-lookahead, D-1 decision, fail-closed, and multi-book disciplines.
- Measure incremental value against frozen crypto-only baselines.
- Produce deterministic, auditable model artifacts and daily inference records.
- Promote a model to exploratory paper trading only after it passes the predefined historical research gate.

### 3.2 Non-Goals

- Predict absolute BTC, ETH, or SOL price levels.
- Use intraday execution, market making, order-book models, or high-frequency labels.
- Use deep reinforcement learning in this phase.
- Use leverage, short altcoins, or derivatives in the initial neural strategy.
- Optimize a large architecture or feature search after observing OOS results.
- Treat six weeks of paper trading as statistical validation of alpha.

## 4. Stage 0 — Baseline and Protocol Freeze

Complete these items before downloading new data or training a model.

### 4.1 Reconcile the existing DSR record

The repository currently contains inconsistent Donchian portfolio DSR(N=32) values: `README.md` reports 0.395, while `research/reports/phase1_report_2026-07-13.md` reports 0.75 and the `strategies/donchian_ensemble.py` header reports 0.751. Rerun `research.metrics.deflated_sharpe` on the frozen deployment-portfolio OOS series with a persisted trial count and SR-variance input, determine which of the two figures that reproduces (the discrepancy is most plausibly a differing `n_trials` or `sr_variance` argument between report generations), regenerate all three documents from that single computation, and freeze the corrected baseline value together with the exact inputs that produced it.

This does not change the current FAIL verdict, but it is required before measuring neural-model incremental value.

### 4.2 Register the experiment families ex ante

Two families are registered together. The **B-family** is the deterministic baseline ladder every neural variant must beat; the **E-family** is the neural feature-ablation ladder. All B and E variants feed the *identical* risk-multiplier mapping of §7.5, so each comparison isolates the forecaster, not the decision rule.

| ID | Overlay input | Purpose |
|---|---|---|
| B0 | None (existing Donchian ensemble) | Frozen non-neural, non-overlay baseline |
| B1 | Trailing 20-day realized volatility (σ20) | Simplest deterministic vol control |
| B2 | HAR-RV forecast of 5-day volatility (OLS, fit on training fold only) | Strongest classical vol forecaster; the literature repeatedly finds it hard for ML to beat |
| B3 | B1 + VIX-percentile threshold | Deterministic cross-asset stress control |
| E1 | Crypto-only TCN overlay | Tests the architecture without cross-asset information |
| E2 | E1 + SPY + QQQ/SPY relative-strength spread | Tests equity risk-state information (QQQ enters only as a spread against SPY; raw QQQ and SPY series are ~0.95 correlated, and feeding both nearly duplicates one input while spending an ablation step on nothing) |
| E3 | E2 + VIX | Tests explicit market-stress information |
| E4 | E3 + GLD | Tests the incremental value of gold |

B1-B3 and E1-E4 are seven trials for family-level multiple-testing purposes (B2's HAR fit is per-fold OLS with fixed 1/5/22-day lags — no hyperparameter search, but it still counts as a trial). Any additional feature bundle, sequence length, label, threshold set, or architecture explored later must be added to the research trial ledger.

Use one fixed union feature schema and identical network shape for E1-E4. Features removed by an ablation are marked unavailable by construction and paired with a zero availability mask; they are not replaced with silently meaningful zero observations.

### 4.3 Freeze the primary decision rule

- Primary comparison: E3 versus B2, E1, and B0. Beating the unscaled Donchian (B0) is necessary but never sufficient; the neural overlay earns its complexity only if it beats the HAR-RV overlay (B2) net of costs.
- Statistical significance of every pairwise overlay comparison uses the Ledoit-Wolf (2008) studentized stationary-bootstrap test on the daily net-return differential, not a naive Sharpe comparison.
- Gold is accepted only if E4 improves on E3 out of sample.
- No model is selected on classification accuracy alone.
- The first model family is a risk overlay on `donchian_ensemble`; TSMOM overlays are out of scope for the initial family.
- If B2 or B3 beats every E-variant, the deterministic overlay is itself a legitimate promotion candidate under the same gate, and the neural track stops there. That outcome is a success of the program, not a failure.

## 5. Stage 1 — Cross-Asset Data Foundation

### 5.1 Initial instruments

| Instrument | Research symbol | Role | Priority |
|---|---|---|---|
| Nasdaq-100 ETF | QQQ | Technology/risk-appetite proxy | Required |
| S&P 500 ETF | SPY | Broad U.S. equity risk proxy | Required |
| Cboe Volatility Index | VIX | Equity stress and forward-volatility proxy | Required |
| Gold ETF | GLD | Gold risk-state proxy with aligned U.S. market hours | Ablation only |

Use ETFs rather than raw index levels for QQQ, SPY, and GLD in the first implementation because they share the same U.S. session calendar and provide executable OHLCV semantics. VIX remains an index input and is never treated as a directly executable position.

Optional second-wave variables, excluded from E1-E4, are the broad U.S. dollar index, 10-year nominal Treasury yield, and 10-year real yield. They require a separate preregistered experiment family because their publication and revision timing differs from exchange-traded market bars.

### 5.2 Provider abstraction

Create a provider interface rather than coupling research code to one vendor:

```python
class CrossAssetDailyProvider(Protocol):
    def fetch_history(self, symbol: str, start: date, end: date) -> DataFrame: ...
    def fetch_latest_completed(self, symbol: str, as_of: Timestamp) -> DataFrame: ...
```

Planning assumption:

- Use an authenticated market-data API with timestamped daily bars for SPY, QQQ, and GLD. Alpaca's historical stock-bars API is a candidate adapter, **but its free tier serves IEX-only data (roughly 2% of consolidated market volume): IEX daily closes can deviate from the official consolidated close, and IEX volume is unusable as a volume feature.** Either use Alpaca with a SIP-feed subscription (the free tier can query SIP history with a ≥15-minute-old `end` parameter, which is compatible with this plan's end-of-day cadence — verify this at implementation time), or prefer a source with official-close EOD semantics (e.g., Tiingo EOD). Record which feed produced each bar in the `source` column.
- Use official Cboe historical VIX data for the VIX adapter: `https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv` (daily OHLC from 1990, free, no key). This file carries a session-date label only, so `available_at` must be assigned by the ingestion process, not read from the file (§5.4).
- Keep a second free provider for recent-window cross-source QC (e.g., Stooq EOD CSV for the ETFs; FRED series `VIXCLS` for VIX).
- Do not make an unofficial web-scraping endpoint the sole production source.

### 5.3 Storage contract

Store one append-safe Parquet series per instrument:

```text
data/store/cross_asset/market/SPY/1d.parquet
data/store/cross_asset/market/QQQ/1d.parquet
data/store/cross_asset/market/GLD/1d.parquet
data/store/cross_asset/index/VIX/1d.parquet
```

Required columns:

| Column | Meaning |
|---|---|
| `session_date` | Exchange session date |
| `bar_start` | Source bar start timestamp with timezone |
| `bar_end` | Source bar end timestamp with timezone |
| `available_at` | Earliest timestamp at which the completed value was usable by the strategy |
| `open`, `high`, `low`, `close` | Raw market values |
| `volume` | ETF volume; nullable for VIX |
| `is_market_open` | Whether the source market held a regular session |
| `source` | Provider identifier |
| `ingested_at` | UTC ingestion timestamp |

Never infer `available_at` solely from a provider's date label. Store it explicitly.

### 5.4 Timing and no-lookahead alignment

The daily job runs shortly after 00:00 UTC (the Mac launchd job fires at 08:10 local, ≈00:10 UTC). The most recent regular U.S. equity close (16:00 ET ≈ 20:00/21:00 UTC depending on DST) is normally available roughly three to four hours earlier. Two provider-timing caveats:

- The official closing auction print and the final VIX value are disseminated minutes after 16:00 ET (Cboe disseminates the final VIX index value at approximately 16:15 ET), and EOD files may be revised later in the evening. Set `available_at` conservatively — e.g., ingestion time on first observation, never earlier than 21:30 UTC for a same-day close — rather than back-dating it to the exchange close.
- Backfilled history gets `available_at = ingested_at` semantics only for training-period rows where the conservative rule above is provably satisfied; the as-of join below is what enforces correctness either way.

Build each model row using an as-of join:

```text
external.available_at <= crypto_decision_timestamp
```

Rules:

- Normalize all internal timestamps to UTC while preserving the original exchange session date.
- On weekends and U.S. holidays, carry the last known market state only as a stale state, not as a new zero-return observation.
- Add `market_closed` and `days_since_last_session` features.
- A forward-filled close must not create a return or volatility observation.
- Daylight-saving transitions must be covered by tests.
- Every research row must persist the external record timestamps that produced it.

### 5.5 Data quality gate

Before feature generation, require:

- no duplicate `(symbol, session_date)` rows;
- valid OHLC relationships for ETF bars;
- monotonic `bar_end` and `available_at` timestamps;
- no observation whose `available_at` is later than the decision time used in that sample;
- documented exchange holidays rather than unexplained gaps;
- recent close-price agreement with a second source within a predefined tolerance;
- explicit freshness status for every daily inference run.

Unexpectedly stale external data must be treated as unknown risk state. It must block new exposure increases, allow reductions, and must not be translated into a forced liquidation signal.

## 6. Stage 2 — Feature Engineering

### 6.1 Principles

- Do not feed raw price levels to the model.
- Fit all means, standard deviations, quantile boundaries, and imputers on the training fold only.
- Use the same feature calculations in research, replay, and daily inference.
- Version the feature schema and store its hash with every checkpoint.
- Keep the initial feature set intentionally small.

### 6.2 Crypto features

Per crypto asset:

- volatility-normalized log returns over 1, 5, 21, 63, 126, and 252 days;
- 20-day and 60-day realized volatility;
- daily high-low range normalized by close;
- 20-day volume z-score;
- distance from Donchian entry and exit bands;
- distance from 20-day and 100-day moving averages;
- funding-rate level and trailing aggregates where historically available;
- optional realized volatility aggregated from complete 1-hour bars.

### 6.3 Cross-asset features

For QQQ, SPY, and GLD:

- 1-day, 5-day, and 20-day log returns;
- 20-day realized volatility;
- 20-day and 60-day drawdown;
- distance from 20-day and 100-day moving averages;
- market-closed flag and days since last completed session.

For VIX:

- log level;
- 1-day and 5-day change;
- expanding training-fold percentile;
- distance from its 20-day moving average.

Cross-market state features:

- 20-day and 60-day BTC/SPY rolling correlation;
- 60-day BTC-to-SPY beta;
- 20-day QQQ/SPY relative-strength spread (log-return difference) as the risk-appetite proxy;
- equity stress interaction: negative SPY return multiplied by VIX percentile;
- a cross-asset-data-valid mask.

Rolling correlations and betas must use only information available at the decision timestamp. The rolling correlation and beta features are also the model's handle on the coupling-regime problem in §2: the network should learn to discount equity state when measured coupling is low, and the regime-sliced evaluation of §10.3 verifies whether it actually does.

Deliberately excluded from E1-E4 and reserved for a separately preregistered second-wave family: U.S. spot-ETF flow features. The repository already collects BTC/ETH/SOL ETF flows (Farside/CoinGlass) for the existing ETF-flow risk gate, and flows are arguably a more direct cross-market signal than index returns — but they have a short history (BTC ETFs listed January 2024), publisher-revision timing that is much harder to pin down than exchange closes, and they already act on the book through the deterministic gate, so mixing them into E1-E4 would confound the ablation ladder.

### 6.4 Feature output contract

Create one deterministic feature table keyed by:

```text
(decision_timestamp, crypto_symbol)
```

Persist:

- raw source record identifiers;
- feature values;
- availability masks;
- scaler version;
- feature schema hash;
- build timestamp.

## 7. Track 1 — Cross-Asset TCN Risk Overlay

### 7.1 Trading role

The model may only reduce the base Donchian target:

```text
final_target[i, t] = donchian_target[i, t] * risk_multiplier[i, t]
0 <= risk_multiplier[i, t] <= 1
```

It may not open a position when Donchian is flat, increase a Donchian weight, short an asset, or use leverage.

### 7.2 Labels

Train a multi-task model with two future-looking labels calculated strictly after the decision timestamp:

1. **Five-day realized volatility**, expressed as log realized volatility.
2. **Five-day tail event**, equal to one when:

   ```text
   min over h=1..5 of (
       cumulative_log_return[t+1:t+h]
       / (sigma20[t] * sqrt(h))
   ) <= -2.0
   ```

   `sigma20[t]` is known at the decision timestamp and is floored at the training fold's 5th percentile of positive `sigma20` observations to avoid unstable division.

Label-supply check (required before training): a −2σ five-day event has an unconditional base rate on the order of 5-10% of days; with roughly 1,400 common OOS bars per asset and five-day overlap, the number of *effectively independent* positive events per test fold is small (tens, not hundreds). Persist the positive-label count per fold; any fold with fewer than 10 positive tail events in validation reports tail-head metrics as "insufficient events" rather than as numbers, and the tail gate for that fold falls back to 1.0 (vol head only).

The tail head's raw sigmoid outputs are not assumed calibrated. Fit an isotonic (or Platt, if isotonic is unstable at this sample size — choose once, ex ante) calibration map on the **validation fold** and apply the §7.5 probability thresholds to calibrated probabilities. The calibration method choice is registered before the first walk-forward run.

Because labels overlap for five days, all train/validation/test boundaries require a five-day embargo.

### 7.3 Fixed initial architecture

Crypto sequence branch:

- lookback: 90 daily observations;
- causal TCN;
- three residual blocks;
- 8 filters per block;
- kernel size 3;
- dilations 1, 2, and 4;
- dropout 0.10.

Cross-asset context branch:

- latest valid cross-asset state plus short rolling summaries;
- two-layer MLP, widths 16 and 8;
- dropout 0.10.

Fusion and heads:

- concatenate both branch embeddings and availability masks;
- one volatility regression head;
- one tail-probability classification head.

This exact architecture is E1-E4. Changing it creates a new trial family.

### 7.4 Loss

```text
loss = huber(predicted_log_vol, realized_log_vol)
       + class_weighted_bce(predicted_tail_probability, tail_label)
       + weight_decay
```

Do not optimize the first TCN directly on portfolio Sharpe. The first track's purpose is a measurable risk forecast and controlled exposure overlay.

### 7.5 Risk multiplier

The mapping from forecast to multiplier is continuous in volatility and discrete only in the tail gate, and the *same mapping* is used by B1, B2, B3, and E1-E4 so that comparisons isolate the forecaster:

```text
vol_component[i, t]  = clip(sigma_ref / sigma_hat[i, t], 0, 1)
risk_multiplier[i, t] = vol_component[i, t] * tail_gate[i, t]
```

- `sigma_hat` is the variant's 5-day volatility forecast (σ20 for B1/B3, HAR-RV for B2, the TCN vol head for E1-E4), annualized consistently.
- `sigma_ref` is the training-fold median of realized 5-day volatility, fixed per fold before the test window opens. A discrete three-level map was considered and rejected: it creates cliff-edge turnover at the thresholds and makes the neural overlay structurally incomparable to standard volatility targeting.
- `tail_gate` uses calibrated tail probability (§7.2): `1.0` below 0.25; `0.5` from 0.25 to below 0.50; `0.0` at or above 0.50. B1/B2 have no tail head, so their `tail_gate` is identically 1.0; B3's gate is `0.5` when the VIX training-fold percentile exceeds 0.90, else 1.0. The extra tail lever available to E-variants is disclosed as part of what is being tested.

These definitions are fixed for B1-B3 and E1-E4. Changing any of them creates a new trial. Apply the existing 2% rebalance threshold after overlay weights are produced, and report overlay-induced turnover separately (a vol-scaled overlay trades every day the forecast moves; the 2% threshold is the only thing standing between the overlay and constant fee bleed — if overlay turnover still erases the gross advantage, §11.2 rejects the track).

### 7.6 Baseline comparisons

Compare E1-E4 against B0 (no overlay), B1 (σ20 targeting), B2 (HAR-RV), and B3 (σ20 + VIX threshold), all defined in §4.2 and all run through the identical §7.5 mapping and cost model.

A neural overlay must beat the best deterministic risk control (in practice, usually B2), not merely the unscaled Donchian strategy.

## 8. Track 2 — Cost-Aware Deep Momentum Network

Start only after Track 1 is fully evaluated and its result is frozen.

### 8.1 Model

- one shared model across BTC, ETH, and SOL;
- one-layer LSTM or GRU;
- hidden width 16;
- 252-day sequence of normalized trend, volatility, volume, funding, and cross-asset context features;
- sigmoid output per asset;
- long/flat spot weights with total gross exposure capped at 1.0.

### 8.2 Objective

```text
net_return[t] = sum_i(weight[i, t] * return[i, t+1])
                - cost[i, t]

loss = -Sharpe(net_return)
       + lambda_turnover * mean(abs(delta_weight))
       + lambda_tail * downside_risk_surrogate
```

Training costs must be no lower than the repository's 10 bps per-side spot fee plus slippage floor. Evaluate a 2x-cost stress case.

This design follows the deep-momentum-network line of Lim, Zohren, and Roberts (2019); its documented extensions — changepoint-detection features (Wood, Roberts, and Zohren 2021) and the Momentum Transformer (Wood, Giegerich, Roberts, and Zohren 2021) — are explicitly out of scope for the first pass and would each be a new preregistered trial family. Note the published DMN results are on 50-100 futures over 25 years; with three assets the loss is dominated by three highly correlated return streams, which is the concrete mechanism behind the §8.3 warning.

### 8.3 Constraint

With only three assets, this track is an engineering feasibility study, not a credible statistical certification program. A broader liquid-asset training panel is strongly preferred before promotion.

## 9. Track 3 — Cross-Sectional Crypto Ranker

Start only after a point-in-time, survivorship-bias-controlled universe exists.

### 9.1 Data prerequisite

- at least 30 liquid spot assets on each training date;
- universe determined from information known at D-1;
- selection based on trailing 30-day executable dollar volume and market-cap constraints;
- delisted and failed assets retained in historical data — Binance Vision is the designated source here, because its bulk archives retain full history for delisted pairs and the repository's backfill stack already speaks it; the universe on each historical date is derived from trailing dollar volume computed out of the archive itself, never from a current listings snapshot;
- stablecoins, wrapped duplicates, leveraged tokens, and non-executable assets excluded by fixed rules.

### 9.2 Initial model

- one shared GRU or LSTM;
- 90-day standardized return sequence;
- hidden width selected from a preregistered set of 5, 10, and 20;
- target: probability that an asset's next-day return exceeds the point-in-time universe median;
- fixed-seed ensemble; never select the best seed after observing OOS performance.

### 9.3 Portfolio rule

- rank assets daily;
- hold the top five long only;
- hold cash when confidence is below a predefined threshold;
- enter inside the top five and exit below rank ten to reduce turnover;
- cap individual weights and total exposure;
- apply the full spot cost model.

Published long/short results are not an expected-return estimate for this long/cash implementation.

## 10. Neural Walk-Forward Protocol

The existing parameter-grid walk-forward code assumes precomputed, path-independent variants and is not sufficient for neural retraining. Add a neural-specific trainer.

### 10.1 Time splits

- minimum initial training history: three years;
- validation window: 182 days;
- OOS test window: 182 days;
- expanding training window;
- retrain every 182 days;
- use identical date boundaries for all assets and model variants;
- apply label-horizon embargo at every boundary.

If three years of clean feature history is unavailable, the track remains blocked for statistical evaluation; shortening the window after seeing results is not allowed.

Expected fold arithmetic, stated ex ante so the gates in §11 are interpretable: with BTC/ETH history from 2019-01, SOL from 2020-08, a three-year minimum training window, and 182-day test windows, the common portfolio-level evaluation yields on the order of **six to eight complete OOS folds**. Every per-fold gate in §11.1 is therefore a coarse screen over single-digit observations, not a statistical test — the Ledoit-Wolf differential test on the pooled daily series (§10.3) is the statistical test.

### 10.2 Reproducibility

- fixed seed list: 17, 29, 43, 71, 101;
- ensemble the fixed seeds by arithmetic mean;
- record individual-seed results and dispersion;
- train on CPU with `torch.use_deterministic_algorithms(True)` and pinned thread counts — at this model size (a few thousand parameters) GPU/MPS training buys nothing and costs bitwise reproducibility;
- pin the exact PyTorch and scikit-learn wheels in the repository's `wheels/` vendoring directory so the training environment is reconstructable offline;
- store code revision, dataset manifest, feature schema hash, scaler hash, model configuration, checkpoint hash, and train/validation/test dates;
- deterministic CPU inference must reproduce persisted weights within tolerance.

### 10.3 Metrics

Predictive metrics:

- volatility MAE and QLIKE, always reported side by side with the HAR-RV (B2) values on the same fold;
- tail-event precision/recall, PR-AUC, calibration error, and Brier score, with the per-fold positive-event count printed next to every number (§7.2).

Economic metrics, always net of costs:

- CAGR;
- annualized volatility;
- Sharpe, Sortino, and Calmar ratios;
- maximum drawdown and expected shortfall;
- turnover and total fees;
- exposure distribution;
- performance by asset, fold, and market regime;
- paired incremental return versus the frozen base strategy, tested with the Ledoit-Wolf (2008) studentized stationary-bootstrap test on the daily net-return differential (overlay minus baseline); this is the primary significance statement for an overlay, because the overlay and base share almost all of their return path and a paired test on the differential has far more power than comparing two standalone Sharpe ratios.

Regime slices are fixed ex ante from the coupling history in §2:

1. pre-2020-03 (decoupled, pre-COVID);
2. 2020-03 to 2022-12 (coupled: QE rally and 2022 tightening bear);
3. 2023-01 to 2025-09 (ETF era);
4. 2025-10 onward (post-flash-crash decoupling, then the 2026 re-coupling).

Every economic metric is reported per slice. Boundaries are calendar-fixed now, not fitted to results later.

Accuracy is a diagnostic, not a promotion gate.

## 11. Research Promotion Gates

### 11.1 Track 1 minimum gate

All conditions must pass:

- portfolio-level OOS net Sharpe exceeds the frozen Donchian baseline (B0) **and** the best deterministic overlay (B1-B3);
- the Ledoit-Wolf stationary-bootstrap test on the daily differential versus B2 rejects "no improvement" at the one-sided 5% level;
- DSR is at least 0.95 after counting all model, feature, and cutoff trials (B1-B3 and E1-E4 at minimum);
- PBO and fold-level degradation do not indicate unstable selection;
- maximum drawdown improves by at least 15% relative, from the frozen baseline, without reducing CAGR by more than 20% relative;
- incremental economic utility is positive in at least 70% of complete OOS folds (per §10.1 this means roughly 5 of 7 folds — a coarse screen, not a test);
- OOS Sharpe remains positive under the 2x transaction-cost stress case;
- the result is not driven solely by one coin, one fold, or one regime slice: it must hold with the single best fold removed, and must not be positive only inside the coupled-regime slices of §10.3;
- E3 beats E1; otherwise cross-asset information has not demonstrated incremental value;
- E4 beats E3 to justify keeping gold features.

Scope of the verdict: passing this gate promotes the overlay to an exploratory paper book; it does **not** certify the composite strategy. The base Donchian portfolio itself failed statistical certification (DSR < 0.95), and no overlay on an uncertified base produces a certified composite. The certification object for any future live decision is the composite (Donchian × overlay) as a whole, re-run through the full Phase 1 gate with the complete trial ledger.

### 11.2 Rejection rules

Reject or redesign the track if:

- any E-variant fails to beat B2 (HAR-RV) on both QLIKE and net portfolio metrics — a neural forecaster that cannot beat a three-coefficient OLS model has no claim on production complexity;
- predictive metrics improve but net portfolio metrics do not;
- performance depends on a single threshold, seed, asset, or fold;
- gold improves in-sample performance but not the E4 versus E3 OOS comparison;
- the model increases turnover enough to lose its gross advantage after costs;
- data alignment cannot be proven from persisted `available_at` timestamps;
- DSR fails after the complete trial count is applied.

## 12. Paper-Trading Rollout

Only a historically gated candidate may become a new paper book.

### 12.1 Book and artifact identity

Proposed book name:

```text
donchian_tcn_risk_overlay
```

Freeze with the book:

- model checkpoint;
- model configuration;
- training dataset manifest;
- feature schema and scalers;
- seed ensemble definition;
- risk-multiplier thresholds;
- frozen historical OOS returns and expected band;
- research trial ledger and final gate report.

### 12.2 Daily inference

```text
update crypto data
  -> update cross-asset data
  -> cross-asset QC and availability check
  -> build D-1 feature snapshot
  -> run deterministic model ensemble
  -> produce risk multiplier
  -> apply multiplier to Donchian target
  -> apply existing event/news/ETF risk gates
  -> paper rebalance
  -> persist inference and notify
```

Persist each run's input timestamps, feature hash, checkpoint hash, raw predictions, calibrated risk state, base weights, final weights, and fallback status.

### 12.3 Fail-closed behavior

- Expected weekend/holiday staleness with a valid market calendar is allowed and explicitly masked.
- Unexpected source staleness or feature-build failure blocks exposure increases.
- Reductions remain allowed.
- Missing cross-asset data is never converted into a zero-weight liquidation signal.
- Existing Donchian and TSMOM paper books remain independent and unaffected.

### 12.4 Observation gate

- Retain the existing six-week check for operational tracking error and incidents.
- Do not interpret six weeks as alpha validation.
- Require at least six months of shadow observation before any live pilot is considered, with the understanding that even six months is a drift and execution check rather than strong statistical proof.

## 13. Monitoring

Add daily and weekly monitoring for:

- cross-asset source freshness and unexpected market-calendar gaps;
- feature missingness and stale-state age;
- feature distribution drift relative to the training set;
- predicted-volatility and tail-probability drift;
- risk-multiplier distribution and time spent at 0, 0.5, and 1;
- model-versus-replay output equality;
- paper-versus-model tracking error;
- turnover, fees, and exposure changes versus Donchian;
- checkpoint, scaler, and feature-schema mismatches;
- fallback frequency and reason codes.

Any artifact-identity mismatch is a P1 incident and blocks new exposure.

## 14. Proposed Repository Changes

The implementation should be split into auditable modules:

```text
data/collectors/cross_asset_daily.py
features/cross_asset.py
research/dl/__init__.py
research/dl/dataset.py
research/dl/models.py
research/dl/train.py
research/dl/walkforward.py
research/dl/evaluation.py
strategies/donchian_tcn_risk_overlay.py
scripts/backfill_cross_asset.py
scripts/update_cross_asset.py
scripts/run_dl_research.py
tests/test_cross_asset_alignment.py
tests/test_cross_asset_qc.py
tests/test_dl_no_lookahead.py
tests/test_dl_reproducibility.py
tests/test_dl_artifact_identity.py
tests/test_overlay_weights.py
```

Configuration additions should cover:

- provider and symbol mapping;
- external-market freshness thresholds;
- exchange calendar and timezone;
- fixed feature schema version;
- model checkpoint path and hash;
- fixed risk thresholds;
- paper-book capital and start date.

Add and pin the minimum research dependencies needed for the implementation:

- PyTorch with CPU inference support;
- scikit-learn for fold-local preprocessing, calibration, and predictive metrics;
- statsmodels (or equivalent OLS) for the HAR-RV baseline;
- a maintained U.S. exchange-calendar library for holidays and daylight-saving transitions (e.g., `exchange_calendars` or `pandas_market_calendars`);
- vendor the pinned wheels in `wheels/` per the repository's existing offline-install convention.

Persist immutable model and report artifacts under:

```text
data/store/models/donchian_tcn_risk_overlay/<model_id>/
research/reports/dl_cross_asset_<run_date>.md
research/reports/dl_cross_asset_trials_<run_date>.csv
```

Provider credentials belong in `.env`; they must never be written to configuration, artifacts, reports, logs, or inference records.

Do not register the paper strategy or add paper capital until the research gate has passed.

## 15. Test Requirements

### 15.1 Timing tests

- reject a U.S. close whose `available_at` is after the crypto decision timestamp;
- correctly align standard-time and daylight-saving-time sessions;
- carry Friday state through the weekend without generating synthetic returns;
- distinguish a valid holiday from an unexpected missing session;
- enforce the five-day embargo around every fold boundary.

### 15.2 Model and feature tests

- identical raw inputs produce identical features and schema hash;
- train-fold scaling never reads validation or test data;
- causal TCN outputs are unchanged when future rows are modified;
- fixed checkpoints reproduce persisted predictions;
- unavailable optional features use masks, not silent zeros;
- overlay weights never exceed base Donchian weights and remain within `[0, 1]`.

### 15.3 Failure tests

- external source timeout blocks adds but not reductions;
- stale unexpected data produces a named incident;
- missing checkpoint, scaler, or schema mismatch fails closed;
- interrupted writes do not corrupt the authoritative artifact or inference record;
- existing books remain runnable when the neural book is disabled.

## 16. Delivery Sequence and Exit Criteria

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | Corrected baseline and trial registry | One authoritative DSR record; experiment family frozen |
| 1 | Cross-asset backfill, updater, storage, QC | Complete aligned history and passing timing/QC tests |
| 2 | Deterministic features | Research/inference parity and no-lookahead tests pass |
| 3 | B1-B3 deterministic overlays + E1-E4 TCN research suite | Full walk-forward report with costs, DSR, PBO, paired Ledoit-Wolf tests, regime slices, and ablations |
| 4 | Track 1 decision | Promote or reject strictly by the predefined gate |
| 5 | Independent paper book | Six-week operational gate plus ongoing shadow observation |
| 6 | Track 2 research | Separate frozen protocol and report |
| 7 | Track 3 data foundation and research | Point-in-time universe and independent report |

## 17. Rollback

The neural track must remain removable without changing the existing strategies:

- disable `donchian_tcn_risk_overlay` in configuration;
- preserve its ledger and artifacts for audit;
- leave `donchian_ensemble` and `tsmom_ensemble` books untouched;
- never overwrite an existing book's trades, baseline, signals, or state;
- if daily inference fails, apply fail-closed exposure rules rather than loading an unverified checkpoint or silently falling back to a different model.

## 18. References

- IMF, [Cryptic Connections: Spillovers between Crypto and Equity Markets](https://www.imf.org/en/publications/global-financial-stability-notes/issues/2022/01/10/cryptic-connections-511776)
- IMF, [The Crypto Cycle and US Monetary Policy](https://www.elibrary.imf.org/view/journals/001/2023/163/article-A001-en.xml)
- Federal Reserve Bank of New York, [The Bitcoin–Macro Disconnect](https://www.newyorkfed.org/research/staff_reports/sr1052.html)
- Baur and Hoang, [The Bitcoin Gold Correlation Puzzle](https://www.sciencedirect.com/science/article/pii/S2214635021001052)
- Lim, Zohren, and Roberts, [Enhancing Time-Series Momentum Strategies Using Deep Neural Networks](https://arxiv.org/abs/1904.04912)
- Bai, Kolter, and Koltun, [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
- Jaquart, Köpke, and Weinhardt, [Machine Learning for Cryptocurrency Market Prediction and Trading](https://www.sciencedirect.com/science/article/pii/S2405918822000174)
- Bailey and López de Prado, [The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- Bailey et al., [The Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- Corsi, [A Simple Approximate Long-Memory Model of Realized Volatility (HAR-RV)](https://academic.oup.com/jfec/article/7/2/174/856522)
- Ledoit and Wolf, [Robust Performance Hypothesis Testing with the Sharpe Ratio](http://www.ledoit.net/jef_2008pdf.pdf)
- Wood, Roberts, and Zohren, [Slow Momentum with Fast Reversion: A Trading Strategy Using Deep Learning and Changepoint Detection](https://arxiv.org/abs/2105.13727)
- Wood, Giegerich, Roberts, and Zohren, [Trading with the Momentum Transformer](https://arxiv.org/abs/2112.08534)
- Cboe, [VIX Index Historical Data (official daily CSV)](https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv)
- Alpaca, [Historical Stock Bars API](https://docs.alpaca.markets/us/v1.4.2/reference/stockbars) and [Market Data FAQ (IEX vs SIP feeds)](https://docs.alpaca.markets/us/docs/market-data-faq)

## 19. v2 Revision Changelog (2026-07-14)

Changes made after critical review, before any implementation or training run (so none of them constitute post-hoc selection):

1. Registered a deterministic baseline family B1-B3 — σ20 vol targeting, **HAR-RV**, and VIX thresholding — sharing the exact §7.5 multiplier mapping with the neural variants; HAR-RV is the canonical hard-to-beat volatility forecaster and its absence was the plan's largest evaluation gap. Trial count for DSR grows from 4 to 7.
2. Replaced the discrete three-level risk multiplier with a continuous vol-target mapping plus a discrete calibrated tail gate, eliminating threshold cliff effects and making neural and deterministic overlays structurally comparable.
3. Replaced raw QQQ inputs with a QQQ/SPY relative-strength spread (raw QQQ and SPY are ~0.95 correlated; feeding both wasted the E2 ablation step).
4. Made the Ledoit-Wolf (2008) studentized stationary-bootstrap test on the paired daily return differential the primary significance test — a paired test on the differential is the statistically correct object for an overlay, where standalone Sharpe comparison is badly underpowered.
5. Added ex-ante regime slices reflecting the documented instability of crypto-equity coupling (decoupled pre-2020, coupled 2020-2022, post-2025-10 flash-crash decoupling, 2026 re-coupling to ~0.96), and required the gate to survive removal of the single best fold and to not hold only in coupled regimes.
6. Flagged the Alpaca free tier as IEX-only (~2% of consolidated volume — unrepresentative closes and unusable volume); required SIP-feed or official-close EOD semantics, pinned the official Cboe VIX CSV endpoint, and tightened `available_at` rules for the ~16:15 ET VIX dissemination.
7. Quantified label supply (tail events per fold) with a fallback to vol-head-only gating on label-starved folds, and required validation-fold probability calibration before fixed thresholds are applied.
8. Clarified the certification object: passing the Track 1 gate promotes an exploratory paper book only; it cannot certify a composite built on a base strategy that itself failed DSR certification.
9. Stated expected fold arithmetic (six to eight OOS folds) so per-fold gates are read as coarse screens, not tests; pinned CPU-deterministic training and `wheels/` vendoring; designated Binance Vision delisted-pair archives as the Track 3 point-in-time universe source; reserved already-collected ETF-flow data as a preregistered second-wave feature family.
