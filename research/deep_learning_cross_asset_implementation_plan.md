# Cross-Asset Deep Learning for Daily Crypto Trading — Implementation Plan

**Status:** Proposed; research only
**Date:** 2026-07-14
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

The current repository has three crypto assets, a strict D-1 decision convention, a cost-aware daily backtester, two independent paper books, and only 1,414 bars in the common portfolio-level out-of-sample window. The current Donchian portfolio is an uncertified research candidate with approximately 0.74 OOS Sharpe and -28.4% maximum drawdown. The TSMOM portfolio is lower risk but also uncertified.

This creates three constraints:

- A large Transformer or a separate neural network per asset is not justified by the sample size.
- New information should first be used to reduce risk in a controlled overlay, not to replace the trading stack end to end.
- Every architecture, feature group, threshold, seed-selection rule, and strategy family expands the research trial count and must be included in DSR/PBO reporting.

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

The repository currently contains inconsistent Donchian portfolio DSR(N=32) values: the README reports 0.395, while the detailed Phase 1 report and strategy header report approximately 0.75/0.751. Identify the authoritative calculation, regenerate affected documents from one source, and freeze the corrected baseline.

This does not change the current FAIL verdict, but it is required before measuring neural-model incremental value.

### 4.2 Register the experiment families ex ante

The initial feature-ablation family is fixed as:

| ID | Model input | Purpose |
|---|---|---|
| E0 | Existing Donchian ensemble | Frozen non-neural baseline |
| E1 | Crypto-only TCN overlay | Tests the model architecture without cross-asset information |
| E2 | E1 + QQQ + SPY | Tests equity risk-state information |
| E3 | E2 + VIX | Tests explicit market-stress information |
| E4 | E3 + GLD | Tests the incremental value of gold |

E1-E4 are four model trials for family-level multiple-testing purposes. Any additional feature bundle, sequence length, label, threshold set, or architecture explored later must be added to the research trial ledger.

Use one fixed union feature schema and identical network shape for E1-E4. Features removed by an ablation are marked unavailable by construction and paired with a zero availability mask; they are not replaced with silently meaningful zero observations.

### 4.3 Freeze the primary decision rule

- Primary comparison: E3 versus E1 and E0.
- Gold is accepted only if E4 improves on E3 out of sample.
- No model is selected on classification accuracy alone.
- The first model family is a risk overlay on `donchian_ensemble`; TSMOM overlays are out of scope for the initial family.

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

- Use an authenticated market-data API with timestamped daily bars for SPY, QQQ, and GLD. Alpaca's historical stock-bars API is a suitable initial adapter.
- Use official Cboe historical VIX data for the VIX adapter.
- Keep a second provider for recent-window cross-source QC.
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

The daily job runs shortly after 00:00 UTC. The most recent regular U.S. close is normally available roughly three to four hours earlier. Build each model row using an as-of join:

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

- 20-day and 60-day BTC/QQQ rolling correlation;
- 20-day and 60-day BTC/SPY rolling correlation;
- 60-day BTC-to-QQQ beta;
- equity stress interaction: negative SPY return multiplied by VIX percentile;
- a cross-asset-data-valid mask.

Rolling correlations and betas must use only information available at the decision timestamp.

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

The initial implementation uses training-fold percentile calibration:

- `1.0`: predicted volatility is at or below the training 80th percentile and tail probability is below 0.25;
- `0.5`: predicted volatility is above the training 80th percentile or tail probability is at least 0.25;
- `0.0`: predicted volatility is above the training 95th percentile or tail probability is at least 0.50.

These thresholds are fixed for E1-E4. Changing any threshold creates a new trial. Apply the existing 2% rebalance threshold after overlay weights are produced.

### 7.6 Baseline comparisons

Compare against:

- Donchian without an overlay;
- deterministic 20-day volatility targeting;
- deterministic VIX thresholding;
- E1-E4 neural variants.

A neural overlay must beat simple deterministic risk controls, not merely the unscaled Donchian strategy.

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

### 8.3 Constraint

With only three assets, this track is an engineering feasibility study, not a credible statistical certification program. A broader liquid-asset training panel is strongly preferred before promotion.

## 9. Track 3 — Cross-Sectional Crypto Ranker

Start only after a point-in-time, survivorship-bias-controlled universe exists.

### 9.1 Data prerequisite

- at least 30 liquid spot assets on each training date;
- universe determined from information known at D-1;
- selection based on trailing 30-day executable dollar volume and market-cap constraints;
- delisted and failed assets retained in historical data;
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

### 10.2 Reproducibility

- fixed seed list: 17, 29, 43, 71, 101;
- ensemble the fixed seeds by arithmetic mean;
- record individual-seed results and dispersion;
- store code revision, dataset manifest, feature schema hash, scaler hash, model configuration, checkpoint hash, and train/validation/test dates;
- deterministic CPU inference must reproduce persisted weights within tolerance.

### 10.3 Metrics

Predictive metrics:

- volatility MAE and QLIKE;
- tail-event precision/recall, PR-AUC, calibration error, and Brier score.

Economic metrics, always net of costs:

- CAGR;
- annualized volatility;
- Sharpe, Sortino, and Calmar ratios;
- maximum drawdown and expected shortfall;
- turnover and total fees;
- exposure distribution;
- performance by asset, fold, and market regime;
- paired incremental return versus the frozen base strategy.

Accuracy is a diagnostic, not a promotion gate.

## 11. Research Promotion Gates

### 11.1 Track 1 minimum gate

All conditions must pass:

- portfolio-level OOS net Sharpe exceeds the frozen Donchian baseline;
- DSR is at least 0.95 after counting all model, feature, and cutoff trials;
- PBO and fold-level degradation do not indicate unstable selection;
- maximum drawdown improves by at least 15% relative, from the frozen baseline, without reducing CAGR by more than 20% relative;
- incremental economic utility is positive in at least 70% of complete OOS folds;
- OOS Sharpe remains positive under the 2x transaction-cost stress case;
- the result is not driven solely by one coin or one crisis regime;
- E3 beats E1; otherwise cross-asset information has not demonstrated incremental value;
- E4 beats E3 to justify keeping gold features.

### 11.2 Rejection rules

Reject or redesign the track if:

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
- scikit-learn for fold-local preprocessing and predictive metrics;
- a maintained U.S. exchange-calendar library for holidays and daylight-saving transitions.

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
| 3 | E1-E4 TCN research suite | Full walk-forward report with costs, DSR, PBO, and ablations |
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
- Alpaca, [Historical Stock Bars API](https://docs.alpaca.markets/us/v1.4.2/reference/stockbars)
