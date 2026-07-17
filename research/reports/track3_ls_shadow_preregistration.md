# Track 3-LS Preregistration — Long/Short Ranker Prospective Shadow Test

**Registered:** 2026-07-17, before any L/S code ran and before any forward data
was observed. **Parent:** `track3_ranker_preregistration.md` (frozen
2026-07-16; long-only product REJECTED at the §6 gate on 2026-07-16 with
significant relative skill vs R1 — see `track3_ranker_2026-07-16.md`).

## Honesty statement (why this is a shadow test, not a backtest family)

The 2023–2026 window already showed us that the GRU's long-minus-baseline
differential was positive; any L/S backtest on that window is contaminated
selection. Therefore this registration commits to **prospective evaluation
only**: the model is frozen now, intended L/S portfolios are logged daily, and
the single preregistered comparison runs on forward data. No historical L/S
backtest result may be used for promotion, sizing, or tuning.

## Mechanical transformation (zero new degrees of freedom)

Everything inherits from the parent registration and its amendments; the only
changes are the mechanical L/S mapping below.

- **Model/training/freeze**: identical GRU family, widths {5,10,20}, seeds
  17/29/43/71/101, frozen training hyperparameters; freeze = expanding train
  ending 182 days ago, last 182 days as validation (early stop + width
  selection), 5-day embargo. Width selection: validation net Sharpe of the
  **L/S portfolio rule below** (the mechanical analog of parent amendment 1);
  ties → smaller width. Refreeze every 182 days, appended to the ledger.
- **Universe**: parent §1 membership (top-50 ADV30, $5M floor, 110-bar
  history, ≥30 names/date). **Short leg additionally requires an active
  Binance USDT-M perp** (enumerated from the Vision `futures/um` archive at
  each freeze and stored in the manifest).
- **Portfolio rule (L/S mechanical transform)**: rank members by ensemble `p`
  daily. Long leg: top-5, **+10%** each, enter only at rank ≤ 5 with
  `p ≥ 0.5`, exit at rank > 10 or `p < 0.5`. Short leg: bottom-5, **−10%**
  each, enter only at rank ≥ N−4 with `p ≤ 0.5` and perp-shortable, exit at
  rank < N−9 or `p > 0.5`. Unfilled slots stay in cash. Gross ≤ 1.0, target
  net exposure 0. 2% rebalance threshold per name.
- **Costs at evaluation**: UM taker 6 bps/side + √-impact slippage from each
  asset's dollar volume; **shorts pay/receive funding** from the recorded
  funding history; 2x-cost stress case included.
- **References logged alongside**: R1-LS (trailing 21-day return replacing
  `p`, same rule, threshold = sign of momentum) and R2 (cash).

## Clarification (2026-07-17, before any forward data)

The two legs are filled **independently** by the rule above; when the model is
directionally one-sided (e.g., no member has `p ≥ 0.5`), the book is one-sided
and net exposure is transiently non-zero. "Target net exposure 0" describes
the balanced design point, not a rebalancing constraint. Thresholds are
inclusive as written (`≥ 0.5` / `≤ 0.5`); exact-0.5 ties are measure-zero in
practice and resolve to both legs by rank.

## Primary evaluation (one preregistered comparison = one trial)

After **≥ 365 days** of logged forward decisions: Ledoit-Wolf one-sided
stationary-bootstrap test at 5% of (a) L/S net return > 0 and (b) L/S net
minus R1-LS net > 0, net of the cost model above, plus the parent §6
structural checks (concentration, turnover rule, 2x-cost stress, regime
slices). Early peeking is diagnostics only, never a stopping or sizing rule.
Passing yields an exploratory paper-book **proposal** under §12 discipline —
not certification.

## Known gaps carried forward

- KLAYUSDT absent from the panel (network-specific stall); irrelevant for
  forward logging (KLAY delisted) but must be noted in the final report.
- Shadow decisions consume Vision-settled bars (T+1), so each decision row is
  logged the morning after; features remain strictly point-in-time.

## Freeze history

- **2026-07-17** — froze `ls_shadow_2026-07-14` (width 10 by val L/S Sharpe {'5': 2.275, '10': 3.045, '20': 2.694}, ckpt `df91f51ab1748289`, train≤2026-01-13, val≤2026-07-14, 937 shortable perps, retrain due 2027-01-12).
