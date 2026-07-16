# DL Research Trial Ledger

Running record of every trial that must count toward any future DSR correction
in the cross-asset DL program (plan §4.2/§11). Machine-generated per-fold rows
live in `dl_cross_asset_trials_<date>.csv`; this file records families,
protocol changes, and prospective registrations.

## Completed families

- **2026-07-14** — Track 1 retrospective family: B1, B2, B3, E1, E2, E3, E4
  (7 trials; 49 fold×variant rows). Verdict: REJECT at the §11.1 gate
  (`dl_cross_asset_2026-07-14.md`). The 2019–2026 sample is SPENT for this
  family: no variant from it may be re-tuned or re-selected on that window.

## Data notes

- **2026-07-15** — Track 3 universe backfilled: 600 eligible symbols attempted,
  599 stored (587 enter the panel with a full 30-day ADV window; 156 of them
  delisted/stale — survivorship control intact). Known gap: **KLAYUSDT**
  consistently stalls from this network (timed out twice under process
  isolation) and is absent from the panel; KLAY was a top-50 asset in
  2021-2023, so any Track 3 backtest over that period must disclose this hole
  or fill it from another network before certification-grade work.

## Prospective registrations

**Track 3 cross-sectional ranker** (registered 2026-07-16, before any model
code or panel contact): full protocol in `track3_ranker_preregistration.md`.
Family = 3 GRU widths {5,10,20} + deterministic momentum baseline R1 = 4
trials, added to the cumulative program count. Universe: top-50 by ADV30 with
a $5M floor, ≥30 names per usable date, point-in-time from the Vision panel
(KLAYUSDT gap must be resolved or quantified before the first run). Gate per
§6 of the preregistration; width selection on validation folds only.

**E2 prospective shadow test** (registered 2026-07-15, before any forward data
was observed; freeze history at the end of this file — the freeze script
appends there):

- Hypothesis: the E2 configuration (crypto + SPY + QQQ/SPY spread, no VIX/GLD)
  improves the donchian book's net risk-adjusted return versus B0 and B2.
  Motivated by its pattern in the rejected retrospective family; that pattern
  is explicitly NOT evidence, only motivation.
- Procedure: freeze the full §10.1-protocol E2 model via
  `scripts.freeze_dl_shadow` (expanding train, last 182d validation, 5-day
  embargo, 5-seed ensemble, fixed thresholds); log daily shadow multipliers for
  E2 plus B1/B2 references and the base donchian weight
  (`data/store/dl_shadow/multipliers.parquet`); refreeze every 182 days per
  protocol, each refreeze appended below. No paper book, no effect on any book.
- Primary evaluation (one preregistered comparison = **one trial**): after
  **≥ 365 days** of logged forward data, Ledoit-Wolf one-sided stationary-
  bootstrap test of the paired daily net differential (E2 overlay minus B0)
  at 5%, with the B2 reference reported alongside; net of the frozen cost
  model with the 2% rebalance threshold. Early peeking at the log is
  diagnostics only and never a stopping/selection rule.
- Timing note: the shadow logger consumes Vision-settled daily bars (published
  T+1), so the row for decision D is typically written the morning after the
  live books traded D (which used the Bitget tail bar). Features remain
  strictly point-in-time (bars ≤ D); the lag affects only when the row is
  logged, not what it could see — a live implementation would be able to act
  on the same information via the tail bar.
- Freeze history:

- **2026-07-15** — froze `e2_shadow_2026-07-14` (variant E2, ckpt `6e72942328f41a39`, train≤2026-01-13, val≤2026-07-14, val tail⁺=77, retrain due 2027-01-12). One prospective trial; evaluation only on data logged after this date.
