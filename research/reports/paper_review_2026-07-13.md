# Paper Review (multi-book) — 2026-07-13

Operational incidents (since 2026-07-12): P0=0 P1=0 P2=1 P3=0 (details in data/store/ops/incidents.parquet); P0 gate: ✅ 0

> Both books are **uncertified research candidates** (Phase 1 revised verdict), so this is exploratory validation; Phase 3 selection discipline (ex-ante): if both books pass → deploy each at half size, do not pick a winner.

## Book: donchian_ensemble (started 2026-07-12, $10,000)

Window: 1 settled day(s) (≈0.1/6 weeks); band baseline: frozen baseline (2026-07-13, 1432 days) bootstrap

| Metric | Paper | Model replay | Note |
|---|---|---|---|
| Cumulative return | 0.08% | 0.08% | diff 0 bps |
| Annualized Sharpe | 0.00 | 0.00 | |
| TE | insufficient sample | — | |
| Expected band 80%/95% | -1.33%~1.47% | ±(-3.12%~3.74%) | in band ✅ |
| Fills/fees | live 0 / catchup 1 | $0.83 | |
| Positions | {'ETHUSDT': 0.4660860565} | cash $9,165.83 | |

- 2026-07-12 buy ETHUSDT 0.466086 @ 1787.94 [catchup]

## Book: tsmom_ensemble (started 2026-07-13, $10,000)

No settled equity records yet, so no statistics available.

