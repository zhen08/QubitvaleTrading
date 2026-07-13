# Paper Review (multi-book) — 2026-07-13

Operational incidents (since 2026-07-12): P0=0 P1=0 P2=0 P3=0 (details in data/store/ops/incidents.parquet); P0 gate: ✅ 0

> Both books are **uncertified research candidates** (Phase 1 revised verdict), so this is exploratory validation; Phase 3 selection discipline (ex-ante): if both books pass → deploy each at half size, do not pick a winner.

## Book: donchian_ensemble (started 2026-07-12, $10,000)

Window: 1 settled day (≈0.1/6 weeks); band baseline: frozen baseline (2026-07-13, 1431 days) bootstrap

| Metric | Paper | Model replay | Note |
|---|---|---|---|
| Cumulative return | -0.05% | 0.08% | diff -13 bps |
| Annualized Sharpe | 0.00 | 0.00 | |
| TE | insufficient sample | — | |
| Expected band 80%/95% | -1.33%~1.47% | ±(-3.11%~3.80%) | in band ✅ |
| Fills/fees | live 1 / catchup 0 | $0.83 | |
| Positions | {'ETHUSDT': 0.4588258829} | cash $9,165.83 | |

- 2026-07-12 buy ETHUSDT 0.458826 @ 1816.23 [live]

## Book: tsmom_ensemble (started 2026-07-13, $10,000)

No settled equity records yet, so no statistics available.
