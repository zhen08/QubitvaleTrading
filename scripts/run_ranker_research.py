"""Run the preregistered Track 3 ranker suite (track3_ranker_preregistration.md).

Usage: python -m scripts.run_ranker_research [--quick]

--quick: 2 seeds instead of 5 — pipeline smoke test only, watermarked NOT VALID.
Produces research/reports/track3_ranker_<date>.md + trials CSV, and persists
per-fold OOS scores to data/store/dl_research/ranker_scores_<date>.parquet.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from data import storeio
from data.collectors.common import REPO_ROOT, load_settings, setup_logging, utc_today
from research.dl.ranker.data import load_ranker_data
from research.dl.ranker.evaluation import evaluate
from research.dl.ranker.model import SEEDS
from research.dl.ranker.walkforward import run_walkforward


def write_report(date, ev, ledger, klay_present: bool, quick: bool) -> str:
    g = ev["GRU_selected"]
    lines = [f"# Track 3 Ranker Research Report — {date}", ""]
    if quick:
        lines += ["> **QUICK MODE — reduced seeds; NOT a valid research result.**", ""]
    lines += [
        f"Protocol: `track3_ranker_preregistration.md` (frozen 2026-07-16). "
        f"OOS {ev['oos_start']} → {ev['oos_end']} ({ev['n_oos_days']} days).",
        "",
        f"**Data disclosure**: KLAYUSDT {'present' if klay_present else 'ABSENT'} "
        "from the panel"
        + ("" if klay_present else
           " — §1 requires this gap be quantified before certification-grade use") + ".",
        "",
        "## Economic summary (net, 2% threshold, per-asset impact costs)",
        "",
        "| variant | Sharpe | CAGR% | MaxDD% | Sortino | ES95% | ann.turnover | 2x-cost SR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, e in [("GRU (val-selected width)", g), ("R0 equal-weight", ev["R0"]),
                    ("R1 21d momentum", ev["R1"])]:
        s = e["summary"]
        lines.append(f"| {name} | {s['sharpe']} | {s['cagr_pct']} | {s['max_dd_pct']} | "
                     f"{s['sortino']} | {s['es95_pct']} | {e.get('ann_turnover', '—')} | "
                     f"{e.get('stress2x_sharpe', '—')} |")
    lines += [
        "", f"Per-width OOS Sharpe: {ev['per_width_oos_sharpe']}",
        f"PBO across widths + R1: {ev['pbo_widths_plus_R1']}",
        f"DSR family(N=4): {g['dsr_family_n4']} · program(N=12): {g['dsr_program_n12']}",
        f"LW vs R1: {g['lw_vs_R1']} · vs R0: {g['lw_vs_R0']}",
        f"Folds positive vs R1: {g['folds_positive_vs_R1']}/{g['folds_total']}; "
        f"Sharpe diff w/o best fold: {g['sharpe_diff_wo_best_fold']}",
        f"Max single-asset share of gross profit: {g['concentration_max_share']}",
        f"Gross edge vs R1 (daily): {g['gross_edge_daily']}; incremental cost: "
        f"{g['incremental_cost_daily']}",
        "", "## Regime slices", "",
        f"- GRU: { {k: v['sharpe'] for k, v in g['regimes'].items() if v['sharpe'] is not None} }",
        f"- R1:  { {k: v['sharpe'] for k, v in ev['R1']['regimes'].items() if v['sharpe'] is not None} }",
        "", "## §6 gate", "", "```json",
        json.dumps(ev["gate"], indent=2, default=str), "```", "",
        "## Fold ledger", "",
        "See companion CSV. Width selection used validation net Sharpe only "
        "(amendment 1); every width's OOS series enters PBO.",
    ]
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    settings = load_settings()
    store = storeio.store_dir(settings)
    date = utc_today().date()

    rd = load_ranker_data(store)
    klay_present = "KLAYUSDT" in rd.ret.columns
    seeds = SEEDS[:2] if args.quick else SEEDS
    scores, ledger = run_walkforward(rd, seeds=seeds)
    if not len(scores["selected"]):
        raise SystemExit("no OOS scores produced")

    out_store = store / "dl_research"
    out_store.mkdir(parents=True, exist_ok=True)
    scores["selected"].reset_index().to_parquet(
        out_store / f"ranker_scores_{date}.parquet", index=False)

    ev = evaluate(rd, scores, ledger)
    out_dir = REPO_ROOT / "research" / "reports"
    pd.DataFrame(ledger).to_csv(out_dir / f"track3_ranker_trials_{date}.csv", index=False)
    report = write_report(date, ev, ledger, klay_present, args.quick)
    path = out_dir / f"track3_ranker_{date}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
