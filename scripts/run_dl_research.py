"""Run the cross-asset DL research suite: B1-B3 + E1-E4 (plan §10/§11).

Usage: python -m scripts.run_dl_research [--variants E1,E3] [--quick]

Produces:
  research/reports/dl_cross_asset_<date>.md          gate report
  research/reports/dl_cross_asset_trials_<date>.csv  per-fold trial ledger
  data/store/dl_research/preds_<date>.parquet        per-row forecasts/multipliers

--quick trains 2 seeds instead of 5 for smoke-testing the pipeline; a quick run
is NOT a valid research result and the report is watermarked accordingly.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from data import storeio
from data.collectors.common import load_settings, setup_logging, utc_today
from features.cross_asset import build_feature_table, persist, schema_hash
from research.dl.evaluation import evaluate, gate_verdict, predictive_metrics
from research.dl.train import SEEDS
from research.dl.walkforward import run_walkforward


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_report(out_dir, date, ev, gates, pred_metrics, ledger, manifest,
                 quick: bool) -> str:
    lines = [f"# Cross-Asset DL Research Report — {date}", ""]
    if quick:
        lines += ["> **QUICK MODE — reduced seed count; NOT a valid research "
                  "result; pipeline smoke test only.**", ""]
    lines += [
        f"Feature schema `{manifest['schema_hash']}` · rows {manifest['rows']} · "
        f"OOS union {ev['oos_start']} → {ev['oos_end']} ({ev['n_oos_days']} days)",
        "",
        "## Economic summary (net of costs, 2% rebalance threshold for all variants)",
        "",
        "| variant | Sharpe | CAGR% | MaxDD% | Sortino | ES95% | DSR(N=7) | 2x-cost SR | "
        "mult<1 frac | LW p vs B0 (SR) | LW p vs B2 (SR) | Δmean p vs B0 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    b0 = ev["B0_summary"]
    lines.append(f"| B0 (no overlay) | {b0['sharpe']} | {b0['cagr_pct']} | "
                 f"{b0['max_dd_pct']} | {b0['sortino']} | {b0['es95_pct']} | — | "
                 f"{_fmt(ev['B0_stress2x_sharpe'])} | — | — | — | — |")
    for v, e in ev["variants"].items():
        s = e["summary"]
        lines.append(
            f"| {v} | {s['sharpe']} | {s['cagr_pct']} | {s['max_dd_pct']} | "
            f"{s['sortino']} | {s['es95_pct']} | {e['dsr_n7']} | "
            f"{_fmt(e['stress2x_sharpe'])} | {e['multiplier_frac_below_1']} | "
            f"{_fmt(e['vs_B0']['sharpe_p'])} | "
            f"{_fmt(e['vs_B2']['sharpe_p']) if e['vs_B2'] else '—'} | "
            f"{_fmt(e['vs_B0']['mean_p'])} |")
    lines += ["", f"PBO across the {len(ev['variants'])} overlay variants: "
              f"{_fmt(ev['pbo_across_variants'])}", ""]

    lines += ["## Per-fold incremental utility vs B0 (coarse screen)", ""]
    for v, e in ev["variants"].items():
        inc = e["incremental"]
        lines.append(f"- **{v}**: {inc['folds_positive']}/{inc['folds_total']} folds "
                     f"positive; Sharpe diff w/o best fold "
                     f"{inc['sharpe_diff_wo_best_fold']}; per-coin ΔSR "
                     f"{e['per_coin_sharpe_diff']}")
    lines += ["", "## Regime slices (ex-ante calendar boundaries)", ""]
    for v, e in ev["variants"].items():
        seg = {k: r["sharpe"] for k, r in e["regimes"].items() if r["sharpe"] is not None}
        lines.append(f"- **{v}**: {seg}")

    lines += ["", "## Predictive metrics (pooled by variant; HAR-RV = B2 reference)", ""]
    pm = pred_metrics.groupby("variant").agg(
        n=("n", "sum"), vol_mae_log=("vol_mae_log", "mean"),
        qlike=("qlike", "mean"), tail_positives=("tail_positives", "sum"),
        tail_pr_auc=("tail_pr_auc", "mean"), tail_brier=("tail_brier", "mean"))
    lines += ["| variant | n | vol MAE(log) | QLIKE | tail⁺ | PR-AUC | Brier |",
              "|---|---|---|---|---|---|---|"]
    for v, r in pm.iterrows():
        lines.append(f"| {v} | {int(r['n'])} | {r['vol_mae_log']:.4f} | "
                     f"{r['qlike']:.4f} | {int(r['tail_positives'])} | "
                     f"{'' if pd.isna(r['tail_pr_auc']) else f'{r.tail_pr_auc:.3f}'} | "
                     f"{'' if pd.isna(r['tail_brier']) else f'{r.tail_brier:.4f}'} |")

    lines += ["", "## §11.1 gate verdict", "", "```json",
              json.dumps(gates, indent=2, default=str), "```", "",
              "## Trial ledger", "",
              f"{len(ledger)} fold×variant trials recorded in the companion CSV. "
              "The DSR trial count for any future promotion decision must include "
              "all seven overlay variants plus any protocol change logged after "
              "this run.", ""]
    return "\n".join(lines)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=None,
                    help="comma-separated subset, e.g. B1,B2,E1")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    settings = load_settings()
    store = storeio.store_dir(settings)
    date = utc_today().date()

    table, manifest = build_feature_table(store, settings["symbols"])
    persist(store, table, manifest)
    assert manifest["schema_hash"] == schema_hash()

    variants = args.variants.split(",") if args.variants else None
    seeds = SEEDS[:2] if args.quick else SEEDS
    preds, ledger = run_walkforward(table, variants, seeds=seeds)
    if preds.empty:
        raise SystemExit("no predictions produced — check data coverage")

    out_store = store / "dl_research"
    out_store.mkdir(parents=True, exist_ok=True)
    storeio.write_parquet(preds, out_store / f"preds_{date}.parquet")

    pred_metrics = predictive_metrics(preds, table)
    ev = evaluate(store, table, preds, settings["symbols"])
    gates = gate_verdict(ev)

    from data.collectors.common import REPO_ROOT
    out_dir = REPO_ROOT / "research" / "reports"
    pd.DataFrame(ledger).to_csv(out_dir / f"dl_cross_asset_trials_{date}.csv", index=False)
    report = write_report(out_dir, date, ev, gates, pred_metrics, ledger,
                          manifest, args.quick)
    path = out_dir / f"dl_cross_asset_{date}.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
