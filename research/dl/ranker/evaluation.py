"""Ranker evaluation + §6 gate (baselines R0/R1/R2, LW tests, DSR, PBO,
regime slices, 2x-cost stress, turnover kill rule, concentration check)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from research import metrics
from research.dl.evaluation import (REGIMES, _es, _sortino, paired_tests,
                                    regime_slices)
from research.dl.ranker import data as D
from research.dl.ranker import portfolio as P
from research.dl.ranker.walkforward import _sim_gru

log = logging.getLogger("qvt.ranker.eval")

N_FAMILY_TRIALS = 4       # 3 widths + R1 (§5)
N_PROGRAM_TRIALS = 12     # + 7 Track 1 trials + 1 E2 prospective (ledger)


def _summary(net: pd.Series) -> dict:
    return {**metrics.summary(net, "1d"), "sortino": _sortino(net),
            "es95_pct": _es(net)}


def _sim_r1(rd: D.RankerData, dates: pd.DatetimeIndex,
            cost_rate: pd.DataFrame | None = None) -> P.SimResult:
    return P.simulate(dates, rd.mom21, rd.member, rd.ret_next,
                      cost_rate if cost_rate is not None else rd.cost_rate,
                      confidence=rd.mom21 > 0)                 # amendment 2


def _sim_r0(rd: D.RankerData, dates: pd.DatetimeIndex,
            cost_rate: pd.DataFrame | None = None) -> P.SimResult:
    flat = rd.member.astype(float)          # constant score: 1/N over all members
    return P.simulate(dates, flat, rd.member, rd.ret_next,
                      cost_rate if cost_rate is not None else rd.cost_rate,
                      equal_weight_all=True)


def _concentration(sim: P.SimResult, rd: D.RankerData) -> float:
    """Largest single-asset share of total positive gross profit (§6)."""
    contrib: dict[str, float] = {}
    for d, names in sim.holdings.items():
        for s in names:
            r = rd.ret_next.at[d, s] if s in rd.ret_next.columns else np.nan
            if np.isfinite(r):
                contrib[s] = contrib.get(s, 0.0) + P.SLOT_WEIGHT * r
    pos = {s: v for s, v in contrib.items() if v > 0}
    total = sum(pos.values())
    return max(pos.values()) / total if total > 0 else 0.0


def evaluate(rd: D.RankerData, scores: dict, ledger: list[dict]) -> dict:
    sel = scores["selected"]
    oos_dates = pd.DatetimeIndex(sorted(sel.index.unique()))
    stress = rd.cost_rate * 2

    sim_sel = _sim_gru(rd, sel, oos_dates)
    sim_sel_2x = _sim_gru(rd, sel, oos_dates, stress)
    sim_r1 = _sim_r1(rd, oos_dates)
    sim_r1_2x = _sim_r1(rd, oos_dates, stress)
    sim_r0 = _sim_r0(rd, oos_dates)
    r2 = pd.Series(0.0, index=oos_dates)

    width_nets = {f"GRU_w{w}": _sim_gru(rd, s, oos_dates).net
                  for w, s in scores["by_width"].items() if len(s)}
    trial_mat = pd.DataFrame({**width_nets, "R1": sim_r1.net})
    sr_var = metrics.trials_sr_variance(trial_mat)

    # turnover kill rule (§6): incremental cost vs gross edge over R1
    gross_edge = float((sim_sel.gross - sim_r1.gross).mean())
    net_edge = float((sim_sel.net - sim_r1.net).mean())
    incr_cost = gross_edge - net_edge

    # per-fold incremental vs R1 + best-fold removal. Fold boundaries are
    # contiguous 182-day test windows by construction, so consecutive blocks
    # over the OOS index reproduce them (last block absorbs a partial fold).
    by_fold = {}
    diff = (sim_sel.net - sim_r1.net).dropna()
    blocks = np.array_split(np.arange(len(diff)), max(1, len(diff) // 182))
    for i, b in enumerate(blocks):
        by_fold[i] = float(diff.iloc[b].mean())
    best = max(by_fold, key=by_fold.get) if by_fold else None
    if best is not None:
        keep = np.concatenate([b for i, b in enumerate(blocks) if i != best])
        sr_wo_best = (metrics.sharpe(sim_sel.net.iloc[keep])
                      - metrics.sharpe(sim_r1.net.iloc[keep]))
    else:
        sr_wo_best = np.nan

    ev = {
        "oos_start": str(oos_dates.min().date()), "oos_end": str(oos_dates.max().date()),
        "n_oos_days": len(oos_dates),
        "GRU_selected": {
            "summary": _summary(sim_sel.net),
            "ann_turnover": round(sim_sel.ann_turnover, 1),
            "stress2x_sharpe": metrics.sharpe(sim_sel_2x.net),
            "dsr_family_n4": round(metrics.deflated_sharpe(sim_sel.net,
                                                           N_FAMILY_TRIALS, sr_var), 3),
            "dsr_program_n12": round(metrics.deflated_sharpe(sim_sel.net,
                                                             N_PROGRAM_TRIALS, sr_var), 3),
            "lw_vs_R1": paired_tests(sim_sel.net, sim_r1.net),
            "lw_vs_R0": paired_tests(sim_sel.net, sim_r0.net),
            "regimes": regime_slices(sim_sel.net),
            "concentration_max_share": round(_concentration(sim_sel, rd), 3),
            "folds_positive_vs_R1": sum(1 for v in by_fold.values() if v > 0),
            "folds_total": len(by_fold),
            "sharpe_diff_wo_best_fold": round(float(sr_wo_best), 3),
            "gross_edge_daily": round(gross_edge, 7),
            "incremental_cost_daily": round(incr_cost, 7),
        },
        "R0": {"summary": _summary(sim_r0.net), "ann_turnover": round(sim_r0.ann_turnover, 1)},
        "R1": {"summary": _summary(sim_r1.net), "ann_turnover": round(sim_r1.ann_turnover, 1),
               "stress2x_sharpe": metrics.sharpe(sim_r1_2x.net),
               "regimes": regime_slices(sim_r1.net)},
        "R2_sharpe": 0.0,
        "per_width_oos_sharpe": {k: metrics.sharpe(v) for k, v in width_nets.items()},
        "pbo_widths_plus_R1": metrics.pbo_cscv(trial_mat),
    }
    g = ev["GRU_selected"]
    ev["gate"] = {k: bool(v) for k, v in {
        "beats_R0_sharpe": g["summary"]["sharpe"] > ev["R0"]["summary"]["sharpe"],
        "beats_R1_sharpe": g["summary"]["sharpe"] > ev["R1"]["summary"]["sharpe"],
        "lw_vs_R1_p<0.05": g["lw_vs_R1"]["sharpe_p"] < 0.05,
        "dsr_family>=0.95": g["dsr_family_n4"] >= 0.95,
        "pbo_not_unstable": (ev["pbo_widths_plus_R1"] is not None
                             and ev["pbo_widths_plus_R1"] == ev["pbo_widths_plus_R1"]
                             and ev["pbo_widths_plus_R1"] < 0.5),
        "folds_positive>=70pct": (g["folds_positive_vs_R1"]
                                  >= 0.7 * max(g["folds_total"], 1)),
        "survives_best_fold_removal": g["sharpe_diff_wo_best_fold"] > 0,
        "stress2x_sharpe>0": g["stress2x_sharpe"] > 0,
        "concentration<=40pct": g["concentration_max_share"] <= 0.40,
        "turnover_rule": not (gross_edge > 0 and incr_cost > 0.5 * gross_edge),
    }.items()}                                        # numpy bools -> JSON-safe
    ev["gate"]["ALL_PASS"] = all(ev["gate"].values())
    ev["nets"] = {"GRU": sim_sel.net, "R0": sim_r0.net, "R1": sim_r1.net, "R2": r2}
    return ev
