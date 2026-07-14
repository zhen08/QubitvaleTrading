"""Evaluation for the overlay research suite (plan §10.3/§11).

Predictive: vol MAE + QLIKE (side by side with B2 on the same folds), tail
PR-AUC/Brier with per-fold positive counts.
Economic: overlay backtest through the repo cost model with the 2% rebalance
threshold applied identically to every variant (including the B0 no-overlay
re-run, so the threshold itself never favors a variant); paired Ledoit-Wolf
style stationary-bootstrap tests on the daily net differential; DSR(N=7)/PBO
across the seven overlay trials; ex-ante regime slices; 2x-cost stress.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data import storeio
from research import metrics
from research.costs import CostModel, SPOT_TAKER
from research.engine import run_backtest
from research.strategies import GRIDS, donchian

log = logging.getLogger("qvt.dl.eval")

REBALANCE_THRESHOLD = 0.02
STRESS_COST = CostModel(fee_bps_side=20.0, slip_floor_bps=2.0, name="spot_stress2x")
N_OVERLAY_TRIALS = 7          # B1-B3 + E1-E4 (§4.2)
TAIL_Z = -2.0

# Ex-ante regime slices (§10.3) — calendar-fixed, never fitted to results.
REGIMES = [
    ("pre2020_decoupled", None, "2020-03-01"),
    ("coupled_qe_qt", "2020-03-01", "2023-01-01"),
    ("etf_era", "2023-01-01", "2025-10-01"),
    ("post_crash_recoupling", "2025-10-01", None),
]


# ---------------- predictive metrics ----------------

def predictive_metrics(preds: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    lab = table[["decision_ts", "symbol", "label_logvol5", "label_minz5",
                 "c_sigma20d"]]
    m = preds.merge(lab, on=["decision_ts", "symbol"], how="left")
    scale = m["c_sigma20d"] / m["c_sigma20d"].clip(lower=m["sigma_floor"])
    m["y_tail"] = ((m["label_minz5"] * scale) <= TAIL_Z).astype(float)
    m.loc[m["label_minz5"].isna(), "y_tail"] = np.nan

    rows = []
    for (variant, fold), g in m.groupby(["variant", "fold"]):
        g = g.dropna(subset=["label_logvol5"])
        if not len(g):
            continue
        rv_true = np.exp(2 * g["label_logvol5"])          # realized variance
        s2_pred = np.maximum(g["sigma_hat"] ** 2, 1e-12)
        row = {
            "variant": variant, "fold": fold, "n": len(g),
            "vol_mae_log": float((np.log(g["sigma_hat"]) - g["label_logvol5"]).abs().mean()),
            "qlike": float((np.log(s2_pred) + rv_true / s2_pred).mean()),
            "tail_positives": int(g["y_tail"].sum()),
        }
        gt = g.dropna(subset=["p_tail_cal", "y_tail"])
        if len(gt) > 20 and gt["y_tail"].nunique() == 2:
            from sklearn.metrics import average_precision_score, brier_score_loss
            row["tail_pr_auc"] = float(average_precision_score(gt["y_tail"], gt["p_tail_cal"]))
            row["tail_brier"] = float(brier_score_loss(gt["y_tail"], gt["p_tail_cal"]))
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in ("tail_pr_auc", "tail_brier"):
        if col not in out.columns:
            out[col] = np.nan
    return out


# ---------------- economic evaluation ----------------

def base_positions(store, symbols: list[str]) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Per-symbol donchian parameter-ensemble weight in [0,1] (deployment form)."""
    out = {}
    for sym in symbols:
        df = pd.read_parquet(storeio.klines_path(store, "spot", sym, "1d"))
        df = df.set_index(pd.to_datetime(df["ts"], utc=True)).sort_index()
        variants = pd.concat({str(p): donchian(df, **p) for p in GRIDS["donchian"]}, axis=1)
        out[sym] = (df, variants.mean(axis=1))
    return out


def apply_threshold(target: pd.Series, threshold: float = REBALANCE_THRESHOLD) -> pd.Series:
    """Trade only when |target - held| > threshold (paper-engine semantics)."""
    t = target.fillna(0.0).to_numpy()
    out = np.empty_like(t)
    held = 0.0
    for i, x in enumerate(t):
        if abs(x - held) > threshold:
            held = x
        out[i] = held
    return pd.Series(out, index=target.index)


def overlay_net_returns(base: dict, preds: pd.DataFrame, variant: str | None,
                        oos_index: pd.DatetimeIndex,
                        cost_model: CostModel = SPOT_TAKER) -> pd.Series:
    """Equal-weight 3-coin portfolio net returns on the OOS union window.

    variant=None -> B0 (no overlay), same threshold treatment.
    Multiplier defaults to 1.0 outside test windows; returns are sliced to the
    OOS union afterwards so pre-OOS behavior is identical across variants.
    """
    legs = {}
    for sym, (df, w) in base.items():
        target = w.copy()
        if variant is not None:
            mv = preds[(preds["variant"] == variant) & (preds["symbol"] == sym)]
            mult = pd.Series(mv["multiplier"].to_numpy(),
                             index=pd.to_datetime(mv["decision_ts"], utc=True)
                             - pd.Timedelta(days=1))       # decision_ts -> bar_date
            mult = mult.reindex(target.index).fillna(1.0)
            target = (target * mult).clip(0.0, 1.0)
        pos = apply_threshold(target)
        res = run_backtest(df, pos, cost_model)
        legs[sym] = res.net
    port = pd.concat(legs, axis=1).mean(axis=1)
    return port.reindex(oos_index).dropna()


def oos_union(preds: pd.DataFrame) -> pd.DatetimeIndex:
    ts = pd.to_datetime(preds["decision_ts"].unique(), utc=True) - pd.Timedelta(days=1)
    return pd.DatetimeIndex(sorted(ts))


# ---------------- paired bootstrap tests ----------------

def _stationary_bootstrap_idx(n: int, rng: np.random.Generator,
                              mean_block: float = 20.0) -> np.ndarray:
    """Politis-Romano stationary bootstrap index sample of length n."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    i = rng.integers(n)
    for k in range(n):
        idx[k] = i
        if rng.random() < p:
            i = rng.integers(n)
        else:
            i = (i + 1) % n
    return idx


def paired_tests(a: pd.Series, b: pd.Series, n_boot: int = 2000,
                 seed: int = 7) -> dict:
    """One-sided tests of H1: a dominates b, on the aligned daily series.

    sharpe_p  — Ledoit-Wolf (2008)-style stationary-bootstrap percentile test
                on the per-period Sharpe difference.
    mean_p    — same machinery on the mean of the paired daily differential
                (the primary overlay statistic: far more power when a and b
                share most of their return path).
    """
    j = pd.concat({"a": a, "b": b}, axis=1).dropna()
    x, y = j["a"].to_numpy(), j["b"].to_numpy()
    n = len(j)
    if n < 100:
        return {"n": n, "sharpe_p": np.nan, "mean_p": np.nan}

    def sr(v):
        s = v.std(ddof=1)
        return v.mean() / s if s > 0 else 0.0

    rng = np.random.default_rng(seed)
    d_sr, d_mu = [], []
    for _ in range(n_boot):
        idx = _stationary_bootstrap_idx(n, rng)
        d_sr.append(sr(x[idx]) - sr(y[idx]))
        d_mu.append(float((x[idx] - y[idx]).mean()))
    return {
        "n": n,
        "sharpe_diff": round(sr(x) - sr(y), 5),
        "sharpe_p": float(np.mean(np.asarray(d_sr) <= 0.0)),
        "mean_diff_daily": round(float((x - y).mean()), 7),
        "mean_p": float(np.mean(np.asarray(d_mu) <= 0.0)),
    }


# ---------------- summary + gate ----------------

def econ_summary(net: pd.Series) -> dict:
    return {**metrics.summary(net, "1d"),
            "sortino": _sortino(net), "es95_pct": _es(net)}


def _sortino(net: pd.Series) -> float:
    d = net[net < 0]
    if len(d) < 3 or d.std(ddof=1) == 0:
        return 0.0
    return round(float(net.mean() / d.std(ddof=1) * np.sqrt(365)), 2)


def _es(net: pd.Series, q: float = 0.05) -> float:
    tail = net[net <= net.quantile(q)]
    return round(100 * float(tail.mean()), 2) if len(tail) else 0.0


def regime_slices(net: pd.Series) -> dict:
    out = {}
    for name, a, b in REGIMES:
        s = net
        if a:
            s = s[s.index >= pd.Timestamp(a, tz="UTC")]
        if b:
            s = s[s.index < pd.Timestamp(b, tz="UTC")]
        out[name] = {"n": len(s), "sharpe": metrics.sharpe(s) if len(s) > 60 else None}
    return out


def fold_incremental(net_v: pd.Series, net_b0: pd.Series,
                     preds_v: pd.DataFrame) -> dict:
    """Per-fold mean daily incremental utility vs B0 (coarse screen, §11.1)."""
    diff = (net_v - net_b0).dropna()
    by_fold = {}
    for fold, g in preds_v.groupby("fold"):
        ts = pd.to_datetime(g["decision_ts"], utc=True) - pd.Timedelta(days=1)
        d = diff.reindex(pd.DatetimeIndex(sorted(ts.unique()))).dropna()
        if len(d):
            by_fold[int(fold)] = float(d.mean())
    pos = sum(1 for v in by_fold.values() if v > 0)
    # best-fold-removal: recompute pooled Sharpe diff without the best fold
    if by_fold:
        best = max(by_fold, key=by_fold.get)
        g = preds_v[preds_v["fold"] == best]
        drop_ts = pd.DatetimeIndex(sorted(
            (pd.to_datetime(g["decision_ts"], utc=True) - pd.Timedelta(days=1)).unique()))
        keep_v = net_v.drop(index=drop_ts, errors="ignore")
        keep_b = net_b0.drop(index=drop_ts, errors="ignore")
        sr_wo_best = metrics.sharpe(keep_v) - metrics.sharpe(keep_b)
    else:
        sr_wo_best = np.nan
    return {"folds_positive": pos, "folds_total": len(by_fold),
            "by_fold": {k: round(v, 7) for k, v in by_fold.items()},
            "sharpe_diff_wo_best_fold": round(float(sr_wo_best), 3)}


def evaluate(store, table: pd.DataFrame, preds: pd.DataFrame,
             symbols: list[str]) -> dict:
    base = base_positions(store, symbols)
    oos = oos_union(preds)
    variants = sorted(preds["variant"].unique())

    nets = {"B0": overlay_net_returns(base, preds, None, oos)}
    nets_stress = {"B0": overlay_net_returns(base, preds, None, oos, STRESS_COST)}
    for v in variants:
        nets[v] = overlay_net_returns(base, preds, v, oos)
        nets_stress[v] = overlay_net_returns(base, preds, v, oos, STRESS_COST)

    mat = pd.DataFrame({v: nets[v] for v in variants})
    sr_var = metrics.trials_sr_variance(mat)

    out = {"oos_start": str(oos.min().date()), "oos_end": str(oos.max().date()),
           "n_oos_days": len(oos), "variants": {}}
    for v in variants:
        pv = preds[preds["variant"] == v]
        entry = {
            "summary": econ_summary(nets[v]),
            "stress2x_sharpe": metrics.sharpe(nets_stress[v]),
            "dsr_n7": round(metrics.deflated_sharpe(nets[v], N_OVERLAY_TRIALS, sr_var), 3),
            "vs_B0": paired_tests(nets[v], nets["B0"]),
            "vs_B2": (paired_tests(nets[v], nets["B2"]) if v != "B2" else None),
            "regimes": regime_slices(nets[v]),
            "incremental": fold_incremental(nets[v], nets["B0"], pv),
            "multiplier_mean": round(float(pv["multiplier"].mean()), 3),
            "multiplier_frac_below_1": round(float((pv["multiplier"] < 0.999).mean()), 3),
            "per_coin_sharpe_diff": _per_coin(base, preds, v, oos),
        }
        out["variants"][v] = entry
    out["B0_summary"] = econ_summary(nets["B0"])
    out["B0_stress2x_sharpe"] = metrics.sharpe(nets_stress["B0"])
    out["pbo_across_variants"] = metrics.pbo_cscv(mat)
    out["nets"] = nets                      # for the report writer
    return out


def _per_coin(base: dict, preds: pd.DataFrame, variant: str,
              oos: pd.DatetimeIndex) -> dict:
    out = {}
    for sym in base:
        single = {sym: base[sym]}
        nv = overlay_net_returns(single, preds, variant, oos)
        nb = overlay_net_returns(single, preds, None, oos)
        out[sym] = round(metrics.sharpe(nv) - metrics.sharpe(nb), 3)
    return out


def gate_verdict(ev: dict, frozen_b0_sharpe: float = 0.738) -> dict:
    """§11.1 Track 1 minimum gate, condition by condition."""
    best_b = max((v for v in ("B1", "B2", "B3") if v in ev["variants"]),
                 key=lambda v: ev["variants"][v]["summary"]["sharpe"], default=None)
    checks = {}
    for name in ("E1", "E2", "E3", "E4"):
        if name not in ev["variants"]:
            continue
        e = ev["variants"][name]
        b0 = ev["B0_summary"]
        checks[name] = {
            "beats_B0_sharpe": e["summary"]["sharpe"] > b0["sharpe"],
            "beats_best_B": (e["summary"]["sharpe"]
                             > ev["variants"][best_b]["summary"]["sharpe"]) if best_b else None,
            "lw_vs_B2_p<0.05": (e["vs_B2"]["sharpe_p"] < 0.05) if e["vs_B2"] else None,
            "dsr_n7>=0.95": e["dsr_n7"] >= 0.95,
            "maxdd_improve>=15pct": (abs(e["summary"]["max_dd_pct"])
                                     <= 0.85 * abs(b0["max_dd_pct"])),
            "cagr_loss<=20pct": (e["summary"]["cagr_pct"]
                                 >= 0.8 * b0["cagr_pct"] if b0["cagr_pct"] > 0 else True),
            "folds_positive>=70pct": (e["incremental"]["folds_positive"]
                                      >= 0.7 * max(e["incremental"]["folds_total"], 1)),
            "stress2x_sharpe>0": e["stress2x_sharpe"] > 0,
            "survives_best_fold_removal": e["incremental"]["sharpe_diff_wo_best_fold"] > 0,
        }
    if {"E1", "E3"} <= set(ev["variants"]):
        checks["E3_beats_E1"] = (ev["variants"]["E3"]["summary"]["sharpe"]
                                 > ev["variants"]["E1"]["summary"]["sharpe"])
    if {"E3", "E4"} <= set(ev["variants"]):
        checks["E4_beats_E3"] = (ev["variants"]["E4"]["summary"]["sharpe"]
                                 > ev["variants"]["E3"]["summary"]["sharpe"])
    passing = {k: all(x for x in v.values() if x is not None)
               for k, v in checks.items() if isinstance(v, dict)}
    return {"per_variant": checks, "any_E_passes_all": any(passing.values()),
            "note": ("Passing promotes an exploratory paper book only; the base "
                     "strategy is uncertified (DSR(N=32)=0.750 < 0.95) and no "
                     "overlay on an uncertified base yields a certified composite.")}
