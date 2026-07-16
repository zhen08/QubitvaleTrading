"""Ranker walk-forward (§2): expanding 730d min train, 182d val, 182d test,
5-day embargo, identical boundaries for every width and baseline. Width is
selected per fold on the validation-window net Sharpe of the §4 portfolio
rule (amendment 1); all widths still produce OOS returns for PBO.

Parallelism: the 15 (width × seed) trainings per fold are independent and run
in a **spawn**-based process pool — fork is unsafe here because the parent has
already run pyarrow's thread pools during data loading, and a forked child can
inherit a held lock and deadlock (observed 2026-07-16: five workers parked on
futexes with zero CPU). Fold tensors are shipped once per worker through the
pool initializer; each training is internally seeded, so scheduling order
cannot affect results and the output is bit-identical to the sequential path.
Worker count via RANKER_WORKERS (default: cores // torch-threads).
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from research import metrics
from research.dl.ranker import data as D
from research.dl.ranker import portfolio as P
from research.dl.ranker.model import (N_THREADS, SEEDS, WIDTHS, TrainedRanker,
                                      predict_ensemble, train_seed)

log = logging.getLogger("qvt.ranker.wf")

TRAIN_MIN_DAYS = 730
VAL_DAYS = 182
TEST_DAYS = 182
EMBARGO_DAYS = 5

_FOLD_TENSORS: dict = {}      # populated in each worker by the pool initializer


def _n_workers() -> int:
    env = os.environ.get("RANKER_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, (os.cpu_count() or 4) // N_THREADS)


def _init_worker(X_tr, y_tr, X_va, y_va) -> None:
    """Spawn initializer: receives the fold tensors once per worker."""
    global _FOLD_TENSORS
    _FOLD_TENSORS = {"X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va}


def _train_task(args: tuple[int, int, int]) -> tuple[int, TrainedRanker]:
    """Runs in a worker; max_epochs travels with the task because spawn
    re-imports modules (parent-side monkeypatches would not propagate)."""
    width, seed, max_epochs = args
    t = _FOLD_TENSORS
    return width, train_seed(seed, width, t["X_tr"], t["y_tr"], t["X_va"], t["y_va"],
                             max_epochs=max_epochs)


def _train_all_widths(X_tr, y_tr, X_va, y_va, widths, seeds) -> dict[int, list[TrainedRanker]]:
    """All (width × seed) trainings for one fold, in parallel; deterministic
    ensemble order is restored by sorting on the fixed seed list."""
    import research.dl.ranker.model as M
    tasks = [(w, s, M.MAX_EPOCHS) for w in widths for s in seeds]
    workers = _n_workers()
    if workers == 1:
        _init_worker(X_tr, y_tr, X_va, y_va)
        results = [_train_task(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers, initializer=_init_worker,
                      initargs=(X_tr, y_tr, X_va, y_va)) as pool:
            results = pool.map(_train_task, tasks)
    by_width: dict[int, list[TrainedRanker]] = {w: [] for w in widths}
    for width, trained in results:
        by_width[width].append(trained)
    order = {s: i for i, s in enumerate(seeds)}
    for w in by_width:
        by_width[w].sort(key=lambda t: order[t.seed])
    return by_width


def fold_schedule(rd: D.RankerData) -> list[dict]:
    t0, t_end = rd.dates.min(), rd.dates.max()
    folds, k = [], 0
    while True:
        train_end = t0 + pd.Timedelta(days=TRAIN_MIN_DAYS + k * TEST_DAYS)
        val_end = train_end + pd.Timedelta(days=VAL_DAYS)
        test_end = min(val_end + pd.Timedelta(days=TEST_DAYS), t_end)
        if val_end >= t_end - pd.Timedelta(days=30):
            break
        folds.append({"fold": k, "train_end": train_end, "val_end": val_end,
                      "test_end": test_end,
                      "complete": bool(test_end == val_end + pd.Timedelta(days=TEST_DAYS))})
        k += 1
    return folds


def _score_frame(meta: pd.MultiIndex, p: np.ndarray, like: pd.DataFrame) -> pd.DataFrame:
    s = pd.Series(p, index=meta).unstack("symbol")
    return s.reindex(columns=like.columns)


def _sim_gru(rd: D.RankerData, score: pd.DataFrame, dates: pd.DatetimeIndex,
             cost_rate: pd.DataFrame | None = None) -> P.SimResult:
    dates = dates[dates.isin(score.index)]
    return P.simulate(dates, score, rd.member, rd.ret_next,
                      cost_rate if cost_rate is not None else rd.cost_rate,
                      confidence=score >= 0.5)


def run_walkforward(rd: D.RankerData, seeds=SEEDS, widths=WIDTHS
                    ) -> tuple[dict, list[dict]]:
    folds = fold_schedule(rd)
    log.info("ranker folds: %d (first test opens %s)",
             len(folds), folds[0]["val_end"].date() if folds else "-")
    emb = pd.Timedelta(days=EMBARGO_DAYS)
    oos_scores: dict[int, list[pd.DataFrame]] = {w: [] for w in widths}
    selected_scores: list[pd.DataFrame] = []
    ledger: list[dict] = []

    for fold in folds:
        train_dates = rd.dates[rd.dates <= fold["train_end"] - emb]
        val_dates = rd.dates[(rd.dates > fold["train_end"])
                             & (rd.dates <= fold["val_end"] - emb)]
        test_dates = rd.dates[(rd.dates > fold["val_end"])
                              & (rd.dates <= fold["test_end"])]
        if len(train_dates) < 400 or len(val_dates) < 60 or len(test_dates) < 30:
            ledger.append({"fold": fold["fold"], "skipped": "insufficient dates"})
            continue

        floor = D.fold_floor(rd, train_dates)
        z = D.normalized_returns(rd, floor)
        zrank = D.cross_rank(z, rd.member)

        tr = D.build_sequences(rd, z, zrank, train_dates, require_label=True)
        va = D.build_sequences(rd, z, zrank, val_dates, require_label=False)
        te = D.build_sequences(rd, z, zrank, test_dates, require_label=False)
        if tr is None or va is None or te is None:
            ledger.append({"fold": fold["fold"], "skipped": "no sequences"})
            continue
        X_tr, y_tr, _ = tr
        X_va, y_va, meta_va = va
        X_te, _, meta_te = te
        X_tr, stats = D.standardize(X_tr)
        X_va, _ = D.standardize(X_va, stats)
        X_te, _ = D.standardize(X_te, stats)

        trained_by_width = _train_all_widths(X_tr, y_tr, X_va, y_va, widths, seeds)
        by_width = {}
        for w in widths:
            trained = trained_by_width[w]
            p_va = predict_ensemble(trained, X_va)
            score_va = _score_frame(meta_va, p_va, rd.member)
            val_sim = _sim_gru(rd, score_va, val_dates)
            val_sharpe = metrics.sharpe(val_sim.net)
            p_te = predict_ensemble(trained, X_te)
            score_te = _score_frame(meta_te, p_te, rd.member)
            oos_scores[w].append(score_te)
            by_width[w] = {"val_sharpe": val_sharpe,
                           "val_loss": float(np.mean([t.val_loss for t in trained])),
                           "score_te": score_te}
            log.info("fold %d width %d: val_sharpe=%.2f val_loss=%.4f",
                     fold["fold"], w, val_sharpe, by_width[w]["val_loss"])

        # amendment 1: max val Sharpe, ties -> smaller width
        sel = max(sorted(widths), key=lambda w: (round(by_width[w]["val_sharpe"], 6), -w))
        selected_scores.append(by_width[sel]["score_te"])
        ledger.append({"fold": fold["fold"], "n_train": len(X_tr),
                       "n_val": len(X_va), "n_test": len(X_te),
                       "sigma_floor": round(floor, 6), "selected_width": sel,
                       **{f"val_sharpe_w{w}": round(by_width[w]["val_sharpe"], 3)
                          for w in widths},
                       **{f"val_loss_w{w}": round(by_width[w]["val_loss"], 5)
                          for w in widths}})

    out = {"selected": pd.concat(selected_scores) if selected_scores else pd.DataFrame(),
           "by_width": {w: (pd.concat(v) if v else pd.DataFrame())
                        for w, v in oos_scores.items()}}
    return out, ledger
