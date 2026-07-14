"""Embargoed expanding walk-forward for B1-B3 + E1-E4 (plan §10.1).

Protocol (ex-ante): >=3y initial training, 182d validation, 182d test,
expanding window, retrain every 182d, 5-day label embargo at every boundary,
identical date boundaries for all variants and assets.

Output: one long DataFrame of per-(decision_ts, symbol) forecasts and risk
multipliers per variant per fold, plus a per-fold diagnostics ledger.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from research.dl import baselines, overlay
from research.dl.dataset import VARIANTS, build_fold, tail_floor
from research.dl.train import SEEDS, predict_ensemble, train_fold

log = logging.getLogger("qvt.dl.wf")

TRAIN_MIN_DAYS = 1095
VAL_DAYS = 182
TEST_DAYS = 182
EMBARGO_DAYS = 5
B_VARIANTS = ("B1", "B2", "B3")


def fold_schedule(table: pd.DataFrame) -> list[dict]:
    """Common fold boundaries from the first fully-valid decision row.

    'Valid' = crypto features complete (252d warmup) for at least one symbol;
    the sequence builder additionally needs 90 contiguous rows, which the
    3-year minimum train window dwarfs.
    """
    valid = table.dropna(subset=["c_ret252_n", "label_logvol5"])
    t0 = valid["decision_ts"].min()
    t_end = table["decision_ts"].max()
    folds, k = [], 0
    while True:
        train_end = t0 + pd.Timedelta(days=TRAIN_MIN_DAYS + k * TEST_DAYS)
        val_end = train_end + pd.Timedelta(days=VAL_DAYS)
        test_end = min(val_end + pd.Timedelta(days=TEST_DAYS), t_end)
        if val_end >= t_end - pd.Timedelta(days=30):
            break
        folds.append({"fold": k, "train_end": train_end,
                      "val_end": val_end, "test_end": test_end,
                      "complete": bool(test_end == val_end + pd.Timedelta(days=TEST_DAYS))})
        k += 1
    return folds


def _b_variant_rows(variant: str, fold: dict, train: pd.DataFrame,
                    test: pd.DataFrame, sigma_ref: float,
                    sigma_floor: float) -> pd.DataFrame:
    if variant == "B2":
        sigma_hat = baselines.HARBaseline().fit(train).predict_sigma(test)
    else:
        sigma_hat = baselines.sigma20_daily(test)
    mult = overlay.risk_multiplier(sigma_hat, sigma_ref)
    if variant == "B3":
        mult = mult * baselines.vix_gate(train, test)
    return pd.DataFrame({
        "variant": variant, "fold": fold["fold"],
        "decision_ts": test.index, "symbol": test["symbol"].to_numpy(),
        "sigma_hat": sigma_hat, "p_tail_cal": np.nan,
        "multiplier": mult, "sigma_floor": sigma_floor,
    })


def run_walkforward(table: pd.DataFrame, variants: list[str] | None = None,
                    seeds=SEEDS) -> tuple[pd.DataFrame, list[dict]]:
    folds = fold_schedule(table)
    log.info("fold schedule: %d folds, first train_end=%s",
             len(folds), folds[0]["train_end"].date() if folds else "-")
    variants = variants or [*B_VARIANTS, *VARIANTS.keys()]
    rows: list[pd.DataFrame] = []
    ledger: list[dict] = []

    for fold in folds:
        # B-family: plain frame splits with the same embargo
        emb = pd.Timedelta(days=EMBARGO_DAYS)
        t = table.set_index("decision_ts")
        train_b = t[t.index <= fold["train_end"] - emb].dropna(subset=["label_logvol5"])
        test_b = t[(t.index > fold["val_end"]) & (t.index <= fold["test_end"])]
        test_b = test_b.dropna(subset=["c_ret252_n"])
        sigma_ref = overlay.sigma_ref_from_train(train_b)
        fold_floor = tail_floor(train_b)

        for variant in variants:
            if variant in B_VARIANTS:
                rows.append(_b_variant_rows(variant, fold, train_b, test_b,
                                            sigma_ref, fold_floor))
                ledger.append({"fold": fold["fold"], "variant": variant,
                               "sigma_ref": round(sigma_ref, 5),
                               "n_test": len(test_b)})
                continue

            fd = build_fold(table, variant, fold["train_end"], fold["val_end"],
                            fold["test_end"], EMBARGO_DAYS)
            if fd is None:
                ledger.append({"fold": fold["fold"], "variant": variant,
                               "skipped": "insufficient data"})
                continue
            trained, diag = train_fold(fd.tensors, seeds)
            n_seq = fd.tensors["train"][0].shape[2]
            n_ctx = fd.tensors["train"][1].shape[1]

            seq_va, ctx_va, yv_va, yt_va, meta_va = fd.tensors["val"]
            _, p_va = predict_ensemble(trained, seq_va, ctx_va, n_seq, n_ctx)
            calibrate, n_pos = overlay.fit_tail_calibrator(p_va, yt_va.numpy())

            seq_te, ctx_te, _, _, meta_te = fd.tensors["test"]
            logvol_te, p_te = predict_ensemble(trained, seq_te, ctx_te, n_seq, n_ctx)
            sigma_hat = np.exp(logvol_te)
            p_cal = calibrate(p_te) if calibrate is not None else None
            mult = overlay.risk_multiplier(sigma_hat, sigma_ref, p_cal)

            rows.append(pd.DataFrame({
                "variant": variant, "fold": fold["fold"],
                "decision_ts": meta_te.get_level_values("decision_ts"),
                "symbol": meta_te.get_level_values("symbol"),
                "sigma_hat": sigma_hat,
                "p_tail_cal": p_cal if p_cal is not None else np.nan,
                "multiplier": mult, "sigma_floor": fd.sigma_floor,
            }))
            ledger.append({"fold": fold["fold"], "variant": variant,
                           "sigma_ref": round(sigma_ref, 5),
                           "n_train": len(fd.tensors["train"][0]),
                           "n_test": len(meta_te),
                           "val_tail_positives": n_pos,
                           "tail_gate_active": calibrate is not None,
                           "sigma_floor": round(fd.sigma_floor, 6),
                           "scaler_hash": fd.scaler.hash(), **diag})
            log.info("fold %d %s done (val_pos=%d, ckpt=%s)",
                     fold["fold"], variant, n_pos, diag["checkpoint_hash"])

    preds = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return preds, ledger
