"""Freeze (or refreeze) the prospective E2 shadow model (plan follow-up, trial ledger).

Usage: python -m scripts.freeze_dl_shadow [--variant E2]

Trains the frozen §10.1 protocol on all data (expanding train, last 182d
validation), persists artifacts under data/store/models/donchian_tcn_risk_overlay/,
prints the model_id to put in config/settings.yaml -> dl_shadow.model_id, and
appends the registration to research/reports/dl_trial_ledger.md.
"""
from __future__ import annotations

import argparse

from data import storeio
from data.collectors.common import REPO_ROOT, load_settings, setup_logging
from features.cross_asset import build_feature_table
from research.dl.freeze import freeze_shadow

LEDGER = REPO_ROOT / "research" / "reports" / "dl_trial_ledger.md"


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="E2")
    args = ap.parse_args()

    settings = load_settings()
    store = storeio.store_dir(settings)
    table, _ = build_feature_table(store, settings["symbols"])
    man = freeze_shadow(store, table, variant=args.variant)

    entry = (f"\n- **{man['frozen_at'][:10]}** — froze `{man['model_id']}` "
             f"(variant {man['variant']}, ckpt `{man['checkpoint_hash']}`, "
             f"train≤{man['train_end'][:10]}, val≤{man['val_end'][:10]}, "
             f"val tail⁺={man['val_tail_positives']}, "
             f"retrain due {man['retrain_due'][:10]}). One prospective trial; "
             "evaluation only on data logged after this date.\n")
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"model_id: {man['model_id']}")
    print(f"checkpoint: {man['checkpoint_hash']}  retrain due: {man['retrain_due'][:10]}")
    print(f"ledger updated: {LEDGER}")


if __name__ == "__main__":
    main()
