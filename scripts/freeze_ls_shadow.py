"""Freeze (or refreeze) the Track 3-LS prospective shadow model.

Usage: python -m scripts.freeze_ls_shadow

Trains the mechanical L/S transform of the frozen ranker protocol on all data
(3 widths × 5 seeds; width selected on validation L/S net Sharpe), persists
artifacts, prints the model_id for config/settings.yaml →
dl_shadow.ranker_ls_model_id, and appends to the preregistration + ledger.
"""
from __future__ import annotations

from data import storeio
from data.collectors.common import REPO_ROOT, load_settings, setup_logging
from research.dl.ranker.freeze import freeze_ls_shadow

PREREG = REPO_ROOT / "research" / "reports" / "track3_ls_shadow_preregistration.md"
LEDGER = REPO_ROOT / "research" / "reports" / "dl_trial_ledger.md"


def main() -> None:
    setup_logging()
    settings = load_settings()
    store = storeio.store_dir(settings)
    man = freeze_ls_shadow(store)

    entry = (f"\n- **{man['frozen_at'][:10]}** — froze `{man['model_id']}` "
             f"(width {man['width']} by val L/S Sharpe {man['val_sharpe_by_width']}, "
             f"ckpt `{man['checkpoint_hash']}`, train≤{man['train_end'][:10]}, "
             f"val≤{man['val_end'][:10]}, {len(man['shortable_um'])} shortable perps, "
             f"retrain due {man['retrain_due'][:10]}).\n")
    with open(PREREG, "a", encoding="utf-8") as f:
        f.write(entry)
    print(f"model_id: {man['model_id']}")
    print(f"width: {man['width']}  checkpoint: {man['checkpoint_hash']}")
    print(f"val sharpes: {man['val_sharpe_by_width']}")


if __name__ == "__main__":
    main()
