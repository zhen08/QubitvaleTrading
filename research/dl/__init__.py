"""Cross-asset deep-learning research suite (plan §7/§10).

Modules:
  dataset     fold-local scaling, sequence tensors, ablation masks, embargo splits
  models      fixed causal TCN + context MLP + two heads (§7.3 — changing it = new trial)
  train       deterministic CPU training, 5-seed ensemble (§10.2)
  baselines   B1 σ20 / B2 HAR-RV / B3 VIX threshold (§4.2)
  overlay     shared §7.5 risk-multiplier mapping + validation-fold calibration
  walkforward embargoed expanding walk-forward orchestration (§10.1)
  evaluation  predictive + economic metrics, Ledoit-Wolf test, DSR, regime slices
"""
