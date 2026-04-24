# Architecture Tuning Summary (2026-04-24)

Scope: targeted stacked/hierarchical changes for losing datasets.

## Code knobs added
The benchmark runner now exposes non-default capacity and loss controls for non-DFFL fuzzy models:

| Flag | Purpose |
|---|---|
| `--stacked-width-scale` | scale stacked hidden widths |
| `--stacked-rule-scale` | scale stacked rule budgets |
| `--hierarchical-width-scale` | scale hierarchical local/aggregate widths |
| `--hierarchical-rule-scale` | scale hierarchical rule budgets |
| `--hierarchical-group-size` | change local feature group size |
| `--stacked-prototype-scoring-mode` | stacked prototype scoring: `max` or `hybrid` |
| `--hierarchical-prototype-scoring-mode` | hierarchical prototype scoring: `max` or `hybrid` |
| `--fuzzy-binary-loss-name` | `bce` or `focal` for shallow/stacked/hierarchical |
| `--fuzzy-binary-focal-gamma` | focal-loss gamma |
| `--fuzzy-binary-auto-pos-weight` | automatic class weighting |
| `--fuzzy-binary-soft-f1-weight` | add differentiable soft-F1 penalty |
| `--fuzzy-regression-loss` | `mse` or `huber` |
| `--fuzzy-huber-delta` | Huber delta |
| `--fuzzy-distill-weight` | label distillation blend weight from ExtraTrees teacher |
| `--fuzzy-distill-models` | fuzzy models receiving distilled targets |
| `--fuzzy-distill-teacher-trees` | ExtraTrees teacher size for distillation |
| `--tune-fuzzy-threshold-calibrated` | calibration-aware threshold tuning for fuzzy models |

Default values reproduce the previous behavior.

## Experiments
| Dataset | Candidate | Result | Compare | Decision |
|---|---|---:|---:|---|
| california_housing | stacked, lr=0.02, width x1.25, rules x1.5, seed 23 | 0.1195 RMSE | previous quick stacked lr=0.02: 0.1146 | reject |
| covtype_binary_20000 | hierarchical, group=3, width x1.5, rules x1.25, seed 23 | 0.7821 F1 | unified stacked mean: 0.8129 | reject |
| covtype_binary_20000 | stacked, focal gamma=1.5, auto pos weight, soft-F1=0.15, seed 23 | 0.8232 F1 | unified seed-23 stacked: 0.8183 | promising single seed |
| covtype_binary_20000 | same stacked focal+soft-F1, 3 seeds | 0.8127 +/- 0.0151 F1 | unified stacked mean: 0.8129 +/- 0.0108 | reject as not robust |
| covtype_binary_20000 | stacked, prototype scoring `hybrid`, seed 23 | 0.8001 F1 | unified seed-23 stacked: 0.8183 | reject |
| covtype_binary_20000 | stacked, focal gamma=1.0, auto pos weight, soft-F1=0.05, tune threshold, seed 23 | 0.8215 F1 | unified seed-23 stacked: 0.8183 | promising single seed |
| covtype_binary_20000 | same stacked focal(g=1.0)+soft-F1, tune threshold, 3 seeds | 0.8047 +/- 0.0175 F1 | unified stacked mean: 0.8129 +/- 0.0108 | reject as not robust |
| covtype_binary_20000 | stacked, BCE + auto pos weight + soft-F1=0.05, tune threshold, seed 23 | 0.7940 F1 | unified seed-23 stacked: 0.8183 | reject |
| covtype_binary_20000 | stacked, distill (ExtraTrees, w=0.35), seed 23 | 0.7933 F1 | unified seed-23 stacked: 0.8183 | reject |
| covtype_binary_20000 | stacked, distill (ExtraTrees, w=0.10), seed 23 | 0.7808 F1 | unified seed-23 stacked: 0.8183 | reject |
| covtype_binary_20000 | stacked, threshold tuning fix (per-model, not DFFL-only), seed 23 | 0.8216 F1 (thr=0.43) | prior quick seed-23 stacked: 0.8183 | promising single seed |
| covtype_binary_20000 | stacked, threshold tuning fix, 3 seeds, full budget | 0.8080 +/- 0.0156 F1 | unified stacked mean: 0.8129 +/- 0.0108 | reject as not robust |
| covtype_binary_20000 | stacked, threshold tuning fix + calibrated, 3 seeds, full budget | 0.8095 +/- 0.0134 F1 | unified stacked mean: 0.8129 +/- 0.0108 | reject as not robust |
| california_housing | stacked, huber loss delta=0.05, seed 23 | 0.1276 RMSE | previous quick stacked lr=0.02: 0.1146 | reject |
| diabetes | DFFL profile=quality, dffl_lr in {0.009, 0.011, 0.013}, seed 23 | 0.1859 / 0.1856 / 0.1900 RMSE | unified DFFL mean: 0.1796 | reject |

## Diagnosis
Increasing capacity did not improve the hard datasets. On `covtype_binary_20000`, wider hierarchical decomposition made the model heavier and worse, which suggests overfitting or weak prototype selection rather than insufficient rule budget. Loss-level tuning improved isolated seeds but failed the three-seed robustness check. Distillation from ExtraTrees degraded quality at tested blend weights. A bug was fixed where `--tune-fuzzy-threshold` affected only DFFL; after fixing per-model tuning, single-seed results improved, but 3-seed means still did not beat the unified baseline due threshold overfitting on one seed. On `diabetes`, small DFFL learning-rate changes in profile `quality` were consistently worse than the current unified result.

## Current recommendation
Keep the current unified numbers for the paper. The next real improvement path is not parameter tuning; it requires a new prototype/rule-selection mechanism or a stronger feature grouping strategy for high-dimensional binary tasks.
