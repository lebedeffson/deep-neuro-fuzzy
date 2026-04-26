# Current Experiment Summary

This directory intentionally keeps a single compact summary instead of many intermediate benchmark dumps.
Exploratory JSON, TXT, Markdown, logs, and temporary sweep folders are not kept in the repository.

## Canonical Results Used in the Draft

The current finalization pass uses two result blocks.

1. Reference comparison on the small and medium tabular benchmark:
   - Models: shallow, stacked, hierarchical, DFFL.
   - Seeds: 19, 23, 29.
   - Metrics: RMSE for regression, F1 for binary classification.
   - Interpretation metrics are treated as structural interpretability/stability, not semantic validation.

2. Large-scale Covtype capacity check:
   - Dataset: `covtype_binary_full`, 581,012 samples, 54 features.
   - Main fuzzy model: capacity-tuned stacked ANFIS.
   - Final stacked configuration: `stacked_width_scale=2.0`, `stacked_rule_scale=2.0`,
     `stacked_final_skip_inputs=24`, `stacked_final_skip_mode=target_corr_diverse`,
     final skip gates enabled, `fuzzy_distill_weight=0.1`, Extra Trees teacher with 400 trees,
     tuned calibrated threshold.
   - Seeds: 19, 23, 29.

## Large-Scale Covtype Full Results

| Model | F1 | Accuracy | Precision | Recall | ROC-AUC | PR-AUC | Rules | Active rules | Active-rule Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic regression | 0.7553 +/- 0.0007 | 0.7571 +/- 0.0006 | 0.7422 +/- 0.0003 | 0.7688 +/- 0.0013 | 0.8281 +/- 0.0012 | 0.8039 +/- 0.0008 | - | - | - |
| histogram gradient boosting | 0.8328 +/- 0.0011 | 0.8314 +/- 0.0011 | 0.8066 +/- 0.0012 | 0.8607 +/- 0.0020 | 0.9183 +/- 0.0010 | 0.9099 +/- 0.0012 | - | - | - |
| MLP | 0.8589 +/- 0.0009 | 0.8586 +/- 0.0033 | 0.8376 +/- 0.0187 | 0.8822 +/- 0.0204 | 0.9377 +/- 0.0026 | 0.9307 +/- 0.0029 | - | - | - |
| hierarchical capacity | 0.8930 +/- 0.0004 | 0.8921 +/- 0.0007 | 0.8645 +/- 0.0050 | 0.9237 +/- 0.0058 | 0.9615 +/- 0.0009 | 0.9573 +/- 0.0013 | 207 | 82.33 +/- 4.19 | 0.2050 |
| stacked w1.5/r1.5 skip16 distill | 0.9095 +/- 0.0025 | 0.9094 +/- 0.0026 | 0.8860 +/- 0.0035 | 0.9343 +/- 0.0018 | 0.9725 +/- 0.0020 | 0.9700 +/- 0.0023 | 75 | 35.33 +/- 2.05 | 0.1529 |
| stacked w2.0/r2.0 skip24 distill | 0.9273 +/- 0.0020 | 0.9281 +/- 0.0020 | 0.9144 +/- 0.0013 | 0.9406 +/- 0.0029 | 0.9817 +/- 0.0008 | 0.9803 +/- 0.0008 | 100 | 49.67 +/- 1.25 | 0.1119 |
| extra trees | 0.9543 +/- 0.0003 | 0.9549 +/- 0.0003 | 0.9430 +/- 0.0006 | 0.9659 +/- 0.0002 | 0.9919 +/- 0.0001 | 0.9912 +/- 0.0002 | - | - | - |
| random forest | 0.9560 +/- 0.0001 | 0.9565 +/- 0.0001 | 0.9441 +/- 0.0002 | 0.9681 +/- 0.0001 | 0.9924 +/- 0.0001 | 0.9918 +/- 0.0001 | - | - | - |

## Interpretation

The strongest fuzzy result is the tuned stacked model with larger hidden/rule capacity and a gated raw-feature
skip. On full Covtype it reaches `F1 = 0.9273 +/- 0.0020`, improving over the previous full stacked setting
(`0.9095 +/- 0.0025`) and over the earlier 200k stacked setting (`0.8856 +/- 0.0011`).

This confirms that the stacked architecture was capacity-limited on large tabular data. Increasing rules and
hidden concepts helps, and raw-feature skip connections reduce information loss between stacked fuzzy stages.
The cost is also clear: the tuned model uses 100 rules, around 50 active rules, and has lower active-rule
Jaccard than smaller configurations.

The tuned stacked model is stronger than logistic regression, histogram gradient boosting, MLP, and the tested
hierarchical fuzzy model on full Covtype. It remains below Random Forest and Extra Trees in raw predictive quality.
The paper should therefore claim a strong quality/complexity/structural-stability trade-off, not dominance over
tree ensembles.

## Reproducibility Command

Final tuned stacked run:

```bash
python examples/run_real_datasets_benchmark.py \
  --datasets covtype_binary_full \
  --seeds 19,23,29 \
  --gpu-only \
  --fuzzy-models stacked \
  --max-epochs 40 \
  --patience 8 \
  --batch-size 4096 \
  --tune-fuzzy-threshold \
  --tune-fuzzy-threshold-calibrated \
  --stacked-width-scale 2.0 \
  --stacked-rule-scale 2.0 \
  --stacked-final-skip-inputs 24 \
  --stacked-final-skip-mode target_corr_diverse \
  --stacked-final-skip-gates \
  --stacked-final-skip-gate-l1-weight 0.0001 \
  --fuzzy-distill-weight 0.1 \
  --fuzzy-distill-models stacked \
  --fuzzy-distill-teacher-trees 400 \
  --output-dir docs/artifacts/latest_covtype_full_stack_w20_r20_skip24_distill010_3s
```

Baseline run:

```bash
python examples/run_real_datasets_benchmark.py \
  --datasets covtype_binary_full \
  --seeds 19,23,29 \
  --fuzzy-models none \
  --output-dir docs/artifacts/latest_covtype_full_baselines_3s
```
