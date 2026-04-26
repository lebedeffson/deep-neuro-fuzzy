# Current Experiment Summary

This directory intentionally keeps a single compact summary instead of many intermediate benchmark dumps.
Old exploratory JSON, TXT, Markdown, logs, and temporary sweep folders were removed from the repository cleanup.

## Canonical Results Used in the Draft

The paper draft reports two result blocks.

1. Main and extended reference comparison:
   - Models: shallow, stacked, hierarchical, DFFL.
   - Seeds: 19, 23, 29.
   - Metrics: RMSE for regression, F1 for binary classification.
   - Interpretation metrics are reported as structural stability, mainly active-rule Jaccard.

2. Large-scale stacked capacity check on `covtype_binary_200000`:
   - Dataset: 200,000 samples from the binary Covtype task.
   - Model: capacity-tuned stacked ANFIS.
   - Configuration: `stacked_width_scale=1.5`, `stacked_rule_scale=1.5`, `stacked_final_skip_inputs=16`,
     `stacked_final_skip_mode=target_corr_diverse`, final skip gates enabled, `fuzzy_distill_weight=0.1`,
     Extra Trees teacher with 400 trees, tuned calibrated threshold.
   - Seeds: 19, 23, 29.

## Large-Scale Covtype Results

| Model | F1 | Accuracy | Precision | Recall | ROC-AUC | PR-AUC | Rules | Active rules | Active-rule Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shallow | 0.7693 +/- 0.0031 | 0.7433 +/- 0.0028 | 0.6846 +/- 0.0024 | 0.8779 +/- 0.0073 | 0.8264 +/- 0.0038 | 0.7855 +/- 0.0042 | 36 | 20.67 +/- 4.71 | 0.1808 |
| stacked reference | 0.8627 +/- 0.0034 | 0.8580 +/- 0.0028 | 0.8163 +/- 0.0109 | 0.9152 +/- 0.0184 | 0.9374 +/- 0.0036 | 0.9280 +/- 0.0055 | 50 | 24.33 +/- 0.47 | 0.1984 |
| hierarchical | 0.8399 +/- 0.0045 | 0.8332 +/- 0.0039 | 0.7895 +/- 0.0050 | 0.8973 +/- 0.0122 | 0.9175 +/- 0.0041 | 0.9060 +/- 0.0049 | 138 | 62.67 +/- 0.94 | 0.2274 |
| DFFL | 0.7971 +/- 0.0022 | 0.7846 +/- 0.0037 | 0.7373 +/- 0.0063 | 0.8676 +/- 0.0048 | 0.8698 +/- 0.0039 | 0.8490 +/- 0.0083 | 144 | 21.00 +/- 1.63 | 0.0415 |
| stacked capacity + skip + distill | 0.8856 +/- 0.0011 | 0.8836 +/- 0.0009 | 0.8504 +/- 0.0005 | 0.9239 +/- 0.0028 | 0.9561 +/- 0.0014 | 0.9507 +/- 0.0017 | 75 | 33.67 +/- 5.79 | 0.1661 |

## Baseline Context on `covtype_binary_200000`

| Baseline | F1 | Accuracy | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| logistic regression | 0.7554 +/- 0.0003 | 0.7560 +/- 0.0007 | 0.8282 +/- 0.0013 | 0.8034 +/- 0.0019 |
| histogram gradient boosting | 0.8308 +/- 0.0012 | 0.8288 +/- 0.0013 | 0.9169 +/- 0.0016 | 0.9086 +/- 0.0019 |
| MLP | 0.8396 +/- 0.0094 | 0.8438 +/- 0.0053 | 0.9266 +/- 0.0022 | 0.9177 +/- 0.0029 |
| random forest | 0.9289 +/- 0.0014 | 0.9292 +/- 0.0015 | 0.9811 +/- 0.0008 | 0.9790 +/- 0.0010 |
| extra trees | 0.9328 +/- 0.0006 | 0.9333 +/- 0.0007 | 0.9825 +/- 0.0005 | 0.9807 +/- 0.0007 |

## Interpretation

The strongest fuzzy result is the capacity-tuned stacked model. It improves over the original stacked reference
from `F1 = 0.8627 +/- 0.0034` to `F1 = 0.8856 +/- 0.0011` on `covtype_binary_200000`.
The improvement comes mainly from larger rule/concept capacity and the final raw-feature skip connection.
Distillation from Extra Trees gives a smaller additional gain.

The tuned stacked model is stronger than logistic regression, histogram gradient boosting, and MLP on this task,
but it remains below Random Forest and Extra Trees in raw predictive quality. The article should therefore keep
claims conservative: the result supports a quality/complexity/structural-stability trade-off, not universal
dominance over tree ensembles.

## Reproducibility Commands

Reference tuned stacked run:

```bash
python examples/run_real_datasets_benchmark.py \
  --datasets covtype_binary_200000 \
  --seeds 19,23,29 \
  --gpu-only \
  --fuzzy-models stacked \
  --max-epochs 40 \
  --patience 8 \
  --batch-size 4096 \
  --tune-fuzzy-threshold \
  --tune-fuzzy-threshold-calibrated \
  --stacked-width-scale 1.5 \
  --stacked-rule-scale 1.5 \
  --stacked-final-skip-inputs 16 \
  --stacked-final-skip-mode target_corr_diverse \
  --stacked-final-skip-gates \
  --stacked-final-skip-gate-l1-weight 0.0001 \
  --fuzzy-distill-weight 0.1 \
  --fuzzy-distill-models stacked \
  --fuzzy-distill-teacher-trees 400 \
  --output-dir docs/artifacts/latest_covtype200k_stacked
```

Baseline context run:

```bash
python examples/run_real_datasets_benchmark.py \
  --datasets covtype_binary_200000 \
  --seeds 19,23,29 \
  --fuzzy-models none \
  --output-dir docs/artifacts/latest_covtype200k_baselines
```
