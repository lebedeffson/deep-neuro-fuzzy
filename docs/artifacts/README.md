# Current Experiment Summary

This directory keeps one compact final summary.
Local raw bundles are stored under `runs/`.

## Two Large-Scale Datasets

We use two large-scale tabular datasets:

1. `covtype_binary_full` (581,012 x 54) as the **main discriminative benchmark**.
2. `kddcup99_binary_full` (4,898,431 x ~122 after preprocessing) as a **feasibility/scalability check**.

## Covtype Full (Main Benchmark)

| Model | F1 |
| --- | ---: |
| Logistic Regression | 0.7551 |
| shallow fuzzy | 0.7700 |
| DFFL final | 0.8413 |
| hist gradient boosting | 0.8321 |
| MLP | 0.8591 |
| hierarchical fuzzy | 0.9033 |
| stacked fuzzy (tuned) | **0.9273 +/- 0.0020** |
| Extra Trees | 0.9543 |
| Random Forest | 0.9560 |

Key promoted stacked metrics on Covtype full:
- `F1 = 0.9273 +/- 0.0020`
- `ROC-AUC = 0.9817 +/- 0.0008`
- `PR-AUC = 0.9803 +/- 0.0008`

Main claim supported by this dataset:
- stacked deep ANFIS is clearly stronger than other fuzzy/neural/linear baselines,
- but still below Random Forest and Extra Trees.

## KDDCup99 Full (Scalability Check)

| Model | F1 |
| --- | ---: |
| shallow fuzzy | 0.9993 |
| stacked fuzzy | 0.9999 |
| hierarchical fuzzy | 0.9998 |
| Logistic Regression | 0.9991 |
| MLP | 0.9999 |
| RF / ExtraTrees / HGB | ~1.0000 |

Interpretation:
- KDD confirms that our pipeline and fuzzy models scale to multi-million rows.
- KDD is too easy/saturated for strong competitive quality claims.

## Paper Wording

Recommended wording:

> We evaluate on two large-scale tabular datasets: Covtype full as the main discriminative benchmark, and KDDCup99 full as a large-scale feasibility/scalability check.

## Reproducibility Commands

Covtype full (promoted stacked run):

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
  --output-dir runs/covtype_full_stacked_w20_r20_skip24_distill010_3s
```
