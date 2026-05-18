# Results to Use: Decision Draft

## Main Paper Core

Use these as the main evidence package.

### 1. Covtype20k: Budgeted KAFN compression

Source: `docs/tables_methods_comparison.csv`

| Budget | Budget-Prune F1 | Gate-L1 F1 | Random-B F1 | Recommended claim |
|---:|---:|---:|---:|---|
| 100 | 0.7812 | 0.7881 | 0.6671 | Gate-L1 is slightly better at small budget |
| 200 | 0.7917 | 0.7916 | 0.6669 | Budget-Prune and Gate-L1 are tied |
| 400 | 0.8044 | 0.7939 | 0.6756 | Budget-Prune is better at larger budget |
| full | 0.8161 | 0.7933 | 0.7998 | Full Budget-Prune upper reference |

Main wording:

Budget-Prune preserves most of full KAFN quality at strong compression, while Random-B remains far below. Gate-L1 is useful as an ablation: it helps at small budgets, but does not dominate structurally selected Budget-Prune at B=400.

### 2. Breast Cancer: transfer to small medical dataset

Source: `docs/tables_methods_comparison.csv`

| Budget | Budget-Prune F1 | Gate-L1 F1 | RuleFit F1 | Recommended claim |
|---:|---:|---:|---:|---|
| 100 | 0.9536 | 0.9617 | 0.9587 | Gate-L1 and RuleFit are close; Gate-L1 leads |
| 200 | 0.9620 | 0.9688 | 0.9603 | Gate-L1 leads |
| 400 | 0.9642 | 0.9700 | 0.9578 | Gate-L1 leads |

Main wording:

On the small medical dataset, Gate-L1 is the strongest KAFN variant and also exceeds RuleFit in mean F1 at all three budgets.

### 3. RuleFit external baseline

Sources:

- `compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_summary.csv`
- `docs/rulefit_config_table.csv`

Use in main text:

| Dataset | Budget | RuleFit F1 | Time |
|---|---:|---:|---:|
| breast_cancer | 100 | 0.9587 | 1.0 s |
| breast_cancer | 200 | 0.9603 | 1.2 s |
| breast_cancer | 400 | 0.9578 | 1.3 s |
| SUSY-200k | 100 | 0.7298 | 388.5 s |
| SUSY-200k | 200 | 0.7322 | 1362.3 s |
| SUSY-200k | 400 | 0.7318 | 15301.2 s |

Main wording:

RuleFit is a proper external interpretable baseline. On Breast Cancer, compact KAFN variants are competitive or better. On SUSY, RuleFit is complete and expensive; use it mainly as an external reference unless KAFN quality rows for SUSY are also included.

## Control Checks

Use these to answer reviewer objections.

### 4. LR on top-K KAFN rules

Source: `docs/tables_lr_topk_baseline.csv`

| Budget | Budget-Prune F1 | LR top-K rules F1 | Delta |
|---:|---:|---:|---:|
| 100 | 0.7785 | 0.7604 | +0.0181 |
| 200 | 0.7889 | 0.7644 | +0.0246 |
| 400 | 0.8044 | 0.7789 | +0.0256 |

Main wording:

Budget-Prune is not merely selecting rule activations for a generic linear classifier; the compact KAFN head remains consistently stronger than logistic regression on the same top-K rule features.

### 5. Rule activation correlation

Source: `docs/tables_rule_activation_correlation.csv`

| Subset | Mean absolute correlation |
|---|---:|
| top_100 | 0.1604 |
| top_200 | 0.1417 |
| top_400 | 0.1518 |
| random_400 | 0.1202 |

Recommended use:

Use as a cautious explanation of redundancy, not as a major result. The correlations are moderate, not extreme.

### 6. Minimal KAFN runtime

Source: `docs/tables_compact_kafn_runtime_minimal.csv`

Use carefully.

Recommended wording:

These runtimes describe the compact KAFN pipeline and should not be presented as a direct apples-to-apples training speed comparison with RuleFit. They are useful to show that the compact KAFN workflow is computationally practical.

## Appendix / Future Q1 Track

### 7. Stable Budget-Prune smoke

Sources:

- `docs/q1_stable_selection_smoke_summary.csv`
- `docs/q1_stable_selection_smoke_sweep.csv`

| Budget | Current Jaccard | Stable Jaccard | Retention |
|---:|---:|---:|---:|
| 100 | 0.2887 | 0.5076 | 0.6997 |
| 200 | 0.3939 | 0.6284 | 0.7715 |
| 400 | 0.6384 | 0.8296 | 0.9284 |

Decision:

Do not make this the main contribution of the current paper yet. It is promising for a Q1 extension, especially at B=400, but it is still a proxy experiment based on importance profiles rather than full quality/fidelity evaluation.

## Do Not Overclaim

- Do not claim Q1-level universal generalization yet.
- Do not claim Stable Budget-Prune is validated as a full method yet.
- Do not claim KAFN is faster than RuleFit in a strict training-time sense.
- Do not use SUSY RuleFit alone to claim KAFN beats RuleFit unless matching SUSY KAFN quality rows are included.
- Do not present Gate-L1 as the main method; present it as a learned sparsification ablation.

## Recommended Story

Current article:

Budgeted pruning of active Routed KAFN rule dictionaries gives a practical quality-interpretability trade-off. On Covtype, Budget-Prune retains strong performance under compression and clearly beats Random-B. On Breast Cancer, learned Gate-L1 sparsification performs best, showing that the best compression regime depends on dataset scale and budget. RuleFit provides an external interpretable reference, while LR top-K, correlation, Jaccard, and local explanations close the main reviewer objections.

Q1 follow-up:

Stable Budget-Prune becomes the next paper only if full H-based experiments confirm that the stability gain at B=400 also preserves F1, fidelity, and explanation faithfulness.
