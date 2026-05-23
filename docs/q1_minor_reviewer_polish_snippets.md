# Q1 minor reviewer polish snippets (v34)

## 1) LORO-BCE positioning at small budgets
`LORO-BCE` is not recommended as a standalone selector at very tight budgets (`B < 50`) due to high instability across runs. In this study, its primary role is as a smooth deletion-aware importance signal used inside `Cluster-Prune`, where grouping of correlated rules stabilizes selection behavior.

## 2) Fidelity/agreement comparison (Covtype, 10-seed post-training bootstrap)
At `B=25`, `Cluster-Prune` and `Budget-Prune` are close on faithfulness (`fidelity 0.138 vs 0.144`, `agreement 0.854 vs 0.848`).
At `B=50` and `B=100`, `Cluster-Prune` improves both metrics (`fidelity 0.076/0.067 vs 0.131/0.120`; `agreement 0.920/0.928 vs 0.861/0.877`), indicating better alignment with the full model under moderate and larger budgets.

## 3) Meta-cluster stability trade-off at B=25
At `B=25` (`tau=0.75`), `meta-cluster Jaccard` for `Cluster-Prune` is lower than for `Budget-Prune` (`0.442` vs `0.591`). This is discussed as a deliberate trade-off: `Cluster-Prune` prioritizes redundancy reduction and quality gain under strict budget, while sacrificing group-level overlap in the smallest-budget regime.

## 4) RuleFit config disclosure for supplementary
RuleFit runs are reported with explicit configuration metadata in supplementary tables:
- implementation: `imodels.RuleFitClassifier`
- split protocol: `train/val/test = 0.6/0.2/0.2`
- seeds: `7, 19, 23, 29, 42, 101` (where applicable)
- budget control: `max_rules in {100, 200, 400}`
- runtime/context table: `docs/tables_rulefit_runtime.csv`
- config log table: `docs/rulefit_config_table.csv`

If an explicit hyperparameter sweep is required by the target venue, add it as a separate supplementary experiment and keep the current table as the baseline reproducibility record.

## 5) Repository tag for exact reproducibility
Code and artifacts should be cited with a fixed release tag (e.g., `v34-final`) in the manuscript and cover letter to avoid ambiguity of moving branch heads.
