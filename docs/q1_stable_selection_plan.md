# Compact KAFN: Two-Track Plan

## Track 1: Current journal article

Scope: empirical and methodological paper on budgeted compression of the active Routed KAFN rule dictionary.

Keep:
- Budget-Prune, Random-B, Gate-L1, RuleFit.
- Subset Jaccard.
- Local explanation table.
- Validation-importance check.
- LR top-K control, rule-correlation control, importance profile, minimal runtime table.

Target: Russian technical journal / applied engineering venue.

Do not expand this version into a full Q1 benchmark. It is already coherent as a compact empirical study.

## Track 2: Q1 extension

Core question:

Can we compress the KAFN rule dictionary while preserving quality, fidelity, and reproducibility of the selected subdictionary?

New method:

Stable Budget-Prune ranks rules by importance, repeated top-B membership, and optional redundancy penalty.

Implemented seed module:

- `src/ruanfis/stable_budget_prune.py`

Initial API:

- `rule_importance`
- `stable_budget_scores`
- `stable_budget_prune`
- `redundancy_aware_budget_indices`
- `weight_only_indices`
- `activation_only_indices`
- `random_budget_indices`
- `sparse_logistic_indices`
- `fidelity_gap`
- `prediction_agreement`

Next implementation steps:

1. Export full KAFN rule activation matrix `H` for train/val/test.
2. Add selectors on `H`: weight-only, activation-only, L1, ElasticNet, greedy forward, Random-B.
3. Add Q1 metrics: fidelity, agreement, faithfulness, redundancy, readability, runtime, memory.
4. Run smoke benchmark on 5 datasets: Covtype, SUSY, Breast Cancer, Adult, Bank Marketing.
5. Scale to 10-15 datasets and add Wilcoxon/Friedman/average ranks.

Decision rule:

Track 1 should be submitted when text is clean. Track 2 should become a new paper only after Stable Budget-Prune improves either subset stability, redundancy, or fidelity under matched budgets.
