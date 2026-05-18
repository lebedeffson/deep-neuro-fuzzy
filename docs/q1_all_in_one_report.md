# Q1 All-in-One Report

- Generated: 2026-05-18 17:07 UTC
- Repo: `/home/lebedeffson/Code/deep-neuro-fuzzy`
- Origin: `https://github.com/lebedeffson/deep-neuro-fuzzy.git`

## Current Core Results

- SUSY B=200 runtime anchor: budget_prune_runtime=25.74s (+/-0.94); selection-after-full-dictionary context (not strict train-speed race).
- SUSY full no-prune runtime: full_no_prune_runtime=24.72s (+/-1.63); n=3; includes training+evaluation with full active dictionary.
- Stable proxy budgets available: [100, 200, 400]
- Stable end-to-end H-validation budgets available: [100, 200, 400]

## Q1 Blocker Status

| Check | Status | Evidence |
| --- | --- | --- |
| Covtype seeds >= 6 for stability claims | FAIL | min seeds on Covtype = 3 |
| Statistical tests (Wilcoxon/Friedman/p-values) | PASS | files: docs/q1_stat_tests_summary.csv, docs/q1_bootstrap_ci_summary.csv |
| Runtime decomposition (train/H/select/refit/infer/memory) | FAIL | stage table present=True, has_NA=True; unified control table lacks explicit stage metrics |
| Public reproducibility link present | PASS | origin remote = https://github.com/lebedeffson/deep-neuro-fuzzy.git |
| Stable Budget-Prune end-to-end across >=3 budgets | PASS | budgets with H-validation = [100, 200, 400] |

## Coverage Gaps

- Covtype RuleFit rows present: 0
- SUSY KAFN quality rows present: 0
- Interpretation: Covtype still has no RuleFit rows; SUSY still lacks matched KAFN quality rows in unified main comparison.

## Immediate Next Runs

1. Covtype: raise seeds to 6-10 for Budget-Prune/Gate-L1/Stable.
2. Build runtime stage table: full-train, H-build, selection, refit, inference, memory.

## Stats Artifacts

- Wilcoxon/paired tests rows: 3
- Bootstrap CI rows: 18
- Files: docs/q1_stat_tests_summary.csv, docs/q1_bootstrap_ci_summary.csv
- Runtime stage table: `docs/tables_runtime_stagewise_context.csv`

## Source Files Used

- `docs/unified_main_methods_table.csv`
- `docs/unified_control_checks_table.csv`
- `docs/tables_stable_h_selection_summary.csv`
- `docs/q1_stat_tests_summary.csv`
- `docs/q1_bootstrap_ci_summary.csv`
- `docs/tables_runtime_stagewise_context.csv`
