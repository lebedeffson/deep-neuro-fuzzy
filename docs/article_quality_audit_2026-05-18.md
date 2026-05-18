# Article Quality Audit

## Verdict

The current evidence is strong enough for a careful unified empirical-method paper, but not yet for an aggressive Q1 claim.

The article can be made high quality if we keep claims narrow:

- Budget-Prune is the main practical method.
- Gate-L1 is a learned sparsification ablation.
- RuleFit is an external interpretable baseline where completed.
- Stable Budget-Prune is a promising stability-aware extension, not fully validated yet.

## What Is Solid

### 1. Covtype20k compression result

Source: `docs/unified_main_methods_table.csv`

Strong points:

- Budget-Prune improves with budget.
- At B=400, Budget-Prune reaches F1 `0.8044`.
- Random-B is much weaker at all budgets:
  - B=100: `0.6671`
  - B=200: `0.6669`
  - B=400: `0.6756`
- This supports the claim that structured importance-based pruning is meaningful.

Safe claim:

> On Covtype20k, Budget-Prune gives a substantially better quality-compactness trade-off than random rule selection, and becomes the strongest compact KAFN variant at B=400.

Avoid:

> Budget-Prune universally dominates Gate-L1.

Because Gate-L1 is slightly better at B=100 and essentially tied at B=200.

### 2. Breast Cancer transfer result

Source: `docs/unified_main_methods_table.csv`

Strong points:

- Six seeds are available.
- Gate-L1 is strongest at all budgets.
- RuleFit is included and competitive, but Gate-L1 has higher mean F1.

Safe claim:

> On the small medical dataset, learned gate sparsification is more effective than deterministic pruning, suggesting that the best compression regime depends on dataset scale and budget.

Avoid:

> Budget-Prune is always the best compact KAFN method.

### 3. LR top-K control

Source: `docs/unified_control_checks_table.csv`

Strong points:

- Same selected top-K rules, but logistic regression is weaker.
- Delta F1:
  - B=100: `+0.0181`
  - B=200: `+0.0246`
  - B=400: `+0.0256`

Safe claim:

> The compact KAFN head is not reducible to a generic logistic model on the same selected rule activations.

This is a strong reviewer-defense result.

### 4. RuleFit external baseline

Sources:

- `compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_summary.csv`
- `docs/rulefit_config_table.csv`

Strong points:

- Breast Cancer RuleFit complete for 6 seeds and B=100/200/400.
- SUSY RuleFit complete for 6 seeds and B=100/200/400.
- Config table exists and documents default settings.

Safe claim:

> RuleFit is included as an external interpretable reference rather than only using internal KAFN ablations.

Avoid:

> KAFN beats RuleFit on SUSY.

Because the current unified table does not include matching SUSY KAFN quality rows.

### 5. Stable Budget-Prune smoke

Source: `docs/q1_stable_selection_smoke_summary.csv`

Strong points:

- Pairwise Jaccard improves:
  - B=100: `0.2887 -> 0.5076`
  - B=200: `0.3939 -> 0.6284`
  - B=400: `0.6384 -> 0.8296`
- At B=400, importance retention is good: `0.9284`.

Safe claim:

> A preliminary stability-aware extension improves selected-subset reproducibility, especially at B=400, while preserving most held-out importance.

Avoid:

> Stable Budget-Prune is fully validated.

Because this is still an importance-profile proxy, not an H-based F1/fidelity evaluation.

## Weak Spots

### A. SUSY is incomplete in the unified quality table

Problem:

`docs/unified_main_methods_table.csv` has SUSY RuleFit rows but no SUSY Budget-Prune/Gate-L1 F1 rows.

Risk:

Reviewer sees a multi-dataset table where SUSY only contains the competitor.

Fix options:

1. Best: recover or generate SUSY KAFN quality rows for B=100/200/400.
2. If not possible: move SUSY RuleFit to a separate external-baseline/runtime table, not the main method comparison table.

### B. Covtype has no RuleFit row

Problem:

RuleFit is available for Breast and SUSY, but not Covtype20k.

Risk:

Reviewer may ask why the main dataset lacks the external baseline.

Fix options:

1. Best: run RuleFit on Covtype20k for B=100/200/400 with 3 seeds.
2. If too slow: explicitly state RuleFit was evaluated on Breast/SUSY as external reference, while Covtype uses internal pruning controls plus LR top-K.

### C. Runtime table can be misread

Problem:

KAFN pipeline time and RuleFit training time are not identical procedures.

Risk:

If we claim speed superiority, reviewer can attack fairness.

Fix:

Only use runtime as computational context. Write:

> These times are not strict training-speed comparisons because the compact KAFN pipeline and RuleFit construct different models.

### D. Stable Budget-Prune needs one decisive practical check

Problem:

Current Stable result is proxy-only.

Fix:

One H-based Covtype B=400 check:

```text
Budget-Prune vs Stable Budget-Prune
seeds: 19, 23, 29
metrics: F1, ROC-AUC, PR-AUC, fidelity gap, prediction agreement, subset Jaccard
```

Promotion rule:

```text
Stable can be main method if:
Jaccard improves
and F1 drop <= 0.01
and fidelity gap is small
```

Otherwise keep it as extension/future work.

## Recommended Article Claims

Use:

1. **Budgeted interpretability:** KAFN active rule dictionaries can be compressed under explicit budgets while retaining useful predictive quality.
2. **Structured pruning matters:** Budget-Prune strongly beats Random-B on Covtype.
3. **Compression regime depends on data:** Gate-L1 wins on small Breast Cancer, while Budget-Prune wins at larger Covtype budget.
4. **Not just feature selection:** LR on top-K rules is weaker than compact KAFN.
5. **External interpretability baseline:** RuleFit is included and compact KAFN is competitive on Breast Cancer.
6. **Stability direction:** Stable Budget-Prune improves subset reproducibility in proxy analysis, motivating a stability-aware extension.

Do not use:

1. Universal superiority over RuleFit.
2. Universal superiority of Budget-Prune over Gate-L1.
3. Full Q1 benchmark claim across many datasets.
4. Strict training-speed superiority over RuleFit.
5. Stable Budget-Prune as fully validated unless H-based check is completed.

## Minimal Pre-Submission Work

Required:

1. Clean main table so SUSY is not misleading.
2. Add limitation paragraph for Stable Budget-Prune proxy.
3. Add method paragraph for Budget-Prune theoretical rationale.
4. Add RuleFit config table or footnote.
5. Add LR top-K control table.

Recommended:

1. H-based Stable Budget-Prune check at Covtype B=400.
2. Covtype RuleFit if runtime is acceptable.
3. Convert `docs/unified_main_methods_table.csv` into manuscript-ready LaTeX/Word table.

## Quality Decision

If no new runs:

Submit as a careful empirical-method paper, likely strong Q2 / technical journal.

If H-based Stable check succeeds:

The work becomes much stronger and can be framed as a unified method:

> Budget-Prune with stability-aware rule selection for compact, reproducible KAFN dictionaries.

If we also add Covtype RuleFit or SUSY KAFN quality rows:

The paper becomes much closer to Q1 positioning, though still not a broad 10-dataset benchmark.
