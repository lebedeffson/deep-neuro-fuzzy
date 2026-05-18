# v16 Article-Ready Manifest

- generated_at: 2026-05-18T11:15:46
- paper_version: v16

## Main Claims
- budgeted compression of active Routed KAFN dictionary
- Random-B as negative control (planned claim; artifact not included in this v16 package)
- subset Jaccard for compact dictionary stability
- RuleFit comparison on Breast/SUSY
- Covtype concept decoding

## Used Tables
- `docs/tables_methods_comparison.csv`
- `docs/tables_fair_active_rules_comparison.csv`
- `docs/tables_importance_split_ablation.csv`
- `docs/tables_subset_jaccard_summary.csv`
- `docs/covtype_local_concept_decoding_clean.csv`
- `docs/rulefit_config_table.csv`
- `docs/tables_rulefit_runtime.csv`
- `docs/rulefit_q1_partial_summary_2026-05-17.md`

## Fair-Active Summary
- B=25: Gate-L1 active=25.0, Budget-Prune active=n/a (matched-budget run only)
- B=50: Gate-L1 active=50.0, Budget-Prune active=n/a (matched-budget run only)
- B=100: Gate-L1 active=100.0, Budget-Prune active=n/a (matched-budget run only)
- B=200: Gate-L1 active=200.0, Budget-Prune active=n/a (matched-budget run only)
- B=400: Gate-L1 active=400.0, Budget-Prune active=n/a (matched-budget run only)

## Importance Split Ablation (F1 mean over seeds 19/23/29)
- B=100: train=0.7595, val=0.7619
- B=200: train=0.7660, val=0.7656
- B=400: train=0.7664, val=0.7665
- conclusion: train/val importance differences are small; main conclusions unchanged.

## Subset Jaccard Summary
- source: `docs/tables_subset_jaccard_summary.csv`
- note: available in this package for Covtype fair-check runs (Budget-Prune matched budgets; Gate-L1 nominal budgets).

## Covtype Local Decoding (cleaned)
- rows: 5 (one row per concept, no duplicates)
- ka_concept_1.c0: contribution=-0.612899; Horizontal_Distance_To_Hydrology IS LOW ; Horizontal_Distance_To_Roadways IS MEDIUM ; Soil_Type_22 IS absent
- ka_concept_2.c5: contribution=-0.541970; ka_concept_1.c0 IS HIGH ; ka_concept_1.c1 IS LOW ; ka_concept_1.c14 IS LOW
- ka_concept_1.c15: contribution=-0.437065; Horizontal_Distance_To_Hydrology IS LOW ; Horizontal_Distance_To_Roadways IS MEDIUM ; Horizontal_Distance_To_Fire_Points IS LOW
- ka_concept_1.c10: contribution=-0.408783; Horizontal_Distance_To_Roadways IS MEDIUM ; Horizontal_Distance_To_Hydrology IS LOW ; Soil_Type_12 IS absent
- ka_concept_1.c5: contribution=-0.390847; Horizontal_Distance_To_Fire_Points IS LOW ; Horizontal_Distance_To_Hydrology IS LOW ; Soil_Type_29 IS absent

## RuleFit@400 SUSY Status (appendix only if partial)
- status: 6/6
- F1: 0.7318 +/- 0.0019
- elapsed_sec: 15301.2 +/- 2157.3
- policy: complete at this snapshot.

## Partial Results
- No partial RuleFit settings at this snapshot.

## Not Claimed
- no Stability Selection implemented in v16 package
- no theoretical convergence guarantee claimed
- no Compact KAFN runtime superiority claim

## NO-LEAKAGE CHECK
- status: OK
- importance split uses only train/val (never test)
