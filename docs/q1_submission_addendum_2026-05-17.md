# Q1 Submission Addendum (targeted fixes)

- generated_at: 2026-05-17T11:15:51
- summary_source: `compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_summary.csv`
- concept_source: `docs/case_level_top5_concepts_export_seed23.csv`

## A. RuleFit main table (only complete settings)

Use this table in the main text. Keep partial settings out of the main comparison table.

| dataset | budget | n_seeds | F1 mean+/-std | ROC-AUC mean+/-std | PR-AUC mean+/-std | elapsed_sec mean+/-std |
|---|---:|---:|---:|---:|---:|---:|
| breast_cancer | 100 | 6 | 0.9587+/-0.0147 | 0.9903+/-0.0081 | 0.9937+/-0.0062 | 1.0+/-0.0 |
| breast_cancer | 200 | 6 | 0.9603+/-0.0098 | 0.9894+/-0.0101 | 0.9924+/-0.0092 | 1.2+/-0.0 |
| breast_cancer | 400 | 6 | 0.9578+/-0.0118 | 0.9875+/-0.0108 | 0.9909+/-0.0107 | 1.3+/-0.0 |
| susy_binary_200000 | 100 | 6 | 0.7298+/-0.0016 | 0.8652+/-0.0007 | 0.8703+/-0.0010 | 388.5+/-65.1 |
| susy_binary_200000 | 200 | 6 | 0.7322+/-0.0015 | 0.8663+/-0.0008 | 0.8715+/-0.0010 | 1362.3+/-229.7 |
| susy_binary_200000 | 400 | 6 | 0.7318+/-0.0019 | 0.8661+/-0.0008 | 0.8712+/-0.0010 | 15301.2+/-2157.3 |

## B. Partial settings for appendix/footnote

No partial RuleFit settings remain in this snapshot.

## C. Local concept decoding (for Table 5 style explanation)

One-object case-level explanation, top-5 concepts, with human-readable predicates.

| concept_name | top_predicate_1 | top_predicate_2 |
|---|---|---|
| s1_8 | mean fractal dimension [x9] IS mid | radius error [x10] IS low |
| s1_18 | worst smoothness [x24] IS high | worst concave points [x27] IS mid |
| s1_19 | worst smoothness [x24] IS high | worst concave points [x27] IS mid |
| s1_21 | worst fractal dimension [x29] IS mid | worst symmetry [x28] IS mid |
| s1_27 | mean radius [x0] IS high | worst perimeter [x22] IS mid |

## D. Ready-to-paste reviewer-facing notes

1. RuleFit on SUSY uses complete evidence at B=100, B=200, and B=400 (6 seeds each).
2. SUSY B=400 is now complete and included in the main comparative table.
3. Local explanation now includes decoded concept predicates (human-readable feature/term form).
4. No new methods were added pre-submission, preserving methodological scope and comparability.
