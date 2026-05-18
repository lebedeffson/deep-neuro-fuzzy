# RuleFit Q1 Partial Summary (snapshot)

- snapshot_generated_at: 2026-05-18T11:15:10
- source: `compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_summary.csv`

## Main Table (for manuscript)

| dataset | budget | n_seeds | F1 mean+/-std | ROC-AUC mean+/-std | PR-AUC mean+/-std | elapsed_sec mean+/-std | status |
|---|---:|---:|---:|---:|---:|---:|---|
| breast_cancer | 100 | 6 | 0.9587+/-0.0147 | 0.9903+/-0.0081 | 0.9937+/-0.0062 | 1.0+/-0.0 | complete |
| breast_cancer | 200 | 6 | 0.9603+/-0.0098 | 0.9894+/-0.0101 | 0.9924+/-0.0092 | 1.2+/-0.0 | complete |
| breast_cancer | 400 | 6 | 0.9578+/-0.0118 | 0.9875+/-0.0108 | 0.9909+/-0.0107 | 1.3+/-0.0 | complete |
| susy_binary_200000 | 100 | 6 | 0.7298+/-0.0016 | 0.8652+/-0.0007 | 0.8703+/-0.0010 | 388.5+/-65.1 | complete |
| susy_binary_200000 | 200 | 6 | 0.7322+/-0.0015 | 0.8663+/-0.0008 | 0.8715+/-0.0010 | 1362.3+/-229.7 | complete |
| susy_binary_200000 | 400 | 6 | 0.7318+/-0.0019 | 0.8661+/-0.0008 | 0.8712+/-0.0010 | 15301.2+/-2157.3 | complete |

## Partial/Appendix Only

No partial RuleFit settings at this snapshot.

## Editorial Notes

1. Main table includes all complete RuleFit settings (`n_seeds=6`).
2. No partial RuleFit settings remain at this snapshot.
3. Do not add new baseline methods before submission; keep scope stable and finalize narrative around current completed settings.
