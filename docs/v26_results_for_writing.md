# V26 Results Pack For Writing

## Covtype Main Comparison (F1)

- B=100: Budget-Prune 0.7812 +/- 0.0083; Gate-L1 0.7881 +/- 0.0043; Random-B 0.6671 +/- 0.0184; RuleFit 0.7469 +/- 0.0036.
- B=200: Budget-Prune 0.7917 +/- 0.0104; Gate-L1 0.7916 +/- 0.0028; Random-B 0.6669 +/- 0.0138; RuleFit 0.7456 +/- 0.0093.
- B=400: Budget-Prune 0.8044 +/- 0.0098; Gate-L1 0.7939 +/- 0.0054; Random-B 0.6756 +/- 0.0151; RuleFit 0.7451 +/- 0.0107.

## Runtime Stage-wise (SUSY B=200, n=3)

- train full dictionary (sec): Full=11.93 +/- 1.55 (n=3); Budget-Prune=11.83 +/- 1.55 (n=3); Gate-L1=13.32 +/- 2.50 (n=3).
- H-build/export (sec): Full=10.41 +/- 0.03 (n=3); Budget-Prune=4.06 +/- 0.10 (n=3); Gate-L1=3.96 +/- 0.50 (n=3).
- selection (sec): Full=0.00 +/- 0.00 (n=3); Budget-Prune=0.03 +/- 0.02 (n=3); Gate-L1=0.03 +/- 0.02 (n=3).
- refit head (sec): Full=0.00 +/- 0.00 (n=3); Budget-Prune=2.31 +/- 0.03 (n=3); Gate-L1=2.70 +/- 0.02 (n=3).
- inference (ms/sample): Full=0.0001 +/- 0.0000 (n=3); Budget-Prune=0.0001 +/- 0.0000 (n=3); Gate-L1=0.0001 +/- 0.0000 (n=3).
- memory peak (MB): Full=2717.77 +/- 12.90 (n=3); Budget-Prune=2438.06 +/- 16.46 (n=3); Gate-L1=2443.55 +/- 19.16 (n=3).

## Added Files

- `docs/rulefit_covtype_detail.csv`
- `docs/rulefit_covtype_summary.csv`
- `docs/tables_importance_distribution_summary.csv`
- `docs/tables_importance_rank_profile.csv`
- `docs/figure_importance_elbow.png`
- `docs/figure_importance_histogram.png`
- `docs/tables_l1_lr_full_h_detail.csv`
- `docs/tables_l1_lr_full_h_summary.csv`
- `docs/covtype_if_then_local_explanation.csv`
- `docs/covtype_if_then_local_explanation.md`
