# Audit Notes

1. KAFN `mean_pairwise_corr` for Budget-Prune (seed=19, B=100) was manually recomputed and matches source table (~0.1424).
2. ANFIS tau=0.95 anomaly is real: with fair thresholding, C-Prune(0.95) has lower meta-jaccard and higher redundancy than C-Prune(0.90).
3. Previous core table mixed thresholds across rows for ANFIS baselines. Use `tables_kafn_anfis_main_table.*` for camera-ready text.
