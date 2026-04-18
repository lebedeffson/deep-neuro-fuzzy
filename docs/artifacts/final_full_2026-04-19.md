# Final Full Comparison (2026-04-19)

Источник сырых артефактов: `artifacts/benchmarks/final_full_2026-04-19` (локально, не в git).

# Final Benchmark Package (2026-04-19)

Source: `pack2_cfgA_auto_lr25_gpu_rtx4060` (full comparison: all RUANFIS models + sklearn baselines).

## california_housing
- Metric: `rmse`
- Best overall: `hist_gradient_boosting_regressor` (sklearn) = `0.0986`
- Best RUANFIS: `ruanfis_stacked_anfis` = `0.1103`

| model | family | value |
| --- | --- | ---: |
| hist_gradient_boosting_regressor | sklearn | 0.0986 |
| extra_trees_regressor | sklearn | 0.1046 |
| random_forest_regressor | sklearn | 0.1064 |
| ruanfis_stacked_anfis | ruanfis | 0.1103 |
| ruanfis_hierarchical_anfis | ruanfis | 0.1153 |
| ruanfis_refined_deep | ruanfis | 0.1273 |
| mlp_regressor | sklearn | 0.1310 |
| linear_regression | sklearn | 0.1484 |
| ruanfis_shallow | ruanfis | 0.1557 |

## covtype_binary_20000
- Metric: `f1`
- Best overall: `extra_trees_classifier` (sklearn) = `0.8561`
- Best RUANFIS: `ruanfis_hierarchical_anfis` = `0.8269`

| model | family | value |
| --- | --- | ---: |
| extra_trees_classifier | sklearn | 0.8561 |
| random_forest_classifier | sklearn | 0.8498 |
| ruanfis_hierarchical_anfis | ruanfis | 0.8269 |
| hist_gradient_boosting_classifier | sklearn | 0.8206 |
| ruanfis_stacked_anfis | ruanfis | 0.8198 |
| mlp_classifier | sklearn | 0.7945 |
| ruanfis_refined_deep | ruanfis | 0.7671 |
| logistic_regression | sklearn | 0.7540 |
| ruanfis_shallow | ruanfis | 0.7408 |

