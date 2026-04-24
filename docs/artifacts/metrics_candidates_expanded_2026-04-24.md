# Expanded Metrics Candidates (actual, 2026-04-24)

Источник: `docs/artifacts/unified_full_7x3_2026-04-24/*_report.json`

Метрики включены только если они есть у **всех моделей** на данном датасете.

## 1) Overall vs best baseline
- Total dataset-metric pairs: **40**
- Our best fuzzy vs best baseline: **wins 6 / ties 8 / losses 26**

| Result | Count | Share |
|---|---:|---:|
| wins | 6 | 15.0% |
| ties | 8 | 20.0% |
| losses | 26 | 65.0% |

## 2) Coverage by dataset (our best fuzzy vs best baseline)
| Dataset | Wins | Ties | Losses | Coverage |
|---|---:|---:|---:|---:|
| breast_cancer | 0 | 1 | 7 | 0/8 (0.0%) |
| california_housing | 0 | 0 | 4 | 0/4 (0.0%) |
| covtype_binary_20000 | 0 | 0 | 4 | 0/4 (0.0%) |
| diabetes | 0 | 0 | 4 | 0/4 (0.0%) |
| digits_binary | 2 | 6 | 0 | 2/8 (25.0%) |
| linnerud_weight | 4 | 0 | 0 | 4/4 (100.0%) |
| wine_binary | 0 | 1 | 7 | 0/8 (0.0%) |

## 3) By fuzzy model (each model vs same best baseline)
| Model | Wins | Ties | Losses | Win rate |
|---|---:|---:|---:|---:|
| shallow | 0 | 0 | 40 | 0.0% |
| stacked | 4 | 5 | 31 | 10.0% |
| hierarchical | 4 | 3 | 33 | 10.0% |
| DFFL | 5 | 6 | 29 | 12.5% |

## 4) Detailed table (our best fuzzy vs best baseline per metric)
| Dataset | Metric | Dir | Best baseline | Baseline value | Best fuzzy | Fuzzy value | Delta (positive=better fuzzy) | Result |
|---|---|---:|---|---:|---|---:|---:|---|
| breast_cancer | accuracy | max | extra_trees_classifier | 0.9883 | stacked | 0.9795 | -0.0088 | loss |
| breast_cancer | brier | min | mlp_classifier | 0.0153 | stacked | 0.0199 | -0.0046 | loss |
| breast_cancer | f1 | max | extra_trees_classifier | 0.9908 | stacked | 0.9841 | -0.0067 | loss |
| breast_cancer | log_loss | min | mlp_classifier | 0.0588 | stacked | 0.0749 | -0.0161 | loss |
| breast_cancer | pr_auc | max | mlp_classifier | 0.9978 | stacked | 0.9966 | -0.0013 | loss |
| breast_cancer | precision | max | extra_trees_classifier | 0.9819 | stacked | 0.9688 | -0.0131 | loss |
| breast_cancer | recall | max | logistic_regression, extra_trees_classifier, mlp_classifier | 1.0000 | stacked | 1.0000 | +0.0000 | tie |
| breast_cancer | roc_auc | max | mlp_classifier | 0.9965 | stacked | 0.9944 | -0.0021 | loss |
| california_housing | mae | min | hist_gradient_boosting_regressor | 0.0656 | hierarchical | 0.0839 | -0.0183 | loss |
| california_housing | mse | min | hist_gradient_boosting_regressor | 0.0094 | stacked | 0.0143 | -0.0048 | loss |
| california_housing | r2 | max | hist_gradient_boosting_regressor | 0.8344 | stacked | 0.7498 | -0.0846 | loss |
| california_housing | rmse | min | hist_gradient_boosting_regressor | 0.0972 | stacked | 0.1193 | -0.0221 | loss |
| covtype_binary_20000 | accuracy | max | extra_trees_classifier | 0.8546 | hierarchical | 0.8148 | -0.0397 | loss |
| covtype_binary_20000 | f1 | max | extra_trees_classifier | 0.8546 | stacked | 0.8129 | -0.0417 | loss |
| covtype_binary_20000 | precision | max | extra_trees_classifier | 0.8338 | hierarchical | 0.8046 | -0.0292 | loss |
| covtype_binary_20000 | recall | max | extra_trees_classifier | 0.8764 | stacked | 0.8368 | -0.0397 | loss |
| diabetes | mae | min | linear_regression | 0.1445 | DFFL | 0.1447 | -0.0003 | loss |
| diabetes | mse | min | linear_regression | 0.0316 | DFFL | 0.0323 | -0.0006 | loss |
| diabetes | r2 | max | linear_regression | 0.5058 | DFFL | 0.4964 | -0.0094 | loss |
| diabetes | rmse | min | linear_regression | 0.1779 | DFFL | 0.1796 | -0.0017 | loss |
| digits_binary | accuracy | max | mlp_classifier | 1.0000 | DFFL | 1.0000 | +0.0000 | tie |
| digits_binary | brier | min | mlp_classifier | 0.0003 | DFFL | 0.0002 | +0.0001 | win |
| digits_binary | f1 | max | mlp_classifier | 1.0000 | DFFL | 1.0000 | +0.0000 | tie |
| digits_binary | log_loss | min | mlp_classifier | 0.0022 | DFFL | 0.0008 | +0.0014 | win |
| digits_binary | pr_auc | max | logistic_regression, random_forest_classifier, extra_trees_classifier, mlp_classifier | 1.0000 | stacked, hierarchical, DFFL | 1.0000 | +0.0000 | tie |
| digits_binary | precision | max | logistic_regression, random_forest_classifier, extra_trees_classifier, mlp_classifier | 1.0000 | stacked, hierarchical, DFFL | 1.0000 | +0.0000 | tie |
| digits_binary | recall | max | mlp_classifier | 1.0000 | DFFL | 1.0000 | +0.0000 | tie |
| digits_binary | roc_auc | max | logistic_regression, random_forest_classifier, extra_trees_classifier, mlp_classifier | 1.0000 | stacked, hierarchical, DFFL | 1.0000 | +0.0000 | tie |
| linnerud_weight | mae | min | hist_gradient_boosting_regressor | 0.2321 | hierarchical | 0.2004 | +0.0317 | win |
| linnerud_weight | mse | min | hist_gradient_boosting_regressor | 0.1162 | hierarchical | 0.0876 | +0.0286 | win |
| linnerud_weight | r2 | max | hist_gradient_boosting_regressor | -0.7910 | hierarchical | 0.1610 | +0.9521 | win |
| linnerud_weight | rmse | min | hist_gradient_boosting_regressor | 0.3058 | hierarchical | 0.2430 | +0.0628 | win |
| wine_binary | accuracy | max | extra_trees_classifier, mlp_classifier | 1.0000 | stacked | 0.9907 | -0.0093 | loss |
| wine_binary | brier | min | mlp_classifier | 0.0085 | stacked | 0.0108 | -0.0023 | loss |
| wine_binary | f1 | max | extra_trees_classifier, mlp_classifier | 1.0000 | stacked | 0.9855 | -0.0145 | loss |
| wine_binary | log_loss | min | mlp_classifier | 0.0458 | stacked | 0.0527 | -0.0069 | loss |
| wine_binary | pr_auc | max | random_forest_classifier, extra_trees_classifier, mlp_classifier | 1.0000 | DFFL | 0.9921 | -0.0079 | loss |
| wine_binary | precision | max | logistic_regression, random_forest_classifier, extra_trees_classifier, hist_gradient_boosting_classifier, mlp_classifier | 1.0000 | stacked | 1.0000 | +0.0000 | tie |
| wine_binary | recall | max | extra_trees_classifier, mlp_classifier | 1.0000 | stacked | 0.9722 | -0.0278 | loss |
| wine_binary | roc_auc | max | random_forest_classifier, extra_trees_classifier, mlp_classifier | 1.0000 | shallow, DFFL | 0.9954 | -0.0046 | loss |

## 5) Closest losses (smallest absolute gap to baseline)
| Dataset | Metric | Best baseline | Best fuzzy | Delta |
|---|---|---:|---:|---:|
| diabetes | mae | 0.1445 | 0.1447 | -0.0003 |
| diabetes | mse | 0.0316 | 0.0323 | -0.0006 |
| breast_cancer | pr_auc | 0.9978 | 0.9966 | -0.0013 |
| diabetes | rmse | 0.1779 | 0.1796 | -0.0017 |
| breast_cancer | roc_auc | 0.9965 | 0.9944 | -0.0021 |
| wine_binary | brier | 0.0085 | 0.0108 | -0.0023 |
| breast_cancer | brier | 0.0153 | 0.0199 | -0.0046 |
| wine_binary | roc_auc | 1.0000 | 0.9954 | -0.0046 |
| california_housing | mse | 0.0094 | 0.0143 | -0.0048 |
| breast_cancer | f1 | 0.9908 | 0.9841 | -0.0067 |
| wine_binary | log_loss | 0.0458 | 0.0527 | -0.0069 |
| wine_binary | pr_auc | 1.0000 | 0.9921 | -0.0079 |
