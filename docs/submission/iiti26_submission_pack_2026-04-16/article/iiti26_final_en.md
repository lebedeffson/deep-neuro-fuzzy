# Deep Neuro-Fuzzy Architectures for Interpretable Learning

## Abstract
This paper presents a controlled comparison of three deep neuro-fuzzy architecture lines within a single reproducible pipeline: Stacked ANFIS, Hierarchical ANFIS, and Deep Fuzzy Feature Learning. The study targets a practical objective that combines predictive performance with transparent internal reasoning and stable rule behavior across repeated runs. Experiments are conducted on a core set of five real tabular datasets and then extended with a large-data block on `california_housing` (`20640` samples) and `covtype_binary_20000` (`20000` samples) with `seed = 19, 23, 29` and expanded classical baselines. In the fuzzy-model group, Hierarchical ANFIS and Stacked ANFIS lead in quality rank, while DFFL remains competitive on selected tasks and shows stronger active-rule stability among deep alternatives.

## Concise Findings
The key result is that architecture selection cannot be reduced to one quality metric. On the main `paper5x3` benchmark, average quality ranks are `2.000` for Hierarchical ANFIS, `2.200` for Stacked ANFIS, `2.800` for DFFL, and `3.000` for Shallow Fuzzy. On the extended combined setup with `7` datasets (`5` core + `2` large), average ranks become `1.857` for Stacked ANFIS, `2.048` for Hierarchical ANFIS, `2.952` for DFFL, and `3.143` for Shallow Fuzzy.

In the large-data block, Stacked ANFIS is the strongest fuzzy architecture on both datasets (`RMSE = 0.1122` on `california_housing`, `F1 = 0.8129` on `covtype_binary_20000`). The best classical baselines are `hist_gradient_boosting_regressor` (`RMSE = 0.0972`) and `extra_trees_classifier` (`F1 = 0.8546`).

At the combined block level (`dataset x seed`, `N = 21`), Friedman test gives `chi2 = 15.57` and `p = 0.00139`, which substantially improves statistical support for architecture-level differences in the extended study.

## Conclusion
The practical conclusion is methodological: deep neuro-fuzzy architecture selection should be treated as a multi-objective decision over quality, rule-base complexity, and explanation stability.

## Reproducibility
Core results: `docs/artifacts/ruanfis_paper5x3_report.json`. DFFL ablations: `docs/artifacts/ruanfis_ablation_baseline_5x3_report.json` and `docs/artifacts/ruanfis_ablation_quality_balanced_5x3_report.json`. Extended large-data block: `docs/artifacts/ruanfis_large_real_2d_3s_v1_report.json`.
