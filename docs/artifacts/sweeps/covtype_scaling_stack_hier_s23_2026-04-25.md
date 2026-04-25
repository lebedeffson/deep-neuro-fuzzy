UNIFIED REAL-DATASET BENCHMARK
datasets: covtype_binary_8000, covtype_binary_20000, covtype_binary_50000, covtype_binary_100000, covtype_binary_200000
seeds: 23
train_noise_sigma (regression only): 0.0000
dffl_profile: quality_auto
feature_geometry: euclidean
binary_heavy_grouping: binary_aware
dffl_rule_swap_ratio_override: None
dffl_rule_swap_min_keep_override: None
dffl_adaptive_rule_swap: True
dffl_adaptive_budget: True
dffl_total_rule_budget_override: None
dffl_bridge_score_interaction_weight_override: None
dffl_bridge_score_stability_weight_override: None
dffl_dataset_overrides: {}
dffl_one_phase: False
dffl_fast_gpu: False
gpu_only: False
fuzzy_models: ruanfis_stacked_anfis, ruanfis_hierarchical_anfis
fuzzy_distill_weight: 0.0
fuzzy_distill_models: none
fuzzy_distill_teacher_trees: 400
tune_fuzzy_threshold: False
tune_fuzzy_threshold_calibrated: False
sklearn_baselines: off

DFFL PROFILE USED: quality_large_cls_plus
DATASET: covtype_binary_8000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis]
  train: accuracy=0.8112, precision=0.8078, recall=0.8044, f1=0.8061, roc_auc=0.8843, pr_auc=0.8595, brier=0.1364, log_loss=0.4231
  test: accuracy=0.7925, precision=0.8003, recall=0.7654, f1=0.7824, roc_auc=0.8631, pr_auc=0.8278, brier=0.1481, log_loss=0.4600
  threshold: 0.50
  structure: total_rules=50.0000, active_rules=22.0000, stages=2.0000, hidden_blocks=2.0000, hidden_concepts=11.0000
  explainability: decision_top1_mass=0.6884, decision_top3_mass=0.8456, decision_entropy=0.7802, hidden_top1_mass=0.5396, hidden_top3_mass=0.6992, hidden_entropy=1.3641
ruanfis_hierarchical_anfis [ruanfis]
  train: accuracy=0.7929, precision=0.7850, recall=0.7924, f1=0.7887, roc_auc=0.8701, pr_auc=0.8489, brier=0.1463, log_loss=0.4503
  test: accuracy=0.7850, precision=0.7876, recall=0.7654, f1=0.7763, roc_auc=0.8462, pr_auc=0.8167, brier=0.1572, log_loss=0.4851
  threshold: 0.50
  structure: total_rules=138.0000, active_rules=61.0000, stages=2.0000, hidden_blocks=15.0000, hidden_concepts=34.0000
  explainability: decision_top1_mass=0.7799, decision_top3_mass=0.9726, decision_entropy=0.5393, hidden_top1_mass=0.8242, hidden_top3_mass=0.9559, hidden_entropy=0.4445

AGGREGATED TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis] over 1 runs
  train: accuracy=0.8112 +/- 0.0000, precision=0.8078 +/- 0.0000, recall=0.8044 +/- 0.0000, f1=0.8061 +/- 0.0000, roc_auc=0.8843 +/- 0.0000, pr_auc=0.8595 +/- 0.0000, brier=0.1364 +/- 0.0000, log_loss=0.4231 +/- 0.0000
  test: accuracy=0.7925 +/- 0.0000, precision=0.8003 +/- 0.0000, recall=0.7654 +/- 0.0000, f1=0.7824 +/- 0.0000, roc_auc=0.8631 +/- 0.0000, pr_auc=0.8278 +/- 0.0000, brier=0.1481 +/- 0.0000, log_loss=0.4600 +/- 0.0000
  structure: total_rules=50.0000 +/- 0.0000, active_rules=22.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=2.0000 +/- 0.0000, hidden_concepts=11.0000 +/- 0.0000
  explainability: decision_top1_mass=0.6884 +/- 0.0000, decision_top3_mass=0.8456 +/- 0.0000, decision_entropy=0.7802 +/- 0.0000, hidden_top1_mass=0.5396 +/- 0.0000, hidden_top3_mass=0.6992 +/- 0.0000, hidden_entropy=1.3641 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000
ruanfis_hierarchical_anfis [ruanfis] over 1 runs
  train: accuracy=0.7929 +/- 0.0000, precision=0.7850 +/- 0.0000, recall=0.7924 +/- 0.0000, f1=0.7887 +/- 0.0000, roc_auc=0.8701 +/- 0.0000, pr_auc=0.8489 +/- 0.0000, brier=0.1463 +/- 0.0000, log_loss=0.4503 +/- 0.0000
  test: accuracy=0.7850 +/- 0.0000, precision=0.7876 +/- 0.0000, recall=0.7654 +/- 0.0000, f1=0.7763 +/- 0.0000, roc_auc=0.8462 +/- 0.0000, pr_auc=0.8167 +/- 0.0000, brier=0.1572 +/- 0.0000, log_loss=0.4851 +/- 0.0000
  structure: total_rules=138.0000 +/- 0.0000, active_rules=61.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=15.0000 +/- 0.0000, hidden_concepts=34.0000 +/- 0.0000
  explainability: decision_top1_mass=0.7799 +/- 0.0000, decision_top3_mass=0.9726 +/- 0.0000, decision_entropy=0.5393 +/- 0.0000, hidden_top1_mass=0.8242 +/- 0.0000, hidden_top3_mass=0.9559 +/- 0.0000, hidden_entropy=0.4445 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_stacked_anfis | ruanfis | 0.7925 +/- 0.0000 | 0.8003 +/- 0.0000 | 0.7654 +/- 0.0000 | 0.7824 +/- 0.0000 | 0.8631 +/- 0.0000 | 0.8278 +/- 0.0000 | 0.1481 +/- 0.0000 | 0.4600 +/- 0.0000 | 50.0000 +/- 0.0000 | 22.0000 +/- 0.0000 | 2.0000 +/- 0.0000 | 11.0000 +/- 0.0000 | 0.6884 +/- 0.0000 | 0.8456 +/- 0.0000 | 0.7802 +/- 0.0000 | 0.5396 +/- 0.0000 | 0.6992 +/- 0.0000 | 1.3641 +/- 0.0000 | 1.0000 | 1.0000 |  |
| ruanfis_hierarchical_anfis | ruanfis | 0.7850 +/- 0.0000 | 0.7876 +/- 0.0000 | 0.7654 +/- 0.0000 | 0.7763 +/- 0.0000 | 0.8462 +/- 0.0000 | 0.8167 +/- 0.0000 | 0.1572 +/- 0.0000 | 0.4851 +/- 0.0000 | 138.0000 +/- 0.0000 | 61.0000 +/- 0.0000 | 15.0000 +/- 0.0000 | 34.0000 +/- 0.0000 | 0.7799 +/- 0.0000 | 0.9726 +/- 0.0000 | 0.5393 +/- 0.0000 | 0.8242 +/- 0.0000 | 0.9559 +/- 0.0000 | 0.4445 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

DFFL PROFILE USED: quality_large_cls_plus
DATASET: covtype_binary_20000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis]
  train: accuracy=0.8270, precision=0.8068, recall=0.8484, f1=0.8271, roc_auc=0.9057, pr_auc=0.8905, brier=0.1228, log_loss=0.3861
  test: accuracy=0.8133, precision=0.7902, recall=0.8400, f1=0.8143, roc_auc=0.8857, pr_auc=0.8604, brier=0.1348, log_loss=0.4219
  threshold: 0.50
  structure: total_rules=50.0000, active_rules=16.0000, stages=2.0000, hidden_blocks=2.0000, hidden_concepts=11.0000
  explainability: decision_top1_mass=0.6790, decision_top3_mass=0.8207, decision_entropy=0.8330, hidden_top1_mass=0.3751, hidden_top3_mass=0.6320, hidden_entropy=1.8121
ruanfis_hierarchical_anfis [ruanfis]
  train: accuracy=0.8070, precision=0.7820, recall=0.8378, f1=0.8089, roc_auc=0.8888, pr_auc=0.8722, brier=0.1347, log_loss=0.4184
  test: accuracy=0.8045, precision=0.7794, recall=0.8354, f1=0.8064, roc_auc=0.8828, pr_auc=0.8540, brier=0.1374, log_loss=0.4257
  threshold: 0.50
  structure: total_rules=138.0000, active_rules=57.0000, stages=2.0000, hidden_blocks=15.0000, hidden_concepts=34.0000
  explainability: decision_top1_mass=0.8364, decision_top3_mass=0.9533, decision_entropy=0.4297, hidden_top1_mass=0.7278, hidden_top3_mass=0.9353, hidden_entropy=0.6528

AGGREGATED TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis] over 1 runs
  train: accuracy=0.8270 +/- 0.0000, precision=0.8068 +/- 0.0000, recall=0.8484 +/- 0.0000, f1=0.8271 +/- 0.0000, roc_auc=0.9057 +/- 0.0000, pr_auc=0.8905 +/- 0.0000, brier=0.1228 +/- 0.0000, log_loss=0.3861 +/- 0.0000
  test: accuracy=0.8133 +/- 0.0000, precision=0.7902 +/- 0.0000, recall=0.8400 +/- 0.0000, f1=0.8143 +/- 0.0000, roc_auc=0.8857 +/- 0.0000, pr_auc=0.8604 +/- 0.0000, brier=0.1348 +/- 0.0000, log_loss=0.4219 +/- 0.0000
  structure: total_rules=50.0000 +/- 0.0000, active_rules=16.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=2.0000 +/- 0.0000, hidden_concepts=11.0000 +/- 0.0000
  explainability: decision_top1_mass=0.6790 +/- 0.0000, decision_top3_mass=0.8207 +/- 0.0000, decision_entropy=0.8330 +/- 0.0000, hidden_top1_mass=0.3751 +/- 0.0000, hidden_top3_mass=0.6320 +/- 0.0000, hidden_entropy=1.8121 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000
ruanfis_hierarchical_anfis [ruanfis] over 1 runs
  train: accuracy=0.8070 +/- 0.0000, precision=0.7820 +/- 0.0000, recall=0.8378 +/- 0.0000, f1=0.8089 +/- 0.0000, roc_auc=0.8888 +/- 0.0000, pr_auc=0.8722 +/- 0.0000, brier=0.1347 +/- 0.0000, log_loss=0.4184 +/- 0.0000
  test: accuracy=0.8045 +/- 0.0000, precision=0.7794 +/- 0.0000, recall=0.8354 +/- 0.0000, f1=0.8064 +/- 0.0000, roc_auc=0.8828 +/- 0.0000, pr_auc=0.8540 +/- 0.0000, brier=0.1374 +/- 0.0000, log_loss=0.4257 +/- 0.0000
  structure: total_rules=138.0000 +/- 0.0000, active_rules=57.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=15.0000 +/- 0.0000, hidden_concepts=34.0000 +/- 0.0000
  explainability: decision_top1_mass=0.8364 +/- 0.0000, decision_top3_mass=0.9533 +/- 0.0000, decision_entropy=0.4297 +/- 0.0000, hidden_top1_mass=0.7278 +/- 0.0000, hidden_top3_mass=0.9353 +/- 0.0000, hidden_entropy=0.6528 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_stacked_anfis | ruanfis | 0.8133 +/- 0.0000 | 0.7902 +/- 0.0000 | 0.8400 +/- 0.0000 | 0.8143 +/- 0.0000 | 0.8857 +/- 0.0000 | 0.8604 +/- 0.0000 | 0.1348 +/- 0.0000 | 0.4219 +/- 0.0000 | 50.0000 +/- 0.0000 | 16.0000 +/- 0.0000 | 2.0000 +/- 0.0000 | 11.0000 +/- 0.0000 | 0.6790 +/- 0.0000 | 0.8207 +/- 0.0000 | 0.8330 +/- 0.0000 | 0.3751 +/- 0.0000 | 0.6320 +/- 0.0000 | 1.8121 +/- 0.0000 | 1.0000 | 1.0000 |  |
| ruanfis_hierarchical_anfis | ruanfis | 0.8045 +/- 0.0000 | 0.7794 +/- 0.0000 | 0.8354 +/- 0.0000 | 0.8064 +/- 0.0000 | 0.8828 +/- 0.0000 | 0.8540 +/- 0.0000 | 0.1374 +/- 0.0000 | 0.4257 +/- 0.0000 | 138.0000 +/- 0.0000 | 57.0000 +/- 0.0000 | 15.0000 +/- 0.0000 | 34.0000 +/- 0.0000 | 0.8364 +/- 0.0000 | 0.9533 +/- 0.0000 | 0.4297 +/- 0.0000 | 0.7278 +/- 0.0000 | 0.9353 +/- 0.0000 | 0.6528 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

DFFL PROFILE USED: quality_large_cls_plus
DATASET: covtype_binary_50000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis]
  train: accuracy=0.8230, precision=0.8109, recall=0.8307, f1=0.8207, roc_auc=0.9066, pr_auc=0.8948, brier=0.1233, log_loss=0.3835
  test: accuracy=0.8137, precision=0.7993, recall=0.8251, f1=0.8120, roc_auc=0.8970, pr_auc=0.8820, brier=0.1294, log_loss=0.4077
  threshold: 0.50
  structure: total_rules=50.0000, active_rules=21.0000, stages=2.0000, hidden_blocks=2.0000, hidden_concepts=11.0000
  explainability: decision_top1_mass=0.6416, decision_top3_mass=0.7826, decision_entropy=0.9285, hidden_top1_mass=0.5036, hidden_top3_mass=0.7683, hidden_entropy=1.4152
ruanfis_hierarchical_anfis [ruanfis]
  train: accuracy=0.8221, precision=0.8147, recall=0.8223, f1=0.8185, roc_auc=0.9048, pr_auc=0.8898, brier=0.1242, log_loss=0.3891
  test: accuracy=0.8148, precision=0.8086, recall=0.8126, f1=0.8106, roc_auc=0.8977, pr_auc=0.8816, brier=0.1291, log_loss=0.4034
  threshold: 0.50
  structure: total_rules=138.0000, active_rules=67.0000, stages=2.0000, hidden_blocks=15.0000, hidden_concepts=34.0000
  explainability: decision_top1_mass=0.8411, decision_top3_mass=0.9777, decision_entropy=0.4078, hidden_top1_mass=0.7646, hidden_top3_mass=0.9574, hidden_entropy=0.5306

AGGREGATED TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis] over 1 runs
  train: accuracy=0.8230 +/- 0.0000, precision=0.8109 +/- 0.0000, recall=0.8307 +/- 0.0000, f1=0.8207 +/- 0.0000, roc_auc=0.9066 +/- 0.0000, pr_auc=0.8948 +/- 0.0000, brier=0.1233 +/- 0.0000, log_loss=0.3835 +/- 0.0000
  test: accuracy=0.8137 +/- 0.0000, precision=0.7993 +/- 0.0000, recall=0.8251 +/- 0.0000, f1=0.8120 +/- 0.0000, roc_auc=0.8970 +/- 0.0000, pr_auc=0.8820 +/- 0.0000, brier=0.1294 +/- 0.0000, log_loss=0.4077 +/- 0.0000
  structure: total_rules=50.0000 +/- 0.0000, active_rules=21.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=2.0000 +/- 0.0000, hidden_concepts=11.0000 +/- 0.0000
  explainability: decision_top1_mass=0.6416 +/- 0.0000, decision_top3_mass=0.7826 +/- 0.0000, decision_entropy=0.9285 +/- 0.0000, hidden_top1_mass=0.5036 +/- 0.0000, hidden_top3_mass=0.7683 +/- 0.0000, hidden_entropy=1.4152 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000
ruanfis_hierarchical_anfis [ruanfis] over 1 runs
  train: accuracy=0.8221 +/- 0.0000, precision=0.8147 +/- 0.0000, recall=0.8223 +/- 0.0000, f1=0.8185 +/- 0.0000, roc_auc=0.9048 +/- 0.0000, pr_auc=0.8898 +/- 0.0000, brier=0.1242 +/- 0.0000, log_loss=0.3891 +/- 0.0000
  test: accuracy=0.8148 +/- 0.0000, precision=0.8086 +/- 0.0000, recall=0.8126 +/- 0.0000, f1=0.8106 +/- 0.0000, roc_auc=0.8977 +/- 0.0000, pr_auc=0.8816 +/- 0.0000, brier=0.1291 +/- 0.0000, log_loss=0.4034 +/- 0.0000
  structure: total_rules=138.0000 +/- 0.0000, active_rules=67.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=15.0000 +/- 0.0000, hidden_concepts=34.0000 +/- 0.0000
  explainability: decision_top1_mass=0.8411 +/- 0.0000, decision_top3_mass=0.9777 +/- 0.0000, decision_entropy=0.4078 +/- 0.0000, hidden_top1_mass=0.7646 +/- 0.0000, hidden_top3_mass=0.9574 +/- 0.0000, hidden_entropy=0.5306 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_stacked_anfis | ruanfis | 0.8137 +/- 0.0000 | 0.7993 +/- 0.0000 | 0.8251 +/- 0.0000 | 0.8120 +/- 0.0000 | 0.8970 +/- 0.0000 | 0.8820 +/- 0.0000 | 0.1294 +/- 0.0000 | 0.4077 +/- 0.0000 | 50.0000 +/- 0.0000 | 21.0000 +/- 0.0000 | 2.0000 +/- 0.0000 | 11.0000 +/- 0.0000 | 0.6416 +/- 0.0000 | 0.7826 +/- 0.0000 | 0.9285 +/- 0.0000 | 0.5036 +/- 0.0000 | 0.7683 +/- 0.0000 | 1.4152 +/- 0.0000 | 1.0000 | 1.0000 |  |
| ruanfis_hierarchical_anfis | ruanfis | 0.8148 +/- 0.0000 | 0.8086 +/- 0.0000 | 0.8126 +/- 0.0000 | 0.8106 +/- 0.0000 | 0.8977 +/- 0.0000 | 0.8816 +/- 0.0000 | 0.1291 +/- 0.0000 | 0.4034 +/- 0.0000 | 138.0000 +/- 0.0000 | 67.0000 +/- 0.0000 | 15.0000 +/- 0.0000 | 34.0000 +/- 0.0000 | 0.8411 +/- 0.0000 | 0.9777 +/- 0.0000 | 0.4078 +/- 0.0000 | 0.7646 +/- 0.0000 | 0.9574 +/- 0.0000 | 0.5306 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

DFFL PROFILE USED: quality_large_cls_plus
DATASET: covtype_binary_100000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis]
  train: accuracy=0.8343, precision=0.8153, recall=0.8535, f1=0.8340, roc_auc=0.9183, pr_auc=0.9080, brier=0.1150, log_loss=0.3587
  test: accuracy=0.8252, precision=0.8081, recall=0.8413, f1=0.8244, roc_auc=0.9090, pr_auc=0.8972, brier=0.1216, log_loss=0.3782
  threshold: 0.50
  structure: total_rules=50.0000, active_rules=22.0000, stages=2.0000, hidden_blocks=2.0000, hidden_concepts=11.0000
  explainability: decision_top1_mass=0.7514, decision_top3_mass=0.9191, decision_entropy=0.6074, hidden_top1_mass=0.4363, hidden_top3_mass=0.6927, hidden_entropy=1.6570
ruanfis_hierarchical_anfis [ruanfis]
  train: accuracy=0.8163, precision=0.8005, recall=0.8301, f1=0.8150, roc_auc=0.8966, pr_auc=0.8786, brier=0.1293, log_loss=0.4002
  test: accuracy=0.8081, precision=0.7942, recall=0.8186, f1=0.8062, roc_auc=0.8906, pr_auc=0.8736, brier=0.1335, log_loss=0.4123
  threshold: 0.50
  structure: total_rules=138.0000, active_rules=70.0000, stages=2.0000, hidden_blocks=15.0000, hidden_concepts=34.0000
  explainability: decision_top1_mass=0.7594, decision_top3_mass=0.9555, decision_entropy=0.5787, hidden_top1_mass=0.7179, hidden_top3_mass=0.9727, hidden_entropy=0.5923

AGGREGATED TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis] over 1 runs
  train: accuracy=0.8343 +/- 0.0000, precision=0.8153 +/- 0.0000, recall=0.8535 +/- 0.0000, f1=0.8340 +/- 0.0000, roc_auc=0.9183 +/- 0.0000, pr_auc=0.9080 +/- 0.0000, brier=0.1150 +/- 0.0000, log_loss=0.3587 +/- 0.0000
  test: accuracy=0.8252 +/- 0.0000, precision=0.8081 +/- 0.0000, recall=0.8413 +/- 0.0000, f1=0.8244 +/- 0.0000, roc_auc=0.9090 +/- 0.0000, pr_auc=0.8972 +/- 0.0000, brier=0.1216 +/- 0.0000, log_loss=0.3782 +/- 0.0000
  structure: total_rules=50.0000 +/- 0.0000, active_rules=22.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=2.0000 +/- 0.0000, hidden_concepts=11.0000 +/- 0.0000
  explainability: decision_top1_mass=0.7514 +/- 0.0000, decision_top3_mass=0.9191 +/- 0.0000, decision_entropy=0.6074 +/- 0.0000, hidden_top1_mass=0.4363 +/- 0.0000, hidden_top3_mass=0.6927 +/- 0.0000, hidden_entropy=1.6570 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000
ruanfis_hierarchical_anfis [ruanfis] over 1 runs
  train: accuracy=0.8163 +/- 0.0000, precision=0.8005 +/- 0.0000, recall=0.8301 +/- 0.0000, f1=0.8150 +/- 0.0000, roc_auc=0.8966 +/- 0.0000, pr_auc=0.8786 +/- 0.0000, brier=0.1293 +/- 0.0000, log_loss=0.4002 +/- 0.0000
  test: accuracy=0.8081 +/- 0.0000, precision=0.7942 +/- 0.0000, recall=0.8186 +/- 0.0000, f1=0.8062 +/- 0.0000, roc_auc=0.8906 +/- 0.0000, pr_auc=0.8736 +/- 0.0000, brier=0.1335 +/- 0.0000, log_loss=0.4123 +/- 0.0000
  structure: total_rules=138.0000 +/- 0.0000, active_rules=70.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=15.0000 +/- 0.0000, hidden_concepts=34.0000 +/- 0.0000
  explainability: decision_top1_mass=0.7594 +/- 0.0000, decision_top3_mass=0.9555 +/- 0.0000, decision_entropy=0.5787 +/- 0.0000, hidden_top1_mass=0.7179 +/- 0.0000, hidden_top3_mass=0.9727 +/- 0.0000, hidden_entropy=0.5923 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_stacked_anfis | ruanfis | 0.8252 +/- 0.0000 | 0.8081 +/- 0.0000 | 0.8413 +/- 0.0000 | 0.8244 +/- 0.0000 | 0.9090 +/- 0.0000 | 0.8972 +/- 0.0000 | 0.1216 +/- 0.0000 | 0.3782 +/- 0.0000 | 50.0000 +/- 0.0000 | 22.0000 +/- 0.0000 | 2.0000 +/- 0.0000 | 11.0000 +/- 0.0000 | 0.7514 +/- 0.0000 | 0.9191 +/- 0.0000 | 0.6074 +/- 0.0000 | 0.4363 +/- 0.0000 | 0.6927 +/- 0.0000 | 1.6570 +/- 0.0000 | 1.0000 | 1.0000 |  |
| ruanfis_hierarchical_anfis | ruanfis | 0.8081 +/- 0.0000 | 0.7942 +/- 0.0000 | 0.8186 +/- 0.0000 | 0.8062 +/- 0.0000 | 0.8906 +/- 0.0000 | 0.8736 +/- 0.0000 | 0.1335 +/- 0.0000 | 0.4123 +/- 0.0000 | 138.0000 +/- 0.0000 | 70.0000 +/- 0.0000 | 15.0000 +/- 0.0000 | 34.0000 +/- 0.0000 | 0.7594 +/- 0.0000 | 0.9555 +/- 0.0000 | 0.5787 +/- 0.0000 | 0.7179 +/- 0.0000 | 0.9727 +/- 0.0000 | 0.5923 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

DFFL PROFILE USED: quality_large_cls_plus
DATASET: covtype_binary_200000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis]
  train: accuracy=0.8472, precision=0.8260, recall=0.8699, f1=0.8474, roc_auc=0.9300, pr_auc=0.9211, brier=0.1059, log_loss=0.3321
  test: accuracy=0.8401, precision=0.8169, recall=0.8664, f1=0.8409, roc_auc=0.9234, pr_auc=0.9116, brier=0.1109, log_loss=0.3477
  threshold: 0.50
  structure: total_rules=50.0000, active_rules=22.0000, stages=2.0000, hidden_blocks=2.0000, hidden_concepts=11.0000
  explainability: decision_top1_mass=0.7234, decision_top3_mass=0.9086, decision_entropy=0.7034, hidden_top1_mass=0.4953, hidden_top3_mass=0.6943, hidden_entropy=1.5323
ruanfis_hierarchical_anfis [ruanfis]
  train: accuracy=0.8444, precision=0.8315, recall=0.8540, f1=0.8426, roc_auc=0.9243, pr_auc=0.9143, brier=0.1099, log_loss=0.3469
  test: accuracy=0.8389, precision=0.8228, recall=0.8536, f1=0.8379, roc_auc=0.9190, pr_auc=0.9075, brier=0.1140, log_loss=0.3576
  threshold: 0.50
  structure: total_rules=138.0000, active_rules=61.0000, stages=2.0000, hidden_blocks=15.0000, hidden_concepts=34.0000
  explainability: decision_top1_mass=0.7294, decision_top3_mass=0.9698, decision_entropy=0.6550, hidden_top1_mass=0.8394, hidden_top3_mass=0.9611, hidden_entropy=0.3931

AGGREGATED TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis] over 1 runs
  train: accuracy=0.8472 +/- 0.0000, precision=0.8260 +/- 0.0000, recall=0.8699 +/- 0.0000, f1=0.8474 +/- 0.0000, roc_auc=0.9300 +/- 0.0000, pr_auc=0.9211 +/- 0.0000, brier=0.1059 +/- 0.0000, log_loss=0.3321 +/- 0.0000
  test: accuracy=0.8401 +/- 0.0000, precision=0.8169 +/- 0.0000, recall=0.8664 +/- 0.0000, f1=0.8409 +/- 0.0000, roc_auc=0.9234 +/- 0.0000, pr_auc=0.9116 +/- 0.0000, brier=0.1109 +/- 0.0000, log_loss=0.3477 +/- 0.0000
  structure: total_rules=50.0000 +/- 0.0000, active_rules=22.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=2.0000 +/- 0.0000, hidden_concepts=11.0000 +/- 0.0000
  explainability: decision_top1_mass=0.7234 +/- 0.0000, decision_top3_mass=0.9086 +/- 0.0000, decision_entropy=0.7034 +/- 0.0000, hidden_top1_mass=0.4953 +/- 0.0000, hidden_top3_mass=0.6943 +/- 0.0000, hidden_entropy=1.5323 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000
ruanfis_hierarchical_anfis [ruanfis] over 1 runs
  train: accuracy=0.8444 +/- 0.0000, precision=0.8315 +/- 0.0000, recall=0.8540 +/- 0.0000, f1=0.8426 +/- 0.0000, roc_auc=0.9243 +/- 0.0000, pr_auc=0.9143 +/- 0.0000, brier=0.1099 +/- 0.0000, log_loss=0.3469 +/- 0.0000
  test: accuracy=0.8389 +/- 0.0000, precision=0.8228 +/- 0.0000, recall=0.8536 +/- 0.0000, f1=0.8379 +/- 0.0000, roc_auc=0.9190 +/- 0.0000, pr_auc=0.9075 +/- 0.0000, brier=0.1140 +/- 0.0000, log_loss=0.3576 +/- 0.0000
  structure: total_rules=138.0000 +/- 0.0000, active_rules=61.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=15.0000 +/- 0.0000, hidden_concepts=34.0000 +/- 0.0000
  explainability: decision_top1_mass=0.7294 +/- 0.0000, decision_top3_mass=0.9698 +/- 0.0000, decision_entropy=0.6550 +/- 0.0000, hidden_top1_mass=0.8394 +/- 0.0000, hidden_top3_mass=0.9611 +/- 0.0000, hidden_entropy=0.3931 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_stacked_anfis | ruanfis | 0.8401 +/- 0.0000 | 0.8169 +/- 0.0000 | 0.8664 +/- 0.0000 | 0.8409 +/- 0.0000 | 0.9234 +/- 0.0000 | 0.9116 +/- 0.0000 | 0.1109 +/- 0.0000 | 0.3477 +/- 0.0000 | 50.0000 +/- 0.0000 | 22.0000 +/- 0.0000 | 2.0000 +/- 0.0000 | 11.0000 +/- 0.0000 | 0.7234 +/- 0.0000 | 0.9086 +/- 0.0000 | 0.7034 +/- 0.0000 | 0.4953 +/- 0.0000 | 0.6943 +/- 0.0000 | 1.5323 +/- 0.0000 | 1.0000 | 1.0000 |  |
| ruanfis_hierarchical_anfis | ruanfis | 0.8389 +/- 0.0000 | 0.8228 +/- 0.0000 | 0.8536 +/- 0.0000 | 0.8379 +/- 0.0000 | 0.9190 +/- 0.0000 | 0.9075 +/- 0.0000 | 0.1140 +/- 0.0000 | 0.3576 +/- 0.0000 | 138.0000 +/- 0.0000 | 61.0000 +/- 0.0000 | 15.0000 +/- 0.0000 | 34.0000 +/- 0.0000 | 0.7294 +/- 0.0000 | 0.9698 +/- 0.0000 | 0.6550 +/- 0.0000 | 0.8394 +/- 0.0000 | 0.9611 +/- 0.0000 | 0.3931 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

CROSS-DATASET FUZZY SUMMARY
| model | avg_rank | wins | avg_rules (mean+/-std) | avg_active_rule_jaccard (mean+/-std) | covtype_binary_8000:f1 (mean+/-std) | covtype_binary_20000:f1 (mean+/-std) | covtype_binary_50000:f1 (mean+/-std) | covtype_binary_100000:f1 (mean+/-std) | covtype_binary_200000:f1 (mean+/-std) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_stacked_anfis | 1.000 | 5 | 50.00 +/- 0.00 | 1.0000 +/- 0.0000 | 0.7824 +/- 0.0000 | 0.8143 +/- 0.0000 | 0.8120 +/- 0.0000 | 0.8244 +/- 0.0000 | 0.8409 +/- 0.0000 |
| ruanfis_hierarchical_anfis | 2.000 | 0 | 138.00 +/- 0.00 | 1.0000 +/- 0.0000 | 0.7763 +/- 0.0000 | 0.8064 +/- 0.0000 | 0.8106 +/- 0.0000 | 0.8062 +/- 0.0000 | 0.8379 +/- 0.0000 |