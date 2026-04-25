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
dffl_one_phase: True
dffl_fast_gpu: True
gpu_only: False
fuzzy_models: ruanfis_refined_deep
fuzzy_distill_weight: 0.0
fuzzy_distill_models: none
fuzzy_distill_teacher_trees: 400
tune_fuzzy_threshold: False
tune_fuzzy_threshold_calibrated: False
sklearn_baselines: off

DFFL PROFILE USED: quality_large_cls_plus_fastgpu
DATASET: covtype_binary_8000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis]
  train: accuracy=0.7483, precision=0.7294, recall=0.7693, f1=0.7489, roc_auc=0.8209, pr_auc=0.7823, brier=0.1721, log_loss=0.5204
  test: accuracy=0.7619, precision=0.7472, recall=0.7731, f1=0.7599, roc_auc=0.8324, pr_auc=0.8079, brier=0.1660, log_loss=0.5035
  threshold: 0.50
  structure: total_rules=103.0000, active_rules=42.0000, stages=2.0000, hidden_blocks=24.0000, hidden_concepts=60.0000
  explainability: decision_top1_mass=0.4310, decision_top3_mass=0.9756, decision_entropy=1.1288, hidden_top1_mass=0.7604, hidden_top3_mass=0.9950, hidden_entropy=0.5191

AGGREGATED TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis] over 1 runs
  train: accuracy=0.7483 +/- 0.0000, precision=0.7294 +/- 0.0000, recall=0.7693 +/- 0.0000, f1=0.7489 +/- 0.0000, roc_auc=0.8209 +/- 0.0000, pr_auc=0.7823 +/- 0.0000, brier=0.1721 +/- 0.0000, log_loss=0.5204 +/- 0.0000
  test: accuracy=0.7619 +/- 0.0000, precision=0.7472 +/- 0.0000, recall=0.7731 +/- 0.0000, f1=0.7599 +/- 0.0000, roc_auc=0.8324 +/- 0.0000, pr_auc=0.8079 +/- 0.0000, brier=0.1660 +/- 0.0000, log_loss=0.5035 +/- 0.0000
  structure: total_rules=103.0000 +/- 0.0000, active_rules=42.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=24.0000 +/- 0.0000, hidden_concepts=60.0000 +/- 0.0000
  explainability: decision_top1_mass=0.4310 +/- 0.0000, decision_top3_mass=0.9756 +/- 0.0000, decision_entropy=1.1288 +/- 0.0000, hidden_top1_mass=0.7604 +/- 0.0000, hidden_top3_mass=0.9950 +/- 0.0000, hidden_entropy=0.5191 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_refined_deep | ruanfis | 0.7619 +/- 0.0000 | 0.7472 +/- 0.0000 | 0.7731 +/- 0.0000 | 0.7599 +/- 0.0000 | 0.8324 +/- 0.0000 | 0.8079 +/- 0.0000 | 0.1660 +/- 0.0000 | 0.5035 +/- 0.0000 | 103.0000 +/- 0.0000 | 42.0000 +/- 0.0000 | 24.0000 +/- 0.0000 | 60.0000 +/- 0.0000 | 0.4310 +/- 0.0000 | 0.9756 +/- 0.0000 | 1.1288 +/- 0.0000 | 0.7604 +/- 0.0000 | 0.9950 +/- 0.0000 | 0.5191 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

DFFL PROFILE USED: quality_large_cls_plus_fastgpu
DATASET: covtype_binary_20000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis]
  train: accuracy=0.7359, precision=0.7205, recall=0.7490, f1=0.7345, roc_auc=0.7978, pr_auc=0.7730, brier=0.1819, log_loss=0.5440
  test: accuracy=0.7377, precision=0.7251, recall=0.7441, f1=0.7345, roc_auc=0.8027, pr_auc=0.7800, brier=0.1806, log_loss=0.5445
  threshold: 0.50
  structure: total_rules=103.0000, active_rules=41.0000, stages=2.0000, hidden_blocks=24.0000, hidden_concepts=60.0000
  explainability: decision_top1_mass=0.7737, decision_top3_mass=0.9940, decision_entropy=0.7013, hidden_top1_mass=0.7932, hidden_top3_mass=0.9994, hidden_entropy=0.4534

AGGREGATED TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis] over 1 runs
  train: accuracy=0.7359 +/- 0.0000, precision=0.7205 +/- 0.0000, recall=0.7490 +/- 0.0000, f1=0.7345 +/- 0.0000, roc_auc=0.7978 +/- 0.0000, pr_auc=0.7730 +/- 0.0000, brier=0.1819 +/- 0.0000, log_loss=0.5440 +/- 0.0000
  test: accuracy=0.7377 +/- 0.0000, precision=0.7251 +/- 0.0000, recall=0.7441 +/- 0.0000, f1=0.7345 +/- 0.0000, roc_auc=0.8027 +/- 0.0000, pr_auc=0.7800 +/- 0.0000, brier=0.1806 +/- 0.0000, log_loss=0.5445 +/- 0.0000
  structure: total_rules=103.0000 +/- 0.0000, active_rules=41.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=24.0000 +/- 0.0000, hidden_concepts=60.0000 +/- 0.0000
  explainability: decision_top1_mass=0.7737 +/- 0.0000, decision_top3_mass=0.9940 +/- 0.0000, decision_entropy=0.7013 +/- 0.0000, hidden_top1_mass=0.7932 +/- 0.0000, hidden_top3_mass=0.9994 +/- 0.0000, hidden_entropy=0.4534 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_refined_deep | ruanfis | 0.7377 +/- 0.0000 | 0.7251 +/- 0.0000 | 0.7441 +/- 0.0000 | 0.7345 +/- 0.0000 | 0.8027 +/- 0.0000 | 0.7800 +/- 0.0000 | 0.1806 +/- 0.0000 | 0.5445 +/- 0.0000 | 103.0000 +/- 0.0000 | 41.0000 +/- 0.0000 | 24.0000 +/- 0.0000 | 60.0000 +/- 0.0000 | 0.7737 +/- 0.0000 | 0.9940 +/- 0.0000 | 0.7013 +/- 0.0000 | 0.7932 +/- 0.0000 | 0.9994 +/- 0.0000 | 0.4534 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

DFFL PROFILE USED: quality_large_cls_plus_fastgpu
DATASET: covtype_binary_50000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis]
  train: accuracy=0.7555, precision=0.7350, recall=0.7797, f1=0.7567, roc_auc=0.8131, pr_auc=0.7607, brier=0.1721, log_loss=0.5230
  test: accuracy=0.7609, precision=0.7412, recall=0.7830, f1=0.7615, roc_auc=0.8178, pr_auc=0.7677, brier=0.1705, log_loss=0.5226
  threshold: 0.50
  structure: total_rules=103.0000, active_rules=40.0000, stages=2.0000, hidden_blocks=24.0000, hidden_concepts=60.0000
  explainability: decision_top1_mass=0.8771, decision_top3_mass=0.9995, decision_entropy=0.4347, hidden_top1_mass=0.7891, hidden_top3_mass=0.9991, hidden_entropy=0.4711

AGGREGATED TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis] over 1 runs
  train: accuracy=0.7555 +/- 0.0000, precision=0.7350 +/- 0.0000, recall=0.7797 +/- 0.0000, f1=0.7567 +/- 0.0000, roc_auc=0.8131 +/- 0.0000, pr_auc=0.7607 +/- 0.0000, brier=0.1721 +/- 0.0000, log_loss=0.5230 +/- 0.0000
  test: accuracy=0.7609 +/- 0.0000, precision=0.7412 +/- 0.0000, recall=0.7830 +/- 0.0000, f1=0.7615 +/- 0.0000, roc_auc=0.8178 +/- 0.0000, pr_auc=0.7677 +/- 0.0000, brier=0.1705 +/- 0.0000, log_loss=0.5226 +/- 0.0000
  structure: total_rules=103.0000 +/- 0.0000, active_rules=40.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=24.0000 +/- 0.0000, hidden_concepts=60.0000 +/- 0.0000
  explainability: decision_top1_mass=0.8771 +/- 0.0000, decision_top3_mass=0.9995 +/- 0.0000, decision_entropy=0.4347 +/- 0.0000, hidden_top1_mass=0.7891 +/- 0.0000, hidden_top3_mass=0.9991 +/- 0.0000, hidden_entropy=0.4711 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_refined_deep | ruanfis | 0.7609 +/- 0.0000 | 0.7412 +/- 0.0000 | 0.7830 +/- 0.0000 | 0.7615 +/- 0.0000 | 0.8178 +/- 0.0000 | 0.7677 +/- 0.0000 | 0.1705 +/- 0.0000 | 0.5226 +/- 0.0000 | 103.0000 +/- 0.0000 | 40.0000 +/- 0.0000 | 24.0000 +/- 0.0000 | 60.0000 +/- 0.0000 | 0.8771 +/- 0.0000 | 0.9995 +/- 0.0000 | 0.4347 +/- 0.0000 | 0.7891 +/- 0.0000 | 0.9991 +/- 0.0000 | 0.4711 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

DFFL PROFILE USED: quality_large_cls_plus_fastgpu
DATASET: covtype_binary_100000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis]
  train: accuracy=0.7715, precision=0.7684, recall=0.7606, f1=0.7645, roc_auc=0.8470, pr_auc=0.8145, brier=0.1577, log_loss=0.4799
  test: accuracy=0.7742, precision=0.7704, recall=0.7648, f1=0.7676, roc_auc=0.8488, pr_auc=0.8185, brier=0.1568, log_loss=0.4774
  threshold: 0.50
  structure: total_rules=103.0000, active_rules=33.0000, stages=2.0000, hidden_blocks=24.0000, hidden_concepts=60.0000
  explainability: decision_top1_mass=0.6696, decision_top3_mass=0.9947, decision_entropy=0.7827, hidden_top1_mass=0.8396, hidden_top3_mass=0.9997, hidden_entropy=0.3802

AGGREGATED TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis] over 1 runs
  train: accuracy=0.7715 +/- 0.0000, precision=0.7684 +/- 0.0000, recall=0.7606 +/- 0.0000, f1=0.7645 +/- 0.0000, roc_auc=0.8470 +/- 0.0000, pr_auc=0.8145 +/- 0.0000, brier=0.1577 +/- 0.0000, log_loss=0.4799 +/- 0.0000
  test: accuracy=0.7742 +/- 0.0000, precision=0.7704 +/- 0.0000, recall=0.7648 +/- 0.0000, f1=0.7676 +/- 0.0000, roc_auc=0.8488 +/- 0.0000, pr_auc=0.8185 +/- 0.0000, brier=0.1568 +/- 0.0000, log_loss=0.4774 +/- 0.0000
  structure: total_rules=103.0000 +/- 0.0000, active_rules=33.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=24.0000 +/- 0.0000, hidden_concepts=60.0000 +/- 0.0000
  explainability: decision_top1_mass=0.6696 +/- 0.0000, decision_top3_mass=0.9947 +/- 0.0000, decision_entropy=0.7827 +/- 0.0000, hidden_top1_mass=0.8396 +/- 0.0000, hidden_top3_mass=0.9997 +/- 0.0000, hidden_entropy=0.3802 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_refined_deep | ruanfis | 0.7742 +/- 0.0000 | 0.7704 +/- 0.0000 | 0.7648 +/- 0.0000 | 0.7676 +/- 0.0000 | 0.8488 +/- 0.0000 | 0.8185 +/- 0.0000 | 0.1568 +/- 0.0000 | 0.4774 +/- 0.0000 | 103.0000 +/- 0.0000 | 33.0000 +/- 0.0000 | 24.0000 +/- 0.0000 | 60.0000 +/- 0.0000 | 0.6696 +/- 0.0000 | 0.9947 +/- 0.0000 | 0.7827 +/- 0.0000 | 0.8396 +/- 0.0000 | 0.9997 +/- 0.0000 | 0.3802 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

DFFL PROFILE USED: quality_large_cls_plus_fastgpu
DATASET: covtype_binary_200000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis]
  train: accuracy=0.7799, precision=0.7585, recall=0.8049, f1=0.7810, roc_auc=0.8564, pr_auc=0.8277, brier=0.1530, log_loss=0.4672
  test: accuracy=0.7777, precision=0.7542, recall=0.8072, f1=0.7798, roc_auc=0.8548, pr_auc=0.8244, brier=0.1541, log_loss=0.4705
  threshold: 0.50
  structure: total_rules=103.0000, active_rules=24.0000, stages=2.0000, hidden_blocks=24.0000, hidden_concepts=60.0000
  explainability: decision_top1_mass=0.5089, decision_top3_mass=0.9995, decision_entropy=1.0356, hidden_top1_mass=0.7972, hidden_top3_mass=0.9941, hidden_entropy=0.4365

AGGREGATED TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis] over 1 runs
  train: accuracy=0.7799 +/- 0.0000, precision=0.7585 +/- 0.0000, recall=0.8049 +/- 0.0000, f1=0.7810 +/- 0.0000, roc_auc=0.8564 +/- 0.0000, pr_auc=0.8277 +/- 0.0000, brier=0.1530 +/- 0.0000, log_loss=0.4672 +/- 0.0000
  test: accuracy=0.7777 +/- 0.0000, precision=0.7542 +/- 0.0000, recall=0.8072 +/- 0.0000, f1=0.7798 +/- 0.0000, roc_auc=0.8548 +/- 0.0000, pr_auc=0.8244 +/- 0.0000, brier=0.1541 +/- 0.0000, log_loss=0.4705 +/- 0.0000
  structure: total_rules=103.0000 +/- 0.0000, active_rules=24.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=24.0000 +/- 0.0000, hidden_concepts=60.0000 +/- 0.0000
  explainability: decision_top1_mass=0.5089 +/- 0.0000, decision_top3_mass=0.9995 +/- 0.0000, decision_entropy=1.0356 +/- 0.0000, hidden_top1_mass=0.7972 +/- 0.0000, hidden_top3_mass=0.9941 +/- 0.0000, hidden_entropy=0.4365 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_refined_deep | ruanfis | 0.7777 +/- 0.0000 | 0.7542 +/- 0.0000 | 0.8072 +/- 0.0000 | 0.7798 +/- 0.0000 | 0.8548 +/- 0.0000 | 0.8244 +/- 0.0000 | 0.1541 +/- 0.0000 | 0.4705 +/- 0.0000 | 103.0000 +/- 0.0000 | 24.0000 +/- 0.0000 | 24.0000 +/- 0.0000 | 60.0000 +/- 0.0000 | 0.5089 +/- 0.0000 | 0.9995 +/- 0.0000 | 1.0356 +/- 0.0000 | 0.7972 +/- 0.0000 | 0.9941 +/- 0.0000 | 0.4365 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

CROSS-DATASET FUZZY SUMMARY
| model | avg_rank | wins | avg_rules (mean+/-std) | avg_active_rule_jaccard (mean+/-std) | covtype_binary_8000:f1 (mean+/-std) | covtype_binary_20000:f1 (mean+/-std) | covtype_binary_50000:f1 (mean+/-std) | covtype_binary_100000:f1 (mean+/-std) | covtype_binary_200000:f1 (mean+/-std) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_refined_deep | 1.000 | 5 | 103.00 +/- 0.00 | 1.0000 +/- 0.0000 | 0.7599 +/- 0.0000 | 0.7345 +/- 0.0000 | 0.7615 +/- 0.0000 | 0.7676 +/- 0.0000 | 0.7798 +/- 0.0000 |