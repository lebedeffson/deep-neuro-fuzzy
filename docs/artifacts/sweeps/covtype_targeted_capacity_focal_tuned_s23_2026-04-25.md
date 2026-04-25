UNIFIED REAL-DATASET BENCHMARK
datasets: covtype_binary_20000, covtype_binary_200000
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
tune_fuzzy_threshold: True
tune_fuzzy_threshold_calibrated: True
sklearn_baselines: off

DFFL PROFILE USED: quality_large_cls_plus
DATASET: covtype_binary_20000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis]
  train: accuracy=0.8569, precision=0.7901, recall=0.9622, f1=0.8677, roc_auc=0.9488, pr_auc=0.9423, brier=0.0926, log_loss=0.3008
  test: accuracy=0.8070, precision=0.7454, recall=0.9174, f1=0.8225, roc_auc=0.8960, pr_auc=0.8757, brier=0.1275, log_loss=0.4118
  threshold: 0.43
  structure: total_rules=125.0000, active_rules=56.0000, stages=2.0000, hidden_blocks=2.0000, hidden_concepts=19.0000
  explainability: decision_top1_mass=0.6435, decision_top3_mass=0.8714, decision_entropy=0.9629, hidden_top1_mass=0.2847, hidden_top3_mass=0.5323, hidden_entropy=2.3585
ruanfis_hierarchical_anfis [ruanfis]
  train: accuracy=0.8353, precision=0.7804, recall=0.9217, f1=0.8452, roc_auc=0.9278, pr_auc=0.9221, brier=0.1098, log_loss=0.3475
  test: accuracy=0.7975, precision=0.7448, recall=0.8892, f1=0.8107, roc_auc=0.8868, pr_auc=0.8643, brier=0.1355, log_loss=0.4302
  threshold: 0.40
  structure: total_rules=196.0000, active_rules=91.0000, stages=2.0000, hidden_blocks=10.0000, hidden_concepts=36.0000
  explainability: decision_top1_mass=0.7093, decision_top3_mass=0.9408, decision_entropy=0.7533, hidden_top1_mass=0.6708, hidden_top3_mass=0.8848, hidden_entropy=0.8620

AGGREGATED TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis] over 1 runs
  train: accuracy=0.8569 +/- 0.0000, precision=0.7901 +/- 0.0000, recall=0.9622 +/- 0.0000, f1=0.8677 +/- 0.0000, roc_auc=0.9488 +/- 0.0000, pr_auc=0.9423 +/- 0.0000, brier=0.0926 +/- 0.0000, log_loss=0.3008 +/- 0.0000
  test: accuracy=0.8070 +/- 0.0000, precision=0.7454 +/- 0.0000, recall=0.9174 +/- 0.0000, f1=0.8225 +/- 0.0000, roc_auc=0.8960 +/- 0.0000, pr_auc=0.8757 +/- 0.0000, brier=0.1275 +/- 0.0000, log_loss=0.4118 +/- 0.0000
  structure: total_rules=125.0000 +/- 0.0000, active_rules=56.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=2.0000 +/- 0.0000, hidden_concepts=19.0000 +/- 0.0000
  explainability: decision_top1_mass=0.6435 +/- 0.0000, decision_top3_mass=0.8714 +/- 0.0000, decision_entropy=0.9629 +/- 0.0000, hidden_top1_mass=0.2847 +/- 0.0000, hidden_top3_mass=0.5323 +/- 0.0000, hidden_entropy=2.3585 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000
ruanfis_hierarchical_anfis [ruanfis] over 1 runs
  train: accuracy=0.8353 +/- 0.0000, precision=0.7804 +/- 0.0000, recall=0.9217 +/- 0.0000, f1=0.8452 +/- 0.0000, roc_auc=0.9278 +/- 0.0000, pr_auc=0.9221 +/- 0.0000, brier=0.1098 +/- 0.0000, log_loss=0.3475 +/- 0.0000
  test: accuracy=0.7975 +/- 0.0000, precision=0.7448 +/- 0.0000, recall=0.8892 +/- 0.0000, f1=0.8107 +/- 0.0000, roc_auc=0.8868 +/- 0.0000, pr_auc=0.8643 +/- 0.0000, brier=0.1355 +/- 0.0000, log_loss=0.4302 +/- 0.0000
  structure: total_rules=196.0000 +/- 0.0000, active_rules=91.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=10.0000 +/- 0.0000, hidden_concepts=36.0000 +/- 0.0000
  explainability: decision_top1_mass=0.7093 +/- 0.0000, decision_top3_mass=0.9408 +/- 0.0000, decision_entropy=0.7533 +/- 0.0000, hidden_top1_mass=0.6708 +/- 0.0000, hidden_top3_mass=0.8848 +/- 0.0000, hidden_entropy=0.8620 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_stacked_anfis | ruanfis | 0.8070 +/- 0.0000 | 0.7454 +/- 0.0000 | 0.9174 +/- 0.0000 | 0.8225 +/- 0.0000 | 0.8960 +/- 0.0000 | 0.8757 +/- 0.0000 | 0.1275 +/- 0.0000 | 0.4118 +/- 0.0000 | 125.0000 +/- 0.0000 | 56.0000 +/- 0.0000 | 2.0000 +/- 0.0000 | 19.0000 +/- 0.0000 | 0.6435 +/- 0.0000 | 0.8714 +/- 0.0000 | 0.9629 +/- 0.0000 | 0.2847 +/- 0.0000 | 0.5323 +/- 0.0000 | 2.3585 +/- 0.0000 | 1.0000 | 1.0000 |  |
| ruanfis_hierarchical_anfis | ruanfis | 0.7975 +/- 0.0000 | 0.7448 +/- 0.0000 | 0.8892 +/- 0.0000 | 0.8107 +/- 0.0000 | 0.8868 +/- 0.0000 | 0.8643 +/- 0.0000 | 0.1355 +/- 0.0000 | 0.4302 +/- 0.0000 | 196.0000 +/- 0.0000 | 91.0000 +/- 0.0000 | 10.0000 +/- 0.0000 | 36.0000 +/- 0.0000 | 0.7093 +/- 0.0000 | 0.9408 +/- 0.0000 | 0.7533 +/- 0.0000 | 0.6708 +/- 0.0000 | 0.8848 +/- 0.0000 | 0.8620 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

DFFL PROFILE USED: quality_large_cls_plus
DATASET: covtype_binary_200000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis]
  train: accuracy=0.9228, precision=0.8927, recall=0.9568, f1=0.9236, roc_auc=0.9825, pr_auc=0.9815, brier=0.0530, log_loss=0.1728
  test: accuracy=0.9045, precision=0.8717, recall=0.9429, f1=0.9059, roc_auc=0.9704, pr_auc=0.9671, brier=0.0672, log_loss=0.2185
  threshold: 0.46
  structure: total_rules=125.0000, active_rules=60.0000, stages=2.0000, hidden_blocks=2.0000, hidden_concepts=19.0000
  explainability: decision_top1_mass=0.6063, decision_top3_mass=0.8429, decision_entropy=1.1136, hidden_top1_mass=0.3096, hidden_top3_mass=0.5611, hidden_entropy=2.2466
ruanfis_hierarchical_anfis [ruanfis]
  train: accuracy=0.8793, precision=0.8508, recall=0.9123, f1=0.8805, roc_auc=0.9544, pr_auc=0.9497, brier=0.0849, log_loss=0.2698
  test: accuracy=0.8717, precision=0.8421, recall=0.9069, f1=0.8733, roc_auc=0.9468, pr_auc=0.9400, brier=0.0914, log_loss=0.2903
  threshold: 0.48
  structure: total_rules=196.0000, active_rules=89.0000, stages=2.0000, hidden_blocks=10.0000, hidden_concepts=36.0000
  explainability: decision_top1_mass=0.7244, decision_top3_mass=0.9596, decision_entropy=0.6780, hidden_top1_mass=0.6821, hidden_top3_mass=0.8553, hidden_entropy=0.9071

AGGREGATED TABULAR BENCHMARK
ruanfis_stacked_anfis [ruanfis] over 1 runs
  train: accuracy=0.9228 +/- 0.0000, precision=0.8927 +/- 0.0000, recall=0.9568 +/- 0.0000, f1=0.9236 +/- 0.0000, roc_auc=0.9825 +/- 0.0000, pr_auc=0.9815 +/- 0.0000, brier=0.0530 +/- 0.0000, log_loss=0.1728 +/- 0.0000
  test: accuracy=0.9045 +/- 0.0000, precision=0.8717 +/- 0.0000, recall=0.9429 +/- 0.0000, f1=0.9059 +/- 0.0000, roc_auc=0.9704 +/- 0.0000, pr_auc=0.9671 +/- 0.0000, brier=0.0672 +/- 0.0000, log_loss=0.2185 +/- 0.0000
  structure: total_rules=125.0000 +/- 0.0000, active_rules=60.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=2.0000 +/- 0.0000, hidden_concepts=19.0000 +/- 0.0000
  explainability: decision_top1_mass=0.6063 +/- 0.0000, decision_top3_mass=0.8429 +/- 0.0000, decision_entropy=1.1136 +/- 0.0000, hidden_top1_mass=0.3096 +/- 0.0000, hidden_top3_mass=0.5611 +/- 0.0000, hidden_entropy=2.2466 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000
ruanfis_hierarchical_anfis [ruanfis] over 1 runs
  train: accuracy=0.8793 +/- 0.0000, precision=0.8508 +/- 0.0000, recall=0.9123 +/- 0.0000, f1=0.8805 +/- 0.0000, roc_auc=0.9544 +/- 0.0000, pr_auc=0.9497 +/- 0.0000, brier=0.0849 +/- 0.0000, log_loss=0.2698 +/- 0.0000
  test: accuracy=0.8717 +/- 0.0000, precision=0.8421 +/- 0.0000, recall=0.9069 +/- 0.0000, f1=0.8733 +/- 0.0000, roc_auc=0.9468 +/- 0.0000, pr_auc=0.9400 +/- 0.0000, brier=0.0914 +/- 0.0000, log_loss=0.2903 +/- 0.0000
  structure: total_rules=196.0000 +/- 0.0000, active_rules=89.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=10.0000 +/- 0.0000, hidden_concepts=36.0000 +/- 0.0000
  explainability: decision_top1_mass=0.7244 +/- 0.0000, decision_top3_mass=0.9596 +/- 0.0000, decision_entropy=0.6780 +/- 0.0000, hidden_top1_mass=0.6821 +/- 0.0000, hidden_top3_mass=0.8553 +/- 0.0000, hidden_entropy=0.9071 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_stacked_anfis | ruanfis | 0.9045 +/- 0.0000 | 0.8717 +/- 0.0000 | 0.9429 +/- 0.0000 | 0.9059 +/- 0.0000 | 0.9704 +/- 0.0000 | 0.9671 +/- 0.0000 | 0.0672 +/- 0.0000 | 0.2185 +/- 0.0000 | 125.0000 +/- 0.0000 | 60.0000 +/- 0.0000 | 2.0000 +/- 0.0000 | 19.0000 +/- 0.0000 | 0.6063 +/- 0.0000 | 0.8429 +/- 0.0000 | 1.1136 +/- 0.0000 | 0.3096 +/- 0.0000 | 0.5611 +/- 0.0000 | 2.2466 +/- 0.0000 | 1.0000 | 1.0000 |  |
| ruanfis_hierarchical_anfis | ruanfis | 0.8717 +/- 0.0000 | 0.8421 +/- 0.0000 | 0.9069 +/- 0.0000 | 0.8733 +/- 0.0000 | 0.9468 +/- 0.0000 | 0.9400 +/- 0.0000 | 0.0914 +/- 0.0000 | 0.2903 +/- 0.0000 | 196.0000 +/- 0.0000 | 89.0000 +/- 0.0000 | 10.0000 +/- 0.0000 | 36.0000 +/- 0.0000 | 0.7244 +/- 0.0000 | 0.9596 +/- 0.0000 | 0.6780 +/- 0.0000 | 0.6821 +/- 0.0000 | 0.8553 +/- 0.0000 | 0.9071 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

CROSS-DATASET FUZZY SUMMARY
| model | avg_rank | wins | avg_rules (mean+/-std) | avg_active_rule_jaccard (mean+/-std) | covtype_binary_20000:f1 (mean+/-std) | covtype_binary_200000:f1 (mean+/-std) |
| --- | --- | --- | --- | --- | --- | --- |
| ruanfis_stacked_anfis | 1.000 | 2 | 125.00 +/- 0.00 | 1.0000 +/- 0.0000 | 0.8225 +/- 0.0000 | 0.9059 +/- 0.0000 |
| ruanfis_hierarchical_anfis | 2.000 | 0 | 196.00 +/- 0.00 | 1.0000 +/- 0.0000 | 0.8107 +/- 0.0000 | 0.8733 +/- 0.0000 |