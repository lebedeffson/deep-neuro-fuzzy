UNIFIED REAL-DATASET BENCHMARK
datasets: covtype_binary_8000
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
dffl_dataset_overrides: {"covtype_binary_8000": {"profile": "quality_large_cls_plus", "residual_enabled": false, "residual_top_features": 0, "top_k_rules": 24, "total_rule_budget": 160, "stagewise_rule_swap_ratio": 0.0, "stagewise_rule_swap_min_keep": 0}}
dffl_one_phase: True
dffl_fast_gpu: False
gpu_only: False
fuzzy_models: ruanfis_refined_deep
fuzzy_distill_weight: 0.0
fuzzy_distill_models: none
fuzzy_distill_teacher_trees: 400
tune_fuzzy_threshold: True
tune_fuzzy_threshold_calibrated: True
sklearn_baselines: off

DFFL PROFILE USED: quality_large_cls_plus
DATASET: covtype_binary_8000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis]
  train: accuracy=0.7554, precision=0.7045, recall=0.8586, f1=0.7740, roc_auc=0.8382, pr_auc=0.8135, brier=0.1623, log_loss=0.4961
  test: accuracy=0.7606, precision=0.7151, recall=0.8462, f1=0.7751, roc_auc=0.8384, pr_auc=0.8132, brier=0.1609, log_loss=0.4970
  threshold: 0.38
  structure: total_rules=147.0000, active_rules=72.0000, stages=2.0000, hidden_blocks=30.0000, hidden_concepts=57.0000
  explainability: decision_top1_mass=0.4178, decision_top3_mass=0.9787, decision_entropy=1.1571, hidden_top1_mass=0.7109, hidden_top3_mass=0.9783, hidden_entropy=0.6536

AGGREGATED TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis] over 1 runs
  train: accuracy=0.7554 +/- 0.0000, precision=0.7045 +/- 0.0000, recall=0.8586 +/- 0.0000, f1=0.7740 +/- 0.0000, roc_auc=0.8382 +/- 0.0000, pr_auc=0.8135 +/- 0.0000, brier=0.1623 +/- 0.0000, log_loss=0.4961 +/- 0.0000
  test: accuracy=0.7606 +/- 0.0000, precision=0.7151 +/- 0.0000, recall=0.8462 +/- 0.0000, f1=0.7751 +/- 0.0000, roc_auc=0.8384 +/- 0.0000, pr_auc=0.8132 +/- 0.0000, brier=0.1609 +/- 0.0000, log_loss=0.4970 +/- 0.0000
  structure: total_rules=147.0000 +/- 0.0000, active_rules=72.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=30.0000 +/- 0.0000, hidden_concepts=57.0000 +/- 0.0000
  explainability: decision_top1_mass=0.4178 +/- 0.0000, decision_top3_mass=0.9787 +/- 0.0000, decision_entropy=1.1571 +/- 0.0000, hidden_top1_mass=0.7109 +/- 0.0000, hidden_top3_mass=0.9783 +/- 0.0000, hidden_entropy=0.6536 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_refined_deep | ruanfis | 0.7606 +/- 0.0000 | 0.7151 +/- 0.0000 | 0.8462 +/- 0.0000 | 0.7751 +/- 0.0000 | 0.8384 +/- 0.0000 | 0.8132 +/- 0.0000 | 0.1609 +/- 0.0000 | 0.4970 +/- 0.0000 | 147.0000 +/- 0.0000 | 72.0000 +/- 0.0000 | 30.0000 +/- 0.0000 | 57.0000 +/- 0.0000 | 0.4178 +/- 0.0000 | 0.9787 +/- 0.0000 | 1.1571 +/- 0.0000 | 0.7109 +/- 0.0000 | 0.9783 +/- 0.0000 | 0.6536 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

CROSS-DATASET FUZZY SUMMARY
| model | avg_rank | wins | avg_rules (mean+/-std) | avg_active_rule_jaccard (mean+/-std) | covtype_binary_8000:f1 (mean+/-std) |
| --- | --- | --- | --- | --- | --- |
| ruanfis_refined_deep | 1.000 | 1 | 147.00 +/- 0.00 | 1.0000 +/- 0.0000 | 0.7751 +/- 0.0000 |