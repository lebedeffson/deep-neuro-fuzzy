UNIFIED REAL-DATASET BENCHMARK
datasets: covtype_binary_20000
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
dffl_dataset_overrides: {"covtype_binary_20000": {"profile": "quality_large_cls_plus", "skip_builtin_budget_policy": true, "bridge_top_pairs": 24, "bridge_max_rules": 12, "bridge_concepts": 2, "bridge_concepts_max": 3, "bridge_concepts_adaptive": true, "aggregate_block_count": 3, "aggregate_overlap": 2, "aggregate_max_rules": 36, "decision_max_rules": 18, "total_rule_budget": 220, "stagewise_rule_swap_ratio": 0.0, "stagewise_rule_swap_min_keep": 0, "pretrain_refinement_rounds": 1, "bridge_validation_gain_weight": 0.45}}
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
DATASET: covtype_binary_20000
task: binary_classification
seeds: 23

SEED 23
TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis]
  train: accuracy=0.7743, precision=0.7377, recall=0.8337, f1=0.7828, roc_auc=0.8446, pr_auc=0.8088, brier=0.1582, log_loss=0.4867
  test: accuracy=0.7775, precision=0.7381, recall=0.8426, f1=0.7869, roc_auc=0.8529, pr_auc=0.8141, brier=0.1553, log_loss=0.4755
  threshold: 0.41
  structure: total_rules=160.0000, active_rules=80.0000, stages=2.0000, hidden_blocks=41.0000, hidden_concepts=92.0000
  explainability: decision_top1_mass=0.9066, decision_top3_mass=0.9996, decision_entropy=0.3173, hidden_top1_mass=0.6961, hidden_top3_mass=0.9851, hidden_entropy=0.6108

AGGREGATED TABULAR BENCHMARK
ruanfis_refined_deep [ruanfis] over 1 runs
  train: accuracy=0.7743 +/- 0.0000, precision=0.7377 +/- 0.0000, recall=0.8337 +/- 0.0000, f1=0.7828 +/- 0.0000, roc_auc=0.8446 +/- 0.0000, pr_auc=0.8088 +/- 0.0000, brier=0.1582 +/- 0.0000, log_loss=0.4867 +/- 0.0000
  test: accuracy=0.7775 +/- 0.0000, precision=0.7381 +/- 0.0000, recall=0.8426 +/- 0.0000, f1=0.7869 +/- 0.0000, roc_auc=0.8529 +/- 0.0000, pr_auc=0.8141 +/- 0.0000, brier=0.1553 +/- 0.0000, log_loss=0.4755 +/- 0.0000
  structure: total_rules=160.0000 +/- 0.0000, active_rules=80.0000 +/- 0.0000, stages=2.0000 +/- 0.0000, hidden_blocks=41.0000 +/- 0.0000, hidden_concepts=92.0000 +/- 0.0000
  explainability: decision_top1_mass=0.9066 +/- 0.0000, decision_top3_mass=0.9996 +/- 0.0000, decision_entropy=0.3173 +/- 0.0000, hidden_top1_mass=0.6961 +/- 0.0000, hidden_top3_mass=0.9851 +/- 0.0000, hidden_entropy=0.6108 +/- 0.0000
  stability: generated_rule_jaccard=1.0000, active_rule_jaccard=1.0000, hidden_generated_rule_jaccard=1.0000, hidden_active_rule_jaccard=1.0000, decision_generated_rule_jaccard=1.0000, decision_active_rule_jaccard=1.0000, layer_active_rule_jaccard=1.0000, layer_generated_rule_jaccard=1.0000

PAPER-READY SUMMARY TABLE
| model | family | test:accuracy | test:precision | test:recall | test:f1 | test:roc_auc | test:pr_auc | test:brier | test:log_loss | rules:total_rules | rules:active_rules | rules:hidden_blocks | rules:hidden_concepts | expl:decision_top1_mass | expl:decision_top3_mass | expl:decision_entropy | expl:hidden_top1_mass | expl:hidden_top3_mass | expl:hidden_entropy | stab:active_rule_jaccard | stab:decision_active_rule_jaccard | stab:layer_active_rule_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ruanfis_refined_deep | ruanfis | 0.7775 +/- 0.0000 | 0.7381 +/- 0.0000 | 0.8426 +/- 0.0000 | 0.7869 +/- 0.0000 | 0.8529 +/- 0.0000 | 0.8141 +/- 0.0000 | 0.1553 +/- 0.0000 | 0.4755 +/- 0.0000 | 160.0000 +/- 0.0000 | 80.0000 +/- 0.0000 | 41.0000 +/- 0.0000 | 92.0000 +/- 0.0000 | 0.9066 +/- 0.0000 | 0.9996 +/- 0.0000 | 0.3173 +/- 0.0000 | 0.6961 +/- 0.0000 | 0.9851 +/- 0.0000 | 0.6108 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 |

CROSS-DATASET FUZZY SUMMARY
| model | avg_rank | wins | avg_rules (mean+/-std) | avg_active_rule_jaccard (mean+/-std) | covtype_binary_20000:f1 (mean+/-std) |
| --- | --- | --- | --- | --- | --- |
| ruanfis_refined_deep | 1.000 | 1 | 160.00 +/- 0.00 | 1.0000 +/- 0.0000 | 0.7869 +/- 0.0000 |