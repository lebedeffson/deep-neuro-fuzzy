# RUN v18 Controls

## 1) Recompute KAFN full-model controls (Covtype, seeds 19/23/29)

```bash
python examples/run_real_datasets_benchmark.py \
  --dataset-suite custom \
  --datasets covtype_binary_20000 \
  --seeds 19,23,29 \
  --fuzzy-models kanfis \
  --skip-sklearn \
  --max-epochs 80 \
  --batch-size 64 \
  --patience 20 \
  --pretrain-epochs 18 \
  --decision-pretrain-epochs 14 \
  --fuzzy-learning-rate 0.02 \
  --fuzzy-distill-weight 0.15 \
  --fuzzy-distill-models kanfis \
  --fuzzy-distill-teacher-trees 400 \
  --tune-fuzzy-threshold \
  --tune-fuzzy-threshold-calibrated \
  --kanfis-depth 2 \
  --kanfis-superposition-terms 12 \
  --kanfis-concept-fan-in 20 \
  --kanfis-routing grouped \
  --kanfis-feature-order teacher_importance \
  --kanfis-projection-count 8 \
  --kanfis-projection-width 4 \
  --kanfis-projection-mode fixed_covtype \
  --kanfis-prune-rules 936 \
  --kanfis-recovery-epochs 12 \
  --kanfis-polish-epochs 0 \
  --output-dir compact_stable_kafn/runs/v18_controls_covtype_full \
  --v18-controls-dir compact_stable_kafn/paper_tables/v18_controls_raw \
  --v18-lr-budgets 100,200,400 \
  --v18-random-state 42
```

## 2) Build final v18 CSV tables for article

```bash
python compact_stable_kafn/scripts/build_v18_tables.py \
  --controls-dir compact_stable_kafn/paper_tables/v18_controls_raw \
  --docs-dir docs \
  --covtype-runs compact_stable_kafn/runs/compact_stable_kafn_covtype20k \
  --susy-runs compact_stable_kafn/runs/compact_stable_kafn_susy200k_q1 \
  --breast-runs compact_stable_kafn/runs/compact_stable_kafn_breast_q1 \
  --budgets 100,200,400 \
  --seeds 19,23,29
```

Outputs:
- `docs/tables_lr_topk_baseline.csv`
- `docs/tables_rule_activation_correlation.csv`
- `docs/tables_importance_profile.csv`
- `docs/tables_compact_kafn_runtime_minimal.csv`
