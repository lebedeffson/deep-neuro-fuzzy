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
  --v18-random-state 42 \
  --v18-export-h-artifacts
```

The `--v18-export-h-artifacts` flag additionally writes per-seed KAFN rule activation matrices and full-model probabilities:

- `compact_stable_kafn/paper_tables/v18_controls_raw/v18_h_artifacts_covtype_binary_20000_seed*.npz`
- `compact_stable_kafn/paper_tables/v18_controls_raw/v18_h_rule_index.csv`

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

## 3) H-based Stable Budget-Prune selection check

Run this after step 1 has produced `v18_h_artifacts_*.npz`.

```bash
python compact_stable_kafn/scripts/evaluate_stable_selection_h_artifacts.py \
  --artifact-dir compact_stable_kafn/paper_tables/v18_controls_raw \
  --dataset covtype_binary_20000 \
  --budgets 400 \
  --out-detail docs/tables_stable_h_selection_detail.csv \
  --out-summary docs/tables_stable_h_selection_summary.csv \
  --stability-top-k-multiplier 3 \
  --std-penalty 0.0
```

Outputs:

- `docs/tables_stable_h_selection_detail.csv`
- `docs/tables_stable_h_selection_summary.csv`

Interpretation rule:

- Promote Stable Budget-Prune into the main method only if it improves selected-subset stability while keeping F1 close to Budget-Prune (`drop <= 0.01`) and keeping fidelity gap small.
- Otherwise, keep it as a stability-aware extension/future-work result.
