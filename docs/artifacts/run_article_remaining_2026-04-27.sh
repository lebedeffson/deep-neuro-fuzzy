#!/usr/bin/env bash
set -euo pipefail
cd /home/lebedeffson/Code/deep-neuro-fuzzy

# 1) Tau sensitivity on paper_main
for TAU in 0.3 0.5 0.7; do
  TAG=${TAU/./}
  OUT="runs/paper_main_fuzzy_tau${TAG}_3s_2026-04-27"
  mkdir -p "${OUT}"
  ./.venv/bin/python examples/run_real_datasets_benchmark.py \
    --dataset-suite paper_main \
    --seeds 19,23,29 \
    --gpu-only \
    --fuzzy-models all \
    --dffl-profile quality_auto \
    --dffl-fast-gpu \
    --max-epochs 8 \
    --pretrain-epochs 3 \
    --decision-pretrain-epochs 2 \
    --refinement-cycles 1 \
    --batch-size 256 \
    --patience 3 \
    --tune-fuzzy-threshold \
    --rule-probability-threshold "${TAU}" \
    --output-dir "${OUT}" \
    > "${OUT}/run.status.log" 2>&1
  echo "tau ${TAU} done -> ${OUT}"
done

# 2) Runtime profile runs on covtype_full (seed 23, one model per run)
run_runtime() {
  local model="$1"
  local out="$2"
  mkdir -p "${out}"
  ./.venv/bin/python examples/run_real_datasets_benchmark.py \
    --datasets covtype_binary_full \
    --seeds 23 \
    --gpu-only \
    --fuzzy-models "${model}" \
    --dffl-profile quality_auto \
    --dffl-fast-gpu \
    --max-epochs 12 \
    --pretrain-epochs 4 \
    --decision-pretrain-epochs 3 \
    --refinement-cycles 1 \
    --batch-size 2048 \
    --patience 4 \
    --tune-fuzzy-threshold \
    --stacked-width-scale 2.14 \
    --stacked-rule-scale 2.14 \
    --hierarchical-width-scale 1.20 \
    --hierarchical-rule-scale 1.20 \
    --rule-probability-threshold 0.5 \
    --output-dir "${out}" \
    > "${out}/run.status.log" 2>&1
}

run_runtime stacked runs/runtime_covtypefull_stacked_capacity_s23_2026-04-27
run_runtime hierarchical runs/runtime_covtypefull_hierarchical_capacity_s23_2026-04-27
run_runtime dffl runs/runtime_covtypefull_dffl_capacity_s23_2026-04-27

# 3) Optional extra large discriminative dataset (SUSY 1M, seed 23)
mkdir -p runs/susy1m_capacity_matched_fuzzy3_s23_2026-04-27
./.venv/bin/python examples/run_real_datasets_benchmark.py \
  --datasets susy_binary_1000000 \
  --seeds 23 \
  --gpu-only \
  --fuzzy-models stacked,hierarchical,dffl \
  --dffl-profile quality_auto \
  --dffl-fast-gpu \
  --max-epochs 12 \
  --pretrain-epochs 4 \
  --decision-pretrain-epochs 3 \
  --refinement-cycles 1 \
  --batch-size 2048 \
  --patience 4 \
  --tune-fuzzy-threshold \
  --stacked-width-scale 2.14 \
  --stacked-rule-scale 2.14 \
  --hierarchical-width-scale 1.20 \
  --hierarchical-rule-scale 1.20 \
  --rule-probability-threshold 0.5 \
  --output-dir runs/susy1m_capacity_matched_fuzzy3_s23_2026-04-27 \
  > runs/susy1m_capacity_matched_fuzzy3_s23_2026-04-27/run.status.log 2>&1

# 4) Build compact support tables
./.venv/bin/python docs/artifacts/build_article_support_tables_2026_04_27.py
