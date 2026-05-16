#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv_run/bin/python}"
OUT_DIR="${OUT_DIR:-compact_stable_kafn/paper_tables/interpretable_baselines_q1}"
SEEDS="${SEEDS:-7,19,23,29,42,101}"
BUDGETS="${BUDGETS:-100,200,400}"

run_part() {
  local dataset="$1"
  local budget="$2"
  echo "[part] dataset=${dataset} budget=${budget} start $(date '+%F %T')"
  "$PYTHON_BIN" compact_stable_kafn/scripts/run_interpretable_baselines.py \
    --datasets "$dataset" \
    --seeds "$SEEDS" \
    --models rulefit \
    --rulefit-budgets "$budget" \
    --resume \
    --checkpoint-every-run \
    --out-dir "$OUT_DIR"
  echo "[part] dataset=${dataset} budget=${budget} done  $(date '+%F %T')"
}

for ds in breast_cancer susy_binary_200000; do
  IFS=',' read -ra b_arr <<< "$BUDGETS"
  for b in "${b_arr[@]}"; do
    run_part "$ds" "$b"
  done
done

echo "[ok] all parts finished"
