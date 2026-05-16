# RuleFit Q1 Remote Run (with ETA + checkpoints)

## 1) Update repo
```bash
git pull
```

## 2) Env
```bash
python -m venv .venv_run
source .venv_run/bin/activate
pip install -U pip
pip install -r requirements.txt -r requirements-extra.txt
```

## 3) Run with live ETA/checkpoints

Single command (all parts):
```bash
bash compact_stable_kafn/scripts/run_rulefit_q1_parts.sh
```

Custom python path/output:
```bash
PYTHON_BIN=.venv_run/bin/python OUT_DIR=compact_stable_kafn/paper_tables/interpretable_baselines_q1 \
bash compact_stable_kafn/scripts/run_rulefit_q1_parts.sh
```

## 4) Where to watch progress

- Per-run results (checkpointed):
`compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_per_run.csv`
- ETA/progress log:
`compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_progress.csv`
- Summary (auto-updated):
`compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_summary.csv`

Quick tail:
```bash
tail -n 20 compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_progress.csv
```

## 5) Resume after interruption

Just run the same command again; completed `(dataset, seed, model, budget)` tasks will be skipped automatically.

