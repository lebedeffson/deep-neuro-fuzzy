# Q1 Review Gap Status (2026-04-26)

This file maps the strict review points to current repository status.

## 1) Editorial / Submission hygiene

| Item | Status | Evidence |
| --- | --- | --- |
| Unresolved placeholders like `[?]` in paper | done | `main(2).tex`, `docs/article_current/main.tex` checked (none found) |
| Figure labels language consistency | done (generator side) | `docs/figures/generate_iiti26_figures.py` emits English titles/labels |
| Single canonical article bundle | done | `docs/article_current/` (`main.tex`, `references.bib`, figures, metrics JSON) |

## 2) Reproducibility protocol completeness

| Item | Status | Evidence |
| --- | --- | --- |
| Dataset presets for reproducible suites | done | `--dataset-suite` (`paper_main`, `paper_extended`, `paper_all`, `q1_large`, `q1_full`) |
| Deterministic multi-seed runs (10-30) | done | `--seed-count` + internal deterministic seed pool |
| Hardware/runtime capture in artifacts | done | reproducibility manifest now includes Python/Torch/CUDA/device/git hash/command |
| Active-rule criterion explicit and configurable | done | `--rule-probability-threshold` wired into eval + manifest |
| Per-dataset runtime in report | done | `runtime_seconds` in dataset protocol entries |

## 3) Experiments/statistics strength

| Item | Status | Notes |
| --- | --- | --- |
| 10+ seeds on larger suite | in progress | Running `q1_large` 10 seeds on GPU |
| Expanded large benchmark beyond toy datasets | partial | Covtype/SUSY/KDD supported; more external suites still needed |
| Fair capacity-matched comparisons | not done | Need dedicated protocol and runs |
| Stronger significance/effect-size reporting | not done | Need script/report section for CI/effect sizes/CD diagrams |

## 4) DFFL methodological clarity

| Item | Status | Notes |
| --- | --- | --- |
| Full formal definition in paper | partial | Core formulas present; stricter full model formalization still needed |
| Clear DFFL vs hierarchical constraints | partial | Narrative exists; stronger formal contrast still needed |
| Full regularizer definitions (`L_sparse`, `L_len`, etc.) | partial | Implemented behavior exists in code; paper needs explicit formulas/defs |

## 5) Interpretability validation

| Item | Status | Notes |
| --- | --- | --- |
| Structural stability metrics | done | active/decision/layer Jaccard implemented |
| Threshold sensitivity for active-rule criterion | partial | CLI support added; dedicated sweep/report pending |
| Multi-case local explanations | not done | Need TP/TN/FP/FN case set |
| Semantic concept validation | not done | Need concept purity/expert validation mapping |

## 6) Immediate next run package (priority)

1. Finish `q1_large` 10-seed run.
2. Run threshold sensitivity (`tau=0.3,0.5,0.7`) on same suite.
3. Produce one combined markdown table for article update (quality + complexity + stability + runtime).

## Active run now

- command: `python examples/run_real_datasets_benchmark.py --dataset-suite q1_large --seed-count 10 --gpu-only --fuzzy-models all --dffl-profile quality_auto --tune-fuzzy-threshold --rule-probability-threshold 0.5 --output-dir runs/q1_large_all_models_10s_2026-04-26`
- status log: `runs/q1_large_all_models_10s_2026-04-26/run.status.log`
