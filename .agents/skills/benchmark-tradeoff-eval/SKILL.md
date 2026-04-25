---
name: benchmark-tradeoff-eval
description: Use for benchmark/evaluation changes, metric-table updates, model-vs-baseline comparisons, and tuning-policy decisions in this repo.
---

# Benchmark + trade-off evaluation rules (ruanfis)

Use this skill when:
- editing `examples/run_real_datasets_benchmark.py` or evaluation scripts;
- updating paper/result tables from benchmark JSON;
- comparing fuzzy models vs sklearn baselines;
- choosing tuning policy for quality/complexity/stability trade-offs.

## Source-of-truth policy

- Keep one canonical run JSON per report block.
- Build markdown/latex tables from that JSON only.
- Do not mix numbers from different run bundles.
- If unified source is incomplete (missing dataset/run), mark table as provisional.

## Reporting policy

- Report `mean ± std` across seeds.
- For regression: primary metric `RMSE` (lower better).
- For binary classification: primary metric `F1` (higher better).
- Structural metrics (rules/Jaccard) label explicitly as structural stability/interpretability.
- Never state metric improvements without dataset-level values.

## Comparison policy

- Always include best baseline value per dataset in comparison tables.
- Use signed delta vs best baseline with clear sign convention.
- Keep claim language conservative when DFFL is not quality leader.
- If fuzzy models lose on quality, state it directly and shift claim to verified strengths (structure/stability/traceability).

## Sweep strategy (performance-efficient)

1. Targeted small sweep (1 seed, 1-2 datasets).
2. Promote only promising configs to 3 seeds.
3. Promote final candidates to unified full run.

Prefer focused tuning:
- diabetes: DFFL profile/LR.
- covtype/california: stacked/hierarchical and threshold/LR/epochs.
- breast/wine: light threshold/restarts only.

## Reproducibility outputs

For important runs, save:
- `report.json`
- `summary.md`
- `reproducibility_manifest.json` / `.md`

Use `--output-dir` to keep bundle consistent.

## Cleanup and consistency

- Avoid proliferating ad-hoc files in `docs/artifacts/`.
- Keep only interpretable run bundles and summaries needed for decisions/paper.
- For each promoted run, ensure table text and manuscript claims point to the same JSON source.
