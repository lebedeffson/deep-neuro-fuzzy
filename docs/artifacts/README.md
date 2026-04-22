# Experiment Artifacts (Current)

This directory contains the current benchmark artifacts used in the paper draft and reproducibility sections.

## Core benchmark bundle (5 datasets, 3 seeds)

- `ruanfis_paper5x3_report.json`
- `ruanfis_paper5x3_report.txt`
- `ruanfis_paper5x3_summary.md`

## DFFL profile ablations

- `ruanfis_ablation_baseline_5x3_report.json`
- `ruanfis_ablation_baseline_5x3_report.txt`
- `ruanfis_ablation_baseline_5x3_summary.md`
- `ruanfis_ablation_quality_balanced_5x3_report.json`
- `ruanfis_ablation_quality_balanced_5x3_report.txt`
- `ruanfis_ablation_quality_balanced_5x3_summary.md`
- `dffl_profile_ablation_5x3.md`

## Large-dataset verification

- `ruanfis_california_3x_gpu_v3_report.json`
- `ruanfis_california_3x_gpu_v3_report.txt`
- `ruanfis_california_3x_gpu_v3_summary.md`

## Large-dataset extension (2 datasets, 3 seeds, expanded baselines)

- `ruanfis_large_real_2d_3s_v1_report.json`
- `ruanfis_large_real_2d_3s_v1_report.txt`
- `ruanfis_large_real_2d_3s_v1_summary.md`

## Cleanup policy

Obsolete temporary GPU logs, intermediate exploratory runs, and ad-hoc local files were removed during repository cleanup to keep only reproducible, article-relevant materials.

## Statistical report generation (Friedman + Wilcoxon + Holm + CI)

Use:

```bash
python examples/generate_article_stats_report.py \
  --input-json docs/artifacts/ruanfis_paper5x3_report.json \
  --model-family ruanfis
```

Outputs are generated next to the source report:

- `<report_stem>_stats.json`
- `<report_stem>_stats.md`

Notes:

- Friedman and Wilcoxon p-values are computed with permutation testing.
- Holm correction is applied to pairwise Wilcoxon p-values.
- CI in rank tables is bootstrap CI for mean rank.

## Reproducibility manifest (full protocol)

The benchmark script can export a full reproducibility manifest (split/seed/preprocessing/budgets/hyperparameters):

```bash
python examples/run_real_datasets_benchmark.py \
  --datasets diabetes,linnerud_weight,breast_cancer,wine_binary,digits_binary \
  --seeds 19,23,29 \
  --output-dir docs/artifacts/latest_run
```

Generated files in `--output-dir`:

- `reproducibility_manifest.md`
- `reproducibility_manifest.json`

## DFFL local-block bottleneck diagnostics (inter-group interaction loss)

Quantifies the limitation “early local decomposition may lose cross-group interactions” with explicit metrics:

- `data_intergroup_interaction_share` (interaction demand in raw features)
- `stage2_intergroup_weighted_share` (how much stage-2 rules actually use inter-group interactions)
- `interaction_recovery_ratio`
- `interaction_loss_index`

Use:

```bash
python examples/diagnose_dffl_intergroup_bottleneck.py \
  --dataset breast_cancer \
  --seed 23 \
  --device cpu \
  --json-output docs/artifacts/dffl_bottleneck_breast_cancer_seed23.json \
  --md-output docs/artifacts/dffl_bottleneck_breast_cancer_seed23.md
```
