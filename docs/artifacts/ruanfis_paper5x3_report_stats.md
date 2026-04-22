# Statistical Report From Benchmark JSON

- generated_utc: `2026-04-21 22:54:51Z`
- source: `docs/artifacts/ruanfis_paper5x3_report.json`
- model_family_filter: `ruanfis`
- selected_models: `ruanfis_hierarchical_anfis, ruanfis_refined_deep, ruanfis_shallow, ruanfis_stacked_anfis`
- confidence: `0.95`
- alpha: `0.05`
- p_value_method: `permutation`
- n_permutations: `5000`
- bootstrap_samples_for_CI: `1000`
- random_seed: `42`

## Dataset-Level Analysis (aggregated means)

- blocks: `5`; datasets: `5`; models: `4`
- Friedman: `chi2=2.040000`, `p=0.632673`, `reject@0.05=False`

### Rank Summary (with CI)

| model | avg_rank | std_rank | ci_low | ci_high | wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| ruanfis_hierarchical_anfis | 2.0000 | 1.0000 | 1.2000 | 2.8000 | 2 |
| ruanfis_stacked_anfis | 2.2000 | 1.3038 | 1.2000 | 3.2000 | 2 |
| ruanfis_refined_deep | 2.8000 | 0.8367 | 2.2000 | 3.4000 | 0 |
| ruanfis_shallow | 3.0000 | 1.4142 | 1.8000 | 4.0000 | 1 |

### Pairwise Wilcoxon + Holm

| model_a | model_b | n_blocks | wilcoxon_stat | p_value | p_holm | reject@0.05 | better_by_rank |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| ruanfis_hierarchical_anfis | ruanfis_refined_deep | 5 | 2.5000 | 0.312500 | 1.000000 | False | ruanfis_hierarchical_anfis |
| ruanfis_hierarchical_anfis | ruanfis_shallow | 5 | 3.5000 | 0.375000 | 1.000000 | False | ruanfis_hierarchical_anfis |
| ruanfis_hierarchical_anfis | ruanfis_stacked_anfis | 5 | 6.5000 | 1.000000 | 1.000000 | False | ruanfis_hierarchical_anfis |
| ruanfis_refined_deep | ruanfis_shallow | 5 | 6.5000 | 1.000000 | 1.000000 | False | ruanfis_refined_deep |
| ruanfis_refined_deep | ruanfis_stacked_anfis | 5 | 5.0000 | 0.687500 | 1.000000 | False | ruanfis_stacked_anfis |
| ruanfis_shallow | ruanfis_stacked_anfis | 5 | 4.5000 | 0.562500 | 1.000000 | False | ruanfis_stacked_anfis |

## Block-Level Analysis (dataset x seed)

- blocks: `15`; datasets: `5`; models: `4`
- Friedman: `chi2=4.572414`, `p=0.214157`, `reject@0.05=False`

### Rank Summary (with CI)

| model | avg_rank | std_rank | ci_low | ci_high | wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| ruanfis_hierarchical_anfis | 2.1333 | 1.0601 | 1.6333 | 2.6667 | 4 |
| ruanfis_stacked_anfis | 2.1667 | 1.2051 | 1.5667 | 2.7667 | 5 |
| ruanfis_shallow | 2.8333 | 1.0293 | 2.3333 | 3.3333 | 1 |
| ruanfis_refined_deep | 2.8667 | 1.0083 | 2.4000 | 3.3667 | 1 |

### Pairwise Wilcoxon + Holm

| model_a | model_b | n_blocks | wilcoxon_stat | p_value | p_holm | reject@0.05 | better_by_rank |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| ruanfis_hierarchical_anfis | ruanfis_refined_deep | 15 | 21.0000 | 0.085449 | 0.512695 | False | ruanfis_hierarchical_anfis |
| ruanfis_hierarchical_anfis | ruanfis_shallow | 15 | 35.5000 | 0.165222 | 0.745239 | False | ruanfis_hierarchical_anfis |
| ruanfis_hierarchical_anfis | ruanfis_stacked_anfis | 15 | 39.0000 | 0.665771 | 1.000000 | False | ruanfis_hierarchical_anfis |
| ruanfis_refined_deep | ruanfis_shallow | 15 | 57.0000 | 0.911133 | 1.000000 | False | ruanfis_shallow |
| ruanfis_refined_deep | ruanfis_stacked_anfis | 15 | 37.0000 | 0.199097 | 0.745239 | False | ruanfis_stacked_anfis |
| ruanfis_shallow | ruanfis_stacked_anfis | 15 | 29.0000 | 0.149048 | 0.745239 | False | ruanfis_stacked_anfis |
