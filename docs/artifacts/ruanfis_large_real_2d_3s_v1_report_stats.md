# Statistical Report From Benchmark JSON

- generated_utc: `2026-04-21 22:55:09Z`
- source: `docs/artifacts/ruanfis_large_real_2d_3s_v1_report.json`
- model_family_filter: `ruanfis`
- selected_models: `ruanfis_hierarchical_anfis, ruanfis_refined_deep, ruanfis_shallow, ruanfis_stacked_anfis`
- confidence: `0.95`
- alpha: `0.05`
- p_value_method: `permutation`
- n_permutations: `3000`
- bootstrap_samples_for_CI: `1000`
- random_seed: `42`

## Dataset-Level Analysis (aggregated means)

- blocks: `2`; datasets: `2`; models: `4`
- Friedman: `chi2=6.000000`, `p=0.038321`, `reject@0.05=True`

### Rank Summary (with CI)

| model | avg_rank | std_rank | ci_low | ci_high | wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| ruanfis_stacked_anfis | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 2 |
| ruanfis_hierarchical_anfis | 2.0000 | 0.0000 | 2.0000 | 2.0000 | 0 |
| ruanfis_refined_deep | 3.0000 | 0.0000 | 3.0000 | 3.0000 | 0 |
| ruanfis_shallow | 4.0000 | 0.0000 | 4.0000 | 4.0000 | 0 |

### Pairwise Wilcoxon + Holm

| model_a | model_b | n_blocks | wilcoxon_stat | p_value | p_holm | reject@0.05 | better_by_rank |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| ruanfis_hierarchical_anfis | ruanfis_refined_deep | 2 | 0.0000 | 0.500000 | 1.000000 | False | ruanfis_hierarchical_anfis |
| ruanfis_hierarchical_anfis | ruanfis_shallow | 2 | 0.0000 | 0.500000 | 1.000000 | False | ruanfis_hierarchical_anfis |
| ruanfis_hierarchical_anfis | ruanfis_stacked_anfis | 2 | 0.0000 | 0.500000 | 1.000000 | False | ruanfis_stacked_anfis |
| ruanfis_refined_deep | ruanfis_shallow | 2 | 0.0000 | 0.500000 | 1.000000 | False | ruanfis_refined_deep |
| ruanfis_refined_deep | ruanfis_stacked_anfis | 2 | 0.0000 | 0.500000 | 1.000000 | False | ruanfis_stacked_anfis |
| ruanfis_shallow | ruanfis_stacked_anfis | 2 | 0.0000 | 0.500000 | 1.000000 | False | ruanfis_stacked_anfis |

## Block-Level Analysis (dataset x seed)

- blocks: `6`; datasets: `2`; models: `4`
- Friedman: `chi2=17.000000`, `p=0.000333`, `reject@0.05=True`

### Rank Summary (with CI)

| model | avg_rank | std_rank | ci_low | ci_high | wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| ruanfis_stacked_anfis | 1.1667 | 0.4082 | 1.0000 | 1.5000 | 5 |
| ruanfis_hierarchical_anfis | 1.8333 | 0.4082 | 1.5000 | 2.0000 | 1 |
| ruanfis_refined_deep | 3.0000 | 0.0000 | 3.0000 | 3.0000 | 0 |
| ruanfis_shallow | 4.0000 | 0.0000 | 4.0000 | 4.0000 | 0 |

### Pairwise Wilcoxon + Holm

| model_a | model_b | n_blocks | wilcoxon_stat | p_value | p_holm | reject@0.05 | better_by_rank |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| ruanfis_hierarchical_anfis | ruanfis_refined_deep | 6 | 0.0000 | 0.031250 | 0.187500 | False | ruanfis_hierarchical_anfis |
| ruanfis_hierarchical_anfis | ruanfis_shallow | 6 | 0.0000 | 0.031250 | 0.187500 | False | ruanfis_hierarchical_anfis |
| ruanfis_hierarchical_anfis | ruanfis_stacked_anfis | 6 | 3.5000 | 0.218750 | 0.218750 | False | ruanfis_stacked_anfis |
| ruanfis_refined_deep | ruanfis_shallow | 6 | 0.0000 | 0.031250 | 0.187500 | False | ruanfis_refined_deep |
| ruanfis_refined_deep | ruanfis_stacked_anfis | 6 | 0.0000 | 0.031250 | 0.187500 | False | ruanfis_stacked_anfis |
| ruanfis_shallow | ruanfis_stacked_anfis | 6 | 0.0000 | 0.031250 | 0.187500 | False | ruanfis_stacked_anfis |
