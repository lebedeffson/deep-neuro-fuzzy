# DFFL Local-Block Bottleneck Diagnostics

- dataset: `breast_cancer`
- seed: `23`
- task_type: `binary_classification`
- dffl_profile: `quality`
- metric: `f1`

## Core indicators

- baseline_metric_test: `0.9796`
- data_intergroup_interaction_share: `0.8742`
- stage2_intergroup_weighted_share: `0.9392`
- interaction_recovery_ratio: `1.0743`
- interaction_loss_index: `0.0000`

## Stage-2 block stats

| block | n_rules | inter_rules | inter_ratio | weighted_inter_share | dominant_inter_share | active_rules | active_inter_rules | active_inter_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dffl_aggregate_0 | 20 | 18 | 0.9000 | 1.0000 | 1.0000 | 3 | 3 | 1.0000 |
| dffl_aggregate_global | 8 | 6 | 0.7500 | 0.8785 | 0.7544 | 3 | 2 | 0.6667 |

## Top inter-group pairs in data

- `mean concave points x worst concave points` (x7, x27): 0.3730
- `worst perimeter x worst concave points` (x22, x27): 0.3633
- `worst radius x worst concave points` (x20, x27): 0.3629
- `mean concave points x worst radius` (x7, x20): 0.3560
- `mean concave points x worst perimeter` (x7, x22): 0.3543
- `mean radius x worst radius` (x0, x20): 0.3538
- `mean perimeter x worst radius` (x2, x20): 0.3521
- `mean radius x worst perimeter` (x0, x22): 0.3478
- `mean perimeter x worst perimeter` (x2, x22): 0.3467
- `mean area x worst radius` (x3, x20): 0.3413

## Top intra-group pairs in data

- `worst radius x worst perimeter` (x20, x22): 0.3663
- `worst radius x worst area` (x20, x23): 0.3498
- `worst perimeter x worst area` (x22, x23): 0.3430
- `mean concavity x mean concave points` (x6, x7): 0.3373
- `mean radius x mean perimeter` (x0, x2): 0.3281
- `mean radius x mean area` (x0, x3): 0.3228
- `worst concavity x worst concave points` (x26, x27): 0.3215
- `mean perimeter x mean area` (x2, x3): 0.3192

## Interpretation

Высокий `data_intergroup_interaction_share` при умеренном/низком `stage2_intergroup_weighted_share` указывает на bottleneck: ранняя локальная декомпозиция теряет часть межгрупповых зависимостей и поздняя агрегация восстанавливает их неполностью.
