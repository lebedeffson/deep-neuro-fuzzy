# Covtype Scaling: DFFL vs Stacked vs Hierarchical

- DFFL source: `covtype_scaling_dffl_fastgpu_s23_2026-04-25.json`
- Stacked/Hier source: `covtype_scaling_stack_hier_s23_2026-04-25.json`
- setup: seed=23; DFFL in fast-gpu one-phase reduced-budget mode; stacked/hier with same epoch budget

| Dataset | N | DFFL F1 | Stacked F1 | Hier F1 | Best Fuzzy | Best External Baseline (if known) | Delta BestFuzzy-Baseline |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| covtype_binary_8000 | 8000 | 0.7599 | 0.7824 | 0.7763 | Stacked (0.7824) | n/a | n/a |
| covtype_binary_20000 | 20000 | 0.7345 | 0.8143 | 0.8064 | Stacked (0.8143) | 0.8546 | -0.0403 |
| covtype_binary_50000 | 50000 | 0.7615 | 0.8120 | 0.8106 | Stacked (0.8120) | n/a | n/a |
| covtype_binary_100000 | 100000 | 0.7676 | 0.8244 | 0.8062 | Stacked (0.8244) | n/a | n/a |
| covtype_binary_200000 | 200000 | 0.7798 | 0.8409 | 0.8379 | Stacked (0.8409) | n/a | n/a |

## Trend Notes
- DFFL: 20k -> 200k: 0.7345 -> 0.7798 (+0.0453)
- Stacked: 20k -> 200k: 0.8143 -> 0.8409 (+0.0266)
- Hierarchical: 20k -> 200k: 0.8064 -> 0.8379 (+0.0315)

## Structural Snapshot (Rules)
| Dataset | DFFL rules/active | Stacked rules/active | Hier rules/active |
| --- | ---: | ---: | ---: |
| covtype_binary_8000 | 103/42 | 50/22 | 138/61 |
| covtype_binary_20000 | 103/41 | 50/16 | 138/57 |
| covtype_binary_50000 | 103/40 | 50/21 | 138/67 |
| covtype_binary_100000 | 103/33 | 50/22 | 138/70 |
| covtype_binary_200000 | 103/24 | 50/22 | 138/61 |
