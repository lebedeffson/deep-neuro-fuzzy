# Covtype: architecture diagnosis (2026-04-25)

## Latest targeted results (seed=23)

| Model | Dataset | Previous F1 | Targeted F1 | Delta |
|---|---:|---:|---:|---:|
| stacked | covtype_binary_20000 | 0.8143 | 0.8225 | +0.0082 |
| hierarchical | covtype_binary_20000 | 0.8064 | 0.8107 | +0.0043 |
| stacked | covtype_binary_200000 | 0.8409 | 0.9059 | +0.0650 |
| hierarchical | covtype_binary_200000 | 0.8379 | 0.8733 | +0.0354 |
| DFFL (fastgpu) | covtype_binary_20000 | 0.7345 | 0.7345 | +0.0000 |
| DFFL (one-phase bridge+) | covtype_binary_20000 | 0.7345 | 0.7869 | +0.0524 |

Reference baseline from paper table on covtype_binary_20000: **0.8546** (Extra Trees).

## What is likely wrong

1. Capacity bottleneck on high-dimensional binary-heavy data.
- Default stacked/hier configs are too compact for covtype; larger width/rule budgets improved F1.

2. Threshold/calibration mismatch.
- Best thresholds moved away from 0.5 (e.g., 0.43/0.40/0.46/0.48), so fixed threshold hurts F1.

3. DFFL remains constrained by early decomposition.
- Even with stronger bridge profile, DFFL stays below stacked on covtype_20000.

4. Found and fixed a real robustness bug.
- In DFFL config generation, aggregate block count could exceed effective rule budget under high-dim caps, producing `max_rules=0` for some blocks.
- Fix applied in `examples/run_real_datasets_benchmark.py`.

## Command profile that improved stacked/hier

- focal loss + auto pos weight + soft F1
- calibrated threshold tuning
- larger width/rule scales
- longer training budget

(See artifact: `covtype_targeted_capacity_focal_tuned_s23_2026-04-25.json`.)
