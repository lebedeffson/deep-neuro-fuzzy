# Covtype Scaling Summary (DFFL Fast-GPU)

- source json: `docs/artifacts/sweeps/covtype_scaling_dffl_fastgpu_s23_2026-04-25.json`
- setup: single-seed (23), fast GPU profile, one-phase, reduced budget for scaling trend

| Dataset | N | F1 | ROC-AUC | LogLoss | Active rules | Total rules |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| covtype_binary_8000 | 8000 | 0.7599 | 0.8324 | 0.5035 | 42.0 | 103.0 |
| covtype_binary_20000 | 20000 | 0.7345 | 0.8027 | 0.5445 | 41.0 | 103.0 |
| covtype_binary_50000 | 50000 | 0.7615 | 0.8178 | 0.5226 | 40.0 | 103.0 |
| covtype_binary_100000 | 100000 | 0.7676 | 0.8488 | 0.4774 | 33.0 | 103.0 |
| covtype_binary_200000 | 200000 | 0.7798 | 0.8548 | 0.4705 | 24.0 | 103.0 |

## Key Deltas
- DFFL F1 at 20k: `0.7345`
- Best external baseline at 20k (known): `0.8546`
- Delta (DFFL - baseline) at 20k: `-0.1201`
- 200k vs 20k (DFFL): `+0.0453` F1

## Quick Read
- In this sweep, best F1 is at `covtype_binary_200000`: `0.7798`.
- Trend from 20k to 200k is positive in this setting, which supports the hypothesis that larger data helps this architecture.
