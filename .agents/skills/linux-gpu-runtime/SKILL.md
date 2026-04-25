---
name: linux-gpu-runtime
description: Use for Linux CUDA runtime behavior, GPU utilization/debugging, and fast benchmark execution in this repo.
---

# Linux + CUDA runtime rules (ruanfis)

Use this skill when:
- GPU is not utilized or runs are unexpectedly slow;
- configuring device/runtime flags for benchmark scripts;
- planning fast local sweeps on CUDA.

## Quick environment checks

```bash
.venv_run/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
.venv_run/bin/python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
nvidia-smi
```

## Runtime policy

- For fuzzy-only CUDA sweeps, prefer `--gpu-only`.
- `--gpu-only` implies GPU device and disables sklearn baselines.
- For baseline-inclusive runs, use CPU/mixed mode and avoid claiming pure-GPU speed.

## Fast validation ladder

1. `.venv_run/bin/python examples/run_real_datasets_benchmark.py --help`
2. 1-seed GPU smoke:
```bash
.venv_run/bin/python examples/run_real_datasets_benchmark.py \
  --datasets covtype_binary_20000 \
  --seeds 23 \
  --gpu-only \
  --output-dir runs/smoke_covtype_gpu_s23
```
3. Then targeted sweeps.

## Stability/perf guardrails

- Start with small dataset subset/seed count before long runs.
- Keep run outputs in dedicated directories; avoid overwriting final artifacts.
- If GPU OOM/instability appears, reduce batch/rule budget before broad code edits.
- If GPU is unexpectedly idle, validate that run is not CPU-fallback (`--gpu-only`, torch CUDA availability, device logs).
- Report real throughput/latency observations; do not assume speedups without measured evidence.
