---
name: repo-bringup
description: Use when repository state is unclear or a new session needs a reliable working baseline before feature work.
---

# Repository bring-up rules (ruanfis)

Use this skill when:
- state is unclear after prior edits;
- imports/entrypoints may be broken;
- you need a clean baseline before new changes.

## Default route

1. Read:
- `README.md`
- `docs/artifacts/README.md`
- `docs/repository_materials_index_ru.md`

2. Inspect entrypoints:
- `examples/run_real_datasets_benchmark.py`
- `examples/run_regression_benchmark.py`
- `examples/generate_article_stats_report.py`

3. Inspect touched library modules in `src/ruanfis/`.

## Verification ladder (cheap to expensive)

1. `.venv_run/bin/python -m pytest -q`
2. `.venv_run/bin/python examples/run_real_datasets_benchmark.py --help`
3. Small smoke run (1 dataset, 1 seed):
```bash
.venv_run/bin/python examples/run_real_datasets_benchmark.py \
  --datasets diabetes \
  --seeds 23 \
  --skip-sklearn \
  --output-dir runs/smoke_diabetes_s23
```
4. Only then longer sweeps.

## Working rules

- Do not start with large refactors.
- If validation fails, localize before stacking fixes.
- Keep docs/artifacts consistent with executed runs.
- Report reality only: no "green" status without actual command evidence.
- Prefer reusable abstractions over copy-paste when touching multiple modules.

## Hygiene rules

- Temporary files should go to explicit temp/artifact paths, not repository root.
- Remove transient task files when they are no longer needed.
- Keep final outputs in documented locations under `docs/artifacts/` or `runs/`.

## Done criteria

Bring-up done when:
- smoke for touched scope is green;
- changed commands/paths are valid for this repo;
- any remaining issues are documented in artifact notes.
- workspace is not polluted by unnecessary temporary files from the task.
