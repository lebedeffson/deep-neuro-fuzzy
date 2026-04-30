# ruanfis

`ruanfis` is a research codebase for interpretable neuro-fuzzy models in PyTorch.
The current paper-facing branch focuses on **Routed Kolmogorov--Arnold Fuzzy
Networks (Routed KAFN)** and compares them with shallow fuzzy, stacked ANFIS,
hierarchical ANFIS, and DFFL baselines.

## Current Paper

Canonical manuscript:

- `main(2).tex`
- bibliography: `docs/iiti26_references_skeleton_en.bib`
- figures: `docs/figures/`

Routed KAFN combines:

- teacher-guided feature routing;
- additive Kolmogorov--Arnold-style fuzzy concept layers;
- binary-aware fuzzy terms for one-hot inputs;
- fixed sparse projection-pursuit channels;
- explicit rule-count and active-rule stability reporting.

The main paper claim is intentionally conservative: KAFN is not presented as a
universal accuracy winner. Its strongest validated point is the
quality--complexity--stability trade-off on `covtype_binary_20000`, where it is
close to stacked fuzzy in F1 while giving much more stable active-rule structure.

## Repository Layout

```text
src/ruanfis/          library code
examples/             benchmark and experiment entrypoints
tests/                automated tests
docs/                 article materials, figures, notes
docs/artifacts/       compact paper-facing result summaries
runs/                 local raw experiment outputs (ignored by git)
artifacts/            local scratch benchmark outputs (ignored by git)
data/                 local datasets/cache (ignored by git)
```

## Installation

```bash
python -m pip install -e .[dev]
```

Minimum supported Python version: `3.11`.

## Important Entrypoints

```bash
python examples/run_real_datasets_benchmark.py --help
python examples/run_regression_benchmark.py
python examples/generate_article_stats_report.py
```

KAFN-focused benchmark example:

```bash
python examples/run_real_datasets_benchmark.py \
  --dataset-suite paper_all \
  --seeds 19,23,29 \
  --fuzzy-models ruanfis_kanfis \
  --gpu-only \
  --kanfis-depth 2 \
  --kanfis-superposition-terms 12 \
  --kanfis-concept-fan-in 20 \
  --kanfis-routing grouped \
  --kanfis-feature-order teacher_importance \
  --kanfis-projection-count 8 \
  --kanfis-projection-width 4 \
  --kanfis-projection-mode fixed_covtype \
  --kanfis-prune-rules 936 \
  --fuzzy-learning-rate 0.020 \
  --fuzzy-distill-weight 0.15 \
  --fuzzy-distill-models kanfis \
  --tune-fuzzy-threshold \
  --tune-fuzzy-threshold-calibrated \
  --output-dir runs/kafn_paper_all_3s
```

## Testing

```bash
pytest -q
```

For the run-oriented environment used in this repo:

```bash
.venv_run/bin/python -m pytest -q
```

## Artifact Policy

Raw runs are intentionally not committed. Keep only compact summaries and
paper-facing artifacts under `docs/artifacts/` or `docs/figures/`.

Do not commit:

- `runs/`
- `artifacts/`
- `data/`
- downloaded datasets such as `susy(2).zip`
- LaTeX build files (`*.aux`, `*.bbl`, `*.log`, ...)

## License

MIT. See [LICENSE](LICENSE).
