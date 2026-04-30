# Deep Neuro-Fuzzy Architectures

Research code, benchmark entrypoints, and compact reproducibility artifacts for
interpretable deep neuro-fuzzy models, with the current work focused on
**Routed Kolmogorov--Arnold Fuzzy Networks (Routed KAFN)**.

The repository implements and compares five fuzzy architectures:

1. shallow fuzzy model;
2. stacked fuzzy model;
3. hierarchical fuzzy model;
4. Deep Fuzzy Feature Learning (DFFL);
5. Routed KAFN.

The Python package is still named `ruanfis` for compatibility, but the project
scope is the full deep neuro-fuzzy benchmark and KAFN evaluation.

## Current Project Materials

The repository intentionally does not store article manuscripts or Word drafts.
It stores code, benchmark configuration, curated result summaries, and figures:

- library code: `src/ruanfis/`
- benchmark entrypoints: `examples/`
- compact artifacts: `docs/artifacts/`
- figures: `docs/figures/`

Main claim: Routed KAFN is not presented as a universal accuracy winner. Its
validated strength is a quality--complexity--stability operating point:
competitive F1 with much more reproducible active fuzzy structure.

## Shared Fuzzy Block

All reference fuzzy architectures use the same rule-activation principle. For a
rule `r`,

```math
w_r = \sigma(g_r)\prod_{(i,j)\in A_r}\mu_{ij}(z_i),
\qquad
\bar{w}_r = \frac{w_r}{\sum_q w_q+\varepsilon}.
```

Here `A_r` is the antecedent structure, `mu_ij` is a membership function, and
`sigma(g_r)` is a learnable global rule gate. Hidden concepts and final Sugeno
outputs are built from normalized activations:

```math
c = \sum_r \bar{w}_r v_r,
\qquad
\hat{y}=\sum_r \bar{w}_r(a_{r0}+a_r^\top z).
```

Structural interpretability is measured by rule/unit count and active-rule
Jaccard stability across repeated seeds.

## Five Architectures

### 1. Shallow Fuzzy

A single fuzzy block over the input followed by a Sugeno decision layer:

```math
\hat{y}_{\mathrm{shallow}} = D(F_1(x)).
```

This is the compact ANFIS-style reference. It is readable, but has limited
representational depth.

### 2. Stacked Fuzzy

Sequential fuzzy blocks refine the representation layer by layer:

```math
z^{(0)}=x,\qquad
z^{(\ell)}=F_\ell(z^{(\ell-1)}),\qquad
\hat{y}_{\mathrm{stacked}}=D(z^{(L)}).
```

This variant is often the strongest fuzzy model by raw F1/RMSE, but its active
rule topology can be unstable across runs.

### 3. Hierarchical Fuzzy

Feature groups are processed by local fuzzy blocks and then aggregated:

```math
h_g=F_g(x_{\mathcal{G}_g}),\qquad
\hat{y}_{\mathrm{hier}}=D(A(h_1,\ldots,h_G)).
```

The hierarchy reduces pressure on one global rule base and gives a controlled
complexity/readability trade-off.

### 4. DFFL

DFFL treats hidden states as concept-oriented fuzzy representations. Local
concepts are aggregated and refined before prediction:

```math
c_g^{(1)}=F_g^{\mathrm{loc}}(x_{\mathcal{G}_g}),\qquad
c^{(2)}=A_{\mathrm{concept}}(c_1^{(1)},\ldots,c_G^{(1)}),\qquad
\hat{y}_{\mathrm{DFFL}}=D(R(c^{(2)})).
```

In the paper protocol, DFFL uses staged concept pretraining, rule re-estimation,
decision pretraining, and final fine-tuning.

### 5. Routed KAFN

Routed KAFN replaces high-order multidimensional fuzzy products with routed
additive one-dimensional predicates. First-stage concepts are

```math
c_q^{(1)}
=\sum_{p\in\mathcal{R}_q}\sum_{t\in\mathcal{T}_p}
\alpha_{qpt}\mu_{pt}(x_p),
\qquad q=1,\ldots,Q.
```

Routes `R_q` are fixed before neural training using Extra-Trees teacher
importance. Continuous variables use LOW/MEDIUM/HIGH terms; binary and one-hot
variables use absent/present terms.

The upper KA fuzzy superposition is

```math
c_k^{(2)}
=\sum_{q\in\mathcal{S}_k}\sum_{t\in\mathcal{U}}
\beta_{kqt}\nu_{qt}(c_q^{(1)}).
```

Optional fixed projection-pursuit channels add limited interaction capacity:

```math
\pi_m(x)=\sum_{p\in\mathcal{P}_m}\gamma_{mp}x_p,
\qquad |\mathcal{P}_m|\le 4.
```

The final decision input is

```math
u(x)=\left[c^{(1)}(x),\ \pi(x),\ c^{(2)}(x),
x_{\mathcal{S}_{\mathrm{skip}}}\right],
\qquad
\hat{p}(y=1\mid x)=\sigma(\theta_0+\theta^\top u(x)).
```

KAFN counts routed unary predicates and fixed projection predicates, not a full
combinatorial ANFIS-style multidimensional rule base.

## Key Paper Results

Focused `covtype_binary_20000` block, three seeds:

| Model | F1 | ROC-AUC | PR-AUC | Active rules/predicates | Jaccard |
|---|---:|---:|---:|---:|---:|
| stacked fuzzy | `0.8222+/-0.0074` | `0.8902+/-0.0066` | `0.8643+/-0.0091` | `20.7` | `0.0978` |
| Routed KAFN, fixed projections | `0.8174+/-0.0101` | `0.8857+/-0.0035` | `0.8605+/-0.0022` | `936` | `0.9167` |
| HistGradientBoosting | `0.8154+/-0.0065` | `0.8988+/-0.0040` | `0.8845+/-0.0039` | - | - |
| Extra Trees | `0.8546+/-0.0026` | `0.9296+/-0.0034` | `0.9181+/-0.0038` | - | - |

Large-scale KAFN validation checks:

| Dataset | F1 | ROC-AUC | PR-AUC | Active predicates | Jaccard |
|---|---:|---:|---:|---:|---:|
| full Covtype | `0.8485+/-0.0030` | `0.9246+/-0.0036` | `0.9135+/-0.0043` | `936` | `0.9696` |
| SUSY-200k | `0.7758+/-0.0018` | `0.8729+/-0.0010` | `0.8773+/-0.0006` | `936` | `0.8639` |
| KDDCup99-10% | `0.9996+/-0.0000` | `1.0000+/-0.0000` | `1.0000+/-0.0000` | `936` | `0.8432` |

Raw run bundles live under `runs/` and are intentionally ignored by git. Current
KAFN source runs include:

- `runs/kafn_paper_all_3s_2026-04-30`
- `runs/kafn_q1_large_3s_2026-04-30`
- `runs/cov20k_kaanifs_proj_p8w4_fixed_3s_2026-04-30`

## Repository Layout

```text
src/ruanfis/          library code for fuzzy blocks and KAFN
examples/             benchmark and experiment entrypoints
tests/                automated tests
docs/                 paper materials, figures, notes
docs/artifacts/       compact paper-facing result summaries
runs/                 local raw experiment outputs, ignored by git
artifacts/            local scratch outputs, ignored by git
data/                 local datasets/cache, ignored by git
```

## Installation

```bash
python -m pip install -e .[dev]
```

Python `>=3.11` is required. The benchmark environment used in this repository
usually uses `.venv_run/bin/python`.

## Main Entrypoints

```bash
python examples/run_real_datasets_benchmark.py --help
python examples/run_regression_benchmark.py
python examples/generate_article_stats_report.py
```

Supported fuzzy-model aliases include:

```text
shallow, stacked, hierarchical, dffl, kanfis
```

The internal model names are `ruanfis_shallow`, `ruanfis_stacked_anfis`,
`ruanfis_hierarchical_anfis`, `ruanfis_refined_deep`, and `ruanfis_kanfis`.

## Reproduce KAFN Paper Runs

Main paper suite:

```bash
.venv_run/bin/python examples/run_real_datasets_benchmark.py \
  --dataset-suite paper_all \
  --seeds 19,23,29 \
  --gpu-only \
  --fuzzy-models kanfis \
  --max-epochs 80 \
  --batch-size 64 \
  --patience 20 \
  --pretrain-epochs 18 \
  --decision-pretrain-epochs 14 \
  --fuzzy-learning-rate 0.02 \
  --fuzzy-distill-weight 0.15 \
  --fuzzy-distill-models kanfis \
  --fuzzy-distill-teacher-trees 400 \
  --tune-fuzzy-threshold \
  --tune-fuzzy-threshold-calibrated \
  --kanfis-depth 2 \
  --kanfis-superposition-terms 12 \
  --kanfis-concept-fan-in 20 \
  --kanfis-routing grouped \
  --kanfis-feature-order teacher_importance \
  --kanfis-projection-count 8 \
  --kanfis-projection-width 4 \
  --kanfis-projection-mode fixed_covtype \
  --kanfis-prune-rules 936 \
  --kanfis-recovery-epochs 12 \
  --kanfis-polish-epochs 0 \
  --output-dir runs/kafn_paper_all_3s
```

Large-scale KAFN validation:

```bash
.venv_run/bin/python examples/run_real_datasets_benchmark.py \
  --dataset-suite q1_large \
  --seeds 19,23,29 \
  --gpu-only \
  --fuzzy-models kanfis \
  --max-epochs 70 \
  --batch-size 256 \
  --patience 16 \
  --pretrain-epochs 18 \
  --decision-pretrain-epochs 14 \
  --fuzzy-learning-rate 0.02 \
  --fuzzy-distill-weight 0.15 \
  --fuzzy-distill-models kanfis \
  --fuzzy-distill-teacher-trees 400 \
  --tune-fuzzy-threshold \
  --tune-fuzzy-threshold-calibrated \
  --kanfis-depth 2 \
  --kanfis-superposition-terms 12 \
  --kanfis-concept-fan-in 20 \
  --kanfis-routing grouped \
  --kanfis-feature-order teacher_importance \
  --kanfis-projection-count 8 \
  --kanfis-projection-width 4 \
  --kanfis-projection-mode fixed_covtype \
  --kanfis-prune-rules 936 \
  --kanfis-recovery-epochs 16 \
  --kanfis-polish-epochs 8 \
  --output-dir runs/kafn_q1_large_3s
```

## Testing

```bash
pytest -q
```

Run-oriented environment:

```bash
.venv_run/bin/python -m pytest -q
```

## Artifact Policy

Commit compact paper-facing summaries under `docs/artifacts/` and figures under
`docs/figures/`. Do not commit raw runs, datasets, logs, or LaTeX build files.

Ignored/local-only paths include:

- `runs/`
- `artifacts/`
- `data/`
- downloaded datasets such as `susy(2).zip`
- build files such as `*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out`, `*.toc`

## License

MIT. See [LICENSE](LICENSE).
