# ruanfis

`ruanfis` is a research-oriented Python library for deep neuro-fuzzy modeling with
layer-wise interpretability.

The project currently focuses on **deep fuzzy feature learning** rather than a fully
general stacked deep ANFIS. Hidden fuzzy stages build interpretable concept
representations, while the final layer performs first-order Sugeno inference.

## Current scope

- transparent fuzzy hidden blocks with interpretable concept outputs;
- hierarchical model builders that structurally control rule growth;
- prototype-based and stage-wise rule initialization;
- stage-wise pretraining and end-to-end fine-tuning in PyTorch;
- local and global explainability utilities;
- serialization, benchmarking, and runnable examples.

## Why this project exists

Classical ANFIS models are interpretable but shallow. Modern deep models are
expressive but often opaque. `ruanfis` explores the middle ground:

`x -> fuzzy concepts -> deeper fuzzy concepts -> prediction`

The implementation is designed to keep hidden representations inspectable while
avoiding the flat combinatorial explosion of a single global rule base.

## Installation

```bash
python -m pip install -e .[dev]
```

Minimum supported Python version: `3.11`.

## Quick start

```python
import torch

from ruanfis import (
    BootstrapConfig,
    DecisionLayerConfig,
    FuzzyVariable,
    GaussianMembership,
    HierarchicalModelConfig,
    StageConfig,
    StagewisePretrainingConfig,
    TransparentBlockConfig,
    build_stagewise_pretrained_hierarchical_model,
)


def var3(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.2, 0.5, 0.8], [0.18, 0.18, 0.18], term_names=["low", "mid", "high"]),
    )


def var2(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"]),
    )


config = HierarchicalModelConfig(
    input_dim=4,
    stages=(
        StageConfig(
            name="stage_1",
            blocks=(
                TransparentBlockConfig(
                    name="left_block",
                    input_indices=(0, 1),
                    variables=(var3("x0"), var3("x1")),
                    n_concepts=2,
                    concept_names=("left_signal", "left_bias"),
                    max_rule_arity=2,
                    max_rules=4,
                    rule_generation_mode="prototype",
                ),
                TransparentBlockConfig(
                    name="right_block",
                    input_indices=(2, 3),
                    variables=(var3("x2"), var3("x3")),
                    n_concepts=2,
                    concept_names=("right_signal", "right_bias"),
                    max_rule_arity=2,
                    max_rules=4,
                    rule_generation_mode="prototype",
                ),
            ),
        ),
    ),
    decision_layer=DecisionLayerConfig(
        name="decision",
        variables=(var2("left_signal"), var2("left_bias"), var2("right_signal"), var2("right_bias")),
        output_dim=1,
        output_names=("target",),
        max_rule_arity=2,
        max_rules=8,
    ),
)

inputs = torch.rand(256, 4)
targets = (
    0.45 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
    + 0.30 * (inputs[:, 2:3] * inputs[:, 3:4])
    + 0.15 * inputs[:, 0:1]
)

model = build_stagewise_pretrained_hierarchical_model(
    config,
    sample_inputs=inputs,
    sample_targets=targets,
    bootstrap_config=BootstrapConfig(decision_task_type="regression"),
    pretraining_config=StagewisePretrainingConfig(task_type="regression"),
)
```

## Repository layout

```text
src/ruanfis/         library code
tests/               automated tests
examples/            runnable demos and benchmark scripts
docs/                mathematical notes and project documentation
```

## Examples

```bash
python examples/train_xor_with_rule_report.py
python examples/train_hierarchical_regression.py
python examples/run_regression_benchmark.py
```

## Testing

```bash
pytest -q
```

## Research status

The implementation is already fully tested and runnable, but it should still be treated
as an active research codebase. The main mathematical claim at the moment is documented
in [docs/deep_fuzzy_feature_learning.md](docs/deep_fuzzy_feature_learning.md).

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
