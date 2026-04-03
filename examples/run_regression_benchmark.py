from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ruanfis import (  # noqa: E402
    BootstrapConfig,
    DecisionLayerConfig,
    FuzzyVariable,
    GaussianMembership,
    HierarchicalModelConfig,
    ShallowFuzzyModelConfig,
    StageConfig,
    StagewisePretrainingConfig,
    TrainingConfig,
    TransparentBlockConfig,
    build_bootstrapped_hierarchical_model,
    build_bootstrapped_shallow_model,
    build_stagewise_pretrained_hierarchical_model,
    format_benchmark_results,
    run_tabular_benchmark,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_config() -> HierarchicalModelConfig:
    return HierarchicalModelConfig(
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


def build_shallow_config() -> ShallowFuzzyModelConfig:
    return ShallowFuzzyModelConfig(
        input_dim=4,
        feature_block=TransparentBlockConfig(
            name="shallow_block",
            input_indices=(0, 1, 2, 3),
            variables=(var3("x0"), var3("x1"), var3("x2"), var3("x3")),
            n_concepts=4,
            concept_names=("signal_0", "signal_1", "signal_2", "signal_3"),
            max_rule_arity=2,
            max_rules=12,
            rule_generation_mode="prototype",
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=(var2("signal_0"), var2("signal_1"), var2("signal_2"), var2("signal_3")),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=8,
        ),
    )


def build_dataset(n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.rand(n_samples, 4)
    targets = (
        0.45 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
        + 0.30 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.15 * inputs[:, 0:1]
    )
    return inputs, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tabular regression benchmark against sklearn baselines.")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    train_inputs, train_targets = build_dataset(384)
    test_inputs, test_targets = build_dataset(128)

    fuzzy_model = build_stagewise_pretrained_hierarchical_model(
        build_config(),
        sample_inputs=train_inputs,
        sample_targets=train_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=25,
            decision_epochs=20,
            learning_rate=0.02,
            batch_size=64,
        ),
    )
    shallow_model = build_bootstrapped_shallow_model(
        build_shallow_config(),
        sample_inputs=train_inputs,
        sample_targets=train_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
    )

    results = run_tabular_benchmark(
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        task_type="regression",
        fuzzy_models={
            "ruanfis_shallow": (
                shallow_model,
                TrainingConfig(
                    task_type="regression",
                    max_epochs=120,
                    learning_rate=0.02,
                    patience=20,
                    batch_size=64,
                ),
            ),
            "ruanfis_stagewise": (
                fuzzy_model,
                TrainingConfig(
                    task_type="regression",
                    max_epochs=120,
                    learning_rate=0.02,
                    patience=20,
                    batch_size=64,
                    concept_orthogonality_weight=1e-3,
                    membership_overlap_weight=1e-3,
                    membership_coverage_weight=1e-3,
                ),
            )
        },
        random_state=args.seed,
    )

    report = format_benchmark_results(results)
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
