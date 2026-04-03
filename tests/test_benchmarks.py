import torch

from ruanfis.benchmarks import format_benchmark_results, run_tabular_benchmark
from ruanfis.bootstrap import (
    BootstrapConfig,
    StagewisePretrainingConfig,
    build_bootstrapped_shallow_model,
    build_stagewise_pretrained_hierarchical_model,
)
from ruanfis.builders import (
    DecisionLayerConfig,
    HierarchicalModelConfig,
    ShallowFuzzyModelConfig,
    StageConfig,
    TransparentBlockConfig,
)
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.trainer import TrainingConfig


def _var3(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.2, 0.5, 0.8], [0.18, 0.18, 0.18], term_names=["low", "mid", "high"]),
    )


def _var2(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"]),
    )


def _benchmark_config() -> HierarchicalModelConfig:
    return HierarchicalModelConfig(
        input_dim=4,
        stages=(
            StageConfig(
                name="stage_1",
                blocks=(
                    TransparentBlockConfig(
                        name="left_block",
                        input_indices=(0, 1),
                        variables=(_var3("x0"), _var3("x1")),
                        n_concepts=2,
                        concept_names=("left_signal", "left_bias"),
                        max_rule_arity=2,
                        max_rules=4,
                        rule_generation_mode="prototype",
                    ),
                    TransparentBlockConfig(
                        name="right_block",
                        input_indices=(2, 3),
                        variables=(_var3("x2"), _var3("x3")),
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
            variables=(
                _var2("left_signal"),
                _var2("left_bias"),
                _var2("right_signal"),
                _var2("right_bias"),
            ),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=8,
        ),
    )


def _shallow_benchmark_config() -> ShallowFuzzyModelConfig:
    return ShallowFuzzyModelConfig(
        input_dim=4,
        feature_block=TransparentBlockConfig(
            name="shallow_block",
            input_indices=(0, 1, 2, 3),
            variables=(_var3("x0"), _var3("x1"), _var3("x2"), _var3("x3")),
            n_concepts=4,
            concept_names=("signal_0", "signal_1", "signal_2", "signal_3"),
            max_rule_arity=2,
            max_rules=12,
            rule_generation_mode="prototype",
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=(_var2("signal_0"), _var2("signal_1"), _var2("signal_2"), _var2("signal_3")),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=8,
        ),
    )


def test_regression_benchmark_runs_with_sklearn_and_fuzzy_models() -> None:
    torch.manual_seed(23)
    train_inputs = torch.rand(192, 4)
    test_inputs = torch.rand(64, 4)
    train_targets = (
        0.45 * torch.sin(torch.pi * train_inputs[:, 0:1] * train_inputs[:, 1:2])
        + 0.30 * (train_inputs[:, 2:3] * train_inputs[:, 3:4])
        + 0.15 * train_inputs[:, 0:1]
    )
    test_targets = (
        0.45 * torch.sin(torch.pi * test_inputs[:, 0:1] * test_inputs[:, 1:2])
        + 0.30 * (test_inputs[:, 2:3] * test_inputs[:, 3:4])
        + 0.15 * test_inputs[:, 0:1]
    )

    fuzzy_model = build_stagewise_pretrained_hierarchical_model(
        _benchmark_config(),
        sample_inputs=train_inputs,
        sample_targets=train_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=15,
            decision_epochs=10,
            learning_rate=0.02,
            batch_size=64,
        ),
    )
    shallow_model = build_bootstrapped_shallow_model(
        _shallow_benchmark_config(),
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
                    max_epochs=50,
                    learning_rate=0.02,
                    patience=10,
                    batch_size=64,
                ),
            ),
            "ruanfis_stagewise": (
                fuzzy_model,
                TrainingConfig(
                    task_type="regression",
                    max_epochs=60,
                    learning_rate=0.02,
                    patience=10,
                    batch_size=64,
                ),
            )
        },
        random_state=23,
    )
    rendered = format_benchmark_results(results)
    names = {result.model_name for result in results}

    assert {
        "linear_regression",
        "random_forest_regressor",
        "mlp_regressor",
        "ruanfis_shallow",
        "ruanfis_stagewise",
    } <= names
    assert "TABULAR BENCHMARK" in rendered
    assert "ruanfis_shallow [ruanfis]" in rendered
    assert "ruanfis_stagewise [ruanfis]" in rendered
