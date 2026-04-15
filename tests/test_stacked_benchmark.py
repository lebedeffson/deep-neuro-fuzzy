import torch

from ruanfis.benchmarks import evaluate_trained_model
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.stacked import (
    StackedAnfisLayerConfig,
    StackedAnfisModelConfig,
    build_stacked_anfis_model,
)
from ruanfis.trainer import FuzzyTrainer, TrainingConfig


def _var3(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.2, 0.5, 0.8], [0.18, 0.18, 0.18], term_names=["low", "mid", "high"]),
    )


def _config() -> StackedAnfisModelConfig:
    return StackedAnfisModelConfig(
        input_dim=4,
        layers=(
            StackedAnfisLayerConfig(
                name="stacked_1",
                variables=(_var3("x0"), _var3("x1"), _var3("x2"), _var3("x3")),
                output_dim=3,
                output_names=("h1", "h2", "h3"),
                max_rule_arity=2,
                max_rules=10,
                rule_generation_mode="enumerate",
            ),
            StackedAnfisLayerConfig(
                name="stacked_2",
                variables=(_var3("h1"), _var3("h2"), _var3("h3")),
                output_dim=1,
                output_names=("target",),
                max_rule_arity=2,
                max_rules=8,
                rule_generation_mode="enumerate",
            ),
        ),
    )


def test_evaluate_trained_model_produces_fuzzy_metrics_for_stacked_anfis() -> None:
    torch.manual_seed(101)
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

    model = build_stacked_anfis_model(_config())
    trainer = FuzzyTrainer(
        model,
        TrainingConfig(
            task_type="regression",
            max_epochs=80,
            learning_rate=0.02,
            patience=15,
            batch_size=64,
            shuffle=False,
        ),
    )
    trainer.fit(train_inputs, train_targets, test_inputs, test_targets)

    result = evaluate_trained_model(
        "ruanfis_stacked_anfis",
        model,
        task_type="regression",
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
    )

    assert result.family == "ruanfis"
    assert "total_rules" in result.structural_metrics
    assert "hidden_blocks" in result.structural_metrics
    assert "decision_top1_mass" in result.explainability_metrics
    assert "hidden_top1_mass" in result.explainability_metrics
    assert "active:all" in result.stability_artifacts
