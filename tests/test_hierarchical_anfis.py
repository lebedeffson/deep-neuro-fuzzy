import torch

from ruanfis.builders import DecisionLayerConfig
from ruanfis.hierarchical_anfis import (
    HierarchicalAnfisBlockConfig,
    HierarchicalAnfisModelConfig,
    HierarchicalAnfisStageConfig,
    build_hierarchical_anfis_model,
)
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.trainer import FuzzyTrainer, TrainingConfig


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


def _config() -> HierarchicalAnfisModelConfig:
    return HierarchicalAnfisModelConfig(
        input_dim=4,
        stages=(
            HierarchicalAnfisStageConfig(
                name="stage_1",
                blocks=(
                    HierarchicalAnfisBlockConfig(
                        name="left",
                        input_indices=(0, 1),
                        variables=(_var3("x0"), _var3("x1")),
                        output_dim=2,
                        output_names=("left_signal", "left_bias"),
                        max_rule_arity=2,
                        max_rules=8,
                    ),
                    HierarchicalAnfisBlockConfig(
                        name="right",
                        input_indices=(2, 3),
                        variables=(_var3("x2"), _var3("x3")),
                        output_dim=2,
                        output_names=("right_signal", "right_bias"),
                        max_rule_arity=2,
                        max_rules=8,
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
            max_rules=10,
        ),
    )


def test_hierarchical_anfis_model_builds_and_trains() -> None:
    torch.manual_seed(211)
    inputs = torch.rand(256, 4)
    targets = (
        0.45 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
        + 0.30 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.15 * inputs[:, 0:1]
    )

    model = build_hierarchical_anfis_model(_config())
    with torch.no_grad():
        initial_loss = torch.nn.functional.mse_loss(model(inputs), targets).item()

    trainer = FuzzyTrainer(
        model,
        TrainingConfig(
            task_type="regression",
            max_epochs=100,
            learning_rate=0.02,
            patience=20,
            batch_size=64,
            shuffle=False,
        ),
    )
    result = trainer.fit(inputs, targets, inputs, targets)

    assert result.validation_loss is not None
    assert result.validation_loss < initial_loss


def test_hierarchical_hidden_blocks_can_use_bounded_outputs() -> None:
    config = HierarchicalAnfisModelConfig(
        input_dim=2,
        stages=(
            HierarchicalAnfisStageConfig(
                name="stage_1",
                blocks=(
                    HierarchicalAnfisBlockConfig(
                        name="hidden",
                        input_indices=(0, 1),
                        variables=(_var3("x0"), _var3("x1")),
                        output_dim=2,
                        output_activation="sigmoid",
                        max_rule_arity=1,
                        max_rules=4,
                    ),
                ),
            ),
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=(_var2("h0"), _var2("h1")),
            output_dim=1,
            max_rule_arity=1,
            max_rules=4,
        ),
    )
    model = build_hierarchical_anfis_model(config)

    features = model.forward_features(torch.rand(8, 2))

    assert torch.all(features >= 0.0)
    assert torch.all(features <= 1.0)
