import pytest
import torch

from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.stacked import (
    StackedAnfisLayerConfig,
    StackedAnfisModel,
    StackedAnfisModelConfig,
    build_stagewise_initialized_stacked_anfis_model,
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
                name="layer_1",
                variables=(_var3("x0"), _var3("x1"), _var3("x2"), _var3("x3")),
                output_dim=3,
                output_names=("h1", "h2", "h3"),
                max_rule_arity=2,
                max_rules=12,
                rule_generation_mode="enumerate",
            ),
            StackedAnfisLayerConfig(
                name="layer_2",
                variables=(_var3("h1"), _var3("h2"), _var3("h3")),
                output_dim=1,
                output_names=("target",),
                max_rule_arity=2,
                max_rules=8,
                rule_generation_mode="enumerate",
            ),
        ),
    )


def test_stacked_builder_constructs_trainable_model() -> None:
    model = build_stacked_anfis_model(_config())

    assert isinstance(model, StackedAnfisModel)
    assert len(model.layers) == 2
    assert model.input_dim == 4
    assert model.output_dim == 1

    outputs = model(torch.rand(5, 4))
    assert outputs.shape == (5, 1)


def test_stacked_hidden_layers_can_use_bounded_outputs() -> None:
    config = StackedAnfisModelConfig(
        input_dim=2,
        layers=(
            StackedAnfisLayerConfig(
                name="hidden",
                variables=(_var3("x0"), _var3("x1")),
                output_dim=2,
                output_activation="sigmoid",
                max_rule_arity=1,
                max_rules=4,
            ),
            StackedAnfisLayerConfig(
                name="decision",
                variables=(_var3("h0"), _var3("h1")),
                output_dim=1,
                max_rule_arity=1,
                max_rules=4,
            ),
        ),
    )
    model = build_stacked_anfis_model(config)

    features = model.forward_features(torch.rand(8, 2))

    assert torch.all(features >= 0.0)
    assert torch.all(features <= 1.0)


def test_stagewise_initialized_stacked_builder_constructs_model() -> None:
    torch.manual_seed(11)
    inputs = torch.rand(48, 4)
    targets = (inputs[:, :1] > 0.5).to(dtype=torch.float32)

    model = build_stagewise_initialized_stacked_anfis_model(
        _config(),
        sample_inputs=inputs,
        sample_targets=targets,
        task_type="binary_classification",
        epochs_per_hidden_layer=1,
        learning_rate=0.01,
        batch_size=16,
        shuffle=False,
    )

    assert isinstance(model, StackedAnfisModel)
    assert model(inputs).shape == (48, 1)


def test_stacked_final_layer_can_receive_raw_skip_inputs() -> None:
    config = StackedAnfisModelConfig(
        input_dim=3,
        final_skip_input_indices=(0, 2),
        layers=(
            StackedAnfisLayerConfig(
                name="hidden",
                variables=(_var3("x0"), _var3("x1"), _var3("x2")),
                output_dim=2,
                max_rule_arity=1,
                max_rules=4,
            ),
            StackedAnfisLayerConfig(
                name="decision",
                variables=(_var3("h0"), _var3("h1"), _var3("x0"), _var3("x2")),
                output_dim=1,
                max_rule_arity=1,
                max_rules=4,
            ),
        ),
    )
    model = build_stacked_anfis_model(config, sample_inputs=torch.rand(12, 3))

    features = model.forward_features(torch.rand(5, 3))

    assert features.shape == (5, 4)
    assert model(torch.rand(5, 3)).shape == (5, 1)


def test_stacked_skip_gates_are_initialized_when_enabled() -> None:
    config = StackedAnfisModelConfig(
        input_dim=3,
        final_skip_input_indices=(0, 2),
        final_skip_gates_enabled=True,
        final_skip_gate_init_logit=1.5,
        layers=(
            StackedAnfisLayerConfig(
                name="hidden",
                variables=(_var3("x0"), _var3("x1"), _var3("x2")),
                output_dim=2,
                max_rule_arity=1,
                max_rules=4,
            ),
            StackedAnfisLayerConfig(
                name="decision",
                variables=(_var3("h0"), _var3("h1"), _var3("x0"), _var3("x2")),
                output_dim=1,
                max_rule_arity=1,
                max_rules=4,
            ),
        ),
    )
    model = build_stacked_anfis_model(config, sample_inputs=torch.rand(12, 3))

    assert model.final_skip_gate_logits is not None
    assert model.final_skip_gate_logits.shape == (2,)


def test_stacked_builder_validates_layer_width() -> None:
    bad_config = StackedAnfisModelConfig(
        input_dim=4,
        layers=(
            StackedAnfisLayerConfig(
                name="layer_1",
                variables=(_var3("x0"), _var3("x1"), _var3("x2"), _var3("x3")),
                output_dim=3,
                max_rule_arity=2,
                max_rules=8,
            ),
            StackedAnfisLayerConfig(
                name="layer_2_bad",
                variables=(_var3("h1"), _var3("h2")),
                output_dim=1,
                max_rule_arity=2,
                max_rules=6,
            ),
        ),
    )

    with pytest.raises(ValueError, match="stacked width"):
        build_stacked_anfis_model(bad_config)


def test_stacked_model_training_reduces_regression_loss() -> None:
    torch.manual_seed(73)
    inputs = torch.rand(256, 4)
    targets = (
        0.45 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
        + 0.30 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.15 * inputs[:, 0:1]
    )

    model = build_stacked_anfis_model(_config())
    with torch.no_grad():
        initial_loss = torch.nn.functional.mse_loss(model(inputs), targets).item()

    trainer = FuzzyTrainer(
        model,
        TrainingConfig(
            task_type="regression",
            max_epochs=120,
            learning_rate=0.02,
            patience=20,
            batch_size=64,
            shuffle=False,
        ),
    )
    result = trainer.fit(inputs, targets, inputs, targets)

    assert result.validation_loss is not None
    assert result.validation_loss < initial_loss
