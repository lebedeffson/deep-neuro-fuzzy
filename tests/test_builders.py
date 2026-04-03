import torch
import pytest

from ruanfis.blocks import BaseFuzzyRuleLayer
from ruanfis.builders import (
    DecisionLayerConfig,
    HierarchicalModelConfig,
    ShallowFuzzyModelConfig,
    StageConfig,
    TransparentBlockConfig,
    build_hierarchical_model,
    build_shallow_fuzzy_model,
)
from ruanfis.exporters import export_model_config_report
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.rules import count_rule_candidates, generate_rule_base
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


def test_rule_counting_matches_combinatorial_formula() -> None:
    assert count_rule_candidates([3, 3, 3], max_rule_arity=2) == 36
    assert count_rule_candidates([3, 3, 3], max_rule_arity=3) == 63

    rule_base = generate_rule_base([3, 3, 3], max_rule_arity=2, max_rules=10, name_prefix="auto")
    assert len(rule_base) == 10
    assert rule_base.names[0] == "auto_0"


def test_hierarchical_builder_controls_rule_growth() -> None:
    config = HierarchicalModelConfig(
        input_dim=6,
        stages=(
            StageConfig(
                name="stage_1",
                blocks=(
                    TransparentBlockConfig(
                        name="block_1",
                        input_indices=(0, 1),
                        variables=(_var3("x0"), _var3("x1")),
                        n_concepts=2,
                        concept_names=("c01_a", "c01_b"),
                        max_rule_arity=2,
                    ),
                    TransparentBlockConfig(
                        name="block_2",
                        input_indices=(2, 3),
                        variables=(_var3("x2"), _var3("x3")),
                        n_concepts=2,
                        concept_names=("c23_a", "c23_b"),
                        max_rule_arity=2,
                    ),
                    TransparentBlockConfig(
                        name="block_3",
                        input_indices=(4, 5),
                        variables=(_var3("x4"), _var3("x5")),
                        n_concepts=2,
                        concept_names=("c45_a", "c45_b"),
                        max_rule_arity=2,
                    ),
                ),
            ),
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=(
                _var2("c01_a"),
                _var2("c01_b"),
                _var2("c23_a"),
                _var2("c23_b"),
                _var2("c45_a"),
                _var2("c45_b"),
            ),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=24,
        ),
    )
    model = build_hierarchical_model(config)

    flat_rule_count = count_rule_candidates([3, 3, 3, 3, 3, 3], max_rule_arity=6)
    assert flat_rule_count == 4095
    assert config.generated_rule_count() == 69

    model_rule_count = sum(
        submodule.n_rules for submodule in model.modules() if isinstance(submodule, BaseFuzzyRuleLayer)
    )
    assert model_rule_count == config.generated_rule_count()
    assert model_rule_count < flat_rule_count

    report = export_model_config_report(config, flat_term_counts=(3, 3, 3, 3, 3, 3))
    assert "MODEL CONFIG REPORT" in report
    assert "Flat full-rule count: 4095" in report
    assert "Hierarchical generated-rule count: 69" in report


def test_hierarchical_builder_model_trains_on_grouped_regression() -> None:
    config = HierarchicalModelConfig(
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
                        max_rules=9,
                    ),
                    TransparentBlockConfig(
                        name="right_block",
                        input_indices=(2, 3),
                        variables=(_var3("x2"), _var3("x3")),
                        n_concepts=2,
                        concept_names=("right_signal", "right_bias"),
                        max_rule_arity=2,
                        max_rules=9,
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
            max_rules=12,
        ),
    )
    model = build_hierarchical_model(config)
    trainer = FuzzyTrainer(
        model,
        TrainingConfig(
            task_type="regression",
            max_epochs=180,
            learning_rate=0.03,
            patience=30,
            batch_size=64,
        ),
    )

    torch.manual_seed(9)
    inputs = torch.rand(256, 4)
    targets = (
        0.45 * torch.sin(torch.pi * inputs[:, :1] * inputs[:, 1:2])
        + 0.35 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.20 * inputs[:, 0:1]
    )
    result = trainer.fit(inputs, targets, inputs, targets)

    assert result.train_metrics["rmse"] < 0.09
    assert result.validation_metrics is not None
    assert result.validation_metrics["rmse"] < 0.09


def test_shallow_builder_creates_single_stage_model_with_expected_rule_budget() -> None:
    config = ShallowFuzzyModelConfig(
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
    sample_inputs = torch.rand(128, 4)

    model = build_shallow_fuzzy_model(config, sample_inputs=sample_inputs)
    hierarchical_config = config.as_hierarchical_config()
    model_rule_count = sum(
        submodule.n_rules for submodule in model.modules() if isinstance(submodule, BaseFuzzyRuleLayer)
    )
    report = export_model_config_report(config, flat_term_counts=(3, 3, 3, 3))

    assert len(model.stages) == 1
    assert len(model.stages[0].blocks) == 1
    assert model_rule_count == hierarchical_config.generated_rule_count()
    assert "Hierarchical generated-rule count:" in report


def test_prototype_builder_requires_sample_inputs() -> None:
    config = HierarchicalModelConfig(
        input_dim=2,
        stages=(
            StageConfig(
                name="stage_1",
                blocks=(
                    TransparentBlockConfig(
                        name="proto_block",
                        input_indices=(0, 1),
                        variables=(_var3("x0"), _var3("x1")),
                        n_concepts=2,
                        concept_names=("signal", "bias"),
                        max_rule_arity=2,
                        max_rules=4,
                        rule_generation_mode="prototype",
                    ),
                ),
            ),
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=(_var2("signal"), _var2("bias")),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=4,
        ),
    )

    with pytest.raises(ValueError, match="sample_inputs"):
        build_hierarchical_model(config)


def test_prototype_builder_reduces_rule_count_and_trains() -> None:
    torch.manual_seed(17)
    inputs = torch.rand(320, 4)
    targets = (
        0.40 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
        + 0.30 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.20 * inputs[:, 0:1]
        + 0.10 * inputs[:, 2:3]
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
                        variables=(_var3("x0"), _var3("x1")),
                        n_concepts=2,
                        concept_names=("left_signal", "left_bias"),
                        max_rule_arity=2,
                        max_rules=4,
                        rule_generation_mode="prototype",
                        prototype_term_limit=2,
                        prototype_sample_size=128,
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
                        prototype_term_limit=2,
                        prototype_sample_size=128,
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

    model = build_hierarchical_model(config, sample_inputs=inputs)
    trainer = FuzzyTrainer(
        model,
        TrainingConfig(
            task_type="regression",
            max_epochs=220,
            learning_rate=0.03,
            patience=35,
            batch_size=64,
        ),
    )
    result = trainer.fit(inputs, targets, inputs, targets)

    model_rule_count = sum(
        submodule.n_rules for submodule in model.modules() if isinstance(submodule, BaseFuzzyRuleLayer)
    )
    full_local_enumeration = 9 + 9 + 10
    report = export_model_config_report(config, flat_term_counts=(3, 3, 3, 3))

    assert model_rule_count == 18
    assert model_rule_count < full_local_enumeration
    assert "rule_generation_mode: prototype" in report
    assert result.validation_metrics is not None
    assert result.validation_metrics["rmse"] < 0.12
