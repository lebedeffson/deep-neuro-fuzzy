import torch

from ruanfis.bootstrap import BootstrapConfig, StagewisePretrainingConfig
from ruanfis.builders import (
    DecisionLayerConfig,
    HierarchicalModelConfig,
    StageConfig,
    TransparentBlockConfig,
)
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.refinement import RefinementLoopConfig, build_refined_hierarchical_model
from ruanfis.trainer import TrainingConfig


def _var2(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"]),
    )


def _var3(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.2, 0.5, 0.8], [0.18, 0.18, 0.18], term_names=["low", "mid", "high"]),
    )


def _config() -> HierarchicalModelConfig:
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


def test_refinement_loop_tracks_cycles_and_selects_best_monitor() -> None:
    torch.manual_seed(41)
    train_inputs = torch.rand(256, 4)
    val_inputs = torch.rand(96, 4)
    train_targets = (
        0.45 * torch.sin(torch.pi * train_inputs[:, 0:1] * train_inputs[:, 1:2])
        + 0.30 * (train_inputs[:, 2:3] * train_inputs[:, 3:4])
        + 0.15 * train_inputs[:, 0:1]
    )
    val_targets = (
        0.45 * torch.sin(torch.pi * val_inputs[:, 0:1] * val_inputs[:, 1:2])
        + 0.30 * (val_inputs[:, 2:3] * val_inputs[:, 3:4])
        + 0.15 * val_inputs[:, 0:1]
    )

    torch.manual_seed(123)
    one_cycle = build_refined_hierarchical_model(
        _config(),
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=val_inputs,
        validation_targets=val_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=14,
            decision_epochs=10,
            refinement_rounds=1,
            learning_rate=0.02,
            batch_size=64,
            shuffle=False,
        ),
        training_config=TrainingConfig(
            task_type="regression",
            max_epochs=60,
            learning_rate=0.02,
            patience=12,
            batch_size=64,
            shuffle=False,
        ),
        refinement_cycles=1,
    )
    torch.manual_seed(123)
    two_cycles = build_refined_hierarchical_model(
        _config(),
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=val_inputs,
        validation_targets=val_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=14,
            decision_epochs=10,
            refinement_rounds=1,
            learning_rate=0.02,
            batch_size=64,
            shuffle=False,
        ),
        training_config=TrainingConfig(
            task_type="regression",
            max_epochs=60,
            learning_rate=0.02,
            patience=12,
            batch_size=64,
            shuffle=False,
        ),
        refinement_cycles=2,
    )

    assert len(one_cycle.cycle_records) == 1
    assert len(two_cycles.cycle_records) == 2
    assert two_cycles.best_monitor_value <= one_cycle.best_monitor_value + 1e-6
    assert two_cycles.cycle_records[two_cycles.best_cycle_index].monitor_value == two_cycles.best_monitor_value
    assert two_cycles.training_result.validation_loss == two_cycles.best_monitor_value
    assert not two_cycles.stopped_early


def test_refinement_loop_supports_automatic_cycle_stopping() -> None:
    torch.manual_seed(52)
    train_inputs = torch.rand(256, 4)
    val_inputs = torch.rand(96, 4)
    train_targets = (
        0.45 * torch.sin(torch.pi * train_inputs[:, 0:1] * train_inputs[:, 1:2])
        + 0.30 * (train_inputs[:, 2:3] * train_inputs[:, 3:4])
        + 0.15 * train_inputs[:, 0:1]
    )
    val_targets = (
        0.45 * torch.sin(torch.pi * val_inputs[:, 0:1] * val_inputs[:, 1:2])
        + 0.30 * (val_inputs[:, 2:3] * val_inputs[:, 3:4])
        + 0.15 * val_inputs[:, 0:1]
    )

    torch.manual_seed(321)
    result = build_refined_hierarchical_model(
        _config(),
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=val_inputs,
        validation_targets=val_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=12,
            decision_epochs=8,
            refinement_rounds=1,
            learning_rate=0.02,
            batch_size=64,
            shuffle=False,
        ),
        training_config=TrainingConfig(
            task_type="regression",
            max_epochs=50,
            learning_rate=0.02,
            patience=10,
            batch_size=64,
            shuffle=False,
        ),
        refinement_loop_config=RefinementLoopConfig(
            max_cycles=4,
            patience=0,
            min_delta=1.0,
        ),
    )

    assert result.cycles_ran == 2
    assert result.stopped_early
