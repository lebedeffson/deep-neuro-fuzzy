import torch
import pytest
from torch import nn

from ruanfis.bootstrap import (
    BootstrapConfig,
    StagewisePretrainingConfig,
    build_bootstrapped_hierarchical_model,
    build_stagewise_pretrained_hierarchical_model,
    initialize_decision_layer_from_samples,
    initialize_transparent_block_from_samples,
    reestimate_hierarchical_model_rule_base,
)
from ruanfis.builders import (
    DecisionLayerConfig,
    HierarchicalModelConfig,
    StageConfig,
    TransparentBlockConfig,
    build_hierarchical_model,
)
from ruanfis.blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.rules import Antecedent, RuleBase, RuleSpec
from ruanfis.trainer import FuzzyTrainer, TrainingConfig


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


def test_hidden_block_bootstrap_creates_nonuniform_concepts_and_rule_gates() -> None:
    torch.manual_seed(3)
    block = TransparentFuzzyBlock(
        name="boot_block",
        variables=[_var2("x0"), _var2("x1")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="ll"),
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="lh"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="hl"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="hh"),
            ]
        ),
        n_concepts=2,
        concept_names=["state_a", "state_b"],
    )
    inputs = torch.tensor(
        [[0.05, 0.10], [0.10, 0.85], [0.80, 0.15], [0.90, 0.95], [0.45, 0.55]],
        dtype=torch.float32,
    )

    initialize_transparent_block_from_samples(block, inputs)

    consequents = block.consequents.detach()
    rule_probabilities = block.rule_probabilities.detach()

    assert torch.all(torch.max(consequents, dim=1).values > 0.8)
    assert torch.all(torch.min(consequents, dim=1).values < 0.2)
    assert float(rule_probabilities.max().item() - rule_probabilities.min().item()) > 0.1


def test_decision_layer_bootstrap_reduces_initial_regression_loss() -> None:
    torch.manual_seed(7)
    layer = SugenoDecisionLayer(
        name="decision",
        variables=[_var2("c0"), _var2("c1")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="low"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="high"),
            ]
        ),
        output_dim=1,
        output_names=["target"],
    )
    inputs = torch.rand(128, 2)
    targets = 0.7 * inputs[:, :1] - 0.2 * inputs[:, 1:2] + 0.15
    loss_fn = nn.MSELoss()

    initial_loss = loss_fn(layer(inputs), targets).item()
    initialize_decision_layer_from_samples(
        layer,
        inputs,
        sample_targets=targets,
        config=BootstrapConfig(decision_task_type="regression"),
    )
    final_loss = loss_fn(layer(inputs), targets).item()

    assert final_loss < initial_loss
    assert final_loss < 1e-8


def test_stagewise_bootstrapped_builder_improves_initial_fit() -> None:
    torch.manual_seed(11)
    inputs = torch.rand(256, 4)
    targets = (
        0.45 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
        + 0.30 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.15 * inputs[:, 0:1]
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

    plain_model = build_hierarchical_model(config, sample_inputs=inputs)
    bootstrapped_model = build_bootstrapped_hierarchical_model(
        config,
        sample_inputs=inputs,
        sample_targets=targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
    )
    loss_fn = nn.MSELoss()

    plain_loss = loss_fn(plain_model(inputs), targets).item()
    bootstrapped_loss = loss_fn(bootstrapped_model(inputs), targets).item()

    assert bootstrapped_loss < plain_loss


def test_stagewise_pretrained_builder_improves_over_bootstrap() -> None:
    torch.manual_seed(13)
    inputs = torch.rand(256, 4)
    targets = (
        0.45 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
        + 0.30 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.15 * inputs[:, 0:1]
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

    bootstrapped_model = build_bootstrapped_hierarchical_model(
        config,
        sample_inputs=inputs,
        sample_targets=targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
    )
    stagewise_model = build_stagewise_pretrained_hierarchical_model(
        config,
        sample_inputs=inputs,
        sample_targets=targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=20,
            decision_epochs=15,
            learning_rate=0.02,
            batch_size=64,
        ),
    )
    loss_fn = nn.MSELoss()

    bootstrapped_loss = loss_fn(bootstrapped_model(inputs), targets).item()
    stagewise_loss = loss_fn(stagewise_model(inputs), targets).item()

    assert stagewise_loss < bootstrapped_loss


def test_stagewise_refinement_rounds_do_not_degrade_best_model() -> None:
    torch.manual_seed(29)
    inputs = torch.rand(256, 4)
    targets = (
        0.42 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
        + 0.28 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.18 * inputs[:, 0:1]
        + 0.07 * inputs[:, 2:3]
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
    loss_fn = nn.MSELoss()

    one_round_model = build_stagewise_pretrained_hierarchical_model(
        config,
        sample_inputs=inputs,
        sample_targets=targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=18,
            decision_epochs=12,
            refinement_rounds=1,
            learning_rate=0.02,
            batch_size=64,
            shuffle=False,
        ),
    )
    refined_model = build_stagewise_pretrained_hierarchical_model(
        config,
        sample_inputs=inputs,
        sample_targets=targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=18,
            decision_epochs=12,
            refinement_rounds=3,
            learning_rate=0.02,
            batch_size=64,
            shuffle=False,
        ),
    )

    one_round_loss = loss_fn(one_round_model(inputs), targets).item()
    refined_loss = loss_fn(refined_model(inputs), targets).item()

    assert refined_loss <= one_round_loss + 1e-6


def test_reestimate_rule_base_rejects_incompatible_reference_model() -> None:
    config = HierarchicalModelConfig(
        input_dim=4,
        stages=(),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=(_var2("x0"), _var2("x1"), _var2("x2"), _var2("x3")),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=4,
        ),
    )
    incompatible_reference = build_hierarchical_model(
        HierarchicalModelConfig(
            input_dim=3,
            stages=(),
            decision_layer=DecisionLayerConfig(
                name="decision",
                variables=(_var2("x0"), _var2("x1"), _var2("x2")),
                output_dim=1,
                output_names=("target",),
                max_rule_arity=2,
                max_rules=4,
            ),
        ),
        sample_inputs=torch.rand(32, 3),
    )

    with pytest.raises(ValueError, match="input_dim"):
        reestimate_hierarchical_model_rule_base(
            config,
            incompatible_reference,
            sample_inputs=torch.rand(32, 4),
            sample_targets=torch.rand(32, 1),
        )


def test_reestimated_rule_base_from_trained_reference_improves_over_bootstrap() -> None:
    torch.manual_seed(31)
    inputs = torch.rand(320, 4)
    targets = (
        0.45 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
        + 0.30 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.15 * inputs[:, 0:1]
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
    pretraining = StagewisePretrainingConfig(
        task_type="regression",
        epochs_per_stage=16,
        decision_epochs=12,
        refinement_rounds=1,
        learning_rate=0.02,
        batch_size=64,
        shuffle=False,
    )

    stagewise_model = build_stagewise_pretrained_hierarchical_model(
        config,
        sample_inputs=inputs,
        sample_targets=targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=pretraining,
    )
    bootstrap_model = build_bootstrapped_hierarchical_model(
        config,
        sample_inputs=inputs,
        sample_targets=targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
    )
    loss_fn = nn.MSELoss()
    bootstrap_loss = loss_fn(bootstrap_model(inputs), targets).item()

    trainer = FuzzyTrainer(
        stagewise_model,
        TrainingConfig(
            task_type="regression",
            max_epochs=80,
            learning_rate=0.02,
            patience=15,
            batch_size=64,
            shuffle=False,
        ),
    )
    trainer.fit(inputs, targets, inputs, targets)

    reestimated_model = reestimate_hierarchical_model_rule_base(
        config,
        stagewise_model,
        sample_inputs=inputs,
        sample_targets=targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=16,
            decision_epochs=12,
            refinement_rounds=2,
            learning_rate=0.02,
            batch_size=64,
            shuffle=False,
        ),
    )
    reestimated_loss = loss_fn(reestimated_model(inputs), targets).item()

    assert reestimated_loss <= bootstrap_loss + 1e-6
