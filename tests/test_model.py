import pytest
import torch

from ruanfis.blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.model import DeepFuzzyFeatureModel
from ruanfis.rules import Antecedent, RuleBase, RuleSpec
from ruanfis.stages import ConnectedFuzzyBlock, FuzzyStage


def _membership_variable(name: str) -> FuzzyVariable:
    return FuzzyVariable(name, GaussianMembership([0.0, 1.0], [0.3, 0.3], term_names=["low", "high"]))


def test_deep_model_returns_trace_for_each_stage() -> None:
    block_a = TransparentFuzzyBlock(
        name="block_a",
        variables=[_membership_variable("x0"), _membership_variable("x1")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="a_low"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="a_high"),
            ]
        ),
        n_concepts=2,
        concept_names=["thermal", "mechanical"],
    )
    block_b = TransparentFuzzyBlock(
        name="block_b",
        variables=[_membership_variable("x2"), _membership_variable("x3")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="b_mixed"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="b_inverse"),
            ]
        ),
        n_concepts=2,
        concept_names=["signal", "balance"],
    )
    stage = FuzzyStage(
        name="feature_stage",
        blocks=[
            ConnectedFuzzyBlock(block_a, input_indices=[0, 1]),
            ConnectedFuzzyBlock(block_b, input_indices=[2, 3]),
        ],
    )
    decision = SugenoDecisionLayer(
        name="decision",
        variables=[
            _membership_variable("thermal"),
            _membership_variable("mechanical"),
            _membership_variable("signal"),
            _membership_variable("balance"),
        ],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(2, 0)), name="safe"),
                RuleSpec((Antecedent(1, 1), Antecedent(3, 1)), name="risky"),
            ]
        ),
        output_dim=1,
        output_names=["score"],
    )
    model = DeepFuzzyFeatureModel(stages=[stage], decision_layer=decision, input_dim=4)

    inputs = torch.rand(5, 4)
    outputs, trace = model.forward_with_trace(inputs)

    assert outputs.shape == (5, 1)
    assert len(trace.stage_traces) == 1
    assert len(trace.stage_traces[0].block_traces) == 2
    assert trace.stage_traces[0].outputs.shape == (5, 4)
    assert trace.decision_trace.outputs.shape == (5, 1)


def test_deep_model_validates_stage_connections() -> None:
    block = TransparentFuzzyBlock(
        name="bad_block",
        variables=[_membership_variable("x0"), _membership_variable("x1")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="low"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="high"),
            ]
        ),
        n_concepts=1,
        concept_names=["concept"],
    )
    stage = FuzzyStage(
        name="bad_stage",
        blocks=[ConnectedFuzzyBlock(block, input_indices=[0, 2])],
    )
    decision = SugenoDecisionLayer(
        name="decision",
        variables=[_membership_variable("concept")],
        rule_base=RuleBase([RuleSpec((Antecedent(0, 0),), name="rule")]),
        output_dim=1,
        output_names=["score"],
    )

    with pytest.raises(ValueError, match="references input index 2"):
        DeepFuzzyFeatureModel(stages=[stage], decision_layer=decision, input_dim=2)
