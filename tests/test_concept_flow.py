import torch

from ruanfis.blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from ruanfis.concept_flow import analyze_concept_flow, format_concept_flows
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.model import DeepFuzzyFeatureModel
from ruanfis.rules import Antecedent, RuleBase, RuleSpec
from ruanfis.stages import ConnectedFuzzyBlock, FuzzyStage


def _var(name: str) -> FuzzyVariable:
    return FuzzyVariable(name, GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"]))


def _build_model() -> DeepFuzzyFeatureModel:
    feature_block = TransparentFuzzyBlock(
        name="feature_block",
        variables=[_var("x0"), _var("x1")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="ll"),
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="lh"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="hl"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="hh"),
            ]
        ),
        n_concepts=2,
        concept_names=["left", "right"],
    )
    stage = FuzzyStage("feature_stage", [ConnectedFuzzyBlock(feature_block, [0, 1])])
    decision = SugenoDecisionLayer(
        name="decision",
        variables=[_var("left"), _var("right")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="low_path"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="high_path"),
            ]
        ),
        output_dim=1,
        output_names=["score"],
    )
    return DeepFuzzyFeatureModel([stage], decision, input_dim=2)


def test_concept_flow_exactly_decomposes_decision_output() -> None:
    torch.manual_seed(6)
    model = _build_model()
    inputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32)

    flows = analyze_concept_flow(model, inputs, top_k_rules=2)

    assert len(flows) == 2
    for flow in flows:
        prediction = torch.tensor(flow.prediction)
        bias = torch.tensor(flow.bias_contribution)
        concept_sum = sum((torch.tensor(item.contribution) for item in flow.decision_concept_contributions), bias)
        assert torch.allclose(prediction, concept_sum, atol=1e-6)
        assert flow.hidden_concepts
        for concept in flow.hidden_concepts:
            contribution_sum = sum(rule.contribution for rule in concept.top_rule_contributions)
            assert contribution_sum <= concept.value + 1e-6


def test_formatted_concept_flow_contains_paths() -> None:
    model = _build_model()
    inputs = torch.tensor([[0.1, 0.9]], dtype=torch.float32)
    flows = analyze_concept_flow(model, inputs, top_k_rules=1)
    rendered = format_concept_flows(flows)

    assert "SAMPLE 0" in rendered
    assert "feature_stage/feature_block/left" in rendered
    assert "decision concept contributions:" in rendered
    assert "top decision rules:" in rendered
