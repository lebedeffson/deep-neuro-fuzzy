import torch
import pytest

from ruanfis.blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from ruanfis.concept_flow import analyze_path_concept_flow, format_path_concept_flows
from ruanfis.explanations import explain_samples, format_sample_explanations
from ruanfis.exporters import export_model_report, export_pruning_report, export_rule_base
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.model import DeepFuzzyFeatureModel
from ruanfis.rules import Antecedent, RuleBase, RuleSpec
from ruanfis.stages import ConnectedFuzzyBlock, FuzzyStage
from ruanfis.trainer import LayerPruningReport, PruningReport


def _var(name: str) -> FuzzyVariable:
    return FuzzyVariable(name, GaussianMembership([0.0, 1.0], [0.3, 0.3], term_names=["low", "high"]))


def test_top_k_explanation_keeps_only_requested_number_of_rules() -> None:
    block = TransparentFuzzyBlock(
        name="feature_block",
        variables=[_var("x0"), _var("x1")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0),), name="r0", gate_init=2.0),
                RuleSpec((Antecedent(1, 0),), name="r1", gate_init=1.0),
                RuleSpec((Antecedent(0, 1),), name="r2", gate_init=0.0),
            ]
        ),
        n_concepts=2,
        concept_names=["c0", "c1"],
    )
    stage = FuzzyStage("feature_stage", [ConnectedFuzzyBlock(block, [0, 1])])
    decision = SugenoDecisionLayer(
        name="decision",
        variables=[_var("c0"), _var("c1")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0),), name="d0", gate_init=1.5),
                RuleSpec((Antecedent(1, 1),), name="d1", gate_init=0.5),
                RuleSpec((Antecedent(0, 1),), name="d2", gate_init=-1.0),
            ]
        ),
        output_dim=1,
        output_names=["score"],
    )
    model = DeepFuzzyFeatureModel([stage], decision, input_dim=2)

    inputs = torch.tensor([[0.1, 0.2], [0.9, 0.7]], dtype=torch.float32)
    _, trace = model.forward_with_trace(inputs, top_k_rules=1)

    feature_weights = trace.stage_traces[0].block_traces[0].normalized_rule_weights
    decision_weights = trace.decision_trace.normalized_rule_weights

    assert torch.all((feature_weights > 0.0).sum(dim=1) == 1)
    assert torch.all((decision_weights > 0.0).sum(dim=1) == 1)


def test_rule_export_uses_human_readable_names() -> None:
    block = TransparentFuzzyBlock(
        name="feature_block",
        variables=[_var("temperature"), _var("pressure")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="mixed"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="inverse"),
            ]
        ),
        n_concepts=2,
        concept_names=["stability", "risk"],
    )
    decision = SugenoDecisionLayer(
        name="decision",
        variables=[_var("stability"), _var("risk")],
        rule_base=RuleBase([RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="final")]),
        output_dim=1,
        output_names=["score"],
    )

    block_lines = export_rule_base(block)
    decision_lines = export_rule_base(decision)

    assert "temperature IS low" in block_lines[0]
    assert "pressure IS high" in block_lines[0]
    assert "stability=" in block_lines[0]
    assert "score =" in decision_lines[0]
    assert "stability" in decision_lines[0]
    assert "risk" in decision_lines[0]
    assert "(mixed)" in block_lines[0]


def test_model_and_pruning_report_export_cover_structure() -> None:
    block = TransparentFuzzyBlock(
        name="feature_block",
        variables=[_var("temperature"), _var("pressure")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="mixed"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="inverse"),
            ]
        ),
        n_concepts=2,
        concept_names=["stability", "risk"],
    )
    stage = FuzzyStage("feature_stage", [ConnectedFuzzyBlock(block, [0, 1])])
    decision = SugenoDecisionLayer(
        name="decision",
        variables=[_var("stability"), _var("risk")],
        rule_base=RuleBase([RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="final")]),
        output_dim=1,
        output_names=["score"],
    )
    model = DeepFuzzyFeatureModel([stage], decision, input_dim=2)
    pruning_report = PruningReport(
        layers=(
            LayerPruningReport(
                layer_name="feature_block",
                active_rules_before=2,
                active_rules_after=1,
                probabilities_before=(0.8, 0.2),
                probabilities_after=(0.8, 0.05),
            ),
        )
    )

    model_report = export_model_report(model)
    pruning_text = export_pruning_report(pruning_report)

    assert "MODEL REPORT" in model_report
    assert "STAGE 0: feature_stage" in model_report
    assert "BLOCK 0: feature_block" in model_report
    assert "DECISION LAYER: decision" in model_report
    assert "RULE 0 (mixed)" in model_report
    assert "PRUNING REPORT" in pruning_text
    assert "feature_block: active_rules 2 -> 1" in pruning_text


def test_sample_explanations_are_human_readable() -> None:
    block = TransparentFuzzyBlock(
        name="feature_block",
        variables=[_var("temperature"), _var("pressure")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="mixed"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="inverse"),
            ]
        ),
        n_concepts=2,
        concept_names=["stability", "risk"],
    )
    stage = FuzzyStage("feature_stage", [ConnectedFuzzyBlock(block, [0, 1])])
    decision = SugenoDecisionLayer(
        name="decision",
        variables=[_var("stability"), _var("risk")],
        rule_base=RuleBase([RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="final")]),
        output_dim=1,
        output_names=["score"],
    )
    model = DeepFuzzyFeatureModel([stage], decision, input_dim=2)
    inputs = torch.tensor([[0.1, 0.9]], dtype=torch.float32)

    explanations = explain_samples(model, inputs, top_k_rules=1)
    rendered = format_sample_explanations(explanations)

    assert len(explanations) == 1
    assert explanations[0].stage_explanations[0].block_explanations[0].top_rules
    assert "SAMPLE 0" in rendered
    assert "temperature IS" in rendered
    assert "pressure IS" in rendered
    assert "decision" in rendered


def test_path_based_concept_flow_reports_hidden_paths() -> None:
    block = TransparentFuzzyBlock(
        name="feature_block",
        variables=[_var("temperature"), _var("pressure")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="mixed"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="inverse"),
            ]
        ),
        n_concepts=2,
        concept_names=["stability", "risk"],
    )
    stage = FuzzyStage("feature_stage", [ConnectedFuzzyBlock(block, [0, 1])])
    decision = SugenoDecisionLayer(
        name="decision",
        variables=[_var("stability"), _var("risk")],
        rule_base=RuleBase([RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="final")]),
        output_dim=1,
        output_names=["score"],
    )
    model = DeepFuzzyFeatureModel([stage], decision, input_dim=2)
    inputs = torch.tensor([[0.1, 0.9]], dtype=torch.float32)

    flows = analyze_path_concept_flow(model, inputs, top_k_rules=2)
    rendered = format_path_concept_flows(flows)

    assert len(flows) == 1
    assert len(flows[0].hidden_paths) == 2
    assert len(flows[0].hidden_rule_paths) == 1
    assert len(flows[0].hidden_paths[0].top_rule_path_contributions) == 2
    assert len(flows[0].hidden_rule_paths[0].top_rule_path_contributions) == 2
    assert "hidden path contributions" in rendered
    assert "aggregated hidden rule paths" in rendered
    assert "feature_stage/feature_block/stability" in rendered

    concept_sum = [0.0]
    for hidden in flows[0].hidden_paths:
        concept_sum[0] += hidden.path_contribution[0]

    rule_sum = [0.0]
    for block_flow in flows[0].hidden_rule_paths:
        for rule in block_flow.top_rule_path_contributions:
            rule_sum[0] += rule.path_contribution[0]

    assert rule_sum[0] == pytest.approx(concept_sum[0], abs=1e-5)
