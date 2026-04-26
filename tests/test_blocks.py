import torch
from torch.nn import functional as F

from ruanfis.blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.rules import Antecedent, RuleBase, RuleSpec


def _build_test_block() -> TransparentFuzzyBlock:
    variables = [
        FuzzyVariable("x0", GaussianMembership([0.0, 1.0], [0.25, 0.25], term_names=["low", "high"])),
        FuzzyVariable("x1", GaussianMembership([0.0, 1.0], [0.25, 0.25], term_names=["low", "high"])),
    ]
    rules = RuleBase(
        [
            RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="both_low", gate_init=0.1),
            RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="both_high", gate_init=-0.2),
        ]
    )
    block = TransparentFuzzyBlock(
        name="latent_block",
        variables=variables,
        rule_base=rules,
        n_concepts=2,
        concept_names=["stability", "risk"],
    )
    with torch.no_grad():
        block.raw_consequents.copy_(
            torch.logit(torch.tensor([[0.2, 0.8], [0.9, 0.1]], dtype=torch.float32), eps=1e-6)
        )
    return block


def test_transparent_block_matches_manual_rule_aggregation() -> None:
    block = _build_test_block()
    inputs = torch.tensor([[0.1, 0.2], [0.85, 0.9]], dtype=torch.float32)

    outputs, trace = block.forward_with_trace(inputs)
    memberships = [variable(inputs[:, index]) for index, variable in enumerate(block.variables)]
    manual_raw = []
    for rule_index, rule in enumerate(block.rule_base.rules):
        log_weight = F.logsigmoid(block.rule_logits[rule_index]).expand(inputs.size(0))
        for antecedent in rule.antecedents:
            degree = memberships[antecedent.variable_index][:, antecedent.term_index].clamp_min(block.epsilon)
            log_weight = log_weight + degree.log()
        manual_raw.append(log_weight.exp())
    manual_raw_weights = torch.stack(manual_raw, dim=1)
    manual_normalized = manual_raw_weights / (manual_raw_weights.sum(dim=1, keepdim=True) + block.epsilon)
    manual_outputs = manual_normalized @ block.consequents

    assert torch.allclose(outputs, manual_outputs, atol=1e-6)
    assert torch.allclose(
        manual_raw_weights[:, 0],
        block.rule_probabilities[0] * memberships[0][:, 0] * memberships[1][:, 0],
        atol=1e-6,
    )
    assert torch.allclose(trace.normalized_rule_weights.sum(dim=1), torch.ones(inputs.size(0)), atol=1e-6)
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)


def test_sugeno_layer_can_bound_hidden_outputs() -> None:
    variables = [
        FuzzyVariable("x0", GaussianMembership([0.0, 1.0], [0.25, 0.25], term_names=["low", "high"])),
    ]
    rules = RuleBase(
        [
            RuleSpec((Antecedent(0, 0),), name="low"),
            RuleSpec((Antecedent(0, 1),), name="high"),
        ]
    )
    layer = SugenoDecisionLayer(
        name="hidden_sugeno",
        variables=variables,
        rule_base=rules,
        output_dim=1,
        output_activation="sigmoid",
    )
    with torch.no_grad():
        layer.rule_bias.fill_(8.0)
        layer.rule_weights.fill_(4.0)

    outputs = layer(torch.tensor([[0.0], [1.0]], dtype=torch.float32))

    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)
