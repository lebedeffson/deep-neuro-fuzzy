import torch

from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.rules import count_rule_candidates, generate_prototype_rule_base


def _var3(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.15, 0.5, 0.85], [0.16, 0.16, 0.16], term_names=["low", "mid", "high"]),
    )


def test_generate_prototype_rule_base_returns_unique_supported_rules() -> None:
    variables = (_var3("x0"), _var3("x1"), _var3("x2"))
    inputs = torch.tensor(
        [
            [0.10, 0.15, 0.20],
            [0.80, 0.25, 0.75],
            [0.55, 0.85, 0.45],
            [0.18, 0.72, 0.82],
            [0.48, 0.42, 0.52],
            [0.87, 0.88, 0.15],
        ],
        dtype=torch.float32,
    )

    rule_base = generate_prototype_rule_base(
        variables=variables,
        inputs=inputs,
        max_rule_arity=2,
        max_rules=6,
        name_prefix="proto",
        top_terms_per_variable=2,
        variable_pool_size=3,
    )

    full_rule_count = count_rule_candidates([3, 3, 3], max_rule_arity=2)
    signatures = {
        tuple((antecedent.variable_index, antecedent.term_index) for antecedent in rule.antecedents)
        for rule in rule_base.rules
    }

    assert len(rule_base) == 6
    assert len(signatures) == len(rule_base)
    assert rule_base.names[0] == "proto_0"
    assert all(1 <= len(rule.antecedents) <= 2 for rule in rule_base.rules)
    assert len(rule_base) < full_rule_count
