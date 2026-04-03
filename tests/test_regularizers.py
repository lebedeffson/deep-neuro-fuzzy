import torch

from ruanfis.blocks import TransparentFuzzyBlock
from ruanfis.memberships import FuzzyVariable, GaussianMembership, GeneralizedBellMembership
from ruanfis.regularizers import (
    concept_binarization_penalty,
    concept_orthogonality_penalty,
    membership_center_order_penalty,
    membership_coverage_penalty,
    membership_overlap_penalty,
)
from ruanfis.rules import Antecedent, RuleBase, RuleSpec


def test_membership_regularizers_support_gaussian_and_bell() -> None:
    gaussian = GaussianMembership([0.0, 0.1], [0.4, 0.4], term_names=["low", "high"])
    bell = GeneralizedBellMembership([0.0, 0.05], [0.5, 0.5], [2.0, 2.0], term_names=["a", "b"])
    model = torch.nn.Module()
    model.gaussian = gaussian
    model.bell = bell

    order_penalty = membership_center_order_penalty(model, min_gap=0.2)
    overlap_penalty = membership_overlap_penalty(model, max_overlap=0.2, num_points=48)
    coverage_penalty = membership_coverage_penalty(model, min_coverage=0.9, num_points=48)

    assert order_penalty.item() > 0.0
    assert overlap_penalty.item() >= 0.0
    assert coverage_penalty.item() >= 0.0


def test_concept_regularizers_are_nonnegative_for_transparent_block() -> None:
    block = TransparentFuzzyBlock(
        name="feature_block",
        variables=[
            FuzzyVariable("x0", GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"])),
            FuzzyVariable("x1", GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"])),
        ],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="ll"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="hh"),
            ]
        ),
        n_concepts=2,
        concept_names=["signal", "risk"],
    )

    orthogonality = concept_orthogonality_penalty(block)
    binarization = concept_binarization_penalty(block)

    assert orthogonality.item() >= 0.0
    assert binarization.item() >= 0.0
