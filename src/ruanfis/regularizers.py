from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import BaseFuzzyRuleLayer, TransparentFuzzyBlock
from .memberships import GaussianMembership, GeneralizedBellMembership


def _zero_for(module: nn.Module) -> Tensor:
    try:
        parameter = next(module.parameters())
        return parameter.new_tensor(0.0)
    except StopIteration:
        return torch.tensor(0.0)


def _iter_rule_probability_tensors(module: nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, BaseFuzzyRuleLayer):
            yield submodule.rule_probabilities
            continue
        rule_probabilities = getattr(submodule, "rule_probabilities", None)
        if rule_probabilities is None:
            continue
        if isinstance(rule_probabilities, Tensor):
            yield rule_probabilities


def rule_sparsity_penalty(module: nn.Module) -> Tensor:
    total = _zero_for(module)
    for probabilities in _iter_rule_probability_tensors(module):
        total = total + probabilities.sum()
    return total


def weighted_rule_length_penalty(module: nn.Module) -> Tensor:
    total = _zero_for(module)
    for submodule in module.modules():
        if isinstance(submodule, BaseFuzzyRuleLayer):
            lengths = torch.tensor(
                submodule.rule_base.lengths,
                dtype=submodule.rule_logits.dtype,
                device=submodule.rule_logits.device,
            )
            total = total + (torch.sigmoid(submodule.rule_logits) * lengths).sum()
    return total


def membership_center_order_penalty(module: nn.Module, min_gap: float = 0.0) -> Tensor:
    total = _zero_for(module)
    target_gap = float(min_gap)
    for submodule in module.modules():
        if isinstance(submodule, (GaussianMembership, GeneralizedBellMembership)) and submodule.n_terms > 1:
            gaps = submodule.centers[1:] - submodule.centers[:-1]
            total = total + torch.relu(target_gap - gaps).sum()
    return total


def concept_orthogonality_penalty(module: nn.Module) -> Tensor:
    total = _zero_for(module)
    for submodule in module.modules():
        if not isinstance(submodule, TransparentFuzzyBlock):
            continue
        if submodule.n_concepts <= 1:
            continue
        concept_vectors = F.normalize(submodule.consequents.transpose(0, 1), p=2.0, dim=1)
        gram = concept_vectors @ concept_vectors.transpose(0, 1)
        identity = torch.eye(
            gram.size(0),
            dtype=gram.dtype,
            device=gram.device,
        )
        total = total + ((gram - identity) ** 2).sum()
    return total


def concept_binarization_penalty(module: nn.Module) -> Tensor:
    total = _zero_for(module)
    for submodule in module.modules():
        if isinstance(submodule, TransparentFuzzyBlock):
            consequents = submodule.consequents
            total = total + (consequents * (1.0 - consequents)).mean()
    return total


def _membership_support_bounds(
    membership: GaussianMembership | GeneralizedBellMembership,
) -> tuple[Tensor, Tensor]:
    centers = membership.centers
    if isinstance(membership, GaussianMembership):
        radii = membership.spreads * 3.0
    else:
        radii = membership.widths * 2.5
    lower = (centers - radii).min()
    upper = (centers + radii).max()
    if float((upper - lower).abs().item()) < 1e-6:
        lower = lower - 1.0
        upper = upper + 1.0
    return lower, upper


def _membership_grid_values(
    membership: GaussianMembership | GeneralizedBellMembership,
    num_points: int,
) -> Tensor:
    if num_points <= 1:
        raise ValueError("num_points must be greater than one.")
    lower, upper = _membership_support_bounds(membership)
    grid = torch.linspace(
        float(lower.detach().cpu().item()),
        float(upper.detach().cpu().item()),
        steps=num_points,
        dtype=membership.centers.dtype,
        device=membership.centers.device,
    )
    return membership(grid)


def membership_overlap_penalty(
    module: nn.Module,
    max_overlap: float = 0.35,
    num_points: int = 64,
) -> Tensor:
    total = _zero_for(module)
    target_overlap = float(max_overlap)
    for submodule in module.modules():
        if not isinstance(submodule, (GaussianMembership, GeneralizedBellMembership)):
            continue
        if submodule.n_terms <= 1:
            continue
        values = _membership_grid_values(submodule, num_points=num_points)
        for left in range(submodule.n_terms - 1):
            right = left + 1
            overlap = (values[:, left] * values[:, right]).mean()
            total = total + torch.relu(overlap - target_overlap)
    return total


def membership_coverage_penalty(
    module: nn.Module,
    min_coverage: float = 0.6,
    num_points: int = 64,
) -> Tensor:
    total = _zero_for(module)
    target_coverage = float(min_coverage)
    for submodule in module.modules():
        if not isinstance(submodule, (GaussianMembership, GeneralizedBellMembership)):
            continue
        values = _membership_grid_values(submodule, num_points=num_points)
        coverage = values.max(dim=1).values.mean()
        total = total + torch.relu(target_coverage - coverage)
    return total


def block_gate_l1_penalty(module: nn.Module) -> Tensor:
    total = _zero_for(module)
    for submodule in module.modules():
        block_gate_logits = getattr(submodule, "block_gate_logits", None)
        if block_gate_logits is not None:
            total = total + torch.sigmoid(block_gate_logits).sum()
        final_skip_gate_logits = getattr(submodule, "final_skip_gate_logits", None)
        if final_skip_gate_logits is not None:
            total = total + torch.sigmoid(final_skip_gate_logits).sum()
    return total


def rule_activation_entropy_penalty(module: nn.Module, epsilon: float = 1e-8) -> Tensor:
    total = _zero_for(module)
    eps = float(max(epsilon, 1e-12))
    for probabilities in _iter_rule_probability_tensors(module):
        probs = probabilities.clamp(min=eps, max=1.0 - eps)
        entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
        total = total + entropy.mean()
    return total


def rule_anchor_stability_penalty(module: nn.Module) -> Tensor:
    total = _zero_for(module)
    for submodule in module.modules():
        anchor = getattr(submodule, "rule_probability_anchor", None)
        if anchor is None:
            continue
        probs = getattr(submodule, "rule_probabilities", None)
        if probs is None:
            continue
        if not isinstance(probs, Tensor):
            continue
        anchor_tensor = anchor.to(device=probs.device, dtype=probs.dtype)
        if anchor_tensor.shape != probs.shape:
            continue
        total = total + F.mse_loss(probs, anchor_tensor, reduction="mean")
    return total
