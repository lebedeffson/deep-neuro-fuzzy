from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Sequence

import torch
from torch import Tensor

from .memberships import FuzzyVariable


@dataclass(frozen=True)
class Antecedent:
    variable_index: int
    term_index: int

    def __post_init__(self) -> None:
        if self.variable_index < 0:
            raise ValueError("Variable indices must be non-negative.")
        if self.term_index < 0:
            raise ValueError("Term indices must be non-negative.")


@dataclass(frozen=True)
class RuleSpec:
    antecedents: tuple[Antecedent, ...]
    name: str | None = None
    gate_init: float = 0.0

    def __post_init__(self) -> None:
        if not self.antecedents:
            raise ValueError("Each rule must contain at least one antecedent.")
        variable_indices = [antecedent.variable_index for antecedent in self.antecedents]
        if len(set(variable_indices)) != len(variable_indices):
            raise ValueError("A rule cannot mention the same variable more than once.")


@dataclass(frozen=True)
class RuleBase:
    rules: tuple[RuleSpec, ...]

    def __init__(self, rules: Sequence[RuleSpec]) -> None:
        rules_tuple = tuple(rules)
        if not rules_tuple:
            raise ValueError("A rule base must contain at least one rule.")
        object.__setattr__(self, "rules", rules_tuple)

    def __len__(self) -> int:
        return len(self.rules)

    @property
    def names(self) -> tuple[str, ...]:
        names: list[str] = []
        for index, rule in enumerate(self.rules):
            names.append(rule.name or f"rule_{index}")
        return tuple(names)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(len(rule.antecedents) for rule in self.rules)

    def validate(self, term_counts: Sequence[int]) -> None:
        for rule_index, rule in enumerate(self.rules):
            for antecedent in rule.antecedents:
                if antecedent.variable_index >= len(term_counts):
                    raise ValueError(
                        f"Rule {rule_index} references variable {antecedent.variable_index}, "
                        f"but only {len(term_counts)} variables are available."
                    )
                if antecedent.term_index >= term_counts[antecedent.variable_index]:
                    raise ValueError(
                        f"Rule {rule_index} references term {antecedent.term_index} of variable "
                        f"{antecedent.variable_index}, but only "
                        f"{term_counts[antecedent.variable_index]} terms are available."
                    )


def count_rule_candidates(term_counts: Sequence[int], max_rule_arity: int | None = None) -> int:
    if not term_counts:
        raise ValueError("term_counts must not be empty.")
    if any(term_count <= 0 for term_count in term_counts):
        raise ValueError("All term counts must be positive.")

    max_arity = len(term_counts) if max_rule_arity is None else max_rule_arity
    if max_arity <= 0:
        raise ValueError("max_rule_arity must be positive.")
    if max_arity > len(term_counts):
        raise ValueError("max_rule_arity cannot exceed the number of variables.")

    total = 0
    for arity in range(1, max_arity + 1):
        for variable_indices in combinations(range(len(term_counts)), arity):
            rule_count = 1
            for variable_index in variable_indices:
                rule_count *= term_counts[variable_index]
            total += rule_count
    return total


def generate_rule_base(
    term_counts: Sequence[int],
    max_rule_arity: int | None = None,
    max_rules: int | None = None,
    gate_init: float = 0.0,
    name_prefix: str = "rule",
) -> RuleBase:
    if max_rules is not None and max_rules <= 0:
        raise ValueError("max_rules must be positive when provided.")

    max_arity = len(term_counts) if max_rule_arity is None else max_rule_arity
    candidate_count = count_rule_candidates(term_counts, max_rule_arity=max_arity)
    target_rule_count = candidate_count if max_rules is None else min(candidate_count, max_rules)

    rules: list[RuleSpec] = []
    for arity in range(1, max_arity + 1):
        for variable_indices in combinations(range(len(term_counts)), arity):
            term_ranges = [range(term_counts[variable_index]) for variable_index in variable_indices]
            for term_indices in product(*term_ranges):
                antecedents = tuple(
                    Antecedent(variable_index=variable_index, term_index=term_index)
                    for variable_index, term_index in zip(variable_indices, term_indices, strict=True)
                )
                rules.append(
                    RuleSpec(
                        antecedents=antecedents,
                        name=f"{name_prefix}_{len(rules)}",
                        gate_init=gate_init,
                    )
                )
                if len(rules) >= target_rule_count:
                    return RuleBase(rules)

    return RuleBase(rules)


def _validate_generation_limits(
    term_counts: Sequence[int],
    max_rule_arity: int | None,
    max_rules: int | None,
) -> tuple[tuple[int, ...], int, int]:
    term_counts_tuple = tuple(int(term_count) for term_count in term_counts)
    if max_rules is not None and max_rules <= 0:
        raise ValueError("max_rules must be positive when provided.")

    max_arity = len(term_counts_tuple) if max_rule_arity is None else max_rule_arity
    candidate_count = count_rule_candidates(term_counts_tuple, max_rule_arity=max_arity)
    target_rule_count = candidate_count if max_rules is None else min(candidate_count, max_rules)
    return term_counts_tuple, max_arity, target_rule_count


def _validate_prototype_inputs(
    variables: Sequence[FuzzyVariable],
    inputs: Tensor,
) -> None:
    if not variables:
        raise ValueError("Prototype-based rule generation requires at least one variable.")
    if inputs.ndim != 2:
        raise ValueError(
            f"Prototype inputs must have shape [samples, features], got {tuple(inputs.shape)}."
        )
    if inputs.size(0) <= 0:
        raise ValueError("Prototype inputs must contain at least one sample.")
    if inputs.size(1) != len(variables):
        raise ValueError(
            f"Expected {len(variables)} prototype features, got {inputs.size(1)}."
        )


def _prototype_membership_scores(
    variables: Sequence[FuzzyVariable],
    inputs: Tensor,
) -> tuple[Tensor, ...]:
    with torch.no_grad():
        detached_inputs = inputs.detach()
        memberships = tuple(
            variable(detached_inputs[:, index]).detach().cpu()
            for index, variable in enumerate(variables)
        )
    return memberships


def _rule_signature(antecedents: tuple[Antecedent, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((antecedent.variable_index, antecedent.term_index) for antecedent in antecedents)


def generate_prototype_rule_base(
    variables: Sequence[FuzzyVariable],
    inputs: Tensor,
    max_rule_arity: int | None = None,
    max_rules: int | None = None,
    gate_init: float = 0.0,
    name_prefix: str = "rule",
    top_terms_per_variable: int = 2,
    variable_pool_size: int | None = None,
) -> RuleBase:
    if top_terms_per_variable <= 0:
        raise ValueError("top_terms_per_variable must be positive.")

    _validate_prototype_inputs(variables, inputs)
    _, max_arity, target_rule_count = _validate_generation_limits(
        [variable.n_terms for variable in variables],
        max_rule_arity=max_rule_arity,
        max_rules=max_rules,
    )
    membership_scores = _prototype_membership_scores(variables, inputs)

    if variable_pool_size is None:
        pool_size = len(variables) if max_arity >= len(variables) else min(len(variables), max_arity + 1)
    else:
        pool_size = int(variable_pool_size)
    if pool_size <= 0:
        raise ValueError("variable_pool_size must be positive when provided.")
    pool_size = max(max_arity, min(pool_size, len(variables)))

    top_term_values: list[Tensor] = []
    top_term_indices: list[Tensor] = []
    for membership in membership_scores:
        k = min(top_terms_per_variable, membership.size(1))
        values, indices = torch.topk(membership, k=k, dim=1)
        top_term_values.append(values)
        top_term_indices.append(indices)

    candidate_scores: dict[tuple[Antecedent, ...], float] = {}
    sample_count = inputs.size(0)
    for sample_index in range(sample_count):
        variable_strengths = torch.tensor(
            [top_term_values[variable_index][sample_index, 0].item() for variable_index in range(len(variables))],
            dtype=torch.float32,
        )
        pooled_variables = torch.argsort(variable_strengths, descending=True).tolist()[:pool_size]

        for arity in range(1, max_arity + 1):
            for variable_indices in combinations(pooled_variables, arity):
                term_index_options = [
                    top_term_indices[variable_index][sample_index].tolist() for variable_index in variable_indices
                ]
                term_value_options = [
                    top_term_values[variable_index][sample_index].tolist() for variable_index in variable_indices
                ]
                for term_choice_indices in product(*[range(len(options)) for options in term_index_options]):
                    score = 1.0
                    antecedent_pairs: list[tuple[int, int]] = []
                    for local_index, choice_index in enumerate(term_choice_indices):
                        variable_index = variable_indices[local_index]
                        term_index = int(term_index_options[local_index][choice_index])
                        score *= float(term_value_options[local_index][choice_index])
                        antecedent_pairs.append((int(variable_index), term_index))
                    antecedents = tuple(
                        Antecedent(variable_index=variable_index, term_index=term_index)
                        for variable_index, term_index in sorted(antecedent_pairs)
                    )
                    previous_score = candidate_scores.get(antecedents)
                    if previous_score is None or score > previous_score:
                        candidate_scores[antecedents] = score

    ranked_candidates = sorted(
        candidate_scores.items(),
        key=lambda item: (-len(item[0]), -item[1], _rule_signature(item[0])),
    )
    if not ranked_candidates:
        raise ValueError("Prototype-based rule generation did not produce any rule candidates.")

    rules = [
        RuleSpec(
            antecedents=antecedents,
            name=f"{name_prefix}_{rule_index}",
            gate_init=gate_init,
        )
        for rule_index, (antecedents, _) in enumerate(ranked_candidates[:target_rule_count])
    ]
    return RuleBase(rules)
