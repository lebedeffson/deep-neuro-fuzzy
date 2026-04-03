from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .memberships import FuzzyVariable
from .rules import RuleBase, RuleSpec
from .traces import BlockTrace, VariableTrace


class BaseFuzzyRuleLayer(nn.Module):
    def __init__(
        self,
        name: str,
        variables: Sequence[FuzzyVariable],
        rule_base: RuleBase | Sequence[RuleSpec],
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if not variables:
            raise ValueError("Each fuzzy layer must contain at least one input variable.")
        self.name = name
        self.variables = nn.ModuleList(variables)
        self.rule_base = rule_base if isinstance(rule_base, RuleBase) else RuleBase(rule_base)
        self.rule_base.validate([variable.n_terms for variable in self.variables])
        self.epsilon = float(epsilon)
        gate_inits = torch.tensor(
            [rule.gate_init for rule in self.rule_base.rules],
            dtype=torch.float32,
        )
        self.rule_logits = nn.Parameter(gate_inits)

    @property
    def input_dim(self) -> int:
        return len(self.variables)

    @property
    def n_rules(self) -> int:
        return len(self.rule_base)

    @property
    def rule_probabilities(self) -> Tensor:
        return torch.sigmoid(self.rule_logits)

    def _validate_inputs(self, inputs: Tensor) -> None:
        if inputs.ndim != 2:
            raise ValueError(f"Expected a batch matrix of shape [batch, features], got {tuple(inputs.shape)}.")
        if inputs.size(1) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features for layer '{self.name}', got {inputs.size(1)}."
            )

    def describe_rule(self, rule_index: int) -> str:
        if rule_index < 0 or rule_index >= self.n_rules:
            raise IndexError(f"Rule index {rule_index} is out of range for layer '{self.name}'.")
        rule = self.rule_base.rules[rule_index]
        parts: list[str] = []
        for antecedent in rule.antecedents:
            variable = self.variables[antecedent.variable_index]
            parts.append(f"{variable.name} IS {variable.term_names[antecedent.term_index]}")
        return " AND ".join(parts)

    def _fuzzify(self, inputs: Tensor) -> tuple[Tensor, ...]:
        self._validate_inputs(inputs)
        return tuple(variable(inputs[:, index]) for index, variable in enumerate(self.variables))

    def _compute_raw_rule_weights(self, memberships: tuple[Tensor, ...]) -> Tensor:
        batch_size = memberships[0].size(0)
        raw_weights: list[Tensor] = []
        for rule_index, rule in enumerate(self.rule_base.rules):
            log_weight = F.logsigmoid(self.rule_logits[rule_index]).expand(batch_size)
            for antecedent in rule.antecedents:
                degree = memberships[antecedent.variable_index][:, antecedent.term_index].clamp_min(self.epsilon)
                log_weight = log_weight + degree.log()
            raw_weights.append(log_weight.exp())
        return torch.stack(raw_weights, dim=1)

    def _normalize_rule_weights(self, raw_rule_weights: Tensor, top_k_rules: int | None = None) -> Tensor:
        if top_k_rules is not None:
            if top_k_rules <= 0:
                raise ValueError("top_k_rules must be positive when provided.")
            if top_k_rules < self.n_rules:
                values, indices = torch.topk(raw_rule_weights, k=top_k_rules, dim=1)
                pruned_weights = torch.zeros_like(raw_rule_weights)
                pruned_weights.scatter_(1, indices, values)
                raw_rule_weights = pruned_weights
        return raw_rule_weights / (raw_rule_weights.sum(dim=1, keepdim=True) + self.epsilon)

    def _build_variable_traces(self, memberships: tuple[Tensor, ...]) -> tuple[VariableTrace, ...]:
        traces: list[VariableTrace] = []
        for variable, values in zip(self.variables, memberships, strict=True):
            traces.append(
                VariableTrace(
                    variable_name=variable.name,
                    term_names=variable.term_names,
                    memberships=values.detach(),
                )
            )
        return tuple(traces)


class TransparentFuzzyBlock(BaseFuzzyRuleLayer):
    def __init__(
        self,
        name: str,
        variables: Sequence[FuzzyVariable],
        rule_base: RuleBase | Sequence[RuleSpec],
        n_concepts: int,
        concept_names: Sequence[str] | None = None,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(name=name, variables=variables, rule_base=rule_base, epsilon=epsilon)
        if n_concepts <= 0:
            raise ValueError("The number of concepts must be positive.")
        if concept_names is not None and len(concept_names) != n_concepts:
            raise ValueError("The number of concept names must match n_concepts.")
        self.n_concepts = int(n_concepts)
        self.output_names = tuple(concept_names) if concept_names is not None else tuple(
            f"{name}_concept_{index}" for index in range(self.n_concepts)
        )
        self.raw_consequents = nn.Parameter(torch.zeros(self.n_rules, self.n_concepts))
        nn.init.normal_(self.raw_consequents, mean=0.0, std=0.15)

    @property
    def output_dim(self) -> int:
        return self.n_concepts

    @property
    def consequents(self) -> Tensor:
        return torch.sigmoid(self.raw_consequents)

    def _run(
        self,
        inputs: Tensor,
        top_k_rules: int | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor, Tensor]:
        memberships = self._fuzzify(inputs)
        raw_rule_weights = self._compute_raw_rule_weights(memberships)
        normalized_rule_weights = self._normalize_rule_weights(raw_rule_weights, top_k_rules=top_k_rules)
        outputs = normalized_rule_weights @ self.consequents
        return outputs, memberships, raw_rule_weights, normalized_rule_weights

    def forward(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        outputs, _, _, _ = self._run(inputs, top_k_rules=top_k_rules)
        return outputs

    def forward_with_trace(
        self,
        inputs: Tensor,
        top_k_rules: int | None = None,
    ) -> tuple[Tensor, BlockTrace]:
        outputs, memberships, raw_rule_weights, normalized_rule_weights = self._run(
            inputs,
            top_k_rules=top_k_rules,
        )
        rule_outputs = self.consequents.unsqueeze(0).expand(inputs.size(0), -1, -1)
        trace = BlockTrace(
            block_name=self.name,
            inputs=inputs.detach(),
            variable_traces=self._build_variable_traces(memberships),
            rule_names=self.rule_base.names,
            raw_rule_weights=raw_rule_weights.detach(),
            normalized_rule_weights=normalized_rule_weights.detach(),
            rule_outputs=rule_outputs.detach(),
            outputs=outputs.detach(),
            output_names=self.output_names,
        )
        return outputs, trace


class SugenoDecisionLayer(BaseFuzzyRuleLayer):
    def __init__(
        self,
        name: str,
        variables: Sequence[FuzzyVariable],
        rule_base: RuleBase | Sequence[RuleSpec],
        output_dim: int,
        output_names: Sequence[str] | None = None,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(name=name, variables=variables, rule_base=rule_base, epsilon=epsilon)
        if output_dim <= 0:
            raise ValueError("The output dimension must be positive.")
        if output_names is not None and len(output_names) != output_dim:
            raise ValueError("The number of output names must match output_dim.")
        self._output_dim = int(output_dim)
        self.output_names = tuple(output_names) if output_names is not None else tuple(
            f"{name}_output_{index}" for index in range(self._output_dim)
        )
        self.rule_bias = nn.Parameter(torch.zeros(self.n_rules, self._output_dim))
        self.rule_weights = nn.Parameter(torch.empty(self.n_rules, self.input_dim, self._output_dim))
        nn.init.xavier_uniform_(self.rule_weights)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _run(
        self,
        inputs: Tensor,
        top_k_rules: int | None = None,
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor, Tensor, Tensor]:
        memberships = self._fuzzify(inputs)
        raw_rule_weights = self._compute_raw_rule_weights(memberships)
        normalized_rule_weights = self._normalize_rule_weights(raw_rule_weights, top_k_rules=top_k_rules)
        rule_outputs = torch.einsum("bi, rio -> bro", inputs, self.rule_weights) + self.rule_bias.unsqueeze(0)
        outputs = (normalized_rule_weights.unsqueeze(-1) * rule_outputs).sum(dim=1)
        return outputs, memberships, raw_rule_weights, normalized_rule_weights, rule_outputs

    def forward(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        outputs, _, _, _, _ = self._run(inputs, top_k_rules=top_k_rules)
        return outputs

    def forward_with_trace(
        self,
        inputs: Tensor,
        top_k_rules: int | None = None,
    ) -> tuple[Tensor, BlockTrace]:
        outputs, memberships, raw_rule_weights, normalized_rule_weights, rule_outputs = self._run(
            inputs,
            top_k_rules=top_k_rules,
        )
        trace = BlockTrace(
            block_name=self.name,
            inputs=inputs.detach(),
            variable_traces=self._build_variable_traces(memberships),
            rule_names=self.rule_base.names,
            raw_rule_weights=raw_rule_weights.detach(),
            normalized_rule_weights=normalized_rule_weights.detach(),
            rule_outputs=rule_outputs.detach(),
            outputs=outputs.detach(),
            output_names=self.output_names,
        )
        return outputs, trace
