from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from .blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from .memberships import FuzzyVariable
from .model import DeepFuzzyFeatureModel
from .rules import count_rule_candidates, generate_prototype_rule_base, generate_rule_base
from .stages import ConnectedFuzzyBlock, FuzzyStage


@dataclass(frozen=True)
class TransparentBlockConfig:
    name: str
    input_indices: tuple[int, ...]
    variables: tuple[FuzzyVariable, ...]
    n_concepts: int
    concept_names: tuple[str, ...] | None = None
    max_rule_arity: int | None = None
    max_rules: int | None = None
    gate_init: float = 0.0
    rule_name_prefix: str | None = None
    rule_generation_mode: str = "enumerate"
    prototype_term_limit: int = 2
    prototype_scoring_mode: str = "max"
    prototype_variable_pool_size: int | None = None
    prototype_sample_size: int | None = 256
    consequent_mode: str = "constant"

    def term_counts(self) -> tuple[int, ...]:
        return tuple(variable.n_terms for variable in self.variables)

    def candidate_rule_count(self) -> int:
        return count_rule_candidates(self.term_counts(), max_rule_arity=self.max_rule_arity)

    def generated_rule_count(self) -> int:
        candidate_count = self.candidate_rule_count()
        return candidate_count if self.max_rules is None else min(candidate_count, self.max_rules)


@dataclass(frozen=True)
class StageConfig:
    name: str
    blocks: tuple[TransparentBlockConfig, ...]
    enable_block_gates: bool = False
    block_gate_init_logit: float = 5.0

    def generated_rule_count(self) -> int:
        return sum(block.generated_rule_count() for block in self.blocks)


@dataclass(frozen=True)
class DecisionLayerConfig:
    name: str
    variables: tuple[FuzzyVariable, ...]
    output_dim: int
    output_names: tuple[str, ...] | None = None
    output_activation: str | None = None
    max_rule_arity: int | None = None
    max_rules: int | None = None
    gate_init: float = 0.0
    rule_name_prefix: str | None = None
    rule_generation_mode: str = "enumerate"
    prototype_term_limit: int = 2
    prototype_scoring_mode: str = "max"
    prototype_variable_pool_size: int | None = None
    prototype_sample_size: int | None = 256

    def term_counts(self) -> tuple[int, ...]:
        return tuple(variable.n_terms for variable in self.variables)

    def candidate_rule_count(self) -> int:
        return count_rule_candidates(self.term_counts(), max_rule_arity=self.max_rule_arity)

    def generated_rule_count(self) -> int:
        candidate_count = self.candidate_rule_count()
        return candidate_count if self.max_rules is None else min(candidate_count, self.max_rules)


@dataclass(frozen=True)
class HierarchicalModelConfig:
    input_dim: int
    stages: tuple[StageConfig, ...]
    decision_layer: DecisionLayerConfig
    final_skip_input_indices: tuple[int, ...] = ()
    final_skip_gates_enabled: bool = False
    final_skip_gate_init_logit: float = 2.0

    def generated_rule_count(self) -> int:
        return sum(stage.generated_rule_count() for stage in self.stages) + self.decision_layer.generated_rule_count()


@dataclass(frozen=True)
class ShallowFuzzyModelConfig:
    input_dim: int
    feature_block: TransparentBlockConfig
    decision_layer: DecisionLayerConfig
    stage_name: str = "shallow_stage"

    def as_hierarchical_config(self) -> HierarchicalModelConfig:
        return HierarchicalModelConfig(
            input_dim=self.input_dim,
            stages=(
                StageConfig(
                    name=self.stage_name,
                    blocks=(self.feature_block,),
                ),
            ),
            decision_layer=self.decision_layer,
            final_skip_input_indices=(),
            final_skip_gates_enabled=False,
            final_skip_gate_init_logit=2.0,
        )

    def generated_rule_count(self) -> int:
        return self.as_hierarchical_config().generated_rule_count()


def _validate_rule_generation_mode(mode: str) -> None:
    if mode not in {"enumerate", "prototype"}:
        raise ValueError(
            f"Unsupported rule_generation_mode={mode!r}. Expected 'enumerate' or 'prototype'."
        )


def _maybe_subsample_inputs(inputs: Tensor, sample_size: int | None) -> Tensor:
    if sample_size is None or inputs.size(0) <= sample_size:
        return inputs
    if sample_size <= 0:
        raise ValueError("prototype_sample_size must be positive when provided.")
    # Deterministic stratified-like sampling: preserve coverage across the data manifold
    # without introducing run-to-run randomness in prototype rule generation.
    if inputs.ndim != 2:
        raise ValueError(f"Expected prototype inputs to be 2D, got shape {tuple(inputs.shape)}.")
    n_samples = inputs.size(0)
    n_features = inputs.size(1)
    if n_features <= 0:
        raise ValueError("prototype inputs must contain at least one feature.")

    # Build a stable 1D score for sorting. We use a fixed weighted projection over
    # the first few features to avoid expensive decomposition steps.
    projection_dim = min(n_features, 8)
    feature_weights = torch.linspace(
        1.0,
        2.0,
        steps=projection_dim,
        device=inputs.device,
        dtype=inputs.dtype,
    )
    projection_scores = inputs[:, :projection_dim] @ feature_weights
    sorted_indices = torch.argsort(projection_scores, descending=False)

    # Take one representative from each quantile bucket for broad coverage.
    bin_edges = torch.linspace(
        0,
        n_samples,
        steps=sample_size + 1,
        device=inputs.device,
        dtype=torch.float64,
    )
    selected_positions: list[int] = []
    for index in range(sample_size):
        left = int(torch.floor(bin_edges[index]).item())
        right = int(torch.floor(bin_edges[index + 1]).item())
        if right <= left:
            right = min(n_samples, left + 1)
        center = (left + right - 1) // 2
        center = max(0, min(n_samples - 1, center))
        selected_positions.append(center)

    selected_indices = sorted_indices[torch.tensor(selected_positions, device=inputs.device, dtype=torch.long)]
    return inputs.index_select(dim=0, index=selected_indices)


def _resolve_module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _build_rule_base_from_config(
    *,
    term_counts: Sequence[int],
    variables: Sequence[FuzzyVariable],
    inputs: Tensor | None,
    max_rule_arity: int | None,
    max_rules: int | None,
    gate_init: float,
    name_prefix: str,
    rule_generation_mode: str,
    prototype_term_limit: int,
    prototype_scoring_mode: str,
    prototype_variable_pool_size: int | None,
    prototype_sample_size: int | None,
):
    _validate_rule_generation_mode(rule_generation_mode)
    if rule_generation_mode == "enumerate":
        return generate_rule_base(
            term_counts,
            max_rule_arity=max_rule_arity,
            max_rules=max_rules,
            gate_init=gate_init,
            name_prefix=name_prefix,
        )

    if inputs is None:
        raise ValueError(
            "Prototype-based rule generation requires sample_inputs to be provided to the builder."
        )
    sampled_inputs = _maybe_subsample_inputs(inputs, prototype_sample_size)
    variable_device = _resolve_module_device(variables[0])
    sampled_inputs = sampled_inputs.to(device=variable_device, dtype=torch.float32)
    return generate_prototype_rule_base(
        variables=variables,
        inputs=sampled_inputs,
        max_rule_arity=max_rule_arity,
        max_rules=max_rules,
        gate_init=gate_init,
        name_prefix=name_prefix,
        top_terms_per_variable=prototype_term_limit,
        prototype_scoring_mode=prototype_scoring_mode,
        variable_pool_size=prototype_variable_pool_size,
    )


def build_transparent_block(
    config: TransparentBlockConfig,
    sample_inputs: Tensor | None = None,
) -> ConnectedFuzzyBlock:
    local_inputs = None if sample_inputs is None else sample_inputs.index_select(
        dim=1,
        index=torch.tensor(config.input_indices, dtype=torch.long, device=sample_inputs.device),
    )
    rule_base = _build_rule_base_from_config(
        term_counts=config.term_counts(),
        variables=config.variables,
        inputs=local_inputs,
        max_rule_arity=config.max_rule_arity,
        max_rules=config.max_rules,
        gate_init=config.gate_init,
        name_prefix=config.rule_name_prefix or config.name,
        rule_generation_mode=config.rule_generation_mode,
        prototype_term_limit=config.prototype_term_limit,
        prototype_scoring_mode=config.prototype_scoring_mode,
        prototype_variable_pool_size=config.prototype_variable_pool_size,
        prototype_sample_size=config.prototype_sample_size,
    )
    block = TransparentFuzzyBlock(
        name=config.name,
        variables=config.variables,
        rule_base=rule_base,
        n_concepts=config.n_concepts,
        concept_names=config.concept_names,
        consequent_mode=config.consequent_mode,
    )
    return ConnectedFuzzyBlock(block=block, input_indices=config.input_indices)


def build_stage(config: StageConfig, sample_inputs: Tensor | None = None) -> FuzzyStage:
    return FuzzyStage(
        name=config.name,
        blocks=[build_transparent_block(block_config, sample_inputs=sample_inputs) for block_config in config.blocks],
        enable_block_gates=config.enable_block_gates,
        block_gate_init_logit=config.block_gate_init_logit,
    )


def build_decision_layer(
    config: DecisionLayerConfig,
    sample_inputs: Tensor | None = None,
) -> SugenoDecisionLayer:
    rule_base = _build_rule_base_from_config(
        term_counts=config.term_counts(),
        variables=config.variables,
        inputs=sample_inputs,
        max_rule_arity=config.max_rule_arity,
        max_rules=config.max_rules,
        gate_init=config.gate_init,
        name_prefix=config.rule_name_prefix or config.name,
        rule_generation_mode=config.rule_generation_mode,
        prototype_term_limit=config.prototype_term_limit,
        prototype_scoring_mode=config.prototype_scoring_mode,
        prototype_variable_pool_size=config.prototype_variable_pool_size,
        prototype_sample_size=config.prototype_sample_size,
    )
    return SugenoDecisionLayer(
        name=config.name,
        variables=config.variables,
        rule_base=rule_base,
        output_dim=config.output_dim,
        output_names=config.output_names,
        output_activation=config.output_activation,
    )


def build_hierarchical_model(
    config: HierarchicalModelConfig,
    sample_inputs: Tensor | None = None,
) -> DeepFuzzyFeatureModel:
    if sample_inputs is not None:
        if sample_inputs.ndim != 2:
            raise ValueError(
                f"sample_inputs must have shape [samples, features], got {tuple(sample_inputs.shape)}."
            )
        if sample_inputs.size(1) != config.input_dim:
            raise ValueError(
                f"Expected sample_inputs with {config.input_dim} features, got {sample_inputs.size(1)}."
            )
        sample_inputs = sample_inputs.detach().cpu()

    stages: list[FuzzyStage] = []
    current_samples = sample_inputs
    original_samples = sample_inputs
    for stage_config in config.stages:
        stage = build_stage(stage_config, sample_inputs=current_samples)
        stages.append(stage)
        if current_samples is not None:
            with torch.no_grad():
                current_samples = stage(current_samples)

    decision_samples = current_samples
    if decision_samples is not None and original_samples is not None and config.final_skip_input_indices:
        skip_indices = torch.tensor(config.final_skip_input_indices, dtype=torch.long, device=original_samples.device)
        decision_samples = torch.cat((decision_samples, original_samples.index_select(1, skip_indices)), dim=1)

    decision_layer = build_decision_layer(config.decision_layer, sample_inputs=decision_samples)
    return DeepFuzzyFeatureModel(
        stages=stages,
        decision_layer=decision_layer,
        input_dim=config.input_dim,
        final_skip_input_indices=config.final_skip_input_indices,
        final_skip_gates_enabled=config.final_skip_gates_enabled,
        final_skip_gate_init_logit=config.final_skip_gate_init_logit,
    )


def build_shallow_fuzzy_model(
    config: ShallowFuzzyModelConfig,
    sample_inputs: Tensor | None = None,
) -> DeepFuzzyFeatureModel:
    return build_hierarchical_model(config.as_hierarchical_config(), sample_inputs=sample_inputs)
