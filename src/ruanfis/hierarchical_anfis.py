from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .builders import DecisionLayerConfig, build_decision_layer
from .memberships import FuzzyVariable
from .model import DeepFuzzyFeatureModel
from .stages import ConnectedFuzzyBlock, FuzzyStage


@dataclass(frozen=True)
class HierarchicalAnfisBlockConfig:
    name: str
    input_indices: tuple[int, ...]
    variables: tuple[FuzzyVariable, ...]
    output_dim: int
    output_names: tuple[str, ...] | None = None
    max_rule_arity: int | None = None
    max_rules: int | None = None
    gate_init: float = 0.0
    rule_name_prefix: str | None = None
    rule_generation_mode: str = "enumerate"
    prototype_term_limit: int = 2
    prototype_variable_pool_size: int | None = None
    prototype_sample_size: int | None = 256


@dataclass(frozen=True)
class HierarchicalAnfisStageConfig:
    name: str
    blocks: tuple[HierarchicalAnfisBlockConfig, ...]


@dataclass(frozen=True)
class HierarchicalAnfisModelConfig:
    input_dim: int
    stages: tuple[HierarchicalAnfisStageConfig, ...]
    decision_layer: DecisionLayerConfig


def _validate_sample_inputs(sample_inputs: Tensor | None, input_dim: int) -> Tensor | None:
    if sample_inputs is None:
        return None
    if sample_inputs.ndim != 2:
        raise ValueError(
            f"sample_inputs must have shape [samples, features], got {tuple(sample_inputs.shape)}."
        )
    if sample_inputs.size(1) != input_dim:
        raise ValueError(
            f"Expected sample_inputs with {input_dim} features, got {sample_inputs.size(1)}."
        )
    return sample_inputs.detach().cpu().to(dtype=torch.float32)


def _as_decision_config(block: HierarchicalAnfisBlockConfig) -> DecisionLayerConfig:
    return DecisionLayerConfig(
        name=block.name,
        variables=block.variables,
        output_dim=block.output_dim,
        output_names=block.output_names,
        max_rule_arity=block.max_rule_arity,
        max_rules=block.max_rules,
        gate_init=block.gate_init,
        rule_name_prefix=block.rule_name_prefix,
        rule_generation_mode=block.rule_generation_mode,
        prototype_term_limit=block.prototype_term_limit,
        prototype_variable_pool_size=block.prototype_variable_pool_size,
        prototype_sample_size=block.prototype_sample_size,
    )


def _build_connected_anfis_block(
    block: HierarchicalAnfisBlockConfig,
    sample_inputs: Tensor | None,
) -> ConnectedFuzzyBlock:
    if len(block.input_indices) != len(block.variables):
        raise ValueError(
            f"Block '{block.name}' expects {len(block.variables)} variables but received "
            f"{len(block.input_indices)} input indices."
        )

    local_inputs = None
    if sample_inputs is not None:
        local_inputs = sample_inputs.index_select(
            dim=1,
            index=torch.tensor(block.input_indices, dtype=torch.long, device=sample_inputs.device),
        )

    layer = build_decision_layer(_as_decision_config(block), sample_inputs=local_inputs)
    return ConnectedFuzzyBlock(layer, input_indices=block.input_indices)


def build_hierarchical_anfis_model(
    config: HierarchicalAnfisModelConfig,
    sample_inputs: Tensor | None = None,
) -> DeepFuzzyFeatureModel:
    current_samples = _validate_sample_inputs(sample_inputs, config.input_dim)

    stages: list[FuzzyStage] = []
    current_width = config.input_dim
    for stage_config in config.stages:
        stage = FuzzyStage(
            name=stage_config.name,
            blocks=[
                _build_connected_anfis_block(block_config, sample_inputs=current_samples)
                for block_config in stage_config.blocks
            ],
        )
        stage.validate_input_dim(current_width)
        stages.append(stage)
        current_width = stage.output_dim

        if current_samples is not None:
            with torch.no_grad():
                current_samples = stage(current_samples)

    if len(config.decision_layer.variables) != current_width:
        raise ValueError(
            f"Decision layer expects {len(config.decision_layer.variables)} variables, "
            f"but final hierarchical width is {current_width}."
        )

    decision_layer = build_decision_layer(config.decision_layer, sample_inputs=current_samples)
    return DeepFuzzyFeatureModel(stages=stages, decision_layer=decision_layer, input_dim=config.input_dim)
