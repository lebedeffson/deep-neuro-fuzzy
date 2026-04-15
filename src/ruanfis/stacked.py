from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .blocks import SugenoDecisionLayer
from .builders import DecisionLayerConfig, build_decision_layer
from .memberships import FuzzyVariable
from .traces import BlockTrace


@dataclass(frozen=True)
class StackedAnfisLayerConfig:
    name: str
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
class StackedAnfisModelConfig:
    input_dim: int
    layers: tuple[StackedAnfisLayerConfig, ...]


class StackedAnfisModel(nn.Module):
    def __init__(
        self,
        layers: tuple[SugenoDecisionLayer, ...],
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("Stacked ANFIS requires at least one layer.")
        self.layers = nn.ModuleList(layers)
        self.input_dim = input_dim

        if self.input_dim is not None:
            current_dim = self.input_dim
            for layer in self.layers:
                if layer.input_dim != current_dim:
                    raise ValueError(
                        f"Layer '{layer.name}' expects {layer.input_dim} inputs, but current stacked width is {current_dim}."
                    )
                current_dim = layer.output_dim

    @property
    def output_dim(self) -> int:
        return self.layers[-1].output_dim

    def forward_features(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        features = inputs
        for layer in self.layers[:-1]:
            features = layer(features, top_k_rules=top_k_rules)
        return features

    def forward(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, top_k_rules=top_k_rules)
        return outputs

    def forward_with_trace(
        self,
        inputs: Tensor,
        top_k_rules: int | None = None,
    ) -> tuple[Tensor, tuple[BlockTrace, ...]]:
        outputs = inputs
        traces: list[BlockTrace] = []
        for layer in self.layers:
            outputs, trace = layer.forward_with_trace(outputs, top_k_rules=top_k_rules)
            traces.append(trace)
        return outputs, tuple(traces)

    def iter_rule_layers(self) -> tuple[SugenoDecisionLayer, ...]:
        return tuple(self.layers)

    def iter_rule_layer_entries(self) -> tuple[tuple[str, SugenoDecisionLayer], ...]:
        return tuple((layer.name, layer) for layer in self.layers)


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


def _as_decision_config(layer: StackedAnfisLayerConfig) -> DecisionLayerConfig:
    return DecisionLayerConfig(
        name=layer.name,
        variables=layer.variables,
        output_dim=layer.output_dim,
        output_names=layer.output_names,
        max_rule_arity=layer.max_rule_arity,
        max_rules=layer.max_rules,
        gate_init=layer.gate_init,
        rule_name_prefix=layer.rule_name_prefix,
        rule_generation_mode=layer.rule_generation_mode,
        prototype_term_limit=layer.prototype_term_limit,
        prototype_variable_pool_size=layer.prototype_variable_pool_size,
        prototype_sample_size=layer.prototype_sample_size,
    )


def build_stacked_anfis_model(
    config: StackedAnfisModelConfig,
    sample_inputs: Tensor | None = None,
) -> StackedAnfisModel:
    current_samples = _validate_sample_inputs(sample_inputs, config.input_dim)

    layers: list[SugenoDecisionLayer] = []
    current_width = config.input_dim
    for layer_config in config.layers:
        if len(layer_config.variables) != current_width:
            raise ValueError(
                f"Layer '{layer_config.name}' expects {len(layer_config.variables)} variables, "
                f"but stacked width before this layer is {current_width}."
            )
        layer = build_decision_layer(_as_decision_config(layer_config), sample_inputs=current_samples)
        layers.append(layer)
        current_width = layer.output_dim

        if current_samples is not None:
            with torch.no_grad():
                current_samples = layer(current_samples)

    return StackedAnfisModel(layers=tuple(layers), input_dim=config.input_dim)
