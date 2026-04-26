from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import SugenoDecisionLayer
from .builders import DecisionLayerConfig, build_decision_layer
from .memberships import FuzzyVariable
from .metrics import TaskType
from .traces import BlockTrace


@dataclass(frozen=True)
class StackedAnfisLayerConfig:
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


@dataclass(frozen=True)
class StackedAnfisModelConfig:
    input_dim: int
    layers: tuple[StackedAnfisLayerConfig, ...]
    final_skip_input_indices: tuple[int, ...] = ()
    final_skip_gates_enabled: bool = False
    final_skip_gate_init_logit: float = 2.0


class StackedAnfisModel(nn.Module):
    def __init__(
        self,
        layers: tuple[SugenoDecisionLayer, ...],
        input_dim: int | None = None,
        final_skip_input_indices: tuple[int, ...] = (),
        final_skip_gates_enabled: bool = False,
        final_skip_gate_init_logit: float = 2.0,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("Stacked ANFIS requires at least one layer.")
        self.layers = nn.ModuleList(layers)
        self.input_dim = input_dim
        self.final_skip_input_indices = tuple(int(index) for index in final_skip_input_indices)
        self.final_skip_gates_enabled = bool(final_skip_gates_enabled)
        self.final_skip_gate_init_logit = float(final_skip_gate_init_logit)
        if len(set(self.final_skip_input_indices)) != len(self.final_skip_input_indices):
            raise ValueError("Final skip input indices must be unique.")
        if any(index < 0 for index in self.final_skip_input_indices):
            raise ValueError("Final skip input indices must be non-negative.")
        self.register_buffer(
            "_final_skip_input_index_tensor",
            torch.tensor(self.final_skip_input_indices, dtype=torch.long),
            persistent=False,
        )
        if self.final_skip_gates_enabled and len(self.final_skip_input_indices) > 0:
            self.final_skip_gate_logits = nn.Parameter(
                torch.full(
                    (len(self.final_skip_input_indices),),
                    fill_value=self.final_skip_gate_init_logit,
                    dtype=torch.float32,
                )
            )
        else:
            self.final_skip_gate_logits = None

        if self.input_dim is not None:
            current_dim = self.input_dim
            for layer_index, layer in enumerate(self.layers):
                is_final = layer_index == len(self.layers) - 1
                expected_dim = current_dim + len(self.final_skip_input_indices) if is_final else current_dim
                if layer.input_dim != expected_dim:
                    raise ValueError(
                        f"Layer '{layer.name}' expects {layer.input_dim} inputs, but current stacked width is "
                        f"{expected_dim}."
                    )
                if is_final:
                    break
                current_dim = layer.output_dim
            if self.final_skip_input_indices and max(self.final_skip_input_indices) >= self.input_dim:
                raise ValueError("Final skip input index is outside the model input dimension.")

    @property
    def output_dim(self) -> int:
        return self.layers[-1].output_dim

    def _append_final_skip_inputs(self, raw_inputs: Tensor, features: Tensor) -> Tensor:
        if not self.final_skip_input_indices:
            return features
        skip_indices = self._final_skip_input_index_tensor.to(device=raw_inputs.device)
        skip_features = raw_inputs.index_select(dim=1, index=skip_indices)
        if self.final_skip_gate_logits is not None:
            gates = torch.sigmoid(self.final_skip_gate_logits).to(device=raw_inputs.device, dtype=skip_features.dtype)
            skip_features = skip_features * gates.unsqueeze(0)
        return torch.cat((features, skip_features), dim=1)

    def forward_features(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        features = inputs
        for layer in self.layers[:-1]:
            features = layer(features, top_k_rules=top_k_rules)
        return self._append_final_skip_inputs(inputs, features)

    def forward(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        return self.layers[-1](self.forward_features(inputs, top_k_rules=top_k_rules), top_k_rules=top_k_rules)

    def forward_with_trace(
        self,
        inputs: Tensor,
        top_k_rules: int | None = None,
    ) -> tuple[Tensor, tuple[BlockTrace, ...]]:
        outputs = inputs
        traces: list[BlockTrace] = []
        for layer in self.layers[:-1]:
            outputs, trace = layer.forward_with_trace(outputs, top_k_rules=top_k_rules)
            traces.append(trace)
        outputs, trace = self.layers[-1].forward_with_trace(
            self._append_final_skip_inputs(inputs, outputs),
            top_k_rules=top_k_rules,
        )
        traces.append(trace)
        return outputs, tuple(traces)

    def iter_rule_layers(self) -> tuple[SugenoDecisionLayer, ...]:
        return tuple(self.layers)

    def iter_rule_layer_entries(self) -> tuple[tuple[str, SugenoDecisionLayer], ...]:
        return tuple((layer.name, layer) for layer in self.layers)


def _validate_sample_inputs(
    sample_inputs: Tensor | None,
    input_dim: int,
    *,
    device: str | torch.device | None = None,
) -> Tensor | None:
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
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return sample_inputs.detach().to(device=target_device, dtype=torch.float32)


def _validate_sample_targets(sample_targets: Tensor, output_dim: int, device: torch.device) -> Tensor:
    if sample_targets.ndim == 1:
        sample_targets = sample_targets.unsqueeze(-1)
    if sample_targets.ndim != 2:
        raise ValueError(
            f"sample_targets must have shape [samples, outputs], got {tuple(sample_targets.shape)}."
        )
    if sample_targets.size(1) != output_dim:
        raise ValueError(f"Expected sample_targets with {output_dim} outputs, got {sample_targets.size(1)}.")
    return sample_targets.detach().to(device=device, dtype=torch.float32)


def _as_decision_config(layer: StackedAnfisLayerConfig) -> DecisionLayerConfig:
    return DecisionLayerConfig(
        name=layer.name,
        variables=layer.variables,
        output_dim=layer.output_dim,
        output_names=layer.output_names,
        output_activation=layer.output_activation,
        max_rule_arity=layer.max_rule_arity,
        max_rules=layer.max_rules,
        gate_init=layer.gate_init,
        rule_name_prefix=layer.rule_name_prefix,
        rule_generation_mode=layer.rule_generation_mode,
        prototype_term_limit=layer.prototype_term_limit,
        prototype_scoring_mode=layer.prototype_scoring_mode,
        prototype_variable_pool_size=layer.prototype_variable_pool_size,
        prototype_sample_size=layer.prototype_sample_size,
    )


def _task_loss(task_type: TaskType, predictions: Tensor, targets: Tensor) -> Tensor:
    if task_type == "regression":
        return F.mse_loss(predictions, targets)
    if task_type == "binary_classification":
        return F.binary_cross_entropy_with_logits(predictions, targets)
    raise ValueError(f"Unsupported task_type={task_type!r}.")


def _iter_batches(inputs: Tensor, targets: Tensor, batch_size: int | None, shuffle: bool):
    effective_batch_size = inputs.size(0) if batch_size is None else int(batch_size)
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    indices = torch.randperm(inputs.size(0), device=inputs.device) if shuffle else torch.arange(
        inputs.size(0),
        device=inputs.device,
    )
    for start in range(0, inputs.size(0), effective_batch_size):
        batch_indices = indices[start : start + effective_batch_size]
        yield inputs.index_select(0, batch_indices), targets.index_select(0, batch_indices)


def _fit_hidden_layer_with_linear_head(
    layer: SugenoDecisionLayer,
    inputs: Tensor,
    targets: Tensor,
    *,
    task_type: TaskType,
    epochs: int,
    learning_rate: float,
    batch_size: int | None,
    shuffle: bool,
) -> None:
    if epochs <= 0:
        return
    layer.train()
    head = nn.Linear(layer.output_dim, targets.size(1)).to(device=inputs.device, dtype=inputs.dtype)
    optimizer = torch.optim.Adam(list(layer.parameters()) + list(head.parameters()), lr=float(learning_rate))
    for _ in range(int(epochs)):
        for batch_inputs, batch_targets in _iter_batches(inputs, targets, batch_size=batch_size, shuffle=shuffle):
            optimizer.zero_grad()
            predictions = head(layer(batch_inputs))
            loss = _task_loss(task_type, predictions, batch_targets)
            loss.backward()
            optimizer.step()
    layer.eval()


def build_stacked_anfis_model(
    config: StackedAnfisModelConfig,
    sample_inputs: Tensor | None = None,
) -> StackedAnfisModel:
    current_samples = _validate_sample_inputs(sample_inputs, config.input_dim)
    original_samples = current_samples

    layers: list[SugenoDecisionLayer] = []
    current_width = config.input_dim
    for layer_index, layer_config in enumerate(config.layers):
        is_final = layer_index == len(config.layers) - 1
        expected_width = current_width + len(config.final_skip_input_indices) if is_final else current_width
        if len(layer_config.variables) != expected_width:
            raise ValueError(
                f"Layer '{layer_config.name}' expects {len(layer_config.variables)} variables, "
                f"but stacked width before this layer is {expected_width}."
            )
        layer_samples = current_samples
        if is_final and current_samples is not None and original_samples is not None and config.final_skip_input_indices:
            skip_indices = torch.tensor(config.final_skip_input_indices, dtype=torch.long, device=original_samples.device)
            layer_samples = torch.cat((current_samples, original_samples.index_select(dim=1, index=skip_indices)), dim=1)
        layer = build_decision_layer(_as_decision_config(layer_config), sample_inputs=layer_samples)
        layers.append(layer)
        current_width = layer.output_dim

        if layer_samples is not None:
            with torch.no_grad():
                current_samples = layer(layer_samples)

    return StackedAnfisModel(
        layers=tuple(layers),
        input_dim=config.input_dim,
        final_skip_input_indices=config.final_skip_input_indices,
        final_skip_gates_enabled=config.final_skip_gates_enabled,
        final_skip_gate_init_logit=config.final_skip_gate_init_logit,
    )


def build_stagewise_initialized_stacked_anfis_model(
    config: StackedAnfisModelConfig,
    sample_inputs: Tensor,
    sample_targets: Tensor,
    *,
    task_type: TaskType,
    epochs_per_hidden_layer: int = 12,
    learning_rate: float = 0.01,
    batch_size: int | None = 64,
    shuffle: bool = True,
    device: str | torch.device | None = None,
) -> StackedAnfisModel:
    """Build stacked ANFIS with upper rules generated from warmed hidden features."""
    current_samples = _validate_sample_inputs(sample_inputs, config.input_dim, device=device)
    if current_samples is None:
        raise ValueError("sample_inputs are required for stagewise stacked initialization.")
    targets = _validate_sample_targets(sample_targets, config.layers[-1].output_dim, current_samples.device)
    if targets.size(0) != current_samples.size(0):
        raise ValueError("sample_inputs and sample_targets must contain the same number of samples.")

    layers: list[SugenoDecisionLayer] = []
    current_width = config.input_dim
    original_samples = current_samples
    for layer_index, layer_config in enumerate(config.layers):
        is_final = layer_index == len(config.layers) - 1
        expected_width = current_width + len(config.final_skip_input_indices) if is_final else current_width
        if len(layer_config.variables) != expected_width:
            raise ValueError(
                f"Layer '{layer_config.name}' expects {len(layer_config.variables)} variables, "
                f"but stacked width before this layer is {expected_width}."
            )
        layer_samples = current_samples
        if is_final and config.final_skip_input_indices:
            skip_indices = torch.tensor(config.final_skip_input_indices, dtype=torch.long, device=original_samples.device)
            layer_samples = torch.cat((current_samples, original_samples.index_select(dim=1, index=skip_indices)), dim=1)
        layer = build_decision_layer(_as_decision_config(layer_config), sample_inputs=layer_samples)
        layer.to(device=current_samples.device)
        is_hidden = layer_index < len(config.layers) - 1
        if is_hidden:
            _fit_hidden_layer_with_linear_head(
                layer,
                current_samples,
                targets,
                task_type=task_type,
                epochs=epochs_per_hidden_layer,
                learning_rate=learning_rate,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        layers.append(layer)
        current_width = layer.output_dim
        with torch.no_grad():
            current_samples = layer(layer_samples)

    return StackedAnfisModel(
        layers=tuple(layers),
        input_dim=config.input_dim,
        final_skip_input_indices=config.final_skip_input_indices,
        final_skip_gates_enabled=config.final_skip_gates_enabled,
        final_skip_gate_init_logit=config.final_skip_gate_init_logit,
    )
