from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import SugenoDecisionLayer
from .stages import FuzzyStage
from .traces import ModelTrace


class DeepFuzzyFeatureModel(nn.Module):
    def __init__(
        self,
        stages: Sequence[FuzzyStage],
        decision_layer: SugenoDecisionLayer,
        input_dim: int | None = None,
        final_skip_input_indices: Sequence[int] = (),
        final_skip_gates_enabled: bool = False,
        final_skip_gate_init_logit: float = 2.0,
        concept_dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.decision_layer = decision_layer
        self.input_dim = input_dim
        self.final_skip_input_indices = tuple(int(index) for index in final_skip_input_indices)
        self.final_skip_gates_enabled = bool(final_skip_gates_enabled)
        self.final_skip_gate_init_logit = float(final_skip_gate_init_logit)
        self.concept_dropout_rate = float(concept_dropout_rate)
        if not 0.0 <= self.concept_dropout_rate < 1.0:
            raise ValueError("concept_dropout_rate must be in [0, 1).")
        if len(set(self.final_skip_input_indices)) != len(self.final_skip_input_indices):
            raise ValueError("Final skip input indices must be unique.")
        if any(index < 0 for index in self.final_skip_input_indices):
            raise ValueError("Final skip input indices must be non-negative.")
        self.register_buffer(
            "_final_skip_input_index_tensor",
            torch.tensor(self.final_skip_input_indices, dtype=torch.long),
            persistent=False,
        )
        if self.final_skip_gates_enabled and self.final_skip_input_indices:
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
            for stage in self.stages:
                stage.validate_input_dim(current_dim)
                current_dim = stage.output_dim
            expected_decision_dim = current_dim + len(self.final_skip_input_indices)
            if expected_decision_dim != self.decision_layer.input_dim:
                raise ValueError(
                    f"The last stage plus final skip produces {expected_decision_dim} features, "
                    f"but the decision layer expects {self.decision_layer.input_dim}."
                )
            if self.final_skip_input_indices and max(self.final_skip_input_indices) >= self.input_dim:
                raise ValueError("Final skip input index is outside the model input dimension.")
        elif self.stages:
            feature_dim = self.stages[-1].output_dim + len(self.final_skip_input_indices)
            if feature_dim != self.decision_layer.input_dim:
                raise ValueError(
                    f"The last stage plus final skip produces {feature_dim} features, but the decision layer expects "
                    f"{self.decision_layer.input_dim}."
                )

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
        for stage in self.stages:
            features = stage(features, top_k_rules=top_k_rules)
            if self.training and self.concept_dropout_rate > 0.0:
                features = F.dropout(features, p=self.concept_dropout_rate, training=True)
        return self._append_final_skip_inputs(inputs, features)

    def forward(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        return self.decision_layer(self.forward_features(inputs, top_k_rules=top_k_rules), top_k_rules=top_k_rules)

    def forward_with_trace(
        self,
        inputs: Tensor,
        top_k_rules: int | None = None,
    ) -> tuple[Tensor, ModelTrace]:
        features = inputs
        stage_traces = []
        for stage in self.stages:
            features, stage_trace = stage.forward_with_trace(features, top_k_rules=top_k_rules)
            stage_traces.append(stage_trace)
        features = self._append_final_skip_inputs(inputs, features)
        outputs, decision_trace = self.decision_layer.forward_with_trace(features, top_k_rules=top_k_rules)
        trace = ModelTrace(stage_traces=tuple(stage_traces), decision_trace=decision_trace)
        return outputs, trace

    def explain(self, inputs: Tensor, top_k_rules: int | None = None) -> ModelTrace:
        _, trace = self.forward_with_trace(inputs, top_k_rules=top_k_rules)
        return trace
