from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn

from .blocks import SugenoDecisionLayer
from .stages import FuzzyStage
from .traces import ModelTrace


class DeepFuzzyFeatureModel(nn.Module):
    def __init__(
        self,
        stages: Sequence[FuzzyStage],
        decision_layer: SugenoDecisionLayer,
        input_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.decision_layer = decision_layer
        self.input_dim = input_dim
        if self.input_dim is not None:
            current_dim = self.input_dim
            for stage in self.stages:
                stage.validate_input_dim(current_dim)
                current_dim = stage.output_dim
            if current_dim != self.decision_layer.input_dim:
                raise ValueError(
                    f"The last stage produces {current_dim} features, but the decision layer expects "
                    f"{self.decision_layer.input_dim}."
                )
        elif self.stages:
            feature_dim = self.stages[-1].output_dim
            if feature_dim != self.decision_layer.input_dim:
                raise ValueError(
                    f"The last stage produces {feature_dim} features, but the decision layer expects "
                    f"{self.decision_layer.input_dim}."
                )

    def forward_features(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        features = inputs
        for stage in self.stages:
            features = stage(features, top_k_rules=top_k_rules)
        return features

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
        outputs, decision_trace = self.decision_layer.forward_with_trace(features, top_k_rules=top_k_rules)
        trace = ModelTrace(stage_traces=tuple(stage_traces), decision_trace=decision_trace)
        return outputs, trace

    def explain(self, inputs: Tensor, top_k_rules: int | None = None) -> ModelTrace:
        _, trace = self.forward_with_trace(inputs, top_k_rules=top_k_rules)
        return trace
