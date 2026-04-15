from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from .blocks import BaseFuzzyRuleLayer
from .traces import StageTrace


class ConnectedFuzzyBlock(nn.Module):
    def __init__(self, block: BaseFuzzyRuleLayer, input_indices: Sequence[int]) -> None:
        super().__init__()
        if len(input_indices) != block.input_dim:
            raise ValueError(
                f"Block '{block.name}' expects {block.input_dim} inputs, got {len(input_indices)} indices."
            )
        if any(index < 0 for index in input_indices):
            raise ValueError("Input indices must be non-negative.")
        if len(set(input_indices)) != len(tuple(input_indices)):
            raise ValueError("Input indices inside a block must be unique.")
        self.block = block
        self.register_buffer(
            "input_indices",
            torch.tensor(tuple(input_indices), dtype=torch.long),
            persistent=False,
        )

    @property
    def name(self) -> str:
        return self.block.name

    @property
    def output_dim(self) -> int:
        return self.block.output_dim

    @property
    def max_input_index(self) -> int:
        return int(self.input_indices.max().item())

    def _select_inputs(self, inputs: Tensor) -> Tensor:
        return inputs.index_select(dim=1, index=self.input_indices)

    def forward(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        return self.block(self._select_inputs(inputs), top_k_rules=top_k_rules)

    def forward_with_trace(self, inputs: Tensor, top_k_rules: int | None = None):
        return self.block.forward_with_trace(self._select_inputs(inputs), top_k_rules=top_k_rules)


class FuzzyStage(nn.Module):
    def __init__(self, name: str, blocks: Sequence[ConnectedFuzzyBlock]) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("A fuzzy stage must contain at least one connected block.")
        self.name = name
        self.blocks = nn.ModuleList(blocks)

    @property
    def output_dim(self) -> int:
        return sum(block.output_dim for block in self.blocks)

    def validate_input_dim(self, input_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError("The stage input dimension must be positive.")
        for block in self.blocks:
            if block.max_input_index >= input_dim:
                raise ValueError(
                    f"Stage '{self.name}' references input index {block.max_input_index}, "
                    f"but only {input_dim} input features are available."
                )

    def forward(self, inputs: Tensor, top_k_rules: int | None = None) -> Tensor:
        outputs = [block(inputs, top_k_rules=top_k_rules) for block in self.blocks]
        return torch.cat(outputs, dim=1)

    def forward_with_trace(
        self,
        inputs: Tensor,
        top_k_rules: int | None = None,
    ) -> tuple[Tensor, StageTrace]:
        outputs = []
        traces = []
        for block in self.blocks:
            block_outputs, block_trace = block.forward_with_trace(inputs, top_k_rules=top_k_rules)
            outputs.append(block_outputs)
            traces.append(block_trace)
        stage_outputs = torch.cat(outputs, dim=1)
        trace = StageTrace(
            stage_name=self.name,
            block_traces=tuple(traces),
            outputs=stage_outputs.detach(),
        )
        return stage_outputs, trace
