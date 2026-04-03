from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class VariableTrace:
    variable_name: str
    term_names: tuple[str, ...]
    memberships: Tensor


@dataclass(frozen=True)
class BlockTrace:
    block_name: str
    inputs: Tensor
    variable_traces: tuple[VariableTrace, ...]
    rule_names: tuple[str, ...]
    raw_rule_weights: Tensor
    normalized_rule_weights: Tensor
    rule_outputs: Tensor
    outputs: Tensor
    output_names: tuple[str, ...]


@dataclass(frozen=True)
class StageTrace:
    stage_name: str
    block_traces: tuple[BlockTrace, ...]
    outputs: Tensor


@dataclass(frozen=True)
class ModelTrace:
    stage_traces: tuple[StageTrace, ...]
    decision_trace: BlockTrace
