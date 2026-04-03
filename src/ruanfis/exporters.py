from __future__ import annotations

from typing import Sequence

from .builders import (
    DecisionLayerConfig,
    HierarchicalModelConfig,
    ShallowFuzzyModelConfig,
    StageConfig,
    TransparentBlockConfig,
)
from .blocks import BaseFuzzyRuleLayer, SugenoDecisionLayer, TransparentFuzzyBlock
from .model import DeepFuzzyFeatureModel
from .rules import count_rule_candidates
from .trainer import PruningReport


def _format_number(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _format_linear_expression(
    output_name: str,
    bias: float,
    coefficients: Sequence[float],
    variable_names: Sequence[str],
    decimals: int,
) -> str:
    terms = [_format_number(bias, decimals)]
    for coefficient, variable_name in zip(coefficients, variable_names, strict=True):
        sign = "+" if coefficient >= 0.0 else "-"
        magnitude = abs(coefficient)
        terms.append(f" {sign} {_format_number(magnitude, decimals)}*{variable_name}")
    return f"{output_name} = {''.join(terms)}"


def export_rule_base(layer: BaseFuzzyRuleLayer, decimals: int = 3) -> list[str]:
    lines: list[str] = []
    gate_probabilities = layer.rule_probabilities.detach().cpu().tolist()

    for rule_index, rule_name in enumerate(layer.rule_base.names):
        antecedent = layer.describe_rule(rule_index)
        gate = _format_number(gate_probabilities[rule_index], decimals)
        label = f"RULE {rule_index} ({rule_name})"

        if isinstance(layer, TransparentFuzzyBlock):
            consequent_values = layer.consequents[rule_index].detach().cpu().tolist()
            consequent = ", ".join(
                f"{concept_name}={_format_number(value, decimals)}"
                for concept_name, value in zip(layer.output_names, consequent_values, strict=True)
            )
            lines.append(f"{label}: IF {antecedent} THEN {consequent} [gate={gate}]")
            continue

        if isinstance(layer, SugenoDecisionLayer):
            variable_names = [variable.name for variable in layer.variables]
            expressions = []
            for output_index, output_name in enumerate(layer.output_names):
                bias = float(layer.rule_bias[rule_index, output_index].detach().cpu().item())
                coefficients = layer.rule_weights[rule_index, :, output_index].detach().cpu().tolist()
                expressions.append(
                    _format_linear_expression(
                        output_name,
                        bias,
                        coefficients,
                        variable_names,
                        decimals=decimals,
                    )
                )
            consequent = "; ".join(expressions)
            lines.append(f"{label}: IF {antecedent} THEN {consequent} [gate={gate}]")
            continue

        lines.append(f"{label}: IF {antecedent} [gate={gate}]")

    return lines


def export_model_report(model: DeepFuzzyFeatureModel, decimals: int = 3) -> str:
    lines: list[str] = ["MODEL REPORT"]

    if model.input_dim is not None:
        lines.append(f"Input dimension: {model.input_dim}")

    if not model.stages:
        lines.append("No hidden stages.")
    else:
        for stage_index, stage in enumerate(model.stages):
            lines.append("")
            lines.append(f"STAGE {stage_index}: {stage.name}")
            lines.append(f"Output dimension: {stage.output_dim}")
            for block_index, connected_block in enumerate(stage.blocks):
                block = connected_block.block
                input_indices = ", ".join(str(int(index)) for index in connected_block.input_indices.tolist())
                output_names = ", ".join(block.output_names)
                lines.append(f"  BLOCK {block_index}: {block.name}")
                lines.append(f"    Inputs: [{input_indices}]")
                lines.append(f"    Outputs: [{output_names}]")
                for rule_line in export_rule_base(block, decimals=decimals):
                    lines.append(f"    {rule_line}")

    lines.append("")
    lines.append(f"DECISION LAYER: {model.decision_layer.name}")
    lines.append(f"Output dimension: {model.decision_layer.output_dim}")
    for rule_line in export_rule_base(model.decision_layer, decimals=decimals):
        lines.append(f"  {rule_line}")

    return "\n".join(lines)


def export_pruning_report(report: PruningReport | None, decimals: int = 3) -> str:
    if report is None:
        return "No pruning report available."

    lines = ["PRUNING REPORT"]
    for layer in report.layers:
        before = ", ".join(_format_number(value, decimals) for value in layer.probabilities_before)
        after = ", ".join(_format_number(value, decimals) for value in layer.probabilities_after)
        lines.append(
            f"{layer.layer_name}: active_rules {layer.active_rules_before} -> {layer.active_rules_after}"
        )
        lines.append(f"  before: [{before}]")
        lines.append(f"  after:  [{after}]")
    return "\n".join(lines)


def _export_block_config(block: TransparentBlockConfig) -> list[str]:
    max_arity = block.max_rule_arity or len(block.variables)
    input_indices = ", ".join(str(index) for index in block.input_indices)
    lines = [
        f"  BLOCK: {block.name}",
        f"    inputs: [{input_indices}]",
        f"    variables: {len(block.variables)}",
        f"    rule_generation_mode: {block.rule_generation_mode}",
        f"    max_rule_arity: {max_arity}",
        f"    candidate_rules: {block.candidate_rule_count()}",
        f"    generated_rules: {block.generated_rule_count()}",
    ]
    if block.rule_generation_mode == "prototype":
        lines.extend(
            [
                f"    prototype_term_limit: {block.prototype_term_limit}",
                f"    prototype_sample_size: {block.prototype_sample_size}",
            ]
        )
    return lines


def _export_stage_config(stage: StageConfig) -> list[str]:
    lines = [f"STAGE: {stage.name}", f"  generated_rules_total: {stage.generated_rule_count()}"]
    for block in stage.blocks:
        lines.extend(_export_block_config(block))
    return lines


def _export_decision_config(config: DecisionLayerConfig) -> list[str]:
    max_arity = config.max_rule_arity or len(config.variables)
    lines = [
        f"DECISION LAYER: {config.name}",
        f"  variables: {len(config.variables)}",
        f"  rule_generation_mode: {config.rule_generation_mode}",
        f"  max_rule_arity: {max_arity}",
        f"  candidate_rules: {config.candidate_rule_count()}",
        f"  generated_rules: {config.generated_rule_count()}",
    ]
    if config.rule_generation_mode == "prototype":
        lines.extend(
            [
                f"  prototype_term_limit: {config.prototype_term_limit}",
                f"  prototype_sample_size: {config.prototype_sample_size}",
            ]
        )
    return lines


def export_model_config_report(
    config: HierarchicalModelConfig | ShallowFuzzyModelConfig,
    flat_term_counts: Sequence[int] | None = None,
) -> str:
    if isinstance(config, ShallowFuzzyModelConfig):
        config = config.as_hierarchical_config()

    lines = ["MODEL CONFIG REPORT", f"Input dimension: {config.input_dim}"]
    if flat_term_counts is not None:
        flat_full_rules = count_rule_candidates(flat_term_counts, max_rule_arity=len(flat_term_counts))
        lines.append(f"Flat full-rule count: {flat_full_rules}")
        lines.append(f"Hierarchical generated-rule count: {config.generated_rule_count()}")
    else:
        lines.append(f"Hierarchical generated-rule count: {config.generated_rule_count()}")

    for stage in config.stages:
        lines.append("")
        lines.extend(_export_stage_config(stage))

    lines.append("")
    lines.extend(_export_decision_config(config.decision_layer))
    return "\n".join(lines)
