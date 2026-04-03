from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .model import DeepFuzzyFeatureModel


@dataclass(frozen=True)
class HiddenRuleContribution:
    rule_name: str
    rule_text: str
    normalized_weight: float
    contribution: float


@dataclass(frozen=True)
class ConceptNodeFlow:
    stage_name: str
    block_name: str
    concept_name: str
    value: float
    top_rule_contributions: tuple[HiddenRuleContribution, ...]


@dataclass(frozen=True)
class DecisionRuleContribution:
    rule_name: str
    rule_text: str
    normalized_weight: float
    contribution: tuple[float, ...]


@dataclass(frozen=True)
class DecisionConceptContribution:
    concept_name: str
    source_stage_name: str | None
    source_block_name: str | None
    concept_value: float
    contribution: tuple[float, ...]


@dataclass(frozen=True)
class SampleConceptFlow:
    sample_index: int
    inputs: tuple[float, ...]
    prediction: tuple[float, ...]
    hidden_concepts: tuple[ConceptNodeFlow, ...]
    bias_contribution: tuple[float, ...]
    decision_concept_contributions: tuple[DecisionConceptContribution, ...]
    top_decision_rules: tuple[DecisionRuleContribution, ...]


@dataclass(frozen=True)
class PathRuleContribution:
    rule_name: str
    rule_text: str
    local_contribution: float
    path_contribution: tuple[float, ...]


@dataclass(frozen=True)
class HiddenConceptPathFlow:
    stage_name: str
    block_name: str
    concept_name: str
    value: float
    downstream_sensitivity: tuple[float, ...]
    path_contribution: tuple[float, ...]
    top_rule_path_contributions: tuple[PathRuleContribution, ...]


@dataclass(frozen=True)
class HiddenRulePathContribution:
    rule_name: str
    rule_text: str
    normalized_weight: float
    local_contributions: tuple[float, ...]
    path_contribution: tuple[float, ...]


@dataclass(frozen=True)
class HiddenBlockRulePathFlow:
    stage_name: str
    block_name: str
    top_rule_path_contributions: tuple[HiddenRulePathContribution, ...]


@dataclass(frozen=True)
class PathSampleConceptFlow:
    sample_index: int
    inputs: tuple[float, ...]
    prediction: tuple[float, ...]
    hidden_paths: tuple[HiddenConceptPathFlow, ...]
    hidden_rule_paths: tuple[HiddenBlockRulePathFlow, ...]


@dataclass(frozen=True)
class RuleChainContribution:
    rule_name: str
    rule_text: str
    attribution_kind: str
    normalized_weight: float
    output_contribution: tuple[float, ...]


@dataclass(frozen=True)
class RuleChainBlockFlow:
    stage_name: str
    block_name: str
    attribution_kind: str
    top_rule_contributions: tuple[RuleChainContribution, ...]


@dataclass(frozen=True)
class SampleRuleChainFlow:
    sample_index: int
    inputs: tuple[float, ...]
    prediction: tuple[float, ...]
    bias_contribution: tuple[float, ...]
    exact_final_rule_contributions: tuple[RuleChainBlockFlow, ...]
    upstream_path_rule_contributions: tuple[RuleChainBlockFlow, ...]


def _concept_sources(model: DeepFuzzyFeatureModel) -> tuple[tuple[str | None, str | None, str], ...]:
    if not model.stages:
        return tuple((None, None, variable.name) for variable in model.decision_layer.variables)

    sources: list[tuple[str | None, str | None, str]] = []
    last_stage = model.stages[-1]
    for connected_block in last_stage.blocks:
        for concept_name in connected_block.block.output_names:
            sources.append((last_stage.name, connected_block.block.name, concept_name))
    return tuple(sources)


def analyze_concept_flow(
    model: DeepFuzzyFeatureModel,
    inputs: Tensor,
    top_k_rules: int = 2,
) -> tuple[SampleConceptFlow, ...]:
    if top_k_rules <= 0:
        raise ValueError("top_k_rules must be positive.")

    predictions, trace = model.forward_with_trace(inputs)
    concept_sources = _concept_sources(model)
    decision_weight_tensor = model.decision_layer.rule_weights.detach()
    decision_bias_tensor = model.decision_layer.rule_bias.detach()

    sample_flows: list[SampleConceptFlow] = []
    for sample_index in range(inputs.size(0)):
        hidden_concepts: list[ConceptNodeFlow] = []
        for stage, stage_trace in zip(model.stages, trace.stage_traces, strict=True):
            for connected_block, block_trace in zip(stage.blocks, stage_trace.block_traces, strict=True):
                weights = block_trace.normalized_rule_weights[sample_index]
                for concept_index, concept_name in enumerate(block_trace.output_names):
                    concept_value = float(block_trace.outputs[sample_index, concept_index].item())
                    rule_contributions = weights * block_trace.rule_outputs[sample_index, :, concept_index]
                    top_values, top_indices = rule_contributions.topk(k=min(top_k_rules, rule_contributions.numel()))
                    top_rules = []
                    for contribution, rule_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
                        top_rules.append(
                            HiddenRuleContribution(
                                rule_name=block_trace.rule_names[rule_index],
                                rule_text=connected_block.block.describe_rule(rule_index),
                                normalized_weight=float(weights[rule_index].item()),
                                contribution=float(contribution),
                            )
                        )
                    hidden_concepts.append(
                        ConceptNodeFlow(
                            stage_name=stage.name,
                            block_name=connected_block.block.name,
                            concept_name=concept_name,
                            value=concept_value,
                            top_rule_contributions=tuple(top_rules),
                        )
                    )

        decision_trace = trace.decision_trace
        decision_inputs = decision_trace.inputs[sample_index]
        normalized_rule_weights = decision_trace.normalized_rule_weights[sample_index]
        decision_rule_outputs = decision_trace.rule_outputs[sample_index]
        bias_contribution = (
            normalized_rule_weights.unsqueeze(-1) * decision_bias_tensor
        ).sum(dim=0)

        concept_contributions: list[DecisionConceptContribution] = []
        for concept_index, concept_name in enumerate(decision_trace.variable_traces):
            contribution = (
                normalized_rule_weights.unsqueeze(-1)
                * decision_weight_tensor[:, concept_index, :]
                * decision_inputs[concept_index]
            ).sum(dim=0)
            source_stage_name, source_block_name, source_name = concept_sources[concept_index]
            concept_contributions.append(
                DecisionConceptContribution(
                    concept_name=source_name,
                    source_stage_name=source_stage_name,
                    source_block_name=source_block_name,
                    concept_value=float(decision_inputs[concept_index].item()),
                    contribution=tuple(float(value) for value in contribution.tolist()),
                )
            )

        top_rule_scores = decision_rule_outputs.abs().sum(dim=1) * normalized_rule_weights
        top_values, top_indices = top_rule_scores.topk(k=min(top_k_rules, top_rule_scores.numel()))
        top_decision_rules = []
        for _, rule_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
            contribution = normalized_rule_weights[rule_index] * decision_rule_outputs[rule_index]
            top_decision_rules.append(
                DecisionRuleContribution(
                    rule_name=decision_trace.rule_names[rule_index],
                    rule_text=model.decision_layer.describe_rule(rule_index),
                    normalized_weight=float(normalized_rule_weights[rule_index].item()),
                    contribution=tuple(float(value) for value in contribution.tolist()),
                )
            )

        sample_flows.append(
            SampleConceptFlow(
                sample_index=sample_index,
                inputs=tuple(float(value) for value in inputs[sample_index].tolist()),
                prediction=tuple(float(value) for value in predictions[sample_index].tolist()),
                hidden_concepts=tuple(hidden_concepts),
                bias_contribution=tuple(float(value) for value in bias_contribution.tolist()),
                decision_concept_contributions=tuple(concept_contributions),
                top_decision_rules=tuple(top_decision_rules),
            )
        )

    return tuple(sample_flows)


def analyze_path_concept_flow(
    model: DeepFuzzyFeatureModel,
    inputs: Tensor,
    top_k_rules: int = 2,
) -> tuple[PathSampleConceptFlow, ...]:
    if top_k_rules <= 0:
        raise ValueError("top_k_rules must be positive.")

    predictions, trace = model.forward_with_trace(inputs, top_k_rules=top_k_rules)
    block_lookup = {
        (stage.name, connected_block.block.name): connected_block.block
        for stage in model.stages
        for connected_block in stage.blocks
    }
    sample_flows: list[PathSampleConceptFlow] = []

    for sample_index in range(inputs.size(0)):
        sample = inputs[sample_index : sample_index + 1].detach().clone().requires_grad_(True)
        current = sample
        hidden_nodes: list[tuple[str, str, tuple[str, ...], Tensor]] = []
        for stage in model.stages:
            block_outputs = []
            for connected_block in stage.blocks:
                block_input = current.index_select(dim=1, index=connected_block.input_indices)
                block_output = connected_block.block(block_input, top_k_rules=top_k_rules)
                hidden_nodes.append(
                    (
                        stage.name,
                        connected_block.block.name,
                        connected_block.block.output_names,
                        block_output,
                    )
                )
                block_outputs.append(block_output)
            current = torch.cat(block_outputs, dim=1)
        sample_prediction = model.decision_layer(current, top_k_rules=top_k_rules)

        hidden_paths: list[HiddenConceptPathFlow] = []
        hidden_rule_paths: list[HiddenBlockRulePathFlow] = []
        block_trace_cursor = 0
        for stage_trace in trace.stage_traces:
            for block_trace in stage_trace.block_traces:
                stage_name, block_name, output_names, block_output = hidden_nodes[block_trace_cursor]
                block_trace_cursor += 1
                weights = block_trace.normalized_rule_weights[sample_index]
                output_dim = sample_prediction.size(1)
                sensitivity_tensor = torch.zeros(
                    block_output.size(1),
                    output_dim,
                    dtype=block_output.dtype,
                    device=block_output.device,
                )

                for concept_index in range(block_output.size(1)):
                    concept_scalar = block_output[0, concept_index]
                    for output_index in range(output_dim):
                        gradient = torch.autograd.grad(
                            sample_prediction[0, output_index],
                            concept_scalar,
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        sensitivity_tensor[concept_index, output_index] = (
                            0.0 if gradient is None else float(gradient.detach().item())
                        )

                local_rule_contributions = weights.unsqueeze(-1) * block_trace.rule_outputs[sample_index]
                block_rule_path_matrix = torch.matmul(local_rule_contributions, sensitivity_tensor)

                for concept_index, concept_name in enumerate(output_names):
                    concept_scalar = block_output[0, concept_index]
                    sensitivities = sensitivity_tensor[concept_index].detach().cpu().tolist()

                    concept_value = float(concept_scalar.detach().item())
                    path_contribution = tuple(concept_value * sensitivity for sensitivity in sensitivities)

                    local_rule_contribution_vector = local_rule_contributions[:, concept_index]
                    local_path_matrix = torch.outer(
                        local_rule_contribution_vector,
                        sensitivity_tensor[concept_index],
                    )
                    local_path_norm = local_path_matrix.abs().sum(dim=1)
                    top_values, top_indices = local_path_norm.topk(k=min(top_k_rules, local_path_norm.numel()))
                    block = block_lookup[(stage_name, block_name)]

                    top_rules = []
                    for _, rule_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
                        top_rules.append(
                            PathRuleContribution(
                                rule_name=block_trace.rule_names[rule_index],
                                rule_text=block.describe_rule(rule_index),
                                local_contribution=float(local_rule_contribution_vector[rule_index].item()),
                                path_contribution=tuple(float(value) for value in local_path_matrix[rule_index].tolist()),
                            )
                        )

                    hidden_paths.append(
                        HiddenConceptPathFlow(
                            stage_name=stage_name,
                            block_name=block_name,
                            concept_name=concept_name,
                            value=concept_value,
                            downstream_sensitivity=tuple(sensitivities),
                            path_contribution=path_contribution,
                            top_rule_path_contributions=tuple(top_rules),
                        )
                    )

                block_rule_path_norm = block_rule_path_matrix.abs().sum(dim=1)
                top_values, top_indices = block_rule_path_norm.topk(k=min(top_k_rules, block_rule_path_norm.numel()))
                block = block_lookup[(stage_name, block_name)]
                top_block_rules = []
                for _, rule_index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
                    top_block_rules.append(
                        HiddenRulePathContribution(
                            rule_name=block_trace.rule_names[rule_index],
                            rule_text=block.describe_rule(rule_index),
                            normalized_weight=float(weights[rule_index].item()),
                            local_contributions=tuple(
                                float(value) for value in local_rule_contributions[rule_index].tolist()
                            ),
                            path_contribution=tuple(
                                float(value) for value in block_rule_path_matrix[rule_index].tolist()
                            ),
                        )
                    )
                hidden_rule_paths.append(
                    HiddenBlockRulePathFlow(
                        stage_name=stage_name,
                        block_name=block_name,
                        top_rule_path_contributions=tuple(top_block_rules),
                    )
                )

        sample_flows.append(
            PathSampleConceptFlow(
                sample_index=sample_index,
                inputs=tuple(float(value) for value in inputs[sample_index].tolist()),
                prediction=tuple(float(value) for value in predictions[sample_index].tolist()),
                hidden_paths=tuple(hidden_paths),
                hidden_rule_paths=tuple(hidden_rule_paths),
            )
        )

    return tuple(sample_flows)


def analyze_rule_chain_flow(
    model: DeepFuzzyFeatureModel,
    inputs: Tensor,
    top_k_rules: int = 2,
) -> tuple[SampleRuleChainFlow, ...]:
    if top_k_rules <= 0:
        raise ValueError("top_k_rules must be positive.")

    predictions, trace = model.forward_with_trace(inputs)
    path_flows = analyze_path_concept_flow(model, inputs, top_k_rules=top_k_rules)
    decision_weight_tensor = model.decision_layer.rule_weights.detach()
    decision_bias_tensor = model.decision_layer.rule_bias.detach()

    if model.stages:
        last_stage = model.stages[-1]
        last_stage_trace = trace.stage_traces[-1]
        final_stage_blocks = {(last_stage.name, connected_block.block.name): connected_block.block for connected_block in last_stage.blocks}
    else:
        last_stage = None
        last_stage_trace = None
        final_stage_blocks = {}

    sample_flows: list[SampleRuleChainFlow] = []
    for sample_index in range(inputs.size(0)):
        decision_trace = trace.decision_trace
        normalized_rule_weights = decision_trace.normalized_rule_weights[sample_index]
        bias_contribution = (
            normalized_rule_weights.unsqueeze(-1) * decision_bias_tensor
        ).sum(dim=0)
        decision_coefficients = (
            normalized_rule_weights.view(-1, 1, 1) * decision_weight_tensor
        ).sum(dim=0)

        exact_final_rule_contributions: list[RuleChainBlockFlow] = []
        if last_stage is not None and last_stage_trace is not None:
            concept_offset = 0
            for connected_block, block_trace in zip(last_stage.blocks, last_stage_trace.block_traces, strict=True):
                block = connected_block.block
                concept_slice = slice(concept_offset, concept_offset + block.output_dim)
                concept_offset += block.output_dim

                local_rule_contributions = (
                    block_trace.normalized_rule_weights[sample_index].unsqueeze(-1)
                    * block_trace.rule_outputs[sample_index]
                )
                rule_output_contributions = torch.matmul(
                    local_rule_contributions,
                    decision_coefficients[concept_slice],
                )
                contribution_norm = rule_output_contributions.abs().sum(dim=1)
                _, top_indices = contribution_norm.topk(k=min(top_k_rules, contribution_norm.numel()))

                top_rules: list[RuleChainContribution] = []
                for rule_index in top_indices.tolist():
                    top_rules.append(
                        RuleChainContribution(
                            rule_name=block_trace.rule_names[rule_index],
                            rule_text=block.describe_rule(rule_index),
                            attribution_kind="exact",
                            normalized_weight=float(block_trace.normalized_rule_weights[sample_index, rule_index].item()),
                            output_contribution=tuple(
                                float(value) for value in rule_output_contributions[rule_index].tolist()
                            ),
                        )
                    )

                exact_final_rule_contributions.append(
                    RuleChainBlockFlow(
                        stage_name=last_stage.name,
                        block_name=block.name,
                        attribution_kind="exact",
                        top_rule_contributions=tuple(top_rules),
                    )
                )

        upstream_path_rule_contributions: list[RuleChainBlockFlow] = []
        if model.stages:
            final_stage_key = last_stage.name if last_stage is not None else None
            for block_flow in path_flows[sample_index].hidden_rule_paths:
                if block_flow.stage_name == final_stage_key:
                    continue
                upstream_path_rule_contributions.append(
                    RuleChainBlockFlow(
                        stage_name=block_flow.stage_name,
                        block_name=block_flow.block_name,
                        attribution_kind="path",
                        top_rule_contributions=tuple(
                            RuleChainContribution(
                                rule_name=rule.rule_name,
                                rule_text=rule.rule_text,
                                attribution_kind="path",
                                normalized_weight=rule.normalized_weight,
                                output_contribution=rule.path_contribution,
                            )
                            for rule in block_flow.top_rule_path_contributions
                        ),
                    )
                )

        sample_flows.append(
            SampleRuleChainFlow(
                sample_index=sample_index,
                inputs=tuple(float(value) for value in inputs[sample_index].tolist()),
                prediction=tuple(float(value) for value in predictions[sample_index].tolist()),
                bias_contribution=tuple(float(value) for value in bias_contribution.tolist()),
                exact_final_rule_contributions=tuple(exact_final_rule_contributions),
                upstream_path_rule_contributions=tuple(upstream_path_rule_contributions),
            )
        )

    return tuple(sample_flows)


def format_concept_flows(flows: tuple[SampleConceptFlow, ...], decimals: int = 4) -> str:
    lines: list[str] = []
    for flow in flows:
        lines.append(f"SAMPLE {flow.sample_index}")
        lines.append(f"  inputs: {tuple(round(value, decimals) for value in flow.inputs)}")
        lines.append(f"  prediction: {tuple(round(value, decimals) for value in flow.prediction)}")
        lines.append("  hidden concepts:")
        for concept in flow.hidden_concepts:
            lines.append(
                "    "
                f"{concept.stage_name}/{concept.block_name}/{concept.concept_name}="
                f"{concept.value:.{decimals}f}"
            )
            for rule in concept.top_rule_contributions:
                lines.append(
                    "      rule: "
                    f"{rule.rule_name} | {rule.rule_text} | "
                    f"weight={rule.normalized_weight:.{decimals}f} | "
                    f"contribution={rule.contribution:.{decimals}f}"
                )
        lines.append(f"  bias contribution: {tuple(round(value, decimals) for value in flow.bias_contribution)}")
        lines.append("  decision concept contributions:")
        for contribution in flow.decision_concept_contributions:
            source = (
                f"{contribution.source_stage_name}/{contribution.source_block_name}"
                if contribution.source_stage_name is not None and contribution.source_block_name is not None
                else "raw_input"
            )
            lines.append(
                "    "
                f"{source}/{contribution.concept_name}: value={contribution.concept_value:.{decimals}f}, "
                f"contribution={tuple(round(value, decimals) for value in contribution.contribution)}"
            )
        lines.append("  top decision rules:")
        for rule in flow.top_decision_rules:
            lines.append(
                "    rule: "
                f"{rule.rule_name} | {rule.rule_text} | "
                f"weight={rule.normalized_weight:.{decimals}f} | "
                f"contribution={tuple(round(value, decimals) for value in rule.contribution)}"
            )
    return "\n".join(lines)


def format_path_concept_flows(flows: tuple[PathSampleConceptFlow, ...], decimals: int = 4) -> str:
    lines: list[str] = []
    for flow in flows:
        lines.append(f"SAMPLE {flow.sample_index}")
        lines.append(f"  inputs: {tuple(round(value, decimals) for value in flow.inputs)}")
        lines.append(f"  prediction: {tuple(round(value, decimals) for value in flow.prediction)}")
        lines.append("  hidden path contributions:")
        for hidden in flow.hidden_paths:
            lines.append(
                "    "
                f"{hidden.stage_name}/{hidden.block_name}/{hidden.concept_name}: "
                f"value={hidden.value:.{decimals}f}, "
                f"sensitivity={tuple(round(value, decimals) for value in hidden.downstream_sensitivity)}, "
                f"path={tuple(round(value, decimals) for value in hidden.path_contribution)}"
            )
            for rule in hidden.top_rule_path_contributions:
                lines.append(
                    "      rule: "
                    f"{rule.rule_name} | {rule.rule_text} | "
                    f"local={rule.local_contribution:.{decimals}f} | "
                    f"path={tuple(round(value, decimals) for value in rule.path_contribution)}"
                )
        lines.append("  aggregated hidden rule paths:")
        for block_flow in flow.hidden_rule_paths:
            lines.append(f"    {block_flow.stage_name}/{block_flow.block_name}")
            for rule in block_flow.top_rule_path_contributions:
                lines.append(
                    "      rule: "
                    f"{rule.rule_name} | {rule.rule_text} | "
                    f"weight={rule.normalized_weight:.{decimals}f} | "
                    f"local={tuple(round(value, decimals) for value in rule.local_contributions)} | "
                    f"path={tuple(round(value, decimals) for value in rule.path_contribution)}"
                )
    return "\n".join(lines)


def format_rule_chain_flows(flows: tuple[SampleRuleChainFlow, ...], decimals: int = 4) -> str:
    lines: list[str] = []
    for flow in flows:
        lines.append(f"SAMPLE {flow.sample_index}")
        lines.append(f"  inputs: {tuple(round(value, decimals) for value in flow.inputs)}")
        lines.append(f"  prediction: {tuple(round(value, decimals) for value in flow.prediction)}")
        lines.append(f"  bias contribution: {tuple(round(value, decimals) for value in flow.bias_contribution)}")
        lines.append("  exact final-stage hidden rule contributions:")
        for block_flow in flow.exact_final_rule_contributions:
            lines.append(f"    {block_flow.stage_name}/{block_flow.block_name} [{block_flow.attribution_kind}]")
            for rule in block_flow.top_rule_contributions:
                lines.append(
                    "      rule: "
                    f"{rule.rule_name} | {rule.rule_text} | "
                    f"weight={rule.normalized_weight:.{decimals}f} | "
                    f"output={tuple(round(value, decimals) for value in rule.output_contribution)}"
                )
        lines.append("  upstream path-based hidden rule contributions:")
        for block_flow in flow.upstream_path_rule_contributions:
            lines.append(f"    {block_flow.stage_name}/{block_flow.block_name} [{block_flow.attribution_kind}]")
            for rule in block_flow.top_rule_contributions:
                lines.append(
                    "      rule: "
                    f"{rule.rule_name} | {rule.rule_text} | "
                    f"weight={rule.normalized_weight:.{decimals}f} | "
                    f"output={tuple(round(value, decimals) for value in rule.output_contribution)}"
                )
    return "\n".join(lines)
