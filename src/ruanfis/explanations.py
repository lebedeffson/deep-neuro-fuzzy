from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from .model import DeepFuzzyFeatureModel


@dataclass(frozen=True)
class RuleExplanation:
    rule_name: str
    rule_text: str
    normalized_weight: float
    outputs: tuple[float, ...]


@dataclass(frozen=True)
class BlockExplanation:
    block_name: str
    outputs: tuple[float, ...]
    top_rules: tuple[RuleExplanation, ...]


@dataclass(frozen=True)
class StageExplanation:
    stage_name: str
    block_explanations: tuple[BlockExplanation, ...]


@dataclass(frozen=True)
class SampleExplanation:
    sample_index: int
    inputs: tuple[float, ...]
    prediction: tuple[float, ...]
    stage_explanations: tuple[StageExplanation, ...]
    decision_explanation: BlockExplanation


@dataclass(frozen=True)
class ConceptExemplar:
    stage_name: str
    block_name: str
    concept_name: str
    top_sample_indices: tuple[int, ...]
    top_scores: tuple[float, ...]


def _extract_block_explanation(
    layer: TransparentFuzzyBlock | SugenoDecisionLayer,
    block_trace,
    sample_index: int,
    top_k_rules: int,
) -> BlockExplanation:
    weights = block_trace.normalized_rule_weights[sample_index]
    top_rule_count = min(top_k_rules, weights.numel())
    top_values, top_indices = weights.topk(k=top_rule_count)

    top_rules = []
    for value, index in zip(top_values.tolist(), top_indices.tolist(), strict=True):
        rule_outputs = tuple(float(output) for output in block_trace.rule_outputs[sample_index, index].tolist())
        top_rules.append(
            RuleExplanation(
                rule_name=block_trace.rule_names[index],
                rule_text=layer.describe_rule(index),
                normalized_weight=float(value),
                outputs=rule_outputs,
            )
        )

    return BlockExplanation(
        block_name=block_trace.block_name,
        outputs=tuple(float(output) for output in block_trace.outputs[sample_index].tolist()),
        top_rules=tuple(top_rules),
    )


def explain_samples(
    model: DeepFuzzyFeatureModel,
    inputs: Tensor,
    top_k_rules: int = 1,
) -> tuple[SampleExplanation, ...]:
    if top_k_rules <= 0:
        raise ValueError("top_k_rules must be positive.")

    predictions, trace = model.forward_with_trace(inputs, top_k_rules=top_k_rules)
    explanations: list[SampleExplanation] = []

    for sample_index in range(inputs.size(0)):
        stage_explanations = []
        for stage, stage_trace in zip(model.stages, trace.stage_traces, strict=True):
            block_explanations = []
            for connected_block, block_trace in zip(stage.blocks, stage_trace.block_traces, strict=True):
                block_explanations.append(
                    _extract_block_explanation(
                        connected_block.block,
                        block_trace,
                        sample_index=sample_index,
                        top_k_rules=top_k_rules,
                    )
                )
            stage_explanations.append(
                StageExplanation(
                    stage_name=stage_trace.stage_name,
                    block_explanations=tuple(block_explanations),
                )
            )

        decision_explanation = _extract_block_explanation(
            model.decision_layer,
            trace.decision_trace,
            sample_index=sample_index,
            top_k_rules=top_k_rules,
        )

        explanations.append(
            SampleExplanation(
                sample_index=sample_index,
                inputs=tuple(float(value) for value in inputs[sample_index].tolist()),
                prediction=tuple(float(value) for value in predictions[sample_index].tolist()),
                stage_explanations=tuple(stage_explanations),
                decision_explanation=decision_explanation,
            )
        )

    return tuple(explanations)


def format_sample_explanations(
    explanations: tuple[SampleExplanation, ...],
    decimals: int = 4,
) -> str:
    lines: list[str] = []
    for explanation in explanations:
        lines.append(f"SAMPLE {explanation.sample_index}")
        lines.append(f"  inputs: {tuple(round(value, decimals) for value in explanation.inputs)}")
        lines.append(f"  prediction: {tuple(round(value, decimals) for value in explanation.prediction)}")
        for stage in explanation.stage_explanations:
            lines.append(f"  stage: {stage.stage_name}")
            for block in stage.block_explanations:
                lines.append(
                    f"    block: {block.block_name} -> outputs "
                    f"{tuple(round(value, decimals) for value in block.outputs)}"
                )
                for rule in block.top_rules:
                    lines.append(
                        "      rule: "
                        f"{rule.rule_name} | {rule.rule_text} | "
                        f"weight={rule.normalized_weight:.{decimals}f} | "
                        f"outputs={tuple(round(value, decimals) for value in rule.outputs)}"
                    )
        lines.append(f"  decision: {explanation.decision_explanation.block_name}")
        for rule in explanation.decision_explanation.top_rules:
            lines.append(
                "    rule: "
                f"{rule.rule_name} | {rule.rule_text} | "
                f"weight={rule.normalized_weight:.{decimals}f} | "
                f"outputs={tuple(round(value, decimals) for value in rule.outputs)}"
            )
    return "\n".join(lines)


def extract_concept_exemplars(
    model: DeepFuzzyFeatureModel,
    inputs: Tensor,
    *,
    top_k: int = 5,
) -> tuple[ConceptExemplar, ...]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    device = next(model.parameters(), torch.empty(0)).device
    model.eval()
    with torch.no_grad():
        _, trace = model.forward_with_trace(inputs.to(device=device, dtype=torch.float32))
    exemplars: list[ConceptExemplar] = []
    for stage_trace in trace.stage_traces:
        for block_trace in stage_trace.block_traces:
            outputs = block_trace.outputs
            concept_names = block_trace.output_names
            concept_count = int(outputs.size(1))
            for concept_index in range(concept_count):
                scores = outputs[:, concept_index].abs()
                k = min(top_k, int(scores.numel()))
                if k <= 0:
                    continue
                values, indices = scores.topk(k=k)
                concept_name = (
                    concept_names[concept_index]
                    if concept_index < len(concept_names)
                    else f"concept_{concept_index}"
                )
                exemplars.append(
                    ConceptExemplar(
                        stage_name=stage_trace.stage_name,
                        block_name=block_trace.block_name,
                        concept_name=concept_name,
                        top_sample_indices=tuple(int(index) for index in indices.tolist()),
                        top_scores=tuple(float(value) for value in values.tolist()),
                    )
                )
    return tuple(exemplars)
