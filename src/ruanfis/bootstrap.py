from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from .builders import (
    HierarchicalModelConfig,
    ShallowFuzzyModelConfig,
    build_decision_layer,
    build_stage,
)
from .model import DeepFuzzyFeatureModel
from .regularizers import (
    concept_binarization_penalty,
    concept_orthogonality_penalty,
    membership_center_order_penalty,
    membership_coverage_penalty,
    membership_overlap_penalty,
    rule_sparsity_penalty,
)
from .stages import FuzzyStage


@dataclass(frozen=True)
class BootstrapConfig:
    hidden_high: float = 0.85
    hidden_low: float = 0.15
    gate_floor: float = 0.25
    gate_ceiling: float = 0.85
    decision_task_type: str = "regression"
    decision_binary_target_clip: float = 0.05


@dataclass(frozen=True)
class StagewisePretrainingConfig:
    task_type: str
    epochs_per_stage: int = 40
    decision_epochs: int = 30
    refinement_rounds: int = 1
    learning_rate: float = 0.02
    batch_size: int | None = 64
    shuffle: bool = True
    rule_sparsity_weight: float = 0.0
    concept_orthogonality_weight: float = 0.0
    concept_binarization_weight: float = 0.0
    membership_order_weight: float = 0.0
    membership_min_gap: float = 0.0
    membership_overlap_weight: float = 0.0
    membership_max_overlap: float = 0.35
    membership_coverage_weight: float = 0.0
    membership_min_coverage: float = 0.6
    classification_target_clip: float = 0.05


def _validate_probability_range(low: float, high: float) -> None:
    if not 0.0 < low < 1.0:
        raise ValueError("Bootstrap probabilities must lie strictly between 0 and 1.")
    if not 0.0 < high < 1.0:
        raise ValueError("Bootstrap probabilities must lie strictly between 0 and 1.")
    if low >= high:
        raise ValueError("The lower bootstrap probability must be smaller than the higher probability.")


def _to_feature_matrix(inputs: Tensor, expected_dim: int, *, name: str) -> Tensor:
    if inputs.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, features], got {tuple(inputs.shape)}.")
    if inputs.size(0) <= 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if inputs.size(1) != expected_dim:
        raise ValueError(f"{name} must contain {expected_dim} features, got {inputs.size(1)}.")
    return inputs.detach().cpu().to(dtype=torch.float32)


def _to_target_matrix(targets: Tensor, output_dim: int, *, name: str) -> Tensor:
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)
    if targets.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, outputs], got {tuple(targets.shape)}.")
    if targets.size(0) <= 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if targets.size(1) != output_dim:
        raise ValueError(f"{name} must contain {output_dim} outputs, got {targets.size(1)}.")
    return targets.detach().cpu().to(dtype=torch.float32)


def _task_loss_from_predictions(task_type: str, predictions: Tensor, targets: Tensor) -> Tensor:
    if task_type == "regression":
        return F.mse_loss(predictions, targets)
    if task_type == "binary_classification":
        return F.binary_cross_entropy_with_logits(predictions, targets)
    raise ValueError(f"Unsupported task_type={task_type!r}. Expected 'regression' or 'binary_classification'.")


def _iter_batches(inputs: Tensor, targets: Tensor, batch_size: int | None, shuffle: bool):
    effective_batch_size = inputs.size(0) if batch_size is None else batch_size
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    indices = torch.randperm(inputs.size(0)) if shuffle else torch.arange(inputs.size(0))
    for start in range(0, inputs.size(0), effective_batch_size):
        batch_indices = indices[start : start + effective_batch_size]
        yield inputs.index_select(0, batch_indices), targets.index_select(0, batch_indices)


def _stage_regularization_penalty(stage: FuzzyStage, config: StagewisePretrainingConfig) -> Tensor:
    penalty = next(stage.parameters()).new_tensor(0.0)
    if config.rule_sparsity_weight > 0.0:
        penalty = penalty + config.rule_sparsity_weight * rule_sparsity_penalty(stage)
    if config.concept_orthogonality_weight > 0.0:
        penalty = penalty + config.concept_orthogonality_weight * concept_orthogonality_penalty(stage)
    if config.concept_binarization_weight > 0.0:
        penalty = penalty + config.concept_binarization_weight * concept_binarization_penalty(stage)
    if config.membership_order_weight > 0.0:
        penalty = penalty + config.membership_order_weight * membership_center_order_penalty(
            stage,
            min_gap=config.membership_min_gap,
        )
    if config.membership_overlap_weight > 0.0:
        penalty = penalty + config.membership_overlap_weight * membership_overlap_penalty(
            stage,
            max_overlap=config.membership_max_overlap,
        )
    if config.membership_coverage_weight > 0.0:
        penalty = penalty + config.membership_coverage_weight * membership_coverage_penalty(
            stage,
            min_coverage=config.membership_min_coverage,
        )
    return penalty


def _fit_stage_with_linear_head(
    stage: FuzzyStage,
    stage_inputs: Tensor,
    targets: Tensor,
    config: StagewisePretrainingConfig,
) -> None:
    stage.train()
    head = nn.Linear(stage.output_dim, targets.size(1))
    optimizer = torch.optim.Adam(
        list(stage.parameters()) + list(head.parameters()),
        lr=config.learning_rate,
    )

    for _ in range(config.epochs_per_stage):
        for batch_inputs, batch_targets in _iter_batches(
            stage_inputs,
            targets,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
        ):
            optimizer.zero_grad()
            features = stage(batch_inputs)
            predictions = head(features)
            loss = _task_loss_from_predictions(config.task_type, predictions, batch_targets)
            loss = loss + _stage_regularization_penalty(stage, config)
            loss.backward()
            optimizer.step()

    stage.eval()


def _fit_decision_layer_on_features(
    layer: SugenoDecisionLayer,
    features: Tensor,
    targets: Tensor,
    config: StagewisePretrainingConfig,
) -> None:
    layer.train()
    optimizer = torch.optim.Adam(layer.parameters(), lr=config.learning_rate)
    for _ in range(config.decision_epochs):
        for batch_features, batch_targets in _iter_batches(
            features,
            targets,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
        ):
            optimizer.zero_grad()
            predictions = layer(batch_features)
            loss = _task_loss_from_predictions(config.task_type, predictions, batch_targets)
            loss.backward()
            optimizer.step()
    layer.eval()


def _compute_stage_reference_inputs(
    model: DeepFuzzyFeatureModel,
    raw_inputs: Tensor,
) -> tuple[Tensor, ...]:
    reference_inputs: list[Tensor] = []
    current = raw_inputs
    for stage in model.stages:
        reference_inputs.append(current)
        with torch.no_grad():
            current = stage(current)
    reference_inputs.append(current)
    return tuple(reference_inputs)


def _support_to_gate_probabilities(
    support: Tensor,
    *,
    floor: float,
    ceiling: float,
) -> Tensor:
    _validate_probability_range(floor, ceiling)
    if support.ndim != 1:
        raise ValueError("Rule support must be a one-dimensional tensor.")
    if support.numel() <= 0:
        raise ValueError("Rule support must not be empty.")

    support = support.detach().cpu().to(dtype=torch.float32)
    min_support = float(support.min().item())
    max_support = float(support.max().item())
    if max_support - min_support < 1e-12:
        return torch.full_like(support, fill_value=(floor + ceiling) * 0.5)
    scaled = (support - min_support) / (max_support - min_support)
    return floor + (ceiling - floor) * scaled


def _cluster_rule_profiles(
    rule_profiles: Tensor,
    *,
    n_clusters: int,
    support: Tensor,
) -> Tensor:
    if rule_profiles.ndim != 2:
        raise ValueError("rule_profiles must have shape [rules, samples].")
    n_rules = rule_profiles.size(0)
    if n_rules <= 0:
        raise ValueError("rule_profiles must contain at least one rule.")
    cluster_count = min(max(int(n_clusters), 1), n_rules)

    normalized_profiles = F.normalize(rule_profiles.detach().cpu().to(dtype=torch.float32), p=2.0, dim=1)
    support = support.detach().cpu().to(dtype=torch.float32)

    first_center = int(torch.argmax(support).item())
    center_indices = [first_center]

    while len(center_indices) < cluster_count:
        centers = normalized_profiles[center_indices]
        distances = torch.cdist(normalized_profiles, centers, p=2.0)
        min_distances = distances.min(dim=1).values
        min_distances[center_indices] = -1.0
        next_index = int(torch.argmax(min_distances).item())
        if next_index in center_indices:
            break
        center_indices.append(next_index)

    centers = normalized_profiles[center_indices]
    assignment_distances = torch.cdist(normalized_profiles, centers, p=2.0)
    return assignment_distances.argmin(dim=1)


def initialize_transparent_block_from_samples(
    block: TransparentFuzzyBlock,
    sample_inputs: Tensor,
    config: BootstrapConfig | None = None,
) -> None:
    bootstrap = config or BootstrapConfig()
    local_inputs = _to_feature_matrix(sample_inputs, block.input_dim, name="sample_inputs")

    with torch.no_grad():
        _, _, raw_rule_weights, normalized_rule_weights = block._run(local_inputs)
        support = raw_rule_weights.mean(dim=0)
        gate_probabilities = _support_to_gate_probabilities(
            support,
            floor=bootstrap.gate_floor,
            ceiling=bootstrap.gate_ceiling,
        )
        block.rule_logits.copy_(torch.logit(gate_probabilities.clamp(1e-4, 1.0 - 1e-4), eps=1e-4))

        rule_profiles = normalized_rule_weights.transpose(0, 1)
        assignments = _cluster_rule_profiles(
            rule_profiles,
            n_clusters=block.n_concepts,
            support=support,
        )

        template = torch.full(
            (block.n_rules, block.n_concepts),
            fill_value=bootstrap.hidden_low,
            dtype=block.raw_consequents.dtype,
        )
        for rule_index, concept_index in enumerate(assignments.tolist()):
            template[rule_index, concept_index] = bootstrap.hidden_high
        block.raw_consequents.copy_(torch.logit(template.clamp(1e-4, 1.0 - 1e-4), eps=1e-4))


def initialize_decision_layer_from_samples(
    layer: SugenoDecisionLayer,
    sample_inputs: Tensor,
    sample_targets: Tensor | None = None,
    config: BootstrapConfig | None = None,
) -> None:
    bootstrap = config or BootstrapConfig()
    local_inputs = _to_feature_matrix(sample_inputs, layer.input_dim, name="sample_inputs")

    with torch.no_grad():
        memberships = layer._fuzzify(local_inputs)
        raw_rule_weights = layer._compute_raw_rule_weights(memberships)
        support = raw_rule_weights.mean(dim=0)
        gate_probabilities = _support_to_gate_probabilities(
            support,
            floor=bootstrap.gate_floor,
            ceiling=bootstrap.gate_ceiling,
        )
        layer.rule_logits.copy_(torch.logit(gate_probabilities.clamp(1e-4, 1.0 - 1e-4), eps=1e-4))

        if sample_targets is None:
            layer.rule_bias.zero_()
            layer.rule_weights.zero_()
            return

        targets = _to_target_matrix(sample_targets, layer.output_dim, name="sample_targets")
        if targets.size(0) != local_inputs.size(0):
            raise ValueError("sample_inputs and sample_targets must contain the same number of samples.")

        if bootstrap.decision_task_type == "binary_classification":
            clip = float(bootstrap.decision_binary_target_clip)
            if not 0.0 < clip < 0.5:
                raise ValueError("decision_binary_target_clip must lie strictly between 0 and 0.5.")
            regression_targets = torch.logit(targets.clamp(clip, 1.0 - clip), eps=clip)
        elif bootstrap.decision_task_type == "regression":
            regression_targets = targets
        else:
            raise ValueError(
                f"Unsupported decision_task_type={bootstrap.decision_task_type!r}. "
                "Expected 'regression' or 'binary_classification'."
            )

        design = torch.cat([torch.ones(local_inputs.size(0), 1), local_inputs], dim=1)
        solution = torch.linalg.lstsq(design, regression_targets).solution
        solution = solution[: design.size(1)]
        bias = solution[0]
        weights = solution[1:]

        layer.rule_bias.copy_(bias.unsqueeze(0).expand_as(layer.rule_bias))
        layer.rule_weights.copy_(weights.unsqueeze(0).expand_as(layer.rule_weights))


def build_bootstrapped_hierarchical_model(
    config: HierarchicalModelConfig,
    sample_inputs: Tensor,
    sample_targets: Tensor | None = None,
    bootstrap_config: BootstrapConfig | None = None,
) -> DeepFuzzyFeatureModel:
    bootstrap = bootstrap_config or BootstrapConfig()
    current_samples = _to_feature_matrix(sample_inputs, config.input_dim, name="sample_inputs")
    target_matrix = None
    if sample_targets is not None:
        target_matrix = _to_target_matrix(sample_targets, config.decision_layer.output_dim, name="sample_targets")
        if target_matrix.size(0) != current_samples.size(0):
            raise ValueError("sample_inputs and sample_targets must contain the same number of samples.")

    stages: list[FuzzyStage] = []
    for stage_config in config.stages:
        stage = build_stage(stage_config, sample_inputs=current_samples)
        for connected_block in stage.blocks:
            local_inputs = current_samples.index_select(dim=1, index=connected_block.input_indices.cpu())
            initialize_transparent_block_from_samples(connected_block.block, local_inputs, config=bootstrap)
        stages.append(stage)
        with torch.no_grad():
            current_samples = stage(current_samples)

    decision_layer = build_decision_layer(config.decision_layer, sample_inputs=current_samples)
    initialize_decision_layer_from_samples(
        decision_layer,
        current_samples,
        sample_targets=target_matrix,
        config=bootstrap,
    )
    return DeepFuzzyFeatureModel(stages=stages, decision_layer=decision_layer, input_dim=config.input_dim)


def build_stagewise_pretrained_hierarchical_model(
    config: HierarchicalModelConfig,
    sample_inputs: Tensor,
    sample_targets: Tensor,
    bootstrap_config: BootstrapConfig | None = None,
    pretraining_config: StagewisePretrainingConfig | None = None,
) -> DeepFuzzyFeatureModel:
    bootstrap = bootstrap_config or BootstrapConfig()
    pretrain = pretraining_config or StagewisePretrainingConfig(task_type=bootstrap.decision_task_type)
    if pretrain.refinement_rounds <= 0:
        raise ValueError("refinement_rounds must be positive.")
    current_samples = _to_feature_matrix(sample_inputs, config.input_dim, name="sample_inputs")
    target_matrix = _to_target_matrix(sample_targets, config.decision_layer.output_dim, name="sample_targets")
    if target_matrix.size(0) != current_samples.size(0):
        raise ValueError("sample_inputs and sample_targets must contain the same number of samples.")

    if pretrain.task_type == "binary_classification":
        clip = float(pretrain.classification_target_clip)
        if not 0.0 < clip < 0.5:
            raise ValueError("classification_target_clip must lie strictly between 0 and 0.5.")
        training_targets = target_matrix.clamp(clip, 1.0 - clip)
    elif pretrain.task_type == "regression":
        training_targets = target_matrix
    else:
        raise ValueError(
            f"Unsupported task_type={pretrain.task_type!r}. Expected 'regression' or 'binary_classification'."
        )

    best_model: DeepFuzzyFeatureModel | None = None
    best_score = float("inf")
    reference_model: DeepFuzzyFeatureModel | None = None

    for refinement_round in range(pretrain.refinement_rounds):
        reference_inputs = None if reference_model is None else _compute_stage_reference_inputs(reference_model, current_samples)
        round_inputs = current_samples
        stages: list[FuzzyStage] = []

        for stage_index, stage_config in enumerate(config.stages):
            generation_inputs = round_inputs if reference_inputs is None else reference_inputs[stage_index]
            stage = build_stage(stage_config, sample_inputs=generation_inputs)
            for connected_block in stage.blocks:
                local_generation_inputs = generation_inputs.index_select(
                    dim=1,
                    index=connected_block.input_indices.cpu(),
                )
                initialize_transparent_block_from_samples(
                    connected_block.block,
                    local_generation_inputs,
                    config=bootstrap,
                )
            _fit_stage_with_linear_head(stage, round_inputs, training_targets, pretrain)
            stages.append(stage)
            with torch.no_grad():
                round_inputs = stage(round_inputs)

        decision_generation_inputs = round_inputs if reference_inputs is None else reference_inputs[-1]
        decision_layer = build_decision_layer(config.decision_layer, sample_inputs=decision_generation_inputs)
        initialize_decision_layer_from_samples(
            decision_layer,
            decision_generation_inputs,
            sample_targets=target_matrix,
            config=bootstrap,
        )
        _fit_decision_layer_on_features(decision_layer, round_inputs, training_targets, pretrain)
        candidate_model = DeepFuzzyFeatureModel(stages=stages, decision_layer=decision_layer, input_dim=config.input_dim)

        with torch.no_grad():
            candidate_score = float(_task_loss_from_predictions(pretrain.task_type, candidate_model(current_samples), training_targets).item())
        if candidate_score < best_score:
            best_score = candidate_score
            best_model = copy.deepcopy(candidate_model)
        reference_model = candidate_model

    if best_model is None:
        raise RuntimeError("Stage-wise pretraining failed to produce a model.")
    return best_model


def build_bootstrapped_shallow_model(
    config: ShallowFuzzyModelConfig,
    sample_inputs: Tensor,
    sample_targets: Tensor | None = None,
    bootstrap_config: BootstrapConfig | None = None,
) -> DeepFuzzyFeatureModel:
    return build_bootstrapped_hierarchical_model(
        config.as_hierarchical_config(),
        sample_inputs=sample_inputs,
        sample_targets=sample_targets,
        bootstrap_config=bootstrap_config,
    )


def build_stagewise_pretrained_shallow_model(
    config: ShallowFuzzyModelConfig,
    sample_inputs: Tensor,
    sample_targets: Tensor,
    bootstrap_config: BootstrapConfig | None = None,
    pretraining_config: StagewisePretrainingConfig | None = None,
) -> DeepFuzzyFeatureModel:
    return build_stagewise_pretrained_hierarchical_model(
        config.as_hierarchical_config(),
        sample_inputs=sample_inputs,
        sample_targets=sample_targets,
        bootstrap_config=bootstrap_config,
        pretraining_config=pretraining_config,
    )
