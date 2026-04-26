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
from .metrics import compute_metrics
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
    decision_wls_ridge: float = 1e-4
    decision_wls_min_effective_weight: float = 1.0


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
    stage_selection_metric: str = "loss"  # "auto" | "loss" | "f1"
    stage_selection_threshold: float = 0.5
    # Keep rule budget fixed while allowing periodic structure refresh from new candidates.
    rule_swap_ratio: float = 0.0
    rule_swap_min_keep: int = 0


def _resolve_device(
    device: str | torch.device | None,
    *,
    fallback: torch.device,
) -> torch.device:
    if device is None:
        return fallback
    return torch.device(device)


def _validate_probability_range(low: float, high: float) -> None:
    if not 0.0 < low < 1.0:
        raise ValueError("Bootstrap probabilities must lie strictly between 0 and 1.")
    if not 0.0 < high < 1.0:
        raise ValueError("Bootstrap probabilities must lie strictly between 0 and 1.")
    if low >= high:
        raise ValueError("The lower bootstrap probability must be smaller than the higher probability.")


def _to_feature_matrix(
    inputs: Tensor,
    expected_dim: int,
    *,
    name: str,
    device: str | torch.device | None = None,
) -> Tensor:
    if inputs.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, features], got {tuple(inputs.shape)}.")
    if inputs.size(0) <= 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if inputs.size(1) != expected_dim:
        raise ValueError(f"{name} must contain {expected_dim} features, got {inputs.size(1)}.")
    target_device = _resolve_device(device, fallback=inputs.device)
    return inputs.detach().to(device=target_device, dtype=torch.float32)


def _to_target_matrix(
    targets: Tensor,
    output_dim: int,
    *,
    name: str,
    device: str | torch.device | None = None,
) -> Tensor:
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)
    if targets.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, outputs], got {tuple(targets.shape)}.")
    if targets.size(0) <= 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if targets.size(1) != output_dim:
        raise ValueError(f"{name} must contain {output_dim} outputs, got {targets.size(1)}.")
    target_device = _resolve_device(device, fallback=targets.device)
    return targets.detach().to(device=target_device, dtype=torch.float32)


def _task_loss_from_predictions(task_type: str, predictions: Tensor, targets: Tensor) -> Tensor:
    if task_type == "regression":
        return F.mse_loss(predictions, targets)
    if task_type == "binary_classification":
        return F.binary_cross_entropy_with_logits(predictions, targets)
    raise ValueError(f"Unsupported task_type={task_type!r}. Expected 'regression' or 'binary_classification'.")


def _resolve_stage_selection_metric(pretrain: StagewisePretrainingConfig) -> str:
    metric = pretrain.stage_selection_metric.strip().lower()
    if metric == "auto":
        return "f1" if pretrain.task_type == "binary_classification" else "loss"
    if metric not in {"loss", "f1"}:
        raise ValueError(
            f"Unsupported stage_selection_metric={pretrain.stage_selection_metric!r}. "
            "Expected 'auto', 'loss' or 'f1'."
        )
    if metric == "f1" and pretrain.task_type != "binary_classification":
        raise ValueError("stage_selection_metric='f1' is only supported for binary_classification.")
    return metric


def _candidate_selection_score(
    pretrain: StagewisePretrainingConfig,
    predictions: Tensor,
    targets: Tensor,
) -> tuple[float, bool]:
    metric = _resolve_stage_selection_metric(pretrain)
    if metric == "loss":
        score = float(_task_loss_from_predictions(pretrain.task_type, predictions, targets).item())
        return score, False

    metrics = compute_metrics(
        "binary_classification",
        predictions,
        targets,
        classification_threshold=float(pretrain.stage_selection_threshold),
    )
    return float(metrics["f1"]), True


def _iter_batches(inputs: Tensor, targets: Tensor, batch_size: int | None, shuffle: bool):
    effective_batch_size = inputs.size(0) if batch_size is None else batch_size
    if effective_batch_size <= 0:
        raise ValueError("batch_size must be positive when provided.")
    indices = (
        torch.randperm(inputs.size(0), device=inputs.device)
        if shuffle
        else torch.arange(inputs.size(0), device=inputs.device)
    )
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
    head = nn.Linear(stage.output_dim, targets.size(1)).to(
        device=stage_inputs.device,
        dtype=stage_inputs.dtype,
    )
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
    layer.to(device=features.device, dtype=features.dtype)
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


def _append_config_final_skip_inputs(
    config: HierarchicalModelConfig,
    raw_inputs: Tensor,
    features: Tensor,
) -> Tensor:
    if not config.final_skip_input_indices:
        return features
    skip_indices = torch.tensor(config.final_skip_input_indices, dtype=torch.long, device=raw_inputs.device)
    return torch.cat((features, raw_inputs.index_select(1, skip_indices)), dim=1)


def _rule_signature_from_layer_rule(rule_spec) -> tuple[tuple[int, int], ...]:
    return tuple((int(ant.variable_index), int(ant.term_index)) for ant in rule_spec.antecedents)


def _select_rule_pairs_to_keep(
    *,
    reference_layer: TransparentFuzzyBlock | SugenoDecisionLayer,
    target_layer: TransparentFuzzyBlock | SugenoDecisionLayer,
    keep_count: int,
) -> list[tuple[int, int]]:
    reference_index_by_signature = {
        _rule_signature_from_layer_rule(rule): int(index)
        for index, rule in enumerate(reference_layer.rule_base.rules)
    }
    candidates: list[tuple[float, int, int]] = []
    reference_probabilities = torch.sigmoid(reference_layer.rule_logits.detach())
    for target_index, target_rule in enumerate(target_layer.rule_base.rules):
        signature = _rule_signature_from_layer_rule(target_rule)
        reference_index = reference_index_by_signature.get(signature)
        if reference_index is None:
            continue
        score = float(reference_probabilities[reference_index].item())
        candidates.append((score, int(reference_index), int(target_index)))

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = candidates[: max(0, min(int(keep_count), len(candidates)))]
    return [(reference_index, target_index) for _, reference_index, target_index in selected]


def _copy_transparent_rule_parameters(
    *,
    reference_layer: TransparentFuzzyBlock,
    target_layer: TransparentFuzzyBlock,
    reference_index: int,
    target_index: int,
) -> None:
    with torch.no_grad():
        target_layer.rule_logits[target_index].copy_(reference_layer.rule_logits[reference_index])
        if reference_layer.raw_consequents.shape[1] == target_layer.raw_consequents.shape[1]:
            target_layer.raw_consequents[target_index].copy_(reference_layer.raw_consequents[reference_index])
        if (
            reference_layer.rule_weights is not None
            and target_layer.rule_weights is not None
            and reference_layer.rule_weights.shape[1:] == target_layer.rule_weights.shape[1:]
        ):
            target_layer.rule_weights[target_index].copy_(reference_layer.rule_weights[reference_index])


def _copy_decision_rule_parameters(
    *,
    reference_layer: SugenoDecisionLayer,
    target_layer: SugenoDecisionLayer,
    reference_index: int,
    target_index: int,
) -> None:
    with torch.no_grad():
        target_layer.rule_logits[target_index].copy_(reference_layer.rule_logits[reference_index])
        if reference_layer.rule_bias.shape[1:] == target_layer.rule_bias.shape[1:]:
            target_layer.rule_bias[target_index].copy_(reference_layer.rule_bias[reference_index])
        if reference_layer.rule_weights.shape[1:] == target_layer.rule_weights.shape[1:]:
            target_layer.rule_weights[target_index].copy_(reference_layer.rule_weights[reference_index])


def _apply_rule_swap_warmstart(
    *,
    reference_layer: TransparentFuzzyBlock | SugenoDecisionLayer | None,
    target_layer: TransparentFuzzyBlock | SugenoDecisionLayer,
    config: StagewisePretrainingConfig,
) -> None:
    if reference_layer is None:
        return
    ratio = float(config.rule_swap_ratio)
    if ratio <= 0.0:
        return
    if ratio > 1.0:
        raise ValueError("rule_swap_ratio must be in [0, 1].")
    min_keep = int(config.rule_swap_min_keep)
    if min_keep < 0:
        raise ValueError("rule_swap_min_keep must be non-negative.")

    keep_count = int(round((1.0 - ratio) * target_layer.n_rules))
    keep_count = min(target_layer.n_rules, max(min_keep, keep_count))
    if keep_count <= 0:
        return

    pairs = _select_rule_pairs_to_keep(
        reference_layer=reference_layer,
        target_layer=target_layer,
        keep_count=keep_count,
    )
    if isinstance(target_layer, TransparentFuzzyBlock) and isinstance(reference_layer, TransparentFuzzyBlock):
        for reference_index, target_index in pairs:
            _copy_transparent_rule_parameters(
                reference_layer=reference_layer,
                target_layer=target_layer,
                reference_index=reference_index,
                target_index=target_index,
            )
        return
    if isinstance(target_layer, SugenoDecisionLayer) and isinstance(reference_layer, SugenoDecisionLayer):
        for reference_index, target_index in pairs:
            _copy_decision_rule_parameters(
                reference_layer=reference_layer,
                target_layer=target_layer,
                reference_index=reference_index,
                target_index=target_index,
            )


def _validate_reference_model_compatibility(
    config: HierarchicalModelConfig,
    reference_model: DeepFuzzyFeatureModel,
) -> None:
    if reference_model.input_dim is not None and reference_model.input_dim != config.input_dim:
        raise ValueError(
            f"reference_model expects input_dim={reference_model.input_dim}, but config uses {config.input_dim}."
        )
    if len(reference_model.stages) != len(config.stages):
        raise ValueError(
            "reference_model and config must contain the same number of hidden stages for rule re-estimation."
        )
    if reference_model.decision_layer.input_dim != len(config.decision_layer.variables):
        raise ValueError(
            "reference_model decision layer input dimension does not match the provided config."
        )


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

    support = support.detach().to(dtype=torch.float32)
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

    normalized_profiles = F.normalize(rule_profiles.detach().to(dtype=torch.float32), p=2.0, dim=1)
    support = support.detach().to(dtype=torch.float32)

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
        _, _, raw_rule_weights, normalized_rule_weights, _ = block._run(local_inputs)
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
            device=block.raw_consequents.device,
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
        normalized_rule_weights = layer._normalize_rule_weights(raw_rule_weights)
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

        targets = _to_target_matrix(
            sample_targets,
            layer.output_dim,
            name="sample_targets",
            device=local_inputs.device,
        )
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

        design = torch.cat(
            [
                torch.ones(
                    local_inputs.size(0),
                    1,
                    device=local_inputs.device,
                    dtype=local_inputs.dtype,
                ),
                local_inputs,
            ],
            dim=1,
        )

        ridge = float(bootstrap.decision_wls_ridge)
        if ridge < 0.0:
            raise ValueError("decision_wls_ridge must be non-negative.")
        min_effective_weight = float(bootstrap.decision_wls_min_effective_weight)
        if min_effective_weight < 0.0:
            raise ValueError("decision_wls_min_effective_weight must be non-negative.")

        def _solve_weighted_linear(sample_weights: Tensor) -> Tensor:
            sqrt_weights = sample_weights.clamp_min(0.0).sqrt().unsqueeze(-1)
            weighted_design = design * sqrt_weights
            weighted_targets = regression_targets * sqrt_weights

            if ridge > 0.0:
                feature_dim = weighted_design.size(1)
                gram = weighted_design.transpose(0, 1) @ weighted_design
                rhs = weighted_design.transpose(0, 1) @ weighted_targets
                regularizer = torch.eye(feature_dim, device=design.device, dtype=design.dtype) * ridge
                try:
                    return torch.linalg.solve(gram + regularizer, rhs)
                except RuntimeError:
                    pass

            solution = torch.linalg.lstsq(weighted_design, weighted_targets).solution
            return solution[: design.size(1)]

        global_solution = _solve_weighted_linear(
            torch.ones(local_inputs.size(0), device=design.device, dtype=design.dtype)
        )

        rule_biases = torch.empty_like(layer.rule_bias)
        rule_weights = torch.empty_like(layer.rule_weights)
        for rule_index in range(layer.n_rules):
            sample_weights = normalized_rule_weights[:, rule_index]
            effective_weight = float(sample_weights.sum().item())
            if effective_weight < min_effective_weight:
                solution = global_solution
            else:
                solution = _solve_weighted_linear(sample_weights)
            rule_biases[rule_index] = solution[0]
            rule_weights[rule_index] = solution[1:]

        layer.rule_bias.copy_(rule_biases)
        layer.rule_weights.copy_(rule_weights)


def build_bootstrapped_hierarchical_model(
    config: HierarchicalModelConfig,
    sample_inputs: Tensor,
    sample_targets: Tensor | None = None,
    bootstrap_config: BootstrapConfig | None = None,
    device: str | torch.device | None = None,
) -> DeepFuzzyFeatureModel:
    bootstrap = bootstrap_config or BootstrapConfig()
    current_samples = _to_feature_matrix(
        sample_inputs,
        config.input_dim,
        name="sample_inputs",
        device=device,
    )
    raw_samples = current_samples
    target_matrix = None
    if sample_targets is not None:
        target_matrix = _to_target_matrix(
            sample_targets,
            config.decision_layer.output_dim,
            name="sample_targets",
            device=current_samples.device,
        )
        if target_matrix.size(0) != current_samples.size(0):
            raise ValueError("sample_inputs and sample_targets must contain the same number of samples.")

    stages: list[FuzzyStage] = []
    compute_device = current_samples.device
    for stage_config in config.stages:
        stage = build_stage(stage_config, sample_inputs=current_samples).to(device=compute_device)
        for connected_block in stage.blocks:
            local_inputs = current_samples.index_select(dim=1, index=connected_block.input_indices)
            initialize_transparent_block_from_samples(connected_block.block, local_inputs, config=bootstrap)
        stages.append(stage)
        with torch.no_grad():
            current_samples = stage(current_samples)

    decision_inputs = _append_config_final_skip_inputs(config, raw_samples, current_samples)
    decision_layer = build_decision_layer(config.decision_layer, sample_inputs=decision_inputs).to(device=compute_device)
    initialize_decision_layer_from_samples(
        decision_layer,
        decision_inputs,
        sample_targets=target_matrix,
        config=bootstrap,
    )
    return DeepFuzzyFeatureModel(
        stages=stages,
        decision_layer=decision_layer,
        input_dim=config.input_dim,
        final_skip_input_indices=config.final_skip_input_indices,
        final_skip_gates_enabled=config.final_skip_gates_enabled,
        final_skip_gate_init_logit=config.final_skip_gate_init_logit,
    )


def build_stagewise_pretrained_hierarchical_model(
    config: HierarchicalModelConfig,
    sample_inputs: Tensor,
    sample_targets: Tensor,
    validation_inputs: Tensor | None = None,
    validation_targets: Tensor | None = None,
    bootstrap_config: BootstrapConfig | None = None,
    pretraining_config: StagewisePretrainingConfig | None = None,
    device: str | torch.device | None = None,
) -> DeepFuzzyFeatureModel:
    bootstrap = bootstrap_config or BootstrapConfig()
    pretrain = pretraining_config or StagewisePretrainingConfig(task_type=bootstrap.decision_task_type)
    if pretrain.refinement_rounds <= 0:
        raise ValueError("refinement_rounds must be positive.")
    current_samples = _to_feature_matrix(
        sample_inputs,
        config.input_dim,
        name="sample_inputs",
        device=device,
    )
    target_matrix = _to_target_matrix(
        sample_targets,
        config.decision_layer.output_dim,
        name="sample_targets",
        device=current_samples.device,
    )
    if target_matrix.size(0) != current_samples.size(0):
        raise ValueError("sample_inputs and sample_targets must contain the same number of samples.")
    if (validation_inputs is None) != (validation_targets is None):
        raise ValueError("validation_inputs and validation_targets must either both be provided or both be omitted.")

    validation_samples: Tensor | None = None
    validation_target_matrix: Tensor | None = None
    if validation_inputs is not None and validation_targets is not None:
        validation_samples = _to_feature_matrix(
            validation_inputs,
            config.input_dim,
            name="validation_inputs",
            device=current_samples.device,
        )
        validation_target_matrix = _to_target_matrix(
            validation_targets,
            config.decision_layer.output_dim,
            name="validation_targets",
            device=current_samples.device,
        )
        if validation_target_matrix.size(0) != validation_samples.size(0):
            raise ValueError("validation_inputs and validation_targets must contain the same number of samples.")

    if pretrain.task_type == "binary_classification":
        clip = float(pretrain.classification_target_clip)
        if not 0.0 < clip < 0.5:
            raise ValueError("classification_target_clip must lie strictly between 0 and 0.5.")
        training_targets = target_matrix.clamp(clip, 1.0 - clip)
        validation_training_targets = (
            validation_target_matrix.clamp(clip, 1.0 - clip) if validation_target_matrix is not None else None
        )
    elif pretrain.task_type == "regression":
        training_targets = target_matrix
        validation_training_targets = validation_target_matrix
    else:
        raise ValueError(
            f"Unsupported task_type={pretrain.task_type!r}. Expected 'regression' or 'binary_classification'."
        )

    return _build_stagewise_from_reference(
        config,
        current_samples,
        target_matrix,
        training_targets,
        validation_samples,
        validation_training_targets,
        bootstrap,
        pretrain,
        initial_reference_model=None,
    )


def _build_stagewise_from_reference(
    config: HierarchicalModelConfig,
    current_samples: Tensor,
    target_matrix: Tensor,
    training_targets: Tensor,
    validation_samples: Tensor | None,
    validation_training_targets: Tensor | None,
    bootstrap: BootstrapConfig,
    pretrain: StagewisePretrainingConfig,
    *,
    initial_reference_model: DeepFuzzyFeatureModel | None,
) -> DeepFuzzyFeatureModel:
    best_model: DeepFuzzyFeatureModel | None = None
    higher_is_better = _resolve_stage_selection_metric(pretrain) == "f1"
    best_score = float("-inf") if higher_is_better else float("inf")
    reference_model: DeepFuzzyFeatureModel | None = initial_reference_model

    for _ in range(pretrain.refinement_rounds):
        compute_device = current_samples.device
        if reference_model is None:
            reference_inputs = None
            reference_blocks_by_stage: list[dict[str, TransparentFuzzyBlock] | None] = [None] * len(config.stages)
            reference_decision_layer = None
        else:
            reference_for_reestimation = copy.deepcopy(reference_model).to(device=compute_device).eval()
            reference_inputs = _compute_stage_reference_inputs(reference_for_reestimation, current_samples)
            reference_blocks_by_stage = []
            for stage in reference_for_reestimation.stages:
                stage_blocks = {str(connected_block.name): connected_block.block for connected_block in stage.blocks}
                reference_blocks_by_stage.append(stage_blocks)
            reference_decision_layer = reference_for_reestimation.decision_layer
        round_inputs = current_samples
        stages: list[FuzzyStage] = []

        for stage_index, stage_config in enumerate(config.stages):
            generation_inputs = round_inputs if reference_inputs is None else reference_inputs[stage_index]
            stage = build_stage(stage_config, sample_inputs=generation_inputs).to(device=compute_device)
            for connected_block in stage.blocks:
                local_generation_inputs = generation_inputs.index_select(
                    dim=1,
                    index=connected_block.input_indices,
                )
                initialize_transparent_block_from_samples(
                    connected_block.block,
                    local_generation_inputs,
                    config=bootstrap,
                )
                reference_block = None
                stage_reference = reference_blocks_by_stage[stage_index]
                if stage_reference is not None:
                    reference_block = stage_reference.get(str(connected_block.name))
                _apply_rule_swap_warmstart(
                    reference_layer=reference_block,
                    target_layer=connected_block.block,
                    config=pretrain,
                )
            _fit_stage_with_linear_head(stage, round_inputs, training_targets, pretrain)
            stages.append(stage)
            with torch.no_grad():
                round_inputs = stage(round_inputs)

        decision_generation_stage_inputs = round_inputs if reference_inputs is None else reference_inputs[-1]
        decision_generation_inputs = _append_config_final_skip_inputs(
            config,
            current_samples,
            decision_generation_stage_inputs,
        )
        decision_layer = build_decision_layer(
            config.decision_layer,
            sample_inputs=decision_generation_inputs,
        ).to(device=compute_device)
        initialize_decision_layer_from_samples(
            decision_layer,
            decision_generation_inputs,
            sample_targets=target_matrix,
            config=bootstrap,
        )
        _apply_rule_swap_warmstart(
            reference_layer=reference_decision_layer,
            target_layer=decision_layer,
            config=pretrain,
        )
        round_decision_inputs = _append_config_final_skip_inputs(config, current_samples, round_inputs)
        _fit_decision_layer_on_features(decision_layer, round_decision_inputs, training_targets, pretrain)
        candidate_model = DeepFuzzyFeatureModel(
            stages=stages,
            decision_layer=decision_layer,
            input_dim=config.input_dim,
            final_skip_input_indices=config.final_skip_input_indices,
            final_skip_gates_enabled=config.final_skip_gates_enabled,
            final_skip_gate_init_logit=config.final_skip_gate_init_logit,
        )

        with torch.no_grad():
            if validation_samples is not None and validation_training_targets is not None:
                score_predictions = candidate_model(validation_samples)
                score_targets = validation_training_targets
            else:
                score_predictions = candidate_model(current_samples)
                score_targets = training_targets
            candidate_score, _ = _candidate_selection_score(pretrain, score_predictions, score_targets)

        improved = candidate_score > best_score if higher_is_better else candidate_score < best_score
        if improved:
            best_score = candidate_score
            best_model = copy.deepcopy(candidate_model)
        reference_model = candidate_model

    if best_model is None:
        raise RuntimeError("Stage-wise pretraining failed to produce a model.")
    return best_model


def reestimate_hierarchical_model_rule_base(
    config: HierarchicalModelConfig,
    reference_model: DeepFuzzyFeatureModel,
    sample_inputs: Tensor,
    sample_targets: Tensor,
    validation_inputs: Tensor | None = None,
    validation_targets: Tensor | None = None,
    bootstrap_config: BootstrapConfig | None = None,
    pretraining_config: StagewisePretrainingConfig | None = None,
    device: str | torch.device | None = None,
) -> DeepFuzzyFeatureModel:
    bootstrap = bootstrap_config or BootstrapConfig()
    pretrain = pretraining_config or StagewisePretrainingConfig(task_type=bootstrap.decision_task_type)
    if pretrain.refinement_rounds <= 0:
        raise ValueError("refinement_rounds must be positive.")

    _validate_reference_model_compatibility(config, reference_model)

    current_samples = _to_feature_matrix(
        sample_inputs,
        config.input_dim,
        name="sample_inputs",
        device=device,
    )
    target_matrix = _to_target_matrix(
        sample_targets,
        config.decision_layer.output_dim,
        name="sample_targets",
        device=current_samples.device,
    )
    if target_matrix.size(0) != current_samples.size(0):
        raise ValueError("sample_inputs and sample_targets must contain the same number of samples.")
    if (validation_inputs is None) != (validation_targets is None):
        raise ValueError("validation_inputs and validation_targets must either both be provided or both be omitted.")

    validation_samples: Tensor | None = None
    validation_target_matrix: Tensor | None = None
    if validation_inputs is not None and validation_targets is not None:
        validation_samples = _to_feature_matrix(
            validation_inputs,
            config.input_dim,
            name="validation_inputs",
            device=current_samples.device,
        )
        validation_target_matrix = _to_target_matrix(
            validation_targets,
            config.decision_layer.output_dim,
            name="validation_targets",
            device=current_samples.device,
        )
        if validation_target_matrix.size(0) != validation_samples.size(0):
            raise ValueError("validation_inputs and validation_targets must contain the same number of samples.")

    if pretrain.task_type == "binary_classification":
        clip = float(pretrain.classification_target_clip)
        if not 0.0 < clip < 0.5:
            raise ValueError("classification_target_clip must lie strictly between 0 and 0.5.")
        training_targets = target_matrix.clamp(clip, 1.0 - clip)
        validation_training_targets = (
            validation_target_matrix.clamp(clip, 1.0 - clip) if validation_target_matrix is not None else None
        )
    elif pretrain.task_type == "regression":
        training_targets = target_matrix
        validation_training_targets = validation_target_matrix
    else:
        raise ValueError(
            f"Unsupported task_type={pretrain.task_type!r}. Expected 'regression' or 'binary_classification'."
        )

    return _build_stagewise_from_reference(
        config,
        current_samples,
        target_matrix,
        training_targets,
        validation_samples,
        validation_training_targets,
        bootstrap,
        pretrain,
        initial_reference_model=reference_model,
    )


def build_bootstrapped_shallow_model(
    config: ShallowFuzzyModelConfig,
    sample_inputs: Tensor,
    sample_targets: Tensor | None = None,
    bootstrap_config: BootstrapConfig | None = None,
    device: str | torch.device | None = None,
) -> DeepFuzzyFeatureModel:
    return build_bootstrapped_hierarchical_model(
        config.as_hierarchical_config(),
        sample_inputs=sample_inputs,
        sample_targets=sample_targets,
        bootstrap_config=bootstrap_config,
        device=device,
    )


def build_stagewise_pretrained_shallow_model(
    config: ShallowFuzzyModelConfig,
    sample_inputs: Tensor,
    sample_targets: Tensor,
    validation_inputs: Tensor | None = None,
    validation_targets: Tensor | None = None,
    bootstrap_config: BootstrapConfig | None = None,
    pretraining_config: StagewisePretrainingConfig | None = None,
    device: str | torch.device | None = None,
) -> DeepFuzzyFeatureModel:
    return build_stagewise_pretrained_hierarchical_model(
        config.as_hierarchical_config(),
        sample_inputs=sample_inputs,
        sample_targets=sample_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        bootstrap_config=bootstrap_config,
        pretraining_config=pretraining_config,
        device=device,
    )


def reestimate_shallow_model_rule_base(
    config: ShallowFuzzyModelConfig,
    reference_model: DeepFuzzyFeatureModel,
    sample_inputs: Tensor,
    sample_targets: Tensor,
    validation_inputs: Tensor | None = None,
    validation_targets: Tensor | None = None,
    bootstrap_config: BootstrapConfig | None = None,
    pretraining_config: StagewisePretrainingConfig | None = None,
    device: str | torch.device | None = None,
) -> DeepFuzzyFeatureModel:
    return reestimate_hierarchical_model_rule_base(
        config.as_hierarchical_config(),
        reference_model=reference_model,
        sample_inputs=sample_inputs,
        sample_targets=sample_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        bootstrap_config=bootstrap_config,
        pretraining_config=pretraining_config,
        device=device,
    )
