from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .blocks import BaseFuzzyRuleLayer
from .metrics import TaskType, compute_metrics
from .regularizers import (
    concept_binarization_penalty,
    concept_orthogonality_penalty,
    membership_coverage_penalty,
    membership_center_order_penalty,
    membership_overlap_penalty,
    rule_sparsity_penalty,
    weighted_rule_length_penalty,
)


@dataclass(frozen=True)
class TrainingConfig:
    task_type: TaskType
    max_epochs: int = 200
    learning_rate: float = 1e-3
    batch_size: int | None = None
    patience: int | None = 20
    min_delta: float = 0.0
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    shuffle: bool = True
    device: str | None = None
    classification_threshold: float = 0.5
    binary_auto_pos_weight: bool = False
    binary_pos_weight: float | None = None
    regression_loss: str = "mse"  # "mse" | "huber"
    huber_delta: float = 1.0
    monitor_metric: str | None = None
    monitor_mode: str | None = None  # "min" | "max"; defaults to "min" when metric is not set
    top_k_rules: int | None = None
    rule_sparsity_weight: float = 0.0
    rule_length_weight: float = 0.0
    concept_orthogonality_weight: float = 0.0
    concept_binarization_weight: float = 0.0
    membership_order_weight: float = 0.0
    membership_min_gap: float = 0.0
    membership_overlap_weight: float = 0.0
    membership_max_overlap: float = 0.35
    membership_coverage_weight: float = 0.0
    membership_min_coverage: float = 0.6
    prune_after_fit: bool = False
    prune_threshold: float = 0.1
    prune_temperature: float = 0.05
    prune_keep_at_least: int = 1


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    train_metrics: dict[str, float]
    validation_loss: float | None = None
    validation_metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class LayerPruningReport:
    layer_name: str
    active_rules_before: int
    active_rules_after: int
    probabilities_before: tuple[float, ...]
    probabilities_after: tuple[float, ...]


@dataclass(frozen=True)
class PruningReport:
    layers: tuple[LayerPruningReport, ...]


@dataclass(frozen=True)
class TrainingResult:
    history: tuple[EpochRecord, ...]
    epochs_ran: int
    best_epoch: int
    monitor_name: str
    best_monitor_value: float
    train_loss: float
    train_metrics: dict[str, float]
    validation_loss: float | None
    validation_metrics: dict[str, float] | None
    pruning_report: PruningReport | None


def soft_prune_rule_layers(
    module: nn.Module,
    threshold: float = 0.1,
    temperature: float = 0.05,
    keep_at_least: int = 1,
) -> PruningReport:
    if not 0.0 < threshold < 1.0:
        raise ValueError("The pruning threshold must lie strictly between 0 and 1.")
    if temperature <= 0.0:
        raise ValueError("The pruning temperature must be positive.")
    if keep_at_least < 0:
        raise ValueError("keep_at_least must be non-negative.")

    reports: list[LayerPruningReport] = []
    for submodule in module.modules():
        if not isinstance(submodule, BaseFuzzyRuleLayer):
            continue
        probabilities_before = torch.sigmoid(submodule.rule_logits.detach())
        softened = probabilities_before * torch.sigmoid((probabilities_before - threshold) / temperature)

        if keep_at_least > 0:
            top_k = min(keep_at_least, softened.numel())
            preserved_indices = torch.topk(probabilities_before, k=top_k).indices
            softened[preserved_indices] = probabilities_before[preserved_indices]

        probabilities_after = softened.clamp(min=1e-4, max=1.0 - 1e-4)
        with torch.no_grad():
            submodule.rule_logits.copy_(torch.logit(probabilities_after, eps=1e-4))

        reports.append(
            LayerPruningReport(
                layer_name=submodule.name,
                active_rules_before=int((probabilities_before >= threshold).sum().item()),
                active_rules_after=int((probabilities_after >= threshold).sum().item()),
                probabilities_before=tuple(float(value) for value in probabilities_before.tolist()),
                probabilities_after=tuple(float(value) for value in probabilities_after.tolist()),
            )
        )

    return PruningReport(layers=tuple(reports))


class FuzzyTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        loss_fn: nn.Module | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.loss_fn = loss_fn or self._default_loss(config.task_type)

    def fit(
        self,
        train_inputs: Tensor,
        train_targets: Tensor,
        validation_inputs: Tensor | None = None,
        validation_targets: Tensor | None = None,
    ) -> TrainingResult:
        has_validation = validation_inputs is not None and validation_targets is not None
        if (validation_inputs is None) != (validation_targets is None):
            raise ValueError("Validation inputs and targets must either both be provided or both be omitted.")

        device = self._resolve_device()
        self.model.to(device)
        train_inputs = train_inputs.to(device=device, dtype=torch.float32)
        train_targets = train_targets.to(device=device, dtype=torch.float32)
        if has_validation:
            validation_inputs = validation_inputs.to(device=device, dtype=torch.float32)
            validation_targets = validation_targets.to(device=device, dtype=torch.float32)

        if self.config.top_k_rules is not None and self.config.top_k_rules <= 0:
            raise ValueError("top_k_rules must be positive when provided.")

        if self.config.task_type == "binary_classification":
            effective_pos_weight: float | None = None
            if self.config.binary_auto_pos_weight:
                flat_targets = train_targets.reshape(-1)
                positive_count = float(flat_targets.sum().item())
                total_count = float(flat_targets.numel())
                negative_count = total_count - positive_count
                if positive_count > 0.0 and negative_count > 0.0:
                    effective_pos_weight = negative_count / positive_count
            elif self.config.binary_pos_weight is not None:
                effective_pos_weight = float(self.config.binary_pos_weight)

            if effective_pos_weight is not None and effective_pos_weight > 0.0:
                pos_weight_tensor = torch.tensor(
                    [effective_pos_weight],
                    device=device,
                    dtype=train_targets.dtype,
                )
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history: list[EpochRecord] = []
        best_state = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        monitor_mode = self._resolve_monitor_mode()
        if monitor_mode == "min":
            best_monitor_value = float("inf")
        elif monitor_mode == "max":
            best_monitor_value = float("-inf")
        else:
            raise ValueError(f"Unsupported monitor mode: {monitor_mode}. Expected 'min' or 'max'.")
        epochs_without_improvement = 0
        monitor_name = self._resolve_monitor_name(has_validation)

        for epoch in range(1, self.config.max_epochs + 1):
            train_result = self._run_training_epoch(train_inputs, train_targets, optimizer)
            validation_result = (
                self.evaluate(validation_inputs, validation_targets)
                if has_validation and validation_inputs is not None and validation_targets is not None
                else None
            )

            monitor_value = self._resolve_monitor_value(
                train_result=train_result,
                validation_result=validation_result,
                has_validation=has_validation,
            )
            history.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=train_result.loss,
                    train_metrics=dict(train_result.metrics),
                    validation_loss=validation_result.loss if validation_result is not None else None,
                    validation_metrics=dict(validation_result.metrics) if validation_result is not None else None,
                )
            )

            improved = (
                monitor_value < best_monitor_value - self.config.min_delta
                if monitor_mode == "min"
                else monitor_value > best_monitor_value + self.config.min_delta
            )
            if improved:
                best_monitor_value = monitor_value
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.config.patience is not None and epochs_without_improvement >= self.config.patience:
                break

        self.model.load_state_dict(best_state)

        pruning_report = None
        if self.config.prune_after_fit:
            pruning_report = soft_prune_rule_layers(
                self.model,
                threshold=self.config.prune_threshold,
                temperature=self.config.prune_temperature,
                keep_at_least=self.config.prune_keep_at_least,
            )

        final_train = self.evaluate(train_inputs, train_targets)
        final_validation = (
            self.evaluate(validation_inputs, validation_targets)
            if has_validation and validation_inputs is not None and validation_targets is not None
            else None
        )

        return TrainingResult(
            history=tuple(history),
            epochs_ran=len(history),
            best_epoch=best_epoch,
            monitor_name=monitor_name,
            best_monitor_value=best_monitor_value,
            train_loss=final_train.loss,
            train_metrics=dict(final_train.metrics),
            validation_loss=final_validation.loss if final_validation is not None else None,
            validation_metrics=dict(final_validation.metrics) if final_validation is not None else None,
            pruning_report=pruning_report,
        )

    def evaluate(self, inputs: Tensor, targets: Tensor) -> EvaluationResult:
        device = self._resolve_device()
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(inputs, top_k_rules=self.config.top_k_rules)
            aligned_targets = self._align_targets(predictions, targets)
            loss = self.loss_fn(predictions, aligned_targets).item()
            metrics = compute_metrics(
                self.config.task_type,
                predictions,
                aligned_targets,
                classification_threshold=self.config.classification_threshold,
            )
        return EvaluationResult(loss=float(loss), metrics=metrics)

    def _resolve_device(self) -> torch.device:
        if self.config.device is not None:
            return torch.device(self.config.device)
        try:
            parameter = next(self.model.parameters())
            return parameter.device
        except StopIteration:
            return torch.device("cpu")

    def _default_loss(self, task_type: TaskType) -> nn.Module:
        if task_type == "regression":
            if self.config.regression_loss == "mse":
                return nn.MSELoss()
            if self.config.regression_loss == "huber":
                return nn.HuberLoss(delta=self.config.huber_delta)
            raise ValueError(
                f"Unsupported regression_loss={self.config.regression_loss!r}. Expected 'mse' or 'huber'."
            )
        if task_type == "binary_classification":
            return nn.BCEWithLogitsLoss()
        raise ValueError(f"Unsupported task type: {task_type}.")

    def _run_training_epoch(
        self,
        inputs: Tensor,
        targets: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> EvaluationResult:
        self.model.train()
        prediction_batches: list[Tensor] = []
        target_batches: list[Tensor] = []
        total_task_loss = 0.0
        total_items = 0

        for batch_inputs, batch_targets in self._iter_batches(inputs, targets):
            optimizer.zero_grad()
            predictions = self.model(batch_inputs, top_k_rules=self.config.top_k_rules)
            aligned_targets = self._align_targets(predictions, batch_targets)
            task_loss = self.loss_fn(predictions, aligned_targets)
            total_loss = task_loss + self._regularization_penalty()
            total_loss.backward()

            if self.config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

            optimizer.step()

            batch_size = batch_inputs.size(0)
            total_task_loss += task_loss.item() * batch_size
            total_items += batch_size
            prediction_batches.append(predictions.detach())
            target_batches.append(aligned_targets.detach())

        stacked_predictions = torch.cat(prediction_batches, dim=0)
        stacked_targets = torch.cat(target_batches, dim=0)
        metrics = compute_metrics(
            self.config.task_type,
            stacked_predictions,
            stacked_targets,
            classification_threshold=self.config.classification_threshold,
        )
        return EvaluationResult(
            loss=float(total_task_loss / max(total_items, 1)),
            metrics=metrics,
        )

    def _resolve_monitor_mode(self) -> str:
        if self.config.monitor_mode is not None:
            return self.config.monitor_mode
        return "min"

    def _resolve_monitor_name(self, has_validation: bool) -> str:
        metric = self.config.monitor_metric
        if metric is None:
            return "validation_loss" if has_validation else "train_loss"
        prefix = "validation" if has_validation else "train"
        return f"{prefix}_{metric}"

    def _resolve_monitor_value(
        self,
        *,
        train_result: EvaluationResult,
        validation_result: EvaluationResult | None,
        has_validation: bool,
    ) -> float:
        metric_name = self.config.monitor_metric
        if metric_name is None:
            return validation_result.loss if validation_result is not None else train_result.loss
        source = validation_result if has_validation and validation_result is not None else train_result
        if metric_name not in source.metrics:
            available = ", ".join(sorted(source.metrics.keys()))
            raise ValueError(
                f"Monitor metric '{metric_name}' is unavailable. Available metrics: {available}."
            )
        return float(source.metrics[metric_name])

    def _regularization_penalty(self) -> Tensor:
        penalty = next(self.model.parameters()).new_tensor(0.0)
        if self.config.rule_sparsity_weight > 0.0:
            penalty = penalty + self.config.rule_sparsity_weight * rule_sparsity_penalty(self.model)
        if self.config.rule_length_weight > 0.0:
            penalty = penalty + self.config.rule_length_weight * weighted_rule_length_penalty(self.model)
        if self.config.concept_orthogonality_weight > 0.0:
            penalty = penalty + self.config.concept_orthogonality_weight * concept_orthogonality_penalty(self.model)
        if self.config.concept_binarization_weight > 0.0:
            penalty = penalty + self.config.concept_binarization_weight * concept_binarization_penalty(self.model)
        if self.config.membership_order_weight > 0.0:
            penalty = penalty + self.config.membership_order_weight * membership_center_order_penalty(
                self.model,
                min_gap=self.config.membership_min_gap,
            )
        if self.config.membership_overlap_weight > 0.0:
            penalty = penalty + self.config.membership_overlap_weight * membership_overlap_penalty(
                self.model,
                max_overlap=self.config.membership_max_overlap,
            )
        if self.config.membership_coverage_weight > 0.0:
            penalty = penalty + self.config.membership_coverage_weight * membership_coverage_penalty(
                self.model,
                min_coverage=self.config.membership_min_coverage,
            )
        return penalty

    def _iter_batches(self, inputs: Tensor, targets: Tensor) -> Iterable[tuple[Tensor, Tensor]]:
        batch_size = self.config.batch_size or inputs.size(0)
        if batch_size <= 0:
            raise ValueError("The batch size must be positive.")
        indices = (
            torch.randperm(inputs.size(0), device=inputs.device)
            if self.config.shuffle
            else torch.arange(inputs.size(0), device=inputs.device)
        )
        for start in range(0, inputs.size(0), batch_size):
            batch_indices = indices[start : start + batch_size]
            yield inputs.index_select(0, batch_indices), targets.index_select(0, batch_indices)

    def _align_targets(self, predictions: Tensor, targets: Tensor) -> Tensor:
        aligned_targets = targets.to(device=predictions.device, dtype=predictions.dtype)
        if predictions.ndim == 2 and predictions.size(-1) == 1 and aligned_targets.ndim == 1:
            aligned_targets = aligned_targets.unsqueeze(-1)
        return aligned_targets
