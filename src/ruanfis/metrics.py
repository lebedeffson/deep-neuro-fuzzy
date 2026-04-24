from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor


TaskType = Literal["regression", "binary_classification"]


def _flatten_pair(predictions: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    if predictions.ndim == 2 and predictions.size(-1) == 1:
        predictions = predictions.squeeze(-1)
    if targets.ndim == 2 and targets.size(-1) == 1:
        targets = targets.squeeze(-1)
    return predictions.reshape(-1), targets.reshape(-1)


def regression_metrics(predictions: Tensor, targets: Tensor) -> dict[str, float]:
    preds, gold = _flatten_pair(predictions, targets)
    diff = preds - gold
    mse = diff.pow(2).mean().item()
    mae = diff.abs().mean().item()
    rmse = math.sqrt(mse)
    centered = gold - gold.mean()
    denominator = centered.pow(2).sum().item()
    numerator = diff.pow(2).sum().item()
    r2 = 1.0 - (numerator / (denominator + 1e-12))
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def binary_classification_metrics(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    preds, gold = _flatten_pair(logits, targets)
    probabilities = torch.sigmoid(preds)
    predicted_labels = (probabilities >= threshold).to(dtype=gold.dtype)
    target_labels = (gold >= 0.5).to(dtype=gold.dtype)
    target_labels_bool = target_labels >= 0.5

    accuracy = (predicted_labels == target_labels).float().mean().item()
    true_positive = ((predicted_labels == 1) & (target_labels == 1)).sum().item()
    false_positive = ((predicted_labels == 1) & (target_labels == 0)).sum().item()
    false_negative = ((predicted_labels == 0) & (target_labels == 1)).sum().item()

    precision = true_positive / (true_positive + false_positive + 1e-12)
    recall = true_positive / (true_positive + false_negative + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    brier = (probabilities - target_labels).pow(2).mean().item()
    probabilities_clipped = probabilities.clamp(1e-7, 1.0 - 1e-7)
    log_loss = (
        -(target_labels * torch.log(probabilities_clipped) + (1.0 - target_labels) * torch.log(1.0 - probabilities_clipped))
    ).mean().item()

    positive_count = int(target_labels_bool.sum().item())
    negative_count = int((~target_labels_bool).sum().item())

    if positive_count == 0 or negative_count == 0:
        roc_auc = 0.5
    else:
        sorted_indices = torch.argsort(probabilities, descending=True)
        sorted_targets = target_labels_bool.index_select(0, sorted_indices).to(dtype=torch.float32)
        true_positive_cum = torch.cumsum(sorted_targets, dim=0)
        false_positive_cum = torch.cumsum(1.0 - sorted_targets, dim=0)
        tpr = torch.cat(
            [torch.zeros(1, dtype=torch.float32, device=sorted_targets.device), true_positive_cum / float(positive_count)]
        )
        fpr = torch.cat(
            [torch.zeros(1, dtype=torch.float32, device=sorted_targets.device), false_positive_cum / float(negative_count)]
        )
        roc_auc = torch.trapezoid(tpr, fpr).item()

    if positive_count == 0:
        pr_auc = 0.0
    else:
        sorted_indices = torch.argsort(probabilities, descending=True)
        sorted_targets = target_labels_bool.index_select(0, sorted_indices).to(dtype=torch.float32)
        ranks = torch.arange(1, sorted_targets.numel() + 1, dtype=torch.float32, device=sorted_targets.device)
        precision_at_k = torch.cumsum(sorted_targets, dim=0) / ranks
        pr_auc = (precision_at_k * sorted_targets).sum().item() / float(positive_count)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier": float(brier),
        "log_loss": float(log_loss),
    }


def compute_metrics(
    task_type: TaskType,
    predictions: Tensor,
    targets: Tensor,
    classification_threshold: float = 0.5,
) -> dict[str, float]:
    if task_type == "regression":
        return regression_metrics(predictions, targets)
    if task_type == "binary_classification":
        return binary_classification_metrics(
            predictions,
            targets,
            threshold=classification_threshold,
        )
    raise ValueError(f"Unsupported task type: {task_type}.")
