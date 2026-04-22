from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from torch import Tensor, nn

from .metrics import TaskType, compute_metrics
from .model import DeepFuzzyFeatureModel
from .stacked import StackedAnfisModel
from .trainer import FuzzyTrainer, TrainingConfig

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
    CatBoostRegressor = None


@dataclass(frozen=True)
class BenchmarkEntryResult:
    model_name: str
    family: str
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    classification_threshold: float | None = None
    structural_metrics: dict[str, float] = field(default_factory=dict)
    explainability_metrics: dict[str, float] = field(default_factory=dict)
    stability_artifacts: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float


@dataclass(frozen=True)
class AggregatedBenchmarkEntryResult:
    model_name: str
    family: str
    runs: int
    train_metrics: dict[str, MetricSummary]
    test_metrics: dict[str, MetricSummary]
    structural_metrics: dict[str, MetricSummary]
    explainability_metrics: dict[str, MetricSummary]
    stability_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiSeedBenchmarkResult:
    seeds: tuple[int, ...]
    per_seed_results: tuple[tuple[BenchmarkEntryResult, ...], ...]
    aggregated_results: tuple[AggregatedBenchmarkEntryResult, ...]


def _to_feature_tensor(values: Tensor) -> Tensor:
    if values.ndim != 2:
        raise ValueError(f"Expected feature matrix [samples, features], got {tuple(values.shape)}.")
    return values.detach().cpu().to(dtype=torch.float32)


def _to_target_tensor(values: Tensor, task_type: TaskType) -> Tensor:
    tensor = values.detach().cpu().to(dtype=torch.float32)
    if task_type == "binary_classification":
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(-1)
        return tensor
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    return tensor


def _sklearn_prediction_to_tensor(task_type: TaskType, estimator, features: Tensor) -> Tensor:
    feature_array = features.numpy()
    if task_type == "regression":
        predictions = estimator.predict(feature_array)
        return torch.as_tensor(predictions, dtype=torch.float32).reshape(-1, 1)

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(feature_array)[:, 1]
        clipped = torch.as_tensor(probabilities, dtype=torch.float32).clamp(1e-5, 1.0 - 1e-5)
        return torch.logit(clipped, eps=1e-5).reshape(-1, 1)
    if hasattr(estimator, "decision_function"):
        logits = estimator.decision_function(feature_array)
        return torch.as_tensor(logits, dtype=torch.float32).reshape(-1, 1)
    predictions = estimator.predict(feature_array)
    clipped = torch.as_tensor(predictions, dtype=torch.float32).clamp(1e-5, 1.0 - 1e-5)
    return torch.logit(clipped, eps=1e-5).reshape(-1, 1)


def _benchmark_sklearn_model(
    model_name: str,
    estimator,
    *,
    task_type: TaskType,
    train_inputs: Tensor,
    train_targets: Tensor,
    test_inputs: Tensor,
    test_targets: Tensor,
    classification_threshold: float,
) -> BenchmarkEntryResult:
    target_array = train_targets.squeeze(-1).numpy()
    estimator.fit(train_inputs.numpy(), target_array)
    train_predictions = _sklearn_prediction_to_tensor(task_type, estimator, train_inputs)
    test_predictions = _sklearn_prediction_to_tensor(task_type, estimator, test_inputs)
    return BenchmarkEntryResult(
        model_name=model_name,
        family="sklearn",
        train_metrics=compute_metrics(
            task_type,
            train_predictions,
            train_targets,
            classification_threshold=classification_threshold,
        ),
        test_metrics=compute_metrics(
            task_type,
            test_predictions,
            test_targets,
            classification_threshold=classification_threshold,
        ),
        classification_threshold=float(classification_threshold) if task_type == "binary_classification" else None,
    )


def _iter_fuzzy_rule_layers(model: nn.Module) -> tuple[nn.Module, ...]:
    if isinstance(model, DeepFuzzyFeatureModel):
        layers = [connected_block.block for stage in model.stages for connected_block in stage.blocks]
        layers.append(model.decision_layer)
        return tuple(layers)
    if isinstance(model, StackedAnfisModel):
        return tuple(model.iter_rule_layers())
    return ()


def _iter_fuzzy_rule_layer_entries(
    model: nn.Module,
) -> tuple[tuple[str, nn.Module], ...]:
    if isinstance(model, DeepFuzzyFeatureModel):
        entries = [
            (f"{stage.name}/{connected_block.block.name}", connected_block.block)
            for stage in model.stages
            for connected_block in stage.blocks
        ]
        entries.append((model.decision_layer.name, model.decision_layer))
        return tuple(entries)
    if isinstance(model, StackedAnfisModel):
        return tuple(model.iter_rule_layer_entries())
    return ()


def _entropy(values: Tensor, epsilon: float = 1e-8) -> Tensor:
    safe_values = values.clamp_min(epsilon)
    return -(safe_values * safe_values.log()).sum(dim=1)


def _jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _pairwise_mean(values: Sequence[float]) -> float:
    if not values:
        return 1.0
    return float(sum(values) / len(values))


def _summarize_fuzzy_model_structure(
    model: nn.Module,
    *,
    rule_probability_threshold: float,
) -> dict[str, float]:
    layers = _iter_fuzzy_rule_layers(model)
    if isinstance(model, DeepFuzzyFeatureModel):
        hidden_blocks = sum(len(stage.blocks) for stage in model.stages)
        hidden_concepts = sum(
            connected_block.block.output_dim for stage in model.stages for connected_block in stage.blocks
        )
        stages = float(len(model.stages))
    else:
        # Stacked ANFIS: every hidden layer is a fuzzy rule layer.
        hidden_blocks = max(len(layers) - 1, 0)
        hidden_concepts = sum(layer.output_dim for layer in layers[:-1])
        stages = float(hidden_blocks)

    total_rules = float(sum(layer.n_rules for layer in layers))
    active_rules = float(
        sum((layer.rule_probabilities.detach() >= rule_probability_threshold).sum().item() for layer in layers)
    )
    return {
        "total_rules": total_rules,
        "active_rules": active_rules,
        "stages": stages,
        "hidden_blocks": float(hidden_blocks),
        "hidden_concepts": float(hidden_concepts),
    }


def _summarize_fuzzy_model_explainability(
    model: nn.Module,
    inputs: Tensor,
    *,
    top_k_rules: int | None = None,
) -> dict[str, float]:
    if inputs.numel() == 0:
        return {}

    device = next(model.parameters(), torch.empty(0)).device
    if isinstance(model, DeepFuzzyFeatureModel):
        model.eval()
        with torch.no_grad():
            _, trace = model.forward_with_trace(
                inputs.to(device=device, dtype=torch.float32),
                top_k_rules=top_k_rules,
            )
        decision_weights = trace.decision_trace.normalized_rule_weights.detach().cpu()
        hidden_weight_tensors = [
            block_trace.normalized_rule_weights.detach().cpu()
            for stage_trace in trace.stage_traces
            for block_trace in stage_trace.block_traces
        ]
    elif isinstance(model, StackedAnfisModel):
        model.eval()
        with torch.no_grad():
            _, traces = model.forward_with_trace(
                inputs.to(device=device, dtype=torch.float32),
                top_k_rules=top_k_rules,
            )
        decision_weights = traces[-1].normalized_rule_weights.detach().cpu()
        hidden_weight_tensors = [trace.normalized_rule_weights.detach().cpu() for trace in traces[:-1]]
    else:
        return {}

    metrics = {
        "decision_top1_mass": float(decision_weights.max(dim=1).values.mean().item()),
        "decision_top3_mass": float(
            decision_weights.topk(k=min(3, decision_weights.size(1)), dim=1).values.sum(dim=1).mean().item()
        ),
        "decision_entropy": float(_entropy(decision_weights).mean().item()),
    }

    hidden_top1_values: list[float] = []
    hidden_top3_values: list[float] = []
    hidden_entropies: list[float] = []
    for weights in hidden_weight_tensors:
        hidden_top1_values.append(float(weights.max(dim=1).values.mean().item()))
        hidden_top3_values.append(
            float(weights.topk(k=min(3, weights.size(1)), dim=1).values.sum(dim=1).mean().item())
        )
        hidden_entropies.append(float(_entropy(weights).mean().item()))

    if hidden_top1_values:
        metrics["hidden_top1_mass"] = float(sum(hidden_top1_values) / len(hidden_top1_values))
        metrics["hidden_top3_mass"] = float(sum(hidden_top3_values) / len(hidden_top3_values))
        metrics["hidden_entropy"] = float(sum(hidden_entropies) / len(hidden_entropies))

    return metrics


def _extract_fuzzy_model_stability_artifacts(
    model: nn.Module,
    *,
    rule_probability_threshold: float,
) -> dict[str, tuple[str, ...]]:
    hidden_generated_rules: list[str] = []
    hidden_active_rules: list[str] = []
    decision_generated_rules: list[str] = []
    decision_active_rules: list[str] = []
    artifacts: dict[str, tuple[str, ...]] = {}

    for layer_path, layer in _iter_fuzzy_rule_layer_entries(model):
        generated_rules = tuple(
            f"{layer_path}::{layer.describe_rule(rule_index)}"
            for rule_index in range(layer.n_rules)
        )
        active_rules = tuple(
            generated_rules[rule_index]
            for rule_index, probability in enumerate(layer.rule_probabilities.detach().cpu().tolist())
            if probability >= rule_probability_threshold
        )
        artifacts[f"generated:{layer_path}"] = generated_rules
        artifacts[f"active:{layer_path}"] = active_rules

        if isinstance(model, DeepFuzzyFeatureModel) and layer is model.decision_layer:
            decision_generated_rules.extend(generated_rules)
            decision_active_rules.extend(active_rules)
        elif isinstance(model, StackedAnfisModel) and layer_path == model.layers[-1].name:
            decision_generated_rules.extend(generated_rules)
            decision_active_rules.extend(active_rules)
        else:
            hidden_generated_rules.extend(generated_rules)
            hidden_active_rules.extend(active_rules)

    artifacts["generated:all"] = tuple([*hidden_generated_rules, *decision_generated_rules])
    artifacts["generated:hidden"] = tuple(hidden_generated_rules)
    artifacts["generated:decision"] = tuple(decision_generated_rules)
    artifacts["active:all"] = tuple([*hidden_active_rules, *decision_active_rules])
    artifacts["active:hidden"] = tuple(hidden_active_rules)
    artifacts["active:decision"] = tuple(decision_active_rules)
    return artifacts


def evaluate_trained_model(
    model_name: str,
    model: nn.Module,
    *,
    task_type: TaskType,
    train_inputs: Tensor,
    train_targets: Tensor,
    test_inputs: Tensor,
    test_targets: Tensor,
    family: str = "ruanfis",
    classification_threshold: float = 0.5,
    rule_probability_threshold: float = 0.5,
    top_k_rules: int | None = None,
) -> BenchmarkEntryResult:
    train_features = _to_feature_tensor(train_inputs)
    test_features = _to_feature_tensor(test_inputs)
    train_gold = _to_target_tensor(train_targets, task_type)
    test_gold = _to_target_tensor(test_targets, task_type)

    device = next(model.parameters(), torch.empty(0)).device

    model.eval()
    with torch.no_grad():
        train_predictions = model(
            train_features.to(device=device, dtype=torch.float32),
            top_k_rules=top_k_rules,
        ).detach().cpu()
        test_predictions = model(
            test_features.to(device=device, dtype=torch.float32),
            top_k_rules=top_k_rules,
        ).detach().cpu()

    structural_metrics: dict[str, float] = {}
    explainability_metrics: dict[str, float] = {}
    stability_artifacts: dict[str, tuple[str, ...]] = {}
    if isinstance(model, (DeepFuzzyFeatureModel, StackedAnfisModel)):
        structural_metrics = _summarize_fuzzy_model_structure(
            model,
            rule_probability_threshold=rule_probability_threshold,
        )
        explainability_metrics = _summarize_fuzzy_model_explainability(
            model,
            test_features,
            top_k_rules=top_k_rules,
        )
        stability_artifacts = _extract_fuzzy_model_stability_artifacts(
            model,
            rule_probability_threshold=rule_probability_threshold,
        )

    return BenchmarkEntryResult(
        model_name=model_name,
        family=family,
        train_metrics=compute_metrics(
            task_type,
            train_predictions,
            train_gold,
            classification_threshold=classification_threshold,
        ),
        test_metrics=compute_metrics(
            task_type,
            test_predictions,
            test_gold,
            classification_threshold=classification_threshold,
        ),
        classification_threshold=float(classification_threshold) if task_type == "binary_classification" else None,
        structural_metrics=structural_metrics,
        explainability_metrics=explainability_metrics,
        stability_artifacts=stability_artifacts,
    )


def _regression_estimators(random_state: int) -> dict[str, object]:
    estimators: dict[str, object] = {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=80,
            random_state=random_state,
        ),
        "extra_trees_regressor": ExtraTreesRegressor(
            n_estimators=120,
            random_state=random_state,
        ),
        "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.06,
            max_depth=6,
            random_state=random_state,
        ),
        "mlp_regressor": MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=300,
            random_state=random_state,
        ),
    }
    if XGBRegressor is not None:
        estimators["xgboost_regressor"] = XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            objective="reg:squarederror",
            random_state=random_state,
            verbosity=0,
        )
    if CatBoostRegressor is not None:
        estimators["catboost_regressor"] = CatBoostRegressor(
            iterations=160,
            depth=4,
            learning_rate=0.08,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
        )
    return estimators


def _binary_estimators(random_state: int) -> dict[str, object]:
    estimators: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=500,
            random_state=random_state,
        ),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=80,
            random_state=random_state,
        ),
        "extra_trees_classifier": ExtraTreesClassifier(
            n_estimators=120,
            random_state=random_state,
        ),
        "hist_gradient_boosting_classifier": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.06,
            max_depth=6,
            random_state=random_state,
        ),
        "mlp_classifier": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=300,
            random_state=random_state,
        ),
    }
    if XGBClassifier is not None:
        estimators["xgboost_classifier"] = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )
    if CatBoostClassifier is not None:
        estimators["catboost_classifier"] = CatBoostClassifier(
            iterations=160,
            depth=4,
            learning_rate=0.08,
            loss_function="Logloss",
            random_seed=random_state,
            verbose=False,
        )
    return estimators


def run_tabular_benchmark(
    *,
    train_inputs: Tensor,
    train_targets: Tensor,
    test_inputs: Tensor,
    test_targets: Tensor,
    task_type: TaskType,
    fuzzy_models: Mapping[str, tuple[nn.Module, TrainingConfig]] | None = None,
    random_state: int = 0,
    classification_threshold: float = 0.5,
    rule_probability_threshold: float = 0.5,
) -> tuple[BenchmarkEntryResult, ...]:
    train_features = _to_feature_tensor(train_inputs)
    test_features = _to_feature_tensor(test_inputs)
    train_gold = _to_target_tensor(train_targets, task_type)
    test_gold = _to_target_tensor(test_targets, task_type)

    estimators = (
        _regression_estimators(random_state)
        if task_type == "regression"
        else _binary_estimators(random_state)
    )

    results: list[BenchmarkEntryResult] = []
    for model_name, estimator in estimators.items():
        results.append(
            _benchmark_sklearn_model(
                model_name,
                estimator,
                task_type=task_type,
                train_inputs=train_features,
                train_targets=train_gold,
                test_inputs=test_features,
                test_targets=test_gold,
                classification_threshold=classification_threshold,
            )
        )

    if fuzzy_models is not None:
        for model_name, (model, config) in fuzzy_models.items():
            trainer = FuzzyTrainer(model, config)
            training_result = trainer.fit(train_features, train_gold, test_features, test_gold)
            evaluated = evaluate_trained_model(
                model_name,
                model,
                task_type=task_type,
                train_inputs=train_features,
                train_targets=train_gold,
                test_inputs=test_features,
                test_targets=test_gold,
                family="ruanfis",
                classification_threshold=classification_threshold,
                rule_probability_threshold=rule_probability_threshold,
            )
            test_metrics = (
                dict(training_result.validation_metrics)
                if training_result.validation_metrics is not None
                else evaluated.test_metrics
            )
            results.append(
                BenchmarkEntryResult(
                    model_name=model_name,
                    family="ruanfis",
                    train_metrics=dict(training_result.train_metrics),
                    test_metrics=test_metrics,
                    classification_threshold=(
                        float(classification_threshold) if task_type == "binary_classification" else None
                    ),
                    structural_metrics=dict(evaluated.structural_metrics),
                    explainability_metrics=dict(evaluated.explainability_metrics),
                    stability_artifacts=dict(evaluated.stability_artifacts),
                )
            )

    return tuple(results)


def _ordered_metric_names(metric_groups: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    ordered_names: list[str] = []
    for metrics in metric_groups:
        for name in metrics.keys():
            if name not in ordered_names:
                ordered_names.append(name)
    return tuple(ordered_names)


def _aggregate_metric_group(metric_groups: Sequence[Mapping[str, float]]) -> dict[str, MetricSummary]:
    summaries: dict[str, MetricSummary] = {}
    for metric_name in _ordered_metric_names(metric_groups):
        values = torch.tensor(
            [float(metrics[metric_name]) for metrics in metric_groups if metric_name in metrics],
            dtype=torch.float32,
        )
        summaries[metric_name] = MetricSummary(
            mean=float(values.mean().item()),
            std=float(values.std(unbiased=False).item()),
        )
    return summaries


def _aggregate_stability_metrics(
    stability_artifacts: Sequence[Mapping[str, tuple[str, ...]]],
) -> dict[str, float]:
    if not stability_artifacts:
        return {}

    all_keys = set().union(*(artifacts.keys() for artifacts in stability_artifacts))
    metrics: dict[str, float] = {}

    named_sets = {
        "generated_rule_jaccard": "generated:all",
        "active_rule_jaccard": "active:all",
        "hidden_generated_rule_jaccard": "generated:hidden",
        "hidden_active_rule_jaccard": "active:hidden",
        "decision_generated_rule_jaccard": "generated:decision",
        "decision_active_rule_jaccard": "active:decision",
    }
    for metric_name, key in named_sets.items():
        if any(key in artifacts for artifacts in stability_artifacts):
            pairwise_scores: list[float] = []
            for left_index in range(len(stability_artifacts)):
                for right_index in range(left_index + 1, len(stability_artifacts)):
                    pairwise_scores.append(
                        _jaccard_similarity(
                            stability_artifacts[left_index].get(key, ()),
                            stability_artifacts[right_index].get(key, ()),
                        )
                    )
            metrics[metric_name] = _pairwise_mean(pairwise_scores)

    active_layer_keys = sorted(key for key in all_keys if key.startswith("active:") and "/" in key)
    if active_layer_keys:
        layer_scores: list[float] = []
        for key in active_layer_keys:
            for left_index in range(len(stability_artifacts)):
                for right_index in range(left_index + 1, len(stability_artifacts)):
                    layer_scores.append(
                        _jaccard_similarity(
                            stability_artifacts[left_index].get(key, ()),
                            stability_artifacts[right_index].get(key, ()),
                        )
                    )
        metrics["layer_active_rule_jaccard"] = _pairwise_mean(layer_scores)

    generated_layer_keys = sorted(key for key in all_keys if key.startswith("generated:") and "/" in key)
    if generated_layer_keys:
        layer_scores = []
        for key in generated_layer_keys:
            for left_index in range(len(stability_artifacts)):
                for right_index in range(left_index + 1, len(stability_artifacts)):
                    layer_scores.append(
                        _jaccard_similarity(
                            stability_artifacts[left_index].get(key, ()),
                            stability_artifacts[right_index].get(key, ()),
                        )
                    )
        metrics["layer_generated_rule_jaccard"] = _pairwise_mean(layer_scores)

    return metrics


def aggregate_benchmark_results(
    per_seed_results: Sequence[Sequence[BenchmarkEntryResult]],
) -> tuple[AggregatedBenchmarkEntryResult, ...]:
    if not per_seed_results:
        return ()

    first_run = tuple(per_seed_results[0])
    if not first_run:
        return ()

    model_order = tuple(entry.model_name for entry in first_run)
    aggregated_results: list[AggregatedBenchmarkEntryResult] = []

    for model_name in model_order:
        model_entries: list[BenchmarkEntryResult] = []
        expected_family: str | None = None
        for run_index, run_results in enumerate(per_seed_results):
            run_mapping = {entry.model_name: entry for entry in run_results}
            if model_name not in run_mapping:
                raise ValueError(f"Missing model '{model_name}' in benchmark run {run_index}.")
            entry = run_mapping[model_name]
            if expected_family is None:
                expected_family = entry.family
            elif entry.family != expected_family:
                raise ValueError(f"Inconsistent family for model '{model_name}' across runs.")
            model_entries.append(entry)

        aggregated_results.append(
            AggregatedBenchmarkEntryResult(
                model_name=model_name,
                family=expected_family or "unknown",
                runs=len(model_entries),
                train_metrics=_aggregate_metric_group([entry.train_metrics for entry in model_entries]),
                test_metrics=_aggregate_metric_group([entry.test_metrics for entry in model_entries]),
                structural_metrics=_aggregate_metric_group([entry.structural_metrics for entry in model_entries]),
                explainability_metrics=_aggregate_metric_group(
                    [entry.explainability_metrics for entry in model_entries]
                ),
                stability_metrics=_aggregate_stability_metrics([entry.stability_artifacts for entry in model_entries]),
            )
        )

    return tuple(aggregated_results)


def run_multi_seed_benchmark(
    *,
    seeds: Sequence[int],
    seed_runner: Callable[[int], Sequence[BenchmarkEntryResult]],
) -> MultiSeedBenchmarkResult:
    seed_tuple = tuple(int(seed) for seed in seeds)
    if not seed_tuple:
        raise ValueError("At least one seed must be provided.")

    per_seed_results = tuple(tuple(seed_runner(seed)) for seed in seed_tuple)
    aggregated_results = aggregate_benchmark_results(per_seed_results)
    return MultiSeedBenchmarkResult(
        seeds=seed_tuple,
        per_seed_results=per_seed_results,
        aggregated_results=aggregated_results,
    )


def _format_number(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def _format_metric_line(metrics: Mapping[str, float], decimals: int) -> str:
    return ", ".join(f"{key}={value:.{decimals}f}" for key, value in metrics.items())


def _format_summary(summary: MetricSummary, decimals: int) -> str:
    return f"{summary.mean:.{decimals}f} +/- {summary.std:.{decimals}f}"


def format_benchmark_results(results: Sequence[BenchmarkEntryResult], decimals: int = 4) -> str:
    lines = ["TABULAR BENCHMARK"]
    for result in results:
        lines.append(f"{result.model_name} [{result.family}]")
        lines.append("  train: " + _format_metric_line(result.train_metrics, decimals))
        lines.append("  test: " + _format_metric_line(result.test_metrics, decimals))
        if result.classification_threshold is not None:
            lines.append(f"  threshold: {result.classification_threshold:.2f}")
        if result.structural_metrics:
            lines.append("  structure: " + _format_metric_line(result.structural_metrics, decimals))
        if result.explainability_metrics:
            lines.append("  explainability: " + _format_metric_line(result.explainability_metrics, decimals))
    return "\n".join(lines)


def format_aggregated_benchmark_results(
    results: Sequence[AggregatedBenchmarkEntryResult],
    decimals: int = 4,
) -> str:
    lines = ["AGGREGATED TABULAR BENCHMARK"]
    for result in results:
        lines.append(f"{result.model_name} [{result.family}] over {result.runs} runs")
        lines.append(
            "  train: "
            + ", ".join(f"{key}={_format_summary(value, decimals)}" for key, value in result.train_metrics.items())
        )
        lines.append(
            "  test: "
            + ", ".join(f"{key}={_format_summary(value, decimals)}" for key, value in result.test_metrics.items())
        )
        if result.structural_metrics:
            lines.append(
                "  structure: "
                + ", ".join(
                    f"{key}={_format_summary(value, decimals)}" for key, value in result.structural_metrics.items()
                )
            )
        if result.explainability_metrics:
            lines.append(
                "  explainability: "
                + ", ".join(
                    f"{key}={_format_summary(value, decimals)}"
                    for key, value in result.explainability_metrics.items()
                )
            )
        if result.stability_metrics:
            lines.append(
                "  stability: "
                + ", ".join(f"{key}={value:.{decimals}f}" for key, value in result.stability_metrics.items())
            )
    return "\n".join(lines)


def serialize_benchmark_results(results: Sequence[BenchmarkEntryResult]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "model_name": result.model_name,
            "family": result.family,
            "train_metrics": dict(result.train_metrics),
            "test_metrics": dict(result.test_metrics),
            "classification_threshold": (
                float(result.classification_threshold) if result.classification_threshold is not None else None
            ),
            "structural_metrics": dict(result.structural_metrics),
            "explainability_metrics": dict(result.explainability_metrics),
            "stability_artifacts": {key: list(value) for key, value in result.stability_artifacts.items()},
        }
        for result in results
    )


def _serialize_metric_summaries(metrics: Mapping[str, MetricSummary]) -> dict[str, dict[str, float]]:
    return {
        name: {"mean": float(summary.mean), "std": float(summary.std)}
        for name, summary in metrics.items()
    }


def serialize_aggregated_benchmark_results(
    results: Sequence[AggregatedBenchmarkEntryResult],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "model_name": result.model_name,
            "family": result.family,
            "runs": result.runs,
            "train_metrics": _serialize_metric_summaries(result.train_metrics),
            "test_metrics": _serialize_metric_summaries(result.test_metrics),
            "structural_metrics": _serialize_metric_summaries(result.structural_metrics),
            "explainability_metrics": _serialize_metric_summaries(result.explainability_metrics),
            "stability_metrics": dict(result.stability_metrics),
        }
        for result in results
    )


def serialize_multi_seed_benchmark_result(result: MultiSeedBenchmarkResult) -> dict[str, object]:
    return {
        "seeds": list(result.seeds),
        "per_seed_results": [
            {"seed": seed, "results": list(serialize_benchmark_results(seed_results))}
            for seed, seed_results in zip(result.seeds, result.per_seed_results, strict=True)
        ],
        "aggregated_results": list(serialize_aggregated_benchmark_results(result.aggregated_results)),
    }


def save_benchmark_results_json(
    results: Sequence[BenchmarkEntryResult],
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(serialize_benchmark_results(results), indent=2),
        encoding="utf-8",
    )


def save_multi_seed_benchmark_results_json(
    result: MultiSeedBenchmarkResult,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(serialize_multi_seed_benchmark_result(result), indent=2),
        encoding="utf-8",
    )


def format_benchmark_markdown_table(
    results: Sequence[BenchmarkEntryResult],
    *,
    split: str = "test",
    decimals: int = 4,
) -> str:
    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'.")
    if not results:
        return "| model | family |\n| --- | --- |"

    metric_key = f"{split}_metrics"
    metric_order: list[str] = []
    for result in results:
        for name in getattr(result, metric_key).keys():
            if name not in metric_order:
                metric_order.append(name)

    header = ["model", "family", *metric_order]
    rows = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for result in results:
        metrics = getattr(result, metric_key)
        row = [result.model_name, result.family]
        for metric_name in metric_order:
            row.append(_format_number(metrics[metric_name], decimals) if metric_name in metrics else "")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def format_aggregated_benchmark_markdown_table(
    results: Sequence[AggregatedBenchmarkEntryResult],
    *,
    split: str = "test",
    decimals: int = 4,
) -> str:
    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'.")
    if not results:
        return "| model | family |\n| --- | --- |"

    metric_key = f"{split}_metrics"
    metric_order: list[str] = []
    for result in results:
        for name in getattr(result, metric_key).keys():
            if name not in metric_order:
                metric_order.append(name)

    header = ["model", "family", *metric_order]
    rows = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for result in results:
        metrics = getattr(result, metric_key)
        row = [result.model_name, result.family]
        for metric_name in metric_order:
            row.append(_format_summary(metrics[metric_name], decimals) if metric_name in metrics else "")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def format_paper_benchmark_markdown_table(
    results: Sequence[AggregatedBenchmarkEntryResult],
    *,
    quality_metric_order: Sequence[str] | None = None,
    structural_metric_order: Sequence[str] | None = None,
    explainability_metric_order: Sequence[str] | None = None,
    stability_metric_order: Sequence[str] | None = None,
    decimals: int = 4,
) -> str:
    if not results:
        return "| model | family |\n| --- | --- |"

    default_structural = ("total_rules", "active_rules", "hidden_blocks", "hidden_concepts")
    default_explainability = (
        "decision_top1_mass",
        "decision_top3_mass",
        "decision_entropy",
        "hidden_top1_mass",
        "hidden_top3_mass",
        "hidden_entropy",
    )
    default_stability = (
        "active_rule_jaccard",
        "decision_active_rule_jaccard",
        "layer_active_rule_jaccard",
    )

    quality_columns = tuple(quality_metric_order or _ordered_metric_names([result.test_metrics for result in results]))
    structural_columns = tuple(
        name
        for name in (structural_metric_order or default_structural)
        if any(name in result.structural_metrics for result in results)
    )
    explainability_columns = tuple(
        name
        for name in (explainability_metric_order or default_explainability)
        if any(name in result.explainability_metrics for result in results)
    )
    stability_columns = tuple(
        name
        for name in (stability_metric_order or default_stability)
        if any(name in result.stability_metrics for result in results)
    )

    header = [
        "model",
        "family",
        *(f"test:{name}" for name in quality_columns),
        *(f"rules:{name}" for name in structural_columns),
        *(f"expl:{name}" for name in explainability_columns),
        *(f"stab:{name}" for name in stability_columns),
    ]
    rows = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]

    for result in results:
        row = [result.model_name, result.family]
        for metric_name in quality_columns:
            row.append(_format_summary(result.test_metrics[metric_name], decimals) if metric_name in result.test_metrics else "")
        for metric_name in structural_columns:
            row.append(
                _format_summary(result.structural_metrics[metric_name], decimals)
                if metric_name in result.structural_metrics
                else ""
            )
        for metric_name in explainability_columns:
            row.append(
                _format_summary(result.explainability_metrics[metric_name], decimals)
                if metric_name in result.explainability_metrics
                else ""
            )
        for metric_name in stability_columns:
            row.append(
                _format_number(result.stability_metrics[metric_name], decimals)
                if metric_name in result.stability_metrics
                else ""
            )
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)


def save_benchmark_markdown_table(
    results: Sequence[BenchmarkEntryResult],
    path: str | Path,
    *,
    split: str = "test",
    decimals: int = 4,
) -> None:
    Path(path).write_text(
        format_benchmark_markdown_table(results, split=split, decimals=decimals),
        encoding="utf-8",
    )


def save_aggregated_benchmark_markdown_table(
    results: Sequence[AggregatedBenchmarkEntryResult],
    path: str | Path,
    *,
    split: str = "test",
    decimals: int = 4,
) -> None:
    Path(path).write_text(
        format_aggregated_benchmark_markdown_table(results, split=split, decimals=decimals),
        encoding="utf-8",
    )


def save_paper_benchmark_markdown_table(
    results: Sequence[AggregatedBenchmarkEntryResult],
    path: str | Path,
    *,
    quality_metric_order: Sequence[str] | None = None,
    structural_metric_order: Sequence[str] | None = None,
    explainability_metric_order: Sequence[str] | None = None,
    stability_metric_order: Sequence[str] | None = None,
    decimals: int = 4,
) -> None:
    Path(path).write_text(
        format_paper_benchmark_markdown_table(
            results,
            quality_metric_order=quality_metric_order,
            structural_metric_order=structural_metric_order,
            explainability_metric_order=explainability_metric_order,
            stability_metric_order=stability_metric_order,
            decimals=decimals,
        ),
        encoding="utf-8",
    )
