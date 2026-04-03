from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from torch import Tensor, nn

from .metrics import TaskType, compute_metrics
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
    )


def _regression_estimators(random_state: int) -> dict[str, object]:
    estimators: dict[str, object] = {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=80,
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
            test_metrics = (
                dict(training_result.validation_metrics)
                if training_result.validation_metrics is not None
                else trainer.evaluate(test_features, test_gold).metrics
            )
            results.append(
                BenchmarkEntryResult(
                    model_name=model_name,
                    family="ruanfis",
                    train_metrics=dict(training_result.train_metrics),
                    test_metrics=test_metrics,
                )
            )

    return tuple(results)


def format_benchmark_results(results: tuple[BenchmarkEntryResult, ...], decimals: int = 4) -> str:
    lines = ["TABULAR BENCHMARK"]
    for result in results:
        lines.append(f"{result.model_name} [{result.family}]")
        lines.append(
            "  train: " + ", ".join(f"{key}={value:.{decimals}f}" for key, value in result.train_metrics.items())
        )
        lines.append(
            "  test: " + ", ".join(f"{key}={value:.{decimals}f}" for key, value in result.test_metrics.items())
        )
    return "\n".join(lines)
