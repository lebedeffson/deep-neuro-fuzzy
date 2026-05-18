from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression


ArrayLike2D = np.ndarray


@dataclass(frozen=True)
class StableBudgetPruneConfig:
    budget: int
    stability_top_k: int | None = None
    std_penalty: float = 0.0
    redundancy_penalty: float = 0.0
    random_state: int = 42


@dataclass(frozen=True)
class StableBudgetScores:
    score: np.ndarray
    mean_importance: np.ndarray
    std_importance: np.ndarray
    topk_frequency: np.ndarray


def rule_importance(activations: ArrayLike2D, weights: np.ndarray) -> np.ndarray:
    h = np.asarray(activations, dtype=np.float64)
    theta = np.asarray(weights, dtype=np.float64).reshape(-1)
    if h.ndim != 2:
        raise ValueError("activations must have shape [n_samples, n_rules].")
    if h.shape[1] != theta.shape[0]:
        raise ValueError("weights length must match the number of rules.")
    return np.abs(theta) * np.mean(np.abs(h), axis=0)


def stable_budget_scores(
    importance_runs: Sequence[np.ndarray],
    *,
    budget: int,
    stability_top_k: int | None = None,
    std_penalty: float = 0.0,
) -> StableBudgetScores:
    if budget <= 0:
        raise ValueError("budget must be positive.")
    values = np.asarray([np.asarray(row, dtype=np.float64).reshape(-1) for row in importance_runs])
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("importance_runs must be a non-empty sequence of 1D arrays.")
    if any(row.shape[0] != values.shape[1] for row in values):
        raise ValueError("all importance arrays must have the same length.")

    n_rules = values.shape[1]
    top_k = min(n_rules, int(stability_top_k or budget))
    hits = np.zeros_like(values, dtype=np.float64)
    for run_index, row in enumerate(values):
        top_indices = np.argsort(-row, kind="stable")[:top_k]
        hits[run_index, top_indices] = 1.0

    mean_importance = values.mean(axis=0)
    std_importance = values.std(axis=0)
    topk_frequency = hits.mean(axis=0)
    score = mean_importance * topk_frequency - float(std_penalty) * std_importance
    return StableBudgetScores(
        score=score,
        mean_importance=mean_importance,
        std_importance=std_importance,
        topk_frequency=topk_frequency,
    )


def top_budget_indices(score: np.ndarray, budget: int) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    if budget <= 0:
        raise ValueError("budget must be positive.")
    k = min(int(budget), values.shape[0])
    return np.argsort(-values, kind="stable")[:k]


def redundancy(activations: ArrayLike2D, indices: Sequence[int]) -> float:
    selected = np.asarray(list(indices), dtype=np.int64)
    if selected.size < 2:
        return 0.0
    h = np.asarray(activations, dtype=np.float64)[:, selected]
    corr = np.corrcoef(h, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.abs(corr[np.triu_indices(corr.shape[0], k=1)])
    return float(upper.mean()) if upper.size else 0.0


def redundancy_aware_budget_indices(
    score: np.ndarray,
    activations: ArrayLike2D,
    budget: int,
    *,
    redundancy_penalty: float,
) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64).reshape(-1)
    h = np.asarray(activations, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != values.shape[0]:
        raise ValueError("activations must have shape [n_samples, len(score)].")
    if budget <= 0:
        raise ValueError("budget must be positive.")

    k = min(int(budget), values.shape[0])
    corr = np.corrcoef(h, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    abs_corr = np.abs(corr)

    selected: list[int] = []
    remaining = set(range(values.shape[0]))
    for _ in range(k):
        if not selected:
            choice = int(np.argmax(values))
        else:
            best_choice = None
            best_value = -np.inf
            for candidate in remaining:
                penalty = float(np.max(abs_corr[candidate, selected]))
                adjusted = float(values[candidate]) - float(redundancy_penalty) * penalty
                if adjusted > best_value:
                    best_value = adjusted
                    best_choice = int(candidate)
            choice = int(best_choice)
        selected.append(choice)
        remaining.remove(choice)
    return np.asarray(selected, dtype=np.int64)


def stable_budget_prune(
    importance_runs: Sequence[np.ndarray],
    *,
    config: StableBudgetPruneConfig,
    reference_activations: ArrayLike2D | None = None,
) -> tuple[np.ndarray, StableBudgetScores]:
    scores = stable_budget_scores(
        importance_runs,
        budget=config.budget,
        stability_top_k=config.stability_top_k,
        std_penalty=config.std_penalty,
    )
    if config.redundancy_penalty > 0.0:
        if reference_activations is None:
            raise ValueError("reference_activations is required for redundancy-aware selection.")
        indices = redundancy_aware_budget_indices(
            scores.score,
            reference_activations,
            config.budget,
            redundancy_penalty=config.redundancy_penalty,
        )
    else:
        indices = top_budget_indices(scores.score, config.budget)
    return indices, scores


def weight_only_indices(weights: np.ndarray, budget: int) -> np.ndarray:
    return top_budget_indices(np.abs(np.asarray(weights, dtype=np.float64).reshape(-1)), budget)


def activation_only_indices(activations: ArrayLike2D, budget: int) -> np.ndarray:
    h = np.asarray(activations, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("activations must have shape [n_samples, n_rules].")
    return top_budget_indices(np.mean(np.abs(h), axis=0), budget)


def random_budget_indices(n_rules: int, budget: int, *, random_state: int = 42) -> np.ndarray:
    if n_rules <= 0:
        raise ValueError("n_rules must be positive.")
    k = min(int(budget), int(n_rules))
    rng = np.random.default_rng(int(random_state))
    return np.sort(rng.choice(int(n_rules), size=k, replace=False))


def sparse_logistic_indices(
    activations: ArrayLike2D,
    targets: np.ndarray,
    budget: int,
    *,
    penalty: str = "l1",
    l1_ratio: float | None = None,
    random_state: int = 42,
) -> np.ndarray:
    h = np.asarray(activations, dtype=np.float64)
    y = (np.asarray(targets).reshape(-1) >= 0.5).astype(np.int64)
    if h.ndim != 2:
        raise ValueError("activations must have shape [n_samples, n_rules].")
    if h.shape[0] != y.shape[0]:
        raise ValueError("targets length must match activations rows.")

    model = LogisticRegression(
        penalty=penalty,
        solver="saga",
        l1_ratio=l1_ratio,
        max_iter=2000,
        random_state=int(random_state),
    )
    model.fit(h, y)
    weights = np.abs(model.coef_).reshape(-1)
    return top_budget_indices(weights, budget)


def fidelity_gap(full_probability: np.ndarray, compact_probability: np.ndarray) -> float:
    full = np.asarray(full_probability, dtype=np.float64).reshape(-1)
    compact = np.asarray(compact_probability, dtype=np.float64).reshape(-1)
    if full.shape != compact.shape:
        raise ValueError("probability arrays must have the same shape.")
    return float(np.mean(np.abs(full - compact)))


def prediction_agreement(full_probability: np.ndarray, compact_probability: np.ndarray, *, threshold: float = 0.5) -> float:
    full = np.asarray(full_probability, dtype=np.float64).reshape(-1) >= float(threshold)
    compact = np.asarray(compact_probability, dtype=np.float64).reshape(-1) >= float(threshold)
    if full.shape != compact.shape:
        raise ValueError("probability arrays must have the same shape.")
    return float(np.mean(full == compact))
