from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class LofoF1Scores:
    baseline_f1: float
    delta_f1: np.ndarray
    f1_without: np.ndarray
    mean_abs_activation: np.ndarray
    top_frequency: np.ndarray | None = None
    mean_rank: np.ndarray | None = None


def logits_from_rules(h: np.ndarray, weights: np.ndarray, bias: float = 0.0) -> np.ndarray:
    h = np.asarray(h, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if h.ndim != 2:
        raise ValueError('h must be a 2D matrix')
    if h.shape[1] != w.shape[0]:
        raise ValueError('weights length must match h.shape[1]')
    return float(bias) + h @ w


def lofo_f1_scores(
    h_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    bias: float = 0.0,
    threshold: float = 0.0,
) -> LofoF1Scores:
    h = np.asarray(h_val, dtype=np.float64)
    y = (np.asarray(y_val).reshape(-1) >= 0.5).astype(np.int64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if h.ndim != 2:
        raise ValueError('h_val must be 2D')
    if h.shape[0] != y.shape[0]:
        raise ValueError('y_val length must match h_val rows')
    if h.shape[1] != w.shape[0]:
        raise ValueError('weights length must match h_val columns')

    z_full = logits_from_rules(h, w, bias=bias)
    pred_full = (z_full >= float(threshold)).astype(np.int64)
    baseline = float(f1_score(y, pred_full, zero_division=0))
    f1_without = np.zeros(h.shape[1], dtype=np.float64)
    for r in range(h.shape[1]):
        z_without = z_full - h[:, r] * w[r]
        pred_without = (z_without >= float(threshold)).astype(np.int64)
        f1_without[r] = float(f1_score(y, pred_without, zero_division=0))
    return LofoF1Scores(
        baseline_f1=baseline,
        delta_f1=baseline - f1_without,
        f1_without=f1_without,
        mean_abs_activation=np.mean(np.abs(h), axis=0),
    )


def top_lofo_f1_indices(scores: LofoF1Scores, budget: int) -> np.ndarray:
    k = min(int(budget), scores.delta_f1.shape[0])
    # Positive F1 drop first; activation is deterministic tie-breaker.
    order = np.lexsort((-scores.mean_abs_activation, -scores.delta_f1))
    return order[:k].astype(np.int64)


def bootstrap_lofo_f1_scores(
    h_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    bias: float = 0.0,
    threshold: float = 0.0,
    *,
    budget: int,
    n_bootstrap: int = 50,
    seed: int = 42,
) -> LofoF1Scores:
    h = np.asarray(h_val, dtype=np.float64)
    y = (np.asarray(y_val).reshape(-1) >= 0.5).astype(np.int64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    n_rules = h.shape[1]
    k = min(int(budget), n_rules)
    rng = np.random.default_rng(int(seed))
    deltas = np.zeros((int(n_bootstrap), n_rules), dtype=np.float64)
    ranks = np.zeros((int(n_bootstrap), n_rules), dtype=np.float64)
    top_hits = np.zeros(n_rules, dtype=np.float64)

    for b in range(max(1, int(n_bootstrap))):
        idx = rng.integers(0, h.shape[0], size=h.shape[0], endpoint=False)
        scores = lofo_f1_scores(h[idx], y[idx], w, bias=bias, threshold=threshold)
        order = top_lofo_f1_indices(scores, n_rules)
        deltas[b] = scores.delta_f1
        for rank, rule_idx in enumerate(order, start=1):
            ranks[b, int(rule_idx)] = rank
            if rank <= k:
                top_hits[int(rule_idx)] += 1.0

    mean_delta = np.mean(deltas, axis=0)
    out = LofoF1Scores(
        baseline_f1=float(lofo_f1_scores(h, y, w, bias=bias, threshold=threshold).baseline_f1),
        delta_f1=mean_delta,
        f1_without=np.zeros(n_rules, dtype=np.float64),
        mean_abs_activation=np.mean(np.abs(h), axis=0),
        top_frequency=top_hits / float(max(1, int(n_bootstrap))),
        mean_rank=np.mean(ranks, axis=0),
    )
    return out


def top_bootstrap_lofo_f1_indices(scores: LofoF1Scores, budget: int) -> np.ndarray:
    if scores.top_frequency is None or scores.mean_rank is None:
        return top_lofo_f1_indices(scores, budget)
    k = min(int(budget), scores.delta_f1.shape[0])
    order = np.lexsort((scores.mean_rank, -scores.delta_f1, -scores.top_frequency))
    return order[:k].astype(np.int64)


def jaccard_similarity(a: Iterable[int] | set[str], b: Iterable[int] | set[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))
