import numpy as np

from ruanfis.stable_budget_prune import (
    StableBudgetPruneConfig,
    activation_only_indices,
    fidelity_gap,
    prediction_agreement,
    random_budget_indices,
    redundancy,
    rule_importance,
    stable_budget_prune,
    stable_budget_scores,
    weight_only_indices,
)


def test_rule_importance_uses_weight_and_mean_activation():
    h = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
    weights = np.array([2.0, -1.0, 0.5])

    values = rule_importance(h, weights)

    np.testing.assert_allclose(values, np.array([4.0, 1.0, 1.0]))


def test_stable_budget_scores_reward_repeated_topk_membership():
    runs = [
        np.array([10.0, 9.0, 1.0]),
        np.array([8.0, 7.0, 6.0]),
        np.array([9.0, 8.0, 1.0]),
    ]

    scores = stable_budget_scores(runs, budget=2)

    np.testing.assert_allclose(scores.topk_frequency, np.array([1.0, 1.0, 0.0]))
    assert list(np.argsort(-scores.score)[:2]) == [0, 1]


def test_redundancy_aware_selection_avoids_duplicate_rules():
    h = np.array(
        [
            [1.0, 1.0, 0.0],
            [0.9, 0.9, 1.0],
            [0.1, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    runs = [np.array([10.0, 9.9, 8.0]), np.array([10.0, 9.9, 8.0])]

    indices, _ = stable_budget_prune(
        runs,
        config=StableBudgetPruneConfig(budget=2, stability_top_k=3, redundancy_penalty=5.0),
        reference_activations=h,
    )

    assert 0 in indices
    assert 2 in indices
    assert redundancy(h, indices) < 0.6


def test_simple_baseline_selectors_are_deterministic():
    h = np.array([[0.0, 2.0, 3.0], [0.0, 1.0, 4.0]])

    assert list(weight_only_indices(np.array([1.0, 3.0, 2.0]), 2)) == [1, 2]
    assert list(activation_only_indices(h, 2)) == [2, 1]
    assert list(random_budget_indices(5, 3, random_state=1)) == [1, 2, 3]


def test_fidelity_metrics():
    full = np.array([0.1, 0.8, 0.7])
    compact = np.array([0.2, 0.6, 0.9])

    assert round(fidelity_gap(full, compact), 6) == round(0.5 / 3.0, 6)
    assert prediction_agreement(full, compact) == 1.0
