from __future__ import annotations

import numpy as np

from compact_stable_kafn.scripts.evaluate_stable_selection_h_artifacts import run


def _write_artifact(path, *, seed: int, importance: np.ndarray) -> None:
    rng = np.random.default_rng(seed)
    n_train, n_test, n_rules = 30, 12, importance.size
    h_train = rng.normal(size=(n_train, n_rules)).astype(np.float32)
    h_test = rng.normal(size=(n_test, n_rules)).astype(np.float32)
    y_train = (h_train[:, 0] + 0.5 * h_train[:, 1] > 0.0).astype(np.float32)
    y_test = (h_test[:, 0] + 0.5 * h_test[:, 1] > 0.0).astype(np.float32)
    full_test_logits = h_test[:, 0] + 0.5 * h_test[:, 1]
    np.savez_compressed(
        path / f"v18_h_artifacts_covtype_binary_20000_seed{seed}.npz",
        h_train=h_train,
        h_test=h_test,
        y_train=y_train,
        y_test=y_test,
        full_test_logits=full_test_logits.astype(np.float32),
        full_test_prob=(1.0 / (1.0 + np.exp(-full_test_logits))).astype(np.float32),
        importance=importance.astype(np.float32),
        rule_key=np.asarray([f"layer_0::r{i}" for i in range(n_rules)]),
        classification_threshold=np.asarray([0.5], dtype=np.float32),
    )


def test_h_artifact_stable_selection_outputs_tables(tmp_path):
    _write_artifact(tmp_path, seed=19, importance=np.asarray([5, 4, 3, 2, 1], dtype=np.float32))
    _write_artifact(tmp_path, seed=23, importance=np.asarray([5, 4, 2, 3, 1], dtype=np.float32))
    _write_artifact(tmp_path, seed=29, importance=np.asarray([5, 4, 2, 1, 3], dtype=np.float32))

    detail = tmp_path / "detail.csv"
    summary = tmp_path / "summary.csv"
    run(
        artifact_dir=tmp_path,
        dataset="covtype_binary_20000",
        budgets=(2,),
        out_detail=detail,
        out_summary=summary,
        stability_top_k_multiplier=2.0,
        std_penalty=0.0,
    )

    assert detail.exists()
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "budget_prune_h_lr" in text
    assert "stable_budget_prune_h_lr" in text
