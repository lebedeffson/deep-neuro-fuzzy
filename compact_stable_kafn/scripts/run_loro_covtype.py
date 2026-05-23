from __future__ import annotations

import argparse
import csv
import heapq
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _to_binary(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y).reshape(-1) >= 0.5).astype(np.int64)


def _load_artifacts(artifact_dir: Path, dataset: str) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for p in sorted(artifact_dir.glob(f"v18_h_artifacts_{dataset}_seed*.npz")):
        z = np.load(p, allow_pickle=True)
        seed = int(p.stem.rsplit("seed", 1)[-1])
        out[seed] = {
            "seed": seed,
            "h_train": z["h_train"],
            "h_test": z["h_test"],
            "y_train": z["y_train"].reshape(-1),
            "y_test": z["y_test"].reshape(-1),
            "full_train_logits": z["full_train_logits"].reshape(-1),
            "full_test_prob": z["full_test_prob"].reshape(-1),
            "importance": z["importance"].astype(np.float64).reshape(-1),
            "rule_key": z["rule_key"].astype(str),
            "threshold": float(np.asarray(z["classification_threshold"]).reshape(-1)[0]),
        }
    return out


def _reconstruct_linear_head(h_train: np.ndarray, full_train_logits: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(h_train, dtype=np.float64)
    y = np.asarray(full_train_logits, dtype=np.float64).reshape(-1)
    xa = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    beta, *_ = np.linalg.lstsq(xa, y, rcond=None)
    bias = float(beta[0])
    w = np.asarray(beta[1:], dtype=np.float64)
    return w, bias


def _compute_loro_scores(
    h_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    bias: float,
    threshold_prob: float,
    old_importance: np.ndarray,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    y = _to_binary(y_val)
    z_full = bias + h_val @ weights
    p_full = _sigmoid(z_full)
    pred_full = (p_full >= float(threshold_prob)).astype(np.int64)
    f1_full = float(f1_score(y, pred_full, zero_division=0))
    bce_full = float(log_loss(y, p_full, labels=[0, 1]))

    n_rules = h_val.shape[1]
    rows: list[dict[str, object]] = []
    delta_f1 = np.zeros(n_rules, dtype=np.float64)
    delta_bce = np.zeros(n_rules, dtype=np.float64)
    for r in range(n_rules):
        z_without = z_full - h_val[:, r] * weights[r]
        p_without = _sigmoid(z_without)
        pred_without = (p_without >= float(threshold_prob)).astype(np.int64)
        f1_without = float(f1_score(y, pred_without, zero_division=0))
        bce_without = float(log_loss(y, p_without, labels=[0, 1]))
        df1 = f1_full - f1_without
        dbce = bce_without - bce_full
        delta_f1[r] = df1
        delta_bce[r] = dbce
        rows.append(
            {
                "rule_id": int(r),
                "delta_f1": float(df1),
                "delta_bce": float(dbce),
                "delta_prob": float(np.mean(np.abs(p_full - p_without))),
                "n_changed_predictions": int(np.sum(pred_full != pred_without)),
                "old_importance": float(old_importance[r]),
                "theta_abs": float(abs(weights[r])),
                "mean_abs_activation": float(np.mean(np.abs(h_val[:, r]))),
                "baseline_f1": f1_full,
                "baseline_bce": bce_full,
            }
        )
    return rows, delta_f1, delta_bce


def _rank_indices(delta_f1: np.ndarray, delta_bce: np.ndarray, old_importance: np.ndarray, mode: str, budget: int) -> np.ndarray:
    n = delta_f1.shape[0]
    idx = list(range(n))
    if mode == "f1":
        idx.sort(key=lambda i: (delta_f1[i], delta_bce[i], old_importance[i]), reverse=True)
    elif mode == "bce":
        idx.sort(key=lambda i: (delta_bce[i], delta_f1[i], old_importance[i]), reverse=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.asarray(idx[: min(int(budget), n)], dtype=np.int64)


def _bce_from_logits(y_bin: np.ndarray, logits: np.ndarray) -> float:
    y = np.asarray(y_bin, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    # Stable logistic loss: log(1 + exp(z)) - y*z
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _score_from_logits(y_bin: np.ndarray, logits: np.ndarray, objective: str, threshold_prob: float) -> float:
    if objective == "bce":
        return -_bce_from_logits(y_bin, logits)
    if objective == "f1":
        p = _sigmoid(logits)
        pred = (p >= float(threshold_prob)).astype(np.int64)
        return float(f1_score(y_bin, pred, zero_division=0))
    raise ValueError(f"Unknown objective: {objective}")


def _lazy_greedy_select(
    h_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    bias: float,
    budget: int,
    threshold_prob: float,
    objective: str,
) -> np.ndarray:
    n_rules = int(h_val.shape[1])
    k = min(int(budget), n_rules)
    if k <= 0:
        return np.asarray([], dtype=np.int64)

    y_bin = _to_binary(y_val)
    cur_logits = np.full(h_val.shape[0], float(bias), dtype=np.float64)
    cur_score = _score_from_logits(y_bin, cur_logits, objective=objective, threshold_prob=threshold_prob)

    selected: list[int] = []
    selected_flag = np.zeros(n_rules, dtype=bool)
    heap: list[tuple[float, int, int]] = []

    # Initial upper-bounds at step 0.
    for r in range(n_rules):
        new_logits = cur_logits + h_val[:, r] * weights[r]
        gain = _score_from_logits(y_bin, new_logits, objective=objective, threshold_prob=threshold_prob) - cur_score
        heapq.heappush(heap, (-float(gain), int(r), 0))

    step = 0
    while len(selected) < k and heap:
        neg_gain, r, stamp = heapq.heappop(heap)
        if selected_flag[r]:
            continue
        if stamp == step:
            # Accept best candidate re-evaluated at current step.
            selected_flag[r] = True
            selected.append(int(r))
            cur_logits = cur_logits + h_val[:, r] * weights[r]
            cur_score = cur_score + (-neg_gain)
            step += 1
            continue

        # Re-evaluate candidate on current subset and push back.
        new_logits = cur_logits + h_val[:, r] * weights[r]
        true_gain = _score_from_logits(y_bin, new_logits, objective=objective, threshold_prob=threshold_prob) - cur_score
        heapq.heappush(heap, (-float(true_gain), int(r), step))

    return np.asarray(selected, dtype=np.int64)


def _eval_subset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    selected_idx: np.ndarray,
    full_test_prob: np.ndarray,
    threshold_prob: float,
) -> dict[str, float]:
    cols = np.asarray(selected_idx, dtype=np.int64).reshape(-1)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(x_train[:, cols], y_train)
    prob = clf.predict_proba(x_test[:, cols])[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    pred_full = (np.asarray(full_test_prob) >= float(threshold_prob)).astype(np.int64)
    return {
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "fidelity_l1_to_full_prob": float(np.mean(np.abs(prob - np.asarray(full_test_prob, dtype=np.float64)))),
        "agreement_to_full": float(np.mean(pred == pred_full)),
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def run(
    *,
    artifact_dir: Path,
    dataset: str,
    budgets: tuple[int, ...],
    out_detail: Path,
    out_summary: Path,
    out_jaccard: Path,
) -> None:
    artifacts = _load_artifacts(artifact_dir, dataset)
    seeds = sorted(artifacts)
    if len(seeds) < 2:
        raise ValueError(f"Need >=2 seeds for jaccard. Found {len(seeds)} in {artifact_dir}")

    detail_rows: list[dict[str, object]] = []
    summary_seed_rows: list[dict[str, object]] = []
    selected_sets: dict[tuple[int, str], list[tuple[int, set[str]]]] = {}

    for seed in seeds:
        art = artifacts[seed]
        x_train_full = np.asarray(art["h_train"], dtype=np.float64)
        y_train_full = _to_binary(np.asarray(art["y_train"]))
        x_test = np.asarray(art["h_test"], dtype=np.float64)
        y_test = _to_binary(np.asarray(art["y_test"]))
        old_importance = np.asarray(art["importance"], dtype=np.float64)
        rule_key = [str(x) for x in art["rule_key"]]
        threshold = float(art["threshold"])
        w_full, b_full = _reconstruct_linear_head(x_train_full, np.asarray(art["full_train_logits"]))

        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.25,
            random_state=int(seed),
            stratify=y_train_full,
        )
        _ = x_tr, y_tr  # explicit: selection uses validation split only.

        rows, df1, dbce = _compute_loro_scores(
            h_val=x_val,
            y_val=y_val,
            weights=w_full,
            bias=b_full,
            threshold_prob=threshold,
            old_importance=old_importance,
        )
        for r in rows:
            detail_rows.append(
                {
                    "dataset": dataset,
                    "seed": int(seed),
                    "rule_id": int(r["rule_id"]),
                    "rule_key": rule_key[int(r["rule_id"])],
                    "delta_f1": r["delta_f1"],
                    "delta_bce": r["delta_bce"],
                    "delta_prob": r["delta_prob"],
                    "n_changed_predictions": r["n_changed_predictions"],
                    "old_importance": r["old_importance"],
                    "theta_abs": r["theta_abs"],
                    "mean_abs_activation": r["mean_abs_activation"],
                    "baseline_f1": r["baseline_f1"],
                    "baseline_bce": r["baseline_bce"],
                }
            )

        for budget in budgets:
            bp_idx = np.argsort(-old_importance, kind="stable")[: min(int(budget), old_importance.shape[0])]
            lf_idx = _rank_indices(df1, dbce, old_importance, mode="f1", budget=int(budget))
            lb_idx = _rank_indices(df1, dbce, old_importance, mode="bce", budget=int(budget))
            sf_idx = _lazy_greedy_select(
                h_val=x_val,
                y_val=y_val,
                weights=w_full,
                bias=b_full,
                budget=int(budget),
                threshold_prob=threshold,
                objective="f1",
            )
            sb_idx = _lazy_greedy_select(
                h_val=x_val,
                y_val=y_val,
                weights=w_full,
                bias=b_full,
                budget=int(budget),
                threshold_prob=threshold,
                objective="bce",
            )
            methods = (
                ("budget_prune", bp_idx),
                ("loro_f1", lf_idx),
                ("loro_bce", lb_idx),
                ("sloro_f1", sf_idx),
                ("sloro_bce", sb_idx),
            )
            for method, idx in methods:
                metrics = _eval_subset(
                    x_train=x_train_full,
                    y_train=y_train_full,
                    x_test=x_test,
                    y_test=y_test,
                    selected_idx=idx,
                    full_test_prob=np.asarray(art["full_test_prob"], dtype=np.float64),
                    threshold_prob=threshold,
                )
                summary_seed_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "budget": int(budget),
                        "method": method,
                        "selected_rules": int(idx.size),
                        **metrics,
                    }
                )
                key = (int(budget), method)
                selected_sets.setdefault(key, [])
                selected_sets[key].append((int(seed), {rule_key[int(i)] for i in idx}))

    summary_rows: list[dict[str, object]] = []
    for budget in budgets:
        for method in ("budget_prune", "loro_f1", "loro_bce", "sloro_f1", "sloro_bce"):
            rows = [r for r in summary_seed_rows if int(r["budget"]) == int(budget) and str(r["method"]) == method]
            if not rows:
                continue
            summary_rows.append(
                {
                    "dataset": dataset,
                    "budget": int(budget),
                    "method": method,
                    "n_seeds": len(rows),
                    "f1_mean": float(mean(float(r["f1"]) for r in rows)),
                    "f1_std": float(pstdev(float(r["f1"]) for r in rows)) if len(rows) > 1 else 0.0,
                    "roc_auc_mean": float(mean(float(r["roc_auc"]) for r in rows)),
                    "roc_auc_std": float(pstdev(float(r["roc_auc"]) for r in rows)) if len(rows) > 1 else 0.0,
                    "pr_auc_mean": float(mean(float(r["pr_auc"]) for r in rows)),
                    "pr_auc_std": float(pstdev(float(r["pr_auc"]) for r in rows)) if len(rows) > 1 else 0.0,
                    "fidelity_mean": float(mean(float(r["fidelity_l1_to_full_prob"]) for r in rows)),
                    "fidelity_std": float(pstdev(float(r["fidelity_l1_to_full_prob"]) for r in rows)) if len(rows) > 1 else 0.0,
                    "agreement_mean": float(mean(float(r["agreement_to_full"]) for r in rows)),
                    "agreement_std": float(pstdev(float(r["agreement_to_full"]) for r in rows)) if len(rows) > 1 else 0.0,
                }
            )

    jaccard_rows: list[dict[str, object]] = []
    for budget in budgets:
        for method in ("budget_prune", "loro_f1", "loro_bce", "sloro_f1", "sloro_bce"):
            sets = selected_sets.get((int(budget), method), [])
            for (sa, a), (sb, b) in combinations(sets, 2):
                jaccard_rows.append(
                    {
                        "dataset": dataset,
                        "budget": int(budget),
                        "method": method,
                        "seed_a": int(sa),
                        "seed_b": int(sb),
                        "jaccard": float(_jaccard(a, b)),
                    }
                )

    _write_csv(
        out_detail,
        [
            "dataset",
            "seed",
            "rule_id",
            "rule_key",
            "delta_f1",
            "delta_bce",
            "delta_prob",
            "n_changed_predictions",
            "old_importance",
            "theta_abs",
            "mean_abs_activation",
            "baseline_f1",
            "baseline_bce",
        ],
        detail_rows,
    )
    _write_csv(
        out_summary,
        [
            "dataset",
            "budget",
            "method",
            "n_seeds",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
            "pr_auc_mean",
            "pr_auc_std",
            "fidelity_mean",
            "fidelity_std",
            "agreement_mean",
            "agreement_std",
        ],
        summary_rows,
    )
    _write_csv(
        out_jaccard,
        ["dataset", "budget", "method", "seed_a", "seed_b", "jaccard"],
        jaccard_rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default="covtype_binary_20000")
    ap.add_argument("--budgets", type=str, default="25,50,100,200,400")
    ap.add_argument("--out-detail", type=Path, default=Path("docs/tables_loro_rule_importance_detail.csv"))
    ap.add_argument("--out-summary", type=Path, default=Path("docs/tables_loro_prune_summary.csv"))
    ap.add_argument("--out-jaccard", type=Path, default=Path("docs/tables_loro_prune_jaccard.csv"))
    args = ap.parse_args()

    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    run(
        artifact_dir=args.artifact_dir,
        dataset=args.dataset,
        budgets=budgets,
        out_detail=args.out_detail,
        out_summary=args.out_summary,
        out_jaccard=args.out_jaccard,
    )
    print(f"Wrote {args.out_detail}")
    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.out_jaccard}")


if __name__ == "__main__":
    main()
