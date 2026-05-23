from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
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
    return np.asarray(beta[1:], dtype=np.float64), float(beta[0])


def _compute_loro_bce_importance(
    h_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray,
    bias: float,
) -> np.ndarray:
    y = _to_binary(y_val)
    z_full = bias + h_val @ weights
    p_full = _sigmoid(z_full)
    bce_full = float(log_loss(y, p_full, labels=[0, 1]))

    n_rules = h_val.shape[1]
    imp = np.zeros(n_rules, dtype=np.float64)
    for r in range(n_rules):
        z_without = z_full - h_val[:, r] * weights[r]
        p_without = _sigmoid(z_without)
        bce_without = float(log_loss(y, p_without, labels=[0, 1]))
        imp[r] = bce_without - bce_full  # correct sign: positive means useful rule
    return imp


def _corr_abs_matrix(x: np.ndarray) -> np.ndarray:
    c = np.corrcoef(np.asarray(x, dtype=np.float64), rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(np.abs(c), 0.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def _cluster_representatives(
    abs_corr: np.ndarray,
    importance: np.ndarray,
    corr_threshold: float,
    budget: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rules = int(importance.shape[0])
    if n_rules == 0:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=bool),
            np.asarray([], dtype=np.int64),
        )

    dist = 1.0 - abs_corr
    dist = np.clip(0.5 * (dist + dist.T), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    cond = squareform(dist, checks=False)
    z = linkage(cond, method="complete")
    labels = fcluster(z, t=float(1.0 - corr_threshold), criterion="distance").astype(np.int64)

    cluster_ids = np.unique(labels)
    reps: list[int] = []
    cluster_scores: list[float] = []
    for cid in cluster_ids:
        members = np.flatnonzero(labels == cid)
        best_local = int(members[np.argmax(importance[members])])
        reps.append(best_local)
        cluster_scores.append(float(importance[best_local]))

    reps_arr = np.asarray(reps, dtype=np.int64)
    scores_arr = np.asarray(cluster_scores, dtype=np.float64)
    order = np.argsort(-scores_arr, kind="stable")
    k = min(int(budget), reps_arr.size)
    selected_rep_idx = order[:k]
    selected_rules = reps_arr[selected_rep_idx]

    selected_clusters_mask = np.zeros(cluster_ids.size, dtype=bool)
    selected_clusters_mask[selected_rep_idx] = True
    return selected_rules, cluster_ids, reps_arr, selected_clusters_mask, labels


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
        "fidelity": float(np.mean(np.abs(prob - np.asarray(full_test_prob, dtype=np.float64)))),
        "agreement": float(np.mean(pred == pred_full)),
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _redundancy_metrics(abs_corr: np.ndarray, selected_idx: np.ndarray) -> tuple[float, float]:
    idx = np.asarray(selected_idx, dtype=np.int64)
    if idx.size <= 1:
        return 0.0, 0.0
    sub = abs_corr[np.ix_(idx, idx)]
    tri = sub[np.triu_indices(sub.shape[0], k=1)]
    if tri.size == 0:
        return 0.0, 0.0
    return float(np.mean(tri)), float(np.max(tri))


def run(
    *,
    artifact_dir: Path,
    dataset: str,
    budgets: tuple[int, ...],
    corr_thresholds: tuple[float, ...],
    out_summary: Path,
    out_detail: Path,
    out_jaccard: Path,
    out_redundancy: Path,
) -> None:
    artifacts = _load_artifacts(artifact_dir, dataset)
    seeds = sorted(artifacts)
    if len(seeds) < 2:
        raise ValueError(f"Need >=2 seeds for jaccard. Found {len(seeds)} in {artifact_dir}")

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    jaccard_rows: list[dict[str, object]] = []
    redundancy_rows: list[dict[str, object]] = []

    selected_sets: dict[tuple[int, str, str], list[tuple[int, set[str]]]] = {}

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

        _, x_val, _, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.25,
            random_state=int(seed),
            stratify=y_train_full,
        )

        loro_bce = _compute_loro_bce_importance(
            h_val=x_val,
            y_val=y_val,
            weights=w_full,
            bias=b_full,
        )
        abs_corr_val = _corr_abs_matrix(x_val)

        for budget in budgets:
            budget_idx = np.argsort(-old_importance, kind="stable")[: min(int(budget), old_importance.shape[0])]
            loro_idx = np.argsort(
                np.rec.fromarrays((-loro_bce, -old_importance), names=("a", "b")),
                kind="stable",
            )[: min(int(budget), old_importance.shape[0])]

            base_methods = (
                ("budget_prune", "na", budget_idx),
                ("loro_bce", "na", loro_idx),
            )

            for method, thr_tag, idx in base_methods:
                metrics = _eval_subset(
                    x_train=x_train_full,
                    y_train=y_train_full,
                    x_test=x_test,
                    y_test=y_test,
                    selected_idx=idx,
                    full_test_prob=np.asarray(art["full_test_prob"], dtype=np.float64),
                    threshold_prob=threshold,
                )
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "budget": int(budget),
                        "method": method,
                        "corr_threshold": thr_tag,
                        "n_clusters": "",
                        "effective_selected_rules": int(idx.size),
                        **metrics,
                    }
                )
                mean_corr, max_corr = _redundancy_metrics(abs_corr_val, idx)
                redundancy_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "budget": int(budget),
                        "method": method,
                        "corr_threshold": thr_tag,
                        "mean_abs_pairwise_corr_selected": mean_corr,
                        "max_abs_pairwise_corr_selected": max_corr,
                    }
                )
                key = (int(budget), method, thr_tag)
                selected_sets.setdefault(key, []).append((int(seed), {rule_key[int(i)] for i in idx}))

            for thr in corr_thresholds:
                cp_idx, cluster_ids, reps, selected_clusters, labels = _cluster_representatives(
                    abs_corr=abs_corr_val,
                    importance=loro_bce,
                    corr_threshold=float(thr),
                    budget=int(budget),
                )

                cluster_to_members: dict[int, np.ndarray] = {}
                for cid in np.unique(labels):
                    cluster_to_members[int(cid)] = np.flatnonzero(labels == cid)

                for j, cid in enumerate(cluster_ids):
                    members = cluster_to_members[int(cid)]
                    rep_id = int(reps[j])
                    detail_rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "budget": int(budget),
                            "corr_threshold": float(thr),
                            "cluster_id": int(cid),
                            "cluster_size": int(members.size),
                            "representative_rule_id": rep_id,
                            "representative_importance": float(loro_bce[rep_id]),
                            "selected": int(bool(selected_clusters[j])),
                        }
                    )

                metrics = _eval_subset(
                    x_train=x_train_full,
                    y_train=y_train_full,
                    x_test=x_test,
                    y_test=y_test,
                    selected_idx=cp_idx,
                    full_test_prob=np.asarray(art["full_test_prob"], dtype=np.float64),
                    threshold_prob=threshold,
                )
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "budget": int(budget),
                        "method": "c_prune_loro_bce",
                        "corr_threshold": float(thr),
                        "n_clusters": int(cluster_ids.size),
                        "effective_selected_rules": int(cp_idx.size),
                        **metrics,
                    }
                )
                mean_corr, max_corr = _redundancy_metrics(abs_corr_val, cp_idx)
                redundancy_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "budget": int(budget),
                        "method": "c_prune_loro_bce",
                        "corr_threshold": float(thr),
                        "mean_abs_pairwise_corr_selected": mean_corr,
                        "max_abs_pairwise_corr_selected": max_corr,
                    }
                )
                key = (int(budget), "c_prune_loro_bce", str(float(thr)))
                selected_sets.setdefault(key, []).append((int(seed), {rule_key[int(i)] for i in cp_idx}))

    # Rule-level jaccard only (no cluster-jaccard claim).
    for (budget, method, thr_tag), sets in selected_sets.items():
        for (sa, a), (sb, b) in combinations(sets, 2):
            jaccard_rows.append(
                {
                    "dataset": dataset,
                    "budget": int(budget),
                    "method": method,
                    "corr_threshold": thr_tag,
                    "seed_a": int(sa),
                    "seed_b": int(sb),
                    "rule_jaccard": float(_jaccard(a, b)),
                }
            )

    _write_csv(
        out_summary,
        [
            "dataset",
            "seed",
            "budget",
            "method",
            "corr_threshold",
            "n_clusters",
            "effective_selected_rules",
            "f1",
            "roc_auc",
            "pr_auc",
            "fidelity",
            "agreement",
        ],
        summary_rows,
    )
    _write_csv(
        out_detail,
        [
            "dataset",
            "seed",
            "budget",
            "corr_threshold",
            "cluster_id",
            "cluster_size",
            "representative_rule_id",
            "representative_importance",
            "selected",
        ],
        detail_rows,
    )
    _write_csv(
        out_jaccard,
        ["dataset", "budget", "method", "corr_threshold", "seed_a", "seed_b", "rule_jaccard"],
        jaccard_rows,
    )
    _write_csv(
        out_redundancy,
        [
            "dataset",
            "seed",
            "budget",
            "method",
            "corr_threshold",
            "mean_abs_pairwise_corr_selected",
            "max_abs_pairwise_corr_selected",
        ],
        redundancy_rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default="covtype_binary_20000")
    ap.add_argument("--budgets", type=str, default="25,50,100")
    ap.add_argument("--corr-thresholds", type=str, default="0.75,0.85,0.90")
    ap.add_argument("--out-summary", type=Path, default=Path("docs/tables_c_prune_summary.csv"))
    ap.add_argument("--out-detail", type=Path, default=Path("docs/tables_c_prune_detail.csv"))
    ap.add_argument("--out-jaccard", type=Path, default=Path("docs/tables_c_prune_jaccard.csv"))
    ap.add_argument("--out-redundancy", type=Path, default=Path("docs/tables_c_prune_redundancy.csv"))
    args = ap.parse_args()

    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    thresholds = tuple(float(x.strip()) for x in args.corr_thresholds.split(",") if x.strip())

    run(
        artifact_dir=args.artifact_dir,
        dataset=args.dataset,
        budgets=budgets,
        corr_thresholds=thresholds,
        out_summary=args.out_summary,
        out_detail=args.out_detail,
        out_jaccard=args.out_jaccard,
        out_redundancy=args.out_redundancy,
    )

    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.out_detail}")
    print(f"Wrote {args.out_jaccard}")
    print(f"Wrote {args.out_redundancy}")


if __name__ == "__main__":
    main()
