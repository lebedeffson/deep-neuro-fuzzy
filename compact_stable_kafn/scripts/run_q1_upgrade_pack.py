from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
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


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _load_artifacts(artifact_dir: Path, dataset: str, seeds: tuple[int, ...]) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for seed in seeds:
        p = artifact_dir / f"v18_h_artifacts_{dataset}_seed{seed}.npz"
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")
        z = np.load(p, allow_pickle=True)
        out[int(seed)] = {
            "seed": int(seed),
            "h_train": np.asarray(z["h_train"], dtype=np.float64),
            "h_test": np.asarray(z["h_test"], dtype=np.float64),
            "y_train": _to_binary(np.asarray(z["y_train"])),
            "y_test": _to_binary(np.asarray(z["y_test"])),
            "full_train_logits": np.asarray(z["full_train_logits"], dtype=np.float64).reshape(-1),
            "full_test_prob": np.asarray(z["full_test_prob"], dtype=np.float64).reshape(-1),
            "importance": np.asarray(z["importance"], dtype=np.float64).reshape(-1),
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
        imp[r] = bce_without - bce_full
    return imp


def _corr_abs_matrix_cols(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(np.asarray(x, dtype=np.float64), rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(np.abs(c), 0.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def _corr_abs_matrix_rows(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(np.asarray(x, dtype=np.float64), rowvar=True)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(np.abs(c), 0.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def _cluster_representatives(
    abs_corr: np.ndarray,
    importance: np.ndarray,
    corr_threshold: float,
    budget: int,
) -> np.ndarray:
    dist = 1.0 - abs_corr
    dist = np.clip(0.5 * (dist + dist.T), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    labels = _cluster_labels_from_distance(dist, float(1.0 - corr_threshold))

    reps: list[int] = []
    scores: list[float] = []
    for cid in np.unique(labels):
        members = np.flatnonzero(labels == cid)
        best_local = int(members[np.argmax(importance[members])])
        reps.append(best_local)
        scores.append(float(importance[best_local]))
    reps_arr = np.asarray(reps, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores_arr, kind="stable")
    return reps_arr[order[: min(int(budget), reps_arr.size)]]


def _cluster_labels_from_distance(dist: np.ndarray, threshold: float) -> np.ndarray:
    # sklearn API changed `affinity` -> `metric`; keep compatibility.
    try:
        cl = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=float(threshold),
            compute_full_tree=True,
        )
    except TypeError:
        cl = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="complete",
            distance_threshold=float(threshold),
            compute_full_tree=True,
        )
    labels = cl.fit_predict(np.asarray(dist, dtype=np.float64))
    return np.asarray(labels, dtype=np.int64) + 1


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
    clf = LogisticRegression(solver="liblinear", max_iter=300)
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


def _eval_subset_linear_head(
    x_test: np.ndarray,
    y_test: np.ndarray,
    selected_idx: np.ndarray,
    weights: np.ndarray,
    bias: float,
    full_test_prob: np.ndarray,
    threshold_prob: float,
) -> dict[str, float]:
    cols = np.asarray(selected_idx, dtype=np.int64).reshape(-1)
    z = float(bias) + np.asarray(x_test, dtype=np.float64)[:, cols] @ np.asarray(weights, dtype=np.float64)[cols]
    prob = _sigmoid(z)
    pred = (prob >= 0.5).astype(np.int64)
    pred_full = (np.asarray(full_test_prob, dtype=np.float64) >= float(threshold_prob)).astype(np.int64)
    return {
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "fidelity": float(np.mean(np.abs(prob - np.asarray(full_test_prob, dtype=np.float64)))),
        "agreement": float(np.mean(pred == pred_full)),
    }


def _quantile_signatures(h_train: np.ndarray, n_ref: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, num=max(8, int(n_ref)), dtype=np.float64)
    n_rules = h_train.shape[1]
    out = np.zeros((n_rules, q.size), dtype=np.float64)
    for r in range(n_rules):
        out[r] = np.quantile(h_train[:, r], q)
    return out


def _build_global_meta_map(
    artifacts: dict[int, dict[str, object]],
    seeds: list[int],
    corr_threshold: float,
    n_ref: int,
) -> dict[tuple[int, int], int]:
    sig_rows: list[np.ndarray] = []
    keys: list[tuple[int, int]] = []
    for seed in seeds:
        sig = _quantile_signatures(np.asarray(artifacts[seed]["h_train"], dtype=np.float64), n_ref=int(n_ref))
        for r in range(sig.shape[0]):
            sig_rows.append(sig[r])
            keys.append((int(seed), int(r)))
    sig_mat = np.vstack(sig_rows)
    abs_corr = _corr_abs_matrix_rows(sig_mat)
    dist = 1.0 - abs_corr
    dist = np.clip(0.5 * (dist + dist.T), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    labels = _cluster_labels_from_distance(dist, float(1.0 - corr_threshold))
    out: dict[tuple[int, int], int] = {}
    for i, k in enumerate(keys):
        out[k] = int(labels[i])
    return out


def _percentile_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.percentile(arr, 100.0 * (alpha / 2.0))), float(
        np.percentile(arr, 100.0 * (1.0 - alpha / 2.0))
    )


def run_bootstrap_ci(
    artifacts: dict[int, dict[str, object]],
    seeds: list[int],
    budgets: tuple[int, ...],
    corr_threshold: float,
    n_boot: int,
    meta_n_ref: int,
    out_samples_csv: Path,
    out_ci_csv: Path,
) -> None:
    per_seed_cache: dict[int, dict[str, object]] = {}
    for seed in seeds:
        art = artifacts[seed]
        x_train = np.asarray(art["h_train"], dtype=np.float64)
        y_train = np.asarray(art["y_train"], dtype=np.int64)
        w_full, b_full = _reconstruct_linear_head(x_train, np.asarray(art["full_train_logits"], dtype=np.float64))
        _, x_val, _, y_val = train_test_split(
            x_train, y_train, test_size=0.25, random_state=int(seed), stratify=y_train
        )
        per_seed_cache[seed] = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": np.asarray(art["h_test"], dtype=np.float64),
            "y_test": np.asarray(art["y_test"], dtype=np.int64),
            "full_test_prob": np.asarray(art["full_test_prob"], dtype=np.float64),
            "threshold": float(art["threshold"]),
            "rule_key": [str(x) for x in art["rule_key"]],
            "x_val": x_val,
            "y_val": y_val,
            "w_full": w_full,
            "b_full": b_full,
        }

    meta_map = _build_global_meta_map(artifacts, seeds, corr_threshold=float(corr_threshold), n_ref=int(meta_n_ref))
    sample_rows: list[dict[str, object]] = []
    jaccard_rows: list[dict[str, object]] = []

    for b in range(int(n_boot)):
        selected_by_seed_method_budget: dict[tuple[int, str, int], np.ndarray] = {}
        for seed in seeds:
            c = per_seed_cache[seed]
            x_val = np.asarray(c["x_val"], dtype=np.float64)
            y_val = np.asarray(c["y_val"], dtype=np.int64)
            rng = np.random.default_rng(seed * 1_000_003 + b)
            idx_bs = rng.integers(0, x_val.shape[0], size=x_val.shape[0], endpoint=False)
            x_bs = x_val[idx_bs]
            y_bs = y_val[idx_bs]

            # Bootstrap-aware proxy for budget_prune
            budget_imp = np.abs(np.asarray(c["w_full"], dtype=np.float64)) * np.mean(x_bs, axis=0)
            loro_bce = _compute_loro_bce_importance(
                h_val=x_bs, y_val=y_bs, weights=np.asarray(c["w_full"], dtype=np.float64), bias=float(c["b_full"])
            )
            abs_corr = _corr_abs_matrix_cols(x_bs)

            for budget in budgets:
                k = min(int(budget), budget_imp.shape[0])
                bp_idx = np.argsort(-budget_imp, kind="stable")[:k]
                lb_idx = np.argsort(
                    np.rec.fromarrays((-loro_bce, -budget_imp), names=("a", "b")), kind="stable"
                )[:k]
                cp_idx = _cluster_representatives(
                    abs_corr=abs_corr, importance=loro_bce, corr_threshold=float(corr_threshold), budget=int(budget)
                )
                method_to_idx = {
                    "budget_prune": bp_idx,
                    "loro_bce": lb_idx,
                    "c_prune_loro_bce": cp_idx,
                }
                for method, sel in method_to_idx.items():
                    selected_by_seed_method_budget[(seed, method, int(budget))] = np.asarray(sel, dtype=np.int64)
                    metrics = _eval_subset(
                        x_train=np.asarray(c["x_train"], dtype=np.float64),
                        y_train=np.asarray(c["y_train"], dtype=np.int64),
                        x_test=np.asarray(c["x_test"], dtype=np.float64),
                        y_test=np.asarray(c["y_test"], dtype=np.int64),
                        selected_idx=np.asarray(sel, dtype=np.int64),
                        full_test_prob=np.asarray(c["full_test_prob"], dtype=np.float64),
                        threshold_prob=float(c["threshold"]),
                    )
                    sample_rows.append(
                        {
                            "bootstrap_iter": int(b),
                            "seed": int(seed),
                            "budget": int(budget),
                            "method": method,
                            "f1": metrics["f1"],
                            "roc_auc": metrics["roc_auc"],
                            "pr_auc": metrics["pr_auc"],
                            "fidelity": metrics["fidelity"],
                            "agreement": metrics["agreement"],
                            "effective_selected_rules": int(np.asarray(sel).size),
                        }
                    )

        # per-iteration pairwise jaccards (rule + meta)
        for budget in budgets:
            for method in ("budget_prune", "loro_bce", "c_prune_loro_bce"):
                sets = []
                for seed in seeds:
                    ids = set(map(int, selected_by_seed_method_budget[(seed, method, int(budget))]))
                    meta = {meta_map[(int(seed), int(r))] for r in ids}
                    sets.append((int(seed), ids, meta))
                pair_rule = []
                pair_meta = []
                for (_, a_ids, a_meta), (_, b_ids, b_meta) in combinations(sets, 2):
                    pair_rule.append(_jaccard(a_ids, b_ids))
                    pair_meta.append(_jaccard(a_meta, b_meta))
                jaccard_rows.append(
                    {
                        "bootstrap_iter": int(b),
                        "budget": int(budget),
                        "method": method,
                        "corr_threshold": float(corr_threshold),
                        "rule_jaccard_mean_pairs": float(np.mean(pair_rule)),
                        "meta_cluster_jaccard_mean_pairs": float(np.mean(pair_meta)),
                    }
                )

    # Aggregate CIs
    ci_rows: list[dict[str, object]] = []
    for budget in budgets:
        for method in ("budget_prune", "loro_bce", "c_prune_loro_bce"):
            sub = [r for r in sample_rows if int(r["budget"]) == int(budget) and str(r["method"]) == method]
            for metric in ("f1", "roc_auc", "pr_auc", "fidelity", "agreement"):
                vals = [float(r[metric]) for r in sub]
                m, lo, hi = _percentile_ci(vals)
                ci_rows.append(
                    {
                        "budget": int(budget),
                        "method": method,
                        "metric": metric,
                        "mean": m,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_samples": int(len(vals)),
                    }
                )
            subj = [r for r in jaccard_rows if int(r["budget"]) == int(budget) and str(r["method"]) == method]
            for metric in ("rule_jaccard_mean_pairs", "meta_cluster_jaccard_mean_pairs"):
                vals = [float(r[metric]) for r in subj]
                m, lo, hi = _percentile_ci(vals)
                ci_rows.append(
                    {
                        "budget": int(budget),
                        "method": method,
                        "metric": metric,
                        "mean": m,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_samples": int(len(vals)),
                    }
                )

    _write_csv(
        out_samples_csv,
        [
            "bootstrap_iter",
            "seed",
            "budget",
            "method",
            "f1",
            "roc_auc",
            "pr_auc",
            "fidelity",
            "agreement",
            "effective_selected_rules",
        ],
        sample_rows,
    )
    _write_csv(
        out_ci_csv,
        ["budget", "method", "metric", "mean", "ci95_low", "ci95_high", "n_samples"],
        ci_rows,
    )


def run_l1lr_stability_vs_cprune(
    artifacts: dict[int, dict[str, object]],
    seeds: list[int],
    budget: int,
    corr_threshold: float,
    meta_n_ref: int,
    out_csv: Path,
) -> None:
    meta_map = _build_global_meta_map(artifacts, seeds, corr_threshold=float(corr_threshold), n_ref=int(meta_n_ref))
    rows: list[dict[str, object]] = []

    selected: dict[str, list[tuple[int, set[int], float]]] = {
        "l1_lr_topk": [],
        "c_prune_loro_bce": [],
    }

    for seed in seeds:
        art = artifacts[seed]
        x_train = np.asarray(art["h_train"], dtype=np.float64)
        y_train = np.asarray(art["y_train"], dtype=np.int64)
        x_test = np.asarray(art["h_test"], dtype=np.float64)
        y_test = np.asarray(art["y_test"], dtype=np.int64)
        full_test_prob = np.asarray(art["full_test_prob"], dtype=np.float64)
        threshold = float(art["threshold"])

        # L1-LR selection by top-|coef| at fixed budget
        l1 = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=4000, random_state=int(seed), n_jobs=1)
        l1.fit(x_train, y_train)
        coef = np.abs(l1.coef_.reshape(-1))
        idx_l1 = np.argsort(-coef, kind="stable")[: min(int(budget), coef.size)]
        met_l1 = _eval_subset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            selected_idx=idx_l1,
            full_test_prob=full_test_prob,
            threshold_prob=threshold,
        )
        selected["l1_lr_topk"].append((int(seed), set(map(int, idx_l1)), float(met_l1["f1"])))

        # C-Prune selection at the same budget/threshold
        w_full, b_full = _reconstruct_linear_head(x_train, np.asarray(art["full_train_logits"], dtype=np.float64))
        _, x_val, _, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=int(seed), stratify=y_train)
        loro_bce = _compute_loro_bce_importance(x_val, y_val, w_full, b_full)
        abs_corr = _corr_abs_matrix_cols(x_val)
        idx_cp = _cluster_representatives(abs_corr, loro_bce, float(corr_threshold), int(budget))
        met_cp = _eval_subset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            selected_idx=idx_cp,
            full_test_prob=full_test_prob,
            threshold_prob=threshold,
        )
        selected["c_prune_loro_bce"].append((int(seed), set(map(int, idx_cp)), float(met_cp["f1"])))

    for method, sets in selected.items():
        f1_mean = float(np.mean([x[2] for x in sets]))
        for (sa, a_ids, _), (sb, b_ids, _) in combinations(sets, 2):
            ma = {meta_map[(int(sa), int(r))] for r in a_ids}
            mb = {meta_map[(int(sb), int(r))] for r in b_ids}
            rows.append(
                {
                    "dataset": "covtype_binary_20000",
                    "budget": int(budget),
                    "method": method,
                    "corr_threshold": float(corr_threshold),
                    "seed_a": int(sa),
                    "seed_b": int(sb),
                    "f1_mean_method": f1_mean,
                    "rule_jaccard": float(_jaccard(a_ids, b_ids)),
                    "meta_cluster_jaccard": float(_jaccard(ma, mb)),
                }
            )

    _write_csv(
        out_csv,
        [
            "dataset",
            "budget",
            "method",
            "corr_threshold",
            "seed_a",
            "seed_b",
            "f1_mean_method",
            "rule_jaccard",
            "meta_cluster_jaccard",
        ],
        rows,
    )


def run_deletion_test(
    artifacts: dict[int, dict[str, object]],
    seed: int,
    budget: int,
    corr_threshold: float,
    n_samples: int,
    out_csv: Path,
) -> None:
    art = artifacts[int(seed)]
    x_train = np.asarray(art["h_train"], dtype=np.float64)
    y_train = np.asarray(art["y_train"], dtype=np.int64)
    x_test = np.asarray(art["h_test"], dtype=np.float64)
    y_test = np.asarray(art["y_test"], dtype=np.int64)
    full_test_prob = np.asarray(art["full_test_prob"], dtype=np.float64)
    rule_key = [str(x) for x in art["rule_key"]]
    w_full, b_full = _reconstruct_linear_head(x_train, np.asarray(art["full_train_logits"], dtype=np.float64))

    _, x_val, _, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=int(seed), stratify=y_train)
    loro_bce = _compute_loro_bce_importance(x_val, y_val, w_full, b_full)
    abs_corr = _corr_abs_matrix_cols(x_val)
    selected_idx = _cluster_representatives(abs_corr, loro_bce, float(corr_threshold), int(budget))

    # compact surrogate
    clf = LogisticRegression(solver="liblinear", max_iter=300)
    clf.fit(x_train[:, selected_idx], y_train)
    coef = clf.coef_.reshape(-1)
    intercept = float(clf.intercept_[0])
    compact_logits = intercept + x_test[:, selected_idx] @ coef
    compact_prob = _sigmoid(compact_logits)

    rng = np.random.default_rng(42)
    sample_ids = rng.choice(x_test.shape[0], size=min(int(n_samples), x_test.shape[0]), replace=False)

    rows: list[dict[str, object]] = []
    for idx in sample_ids:
        y = int(y_test[int(idx)])
        deltas = x_test[int(idx), selected_idx] * coef
        order = np.argsort(-np.abs(deltas), kind="stable")[:5]
        z_del = float(compact_logits[int(idx)] - np.sum(deltas[order]))
        p_del = float(_sigmoid(z_del))
        bce_full = float(log_loss([y], [float(compact_prob[int(idx)])], labels=[0, 1]))
        bce_del = float(log_loss([y], [p_del], labels=[0, 1]))
        top_rules = []
        for j in order:
            rid = int(selected_idx[int(j)])
            top_rules.append(f"{rule_key[rid]}:{deltas[int(j)]:+.4f}")
        rows.append(
            {
                "dataset": "covtype_binary_20000",
                "seed": int(seed),
                "budget": int(budget),
                "corr_threshold": float(corr_threshold),
                "sample_index": int(idx),
                "y_true": int(y),
                "full_prob": float(full_test_prob[int(idx)]),
                "compact_prob": float(compact_prob[int(idx)]),
                "compact_prob_without_top5": float(p_del),
                "deletion_delta_bce_top5": float(bce_del - bce_full),
                "top5_selected_rule_contribs": " | ".join(top_rules),
            }
        )

    _write_csv(
        out_csv,
        [
            "dataset",
            "seed",
            "budget",
            "corr_threshold",
            "sample_index",
            "y_true",
            "full_prob",
            "compact_prob",
            "compact_prob_without_top5",
            "deletion_delta_bce_top5",
            "top5_selected_rule_contribs",
        ],
        rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default="covtype_binary_20000")
    ap.add_argument("--seeds", type=str, default="19,23,29")
    ap.add_argument("--budgets", type=str, default="25,50,100")
    ap.add_argument("--corr-threshold", type=float, default=0.75)
    ap.add_argument("--meta-n-ref", type=int, default=500)
    ap.add_argument("--bootstrap-iters", type=int, default=100)
    ap.add_argument(
        "--out-bootstrap-samples",
        type=Path,
        default=Path("docs/tables_c_prune_bootstrap_samples.csv"),
    )
    ap.add_argument(
        "--out-bootstrap-ci",
        type=Path,
        default=Path("docs/tables_c_prune_bootstrap_ci.csv"),
    )
    ap.add_argument(
        "--out-l1lr-stability",
        type=Path,
        default=Path("docs/tables_l1lr_stability_vs_cprune.csv"),
    )
    ap.add_argument(
        "--out-deletion-test",
        type=Path,
        default=Path("docs/tables_c_prune_deletion_test.csv"),
    )
    ap.add_argument("--deletion-seed", type=int, default=23)
    ap.add_argument("--deletion-budget", type=int, default=100)
    ap.add_argument("--deletion-n-samples", type=int, default=5)
    args = ap.parse_args()

    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    artifacts = _load_artifacts(
        artifact_dir=Path(args.artifact_dir),
        dataset=str(args.dataset),
        seeds=seeds,
    )
    seed_list = sorted(artifacts.keys())

    run_bootstrap_ci(
        artifacts=artifacts,
        seeds=seed_list,
        budgets=budgets,
        corr_threshold=float(args.corr_threshold),
        n_boot=int(args.bootstrap_iters),
        meta_n_ref=int(args.meta_n_ref),
        out_samples_csv=Path(args.out_bootstrap_samples),
        out_ci_csv=Path(args.out_bootstrap_ci),
    )

    run_l1lr_stability_vs_cprune(
        artifacts=artifacts,
        seeds=seed_list,
        budget=100,
        corr_threshold=float(args.corr_threshold),
        meta_n_ref=int(args.meta_n_ref),
        out_csv=Path(args.out_l1lr_stability),
    )

    run_deletion_test(
        artifacts=artifacts,
        seed=int(args.deletion_seed),
        budget=int(args.deletion_budget),
        corr_threshold=float(args.corr_threshold),
        n_samples=int(args.deletion_n_samples),
        out_csv=Path(args.out_deletion_test),
    )

    print(f"Wrote {args.out_bootstrap_samples}")
    print(f"Wrote {args.out_bootstrap_ci}")
    print(f"Wrote {args.out_l1lr_stability}")
    print(f"Wrote {args.out_deletion_test}")


if __name__ == "__main__":
    main()
