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


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


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
    imp = np.zeros(h_val.shape[1], dtype=np.float64)
    for r in range(h_val.shape[1]):
        z_without = z_full - h_val[:, r] * weights[r]
        p_without = _sigmoid(z_without)
        imp[r] = float(log_loss(y, p_without, labels=[0, 1])) - bce_full
    return imp


def _corr_abs_matrix(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(np.asarray(x, dtype=np.float64), rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(np.abs(c), 0.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def _cluster_representatives(abs_corr: np.ndarray, importance: np.ndarray, tau: float, budget: int) -> np.ndarray:
    dist = 1.0 - abs_corr
    dist = np.clip(0.5 * (dist + dist.T), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(z, t=float(1.0 - tau), criterion="distance").astype(np.int64)

    reps: list[int] = []
    scores: list[float] = []
    for cid in np.unique(labels):
        members = np.flatnonzero(labels == cid)
        j = int(members[np.argmax(importance[members])])
        reps.append(j)
        scores.append(float(importance[j]))
    reps_arr = np.asarray(reps, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores_arr, kind="stable")
    return reps_arr[order[: min(int(budget), reps_arr.size)]]


def _eval_subset_prob(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    selected_idx: np.ndarray,
) -> np.ndarray:
    clf = LogisticRegression(solver="liblinear", max_iter=300)
    clf.fit(x_train[:, selected_idx], y_train)
    return clf.predict_proba(x_test[:, selected_idx])[:, 1]


def _metrics_from_prob(
    y_true: np.ndarray,
    prob: np.ndarray,
    full_prob: np.ndarray,
    full_label: np.ndarray,
) -> dict[str, float]:
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "fidelity_l1_to_full_prob": float(np.mean(np.abs(prob - full_prob))),
        "agreement_to_full": float(np.mean(pred == full_label)),
        "bce": float(log_loss(y_true, prob, labels=[0, 1])),
    }


def _percentile_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float, float]:
    a = np.asarray(values, dtype=np.float64)
    return float(np.mean(a)), float(np.percentile(a, 100.0 * alpha / 2.0)), float(
        np.percentile(a, 100.0 * (1.0 - alpha / 2.0))
    )


def _fast_f1(y_true: np.ndarray, prob: np.ndarray, thr: float = 0.5) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    p = (np.asarray(prob, dtype=np.float64) >= thr).astype(np.int64)
    tp = int(np.sum((p == 1) & (y == 1)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    den = (2 * tp + fp + fn)
    return float((2 * tp) / den) if den > 0 else 0.0


def _load_artifacts(artifact_dir: Path, dataset: str, seeds: tuple[int, ...]) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for seed in seeds:
        p = artifact_dir / f"v18_h_artifacts_{dataset}_seed{seed}.npz"
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")
        z = np.load(p, allow_pickle=True)
        out[int(seed)] = {
            "h_train": np.asarray(z["h_train"], dtype=np.float64),
            "h_test": np.asarray(z["h_test"], dtype=np.float64),
            "y_train": _to_binary(np.asarray(z["y_train"])),
            "y_test": _to_binary(np.asarray(z["y_test"])),
            "full_train_logits": np.asarray(z["full_train_logits"], dtype=np.float64).reshape(-1),
            "full_test_prob": np.asarray(z["full_test_prob"], dtype=np.float64).reshape(-1),
            "old_importance": np.asarray(z["importance"], dtype=np.float64).reshape(-1),
            "threshold": float(np.asarray(z["classification_threshold"]).reshape(-1)[0]),
            "rule_key": z["rule_key"].astype(str),
        }
    return out


def run(
    *,
    artifact_dir: Path,
    dataset: str,
    seeds: tuple[int, ...],
    budgets: tuple[int, ...],
    tau: float,
    n_boot_select: int,
    n_boot_test: int,
    out_selection_summary: Path,
    out_selection_ci: Path,
    out_paired_test_bootstrap: Path,
    out_l1lr_stability: Path,
    out_faithfulness_summary: Path,
    out_faithfulness_detail: Path,
) -> None:
    artifacts = _load_artifacts(artifact_dir, dataset, seeds)

    # cache per seed
    cache: dict[int, dict[str, object]] = {}
    selected_rules: dict[tuple[int, str, int], np.ndarray] = {}
    probs: dict[tuple[int, str, int], np.ndarray] = {}
    full_probs: dict[int, np.ndarray] = {}
    full_labels: dict[int, np.ndarray] = {}
    y_tests: dict[int, np.ndarray] = {}
    w_fulls: dict[int, np.ndarray] = {}
    b_fulls: dict[int, float] = {}

    for seed in seeds:
        art = artifacts[int(seed)]
        x_train = np.asarray(art["h_train"], dtype=np.float64)
        y_train = np.asarray(art["y_train"], dtype=np.int64)
        x_test = np.asarray(art["h_test"], dtype=np.float64)
        y_test = np.asarray(art["y_test"], dtype=np.int64)
        full_prob = np.asarray(art["full_test_prob"], dtype=np.float64)
        threshold = float(art["threshold"])
        full_label = (full_prob >= threshold).astype(np.int64)
        w_full, b_full = _reconstruct_linear_head(x_train, np.asarray(art["full_train_logits"], dtype=np.float64))
        _, x_val, _, y_val = train_test_split(
            x_train, y_train, test_size=0.25, random_state=int(seed), stratify=y_train
        )

        cache[int(seed)] = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "x_val": x_val,
            "y_val": y_val,
            "threshold": threshold,
            "old_importance": np.asarray(art["old_importance"], dtype=np.float64),
            "rule_key": art["rule_key"],
        }
        full_probs[int(seed)] = full_prob
        full_labels[int(seed)] = full_label
        y_tests[int(seed)] = y_test
        w_fulls[int(seed)] = w_full
        b_fulls[int(seed)] = b_full

        # fixed selections (main runs)
        imp_loro = _compute_loro_bce_importance(x_val, y_val, w_full, b_full)
        abs_corr_val = _corr_abs_matrix(x_val)
        for budget in budgets:
            k = min(int(budget), imp_loro.size)
            bp_idx = np.argsort(-np.asarray(art["old_importance"], dtype=np.float64), kind="stable")[:k]
            lb_idx = np.argsort(
                np.rec.fromarrays((-imp_loro, -np.asarray(art["old_importance"], dtype=np.float64)), names=("a", "b")),
                kind="stable",
            )[:k]
            cp_idx = _cluster_representatives(abs_corr_val, imp_loro, float(tau), int(budget))
            selected_rules[(int(seed), "budget_prune", int(budget))] = bp_idx
            selected_rules[(int(seed), "loro_bce", int(budget))] = lb_idx
            selected_rules[(int(seed), "c_prune_loro_bce", int(budget))] = cp_idx
            for method, idx in (
                ("budget_prune", bp_idx),
                ("loro_bce", lb_idx),
                ("c_prune_loro_bce", cp_idx),
            ):
                probs[(int(seed), method, int(budget))] = _eval_subset_prob(x_train, y_train, x_test, idx)

    # A) Selection bootstrap (validation resampling)
    sel_rows: list[dict[str, object]] = []
    rng_master = np.random.default_rng(20260523)
    for b in range(int(n_boot_select)):
        for seed in seeds:
            c = cache[int(seed)]
            x_val = np.asarray(c["x_val"], dtype=np.float64)
            y_val = np.asarray(c["y_val"], dtype=np.int64)
            bs = rng_master.integers(0, x_val.shape[0], size=x_val.shape[0], endpoint=False)
            xb = x_val[bs]
            yb = y_val[bs]
            w_full = w_fulls[int(seed)]
            b_full = b_fulls[int(seed)]
            imp_loro = _compute_loro_bce_importance(xb, yb, w_full, b_full)
            abs_corr = _corr_abs_matrix(xb)
            old_imp = np.asarray(c["old_importance"], dtype=np.float64)

            for budget in budgets:
                k = min(int(budget), old_imp.size)
                bp_idx = np.argsort(-old_imp, kind="stable")[:k]
                lb_idx = np.argsort(np.rec.fromarrays((-imp_loro, -old_imp), names=("a", "b")), kind="stable")[:k]
                cp_idx = _cluster_representatives(abs_corr, imp_loro, float(tau), int(budget))

                for method, idx in (
                    ("budget_prune", bp_idx),
                    ("loro_bce", lb_idx),
                    ("c_prune_loro_bce", cp_idx),
                ):
                    prob = _eval_subset_prob(
                        np.asarray(c["x_train"], dtype=np.float64),
                        np.asarray(c["y_train"], dtype=np.int64),
                        np.asarray(c["x_test"], dtype=np.float64),
                        np.asarray(idx, dtype=np.int64),
                    )
                    met = _metrics_from_prob(
                        y_true=np.asarray(c["y_test"], dtype=np.int64),
                        prob=prob,
                        full_prob=full_probs[int(seed)],
                        full_label=full_labels[int(seed)],
                    )
                    sel_rows.append(
                        {
                            "bootstrap_iter": int(b),
                            "dataset": dataset,
                            "seed": int(seed),
                            "budget": int(budget),
                            "method": method,
                            "f1": met["f1"],
                            "roc_auc": met["roc_auc"],
                            "pr_auc": met["pr_auc"],
                            "fidelity": met["fidelity_l1_to_full_prob"],
                            "agreement": met["agreement_to_full"],
                            "effective_selected_rules": int(np.asarray(idx).size),
                        }
                    )

    # summaries from selection bootstrap
    sum_rows: list[dict[str, object]] = []
    ci_rows: list[dict[str, object]] = []
    for budget in budgets:
        for method in ("budget_prune", "loro_bce", "c_prune_loro_bce"):
            sub = [r for r in sel_rows if int(r["budget"]) == int(budget) and str(r["method"]) == method]
            row = {"dataset": dataset, "budget": int(budget), "method": method, "n_samples": len(sub)}
            for metric in ("f1", "roc_auc", "pr_auc", "fidelity", "agreement"):
                vals = [float(r[metric]) for r in sub]
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
                m, lo, hi = _percentile_ci(vals)
                ci_rows.append(
                    {
                        "dataset": dataset,
                        "budget": int(budget),
                        "method": method,
                        "metric": metric,
                        "mean": m,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_samples": len(vals),
                    }
                )
            row["effective_selected_rules_mean"] = float(
                np.mean([float(r["effective_selected_rules"]) for r in sub])
            )
            sum_rows.append(row)

    _write_csv(
        out_selection_summary,
        [
            "dataset",
            "budget",
            "method",
            "n_samples",
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
            "effective_selected_rules_mean",
        ],
        sum_rows,
    )
    _write_csv(
        out_selection_ci,
        ["dataset", "budget", "method", "metric", "mean", "ci95_low", "ci95_high", "n_samples"],
        ci_rows,
    )

    # B) paired test bootstrap on fixed selected rules
    pair_rows: list[dict[str, object]] = []
    method_pairs = [
        ("c_prune_loro_bce", "budget_prune"),
        ("c_prune_loro_bce", "loro_bce"),
        ("loro_bce", "budget_prune"),
    ]
    rng = np.random.default_rng(123456)
    for budget in budgets:
        for ma, mb in method_pairs:
            deltas = {m: [] for m in ("f1", "bce", "fidelity", "agreement")}
            for _ in range(int(n_boot_test)):
                seed_deltas = {m: [] for m in ("f1", "bce", "fidelity", "agreement")}
                for seed in seeds:
                    y = y_tests[int(seed)]
                    n = y.shape[0]
                    idx = rng.integers(0, n, size=n, endpoint=False)
                    pa = probs[(int(seed), ma, int(budget))][idx]
                    pb = probs[(int(seed), mb, int(budget))][idx]
                    yb = y[idx]
                    fprob = full_probs[int(seed)][idx]
                    flabel = full_labels[int(seed)][idx]
                    pa_clip = np.clip(pa, 1e-9, 1.0 - 1e-9)
                    pb_clip = np.clip(pb, 1e-9, 1.0 - 1e-9)
                    ybf = yb.astype(np.float64)
                    bce_a = float(-np.mean(ybf * np.log(pa_clip) + (1.0 - ybf) * np.log(1.0 - pa_clip)))
                    bce_b = float(-np.mean(ybf * np.log(pb_clip) + (1.0 - ybf) * np.log(1.0 - pb_clip)))
                    f1_a = _fast_f1(yb, pa)
                    f1_b = _fast_f1(yb, pb)
                    pred_a = (pa >= 0.5).astype(np.int64)
                    pred_b = (pb >= 0.5).astype(np.int64)
                    seed_deltas["f1"].append(float(f1_a - f1_b))
                    seed_deltas["bce"].append(float(bce_a - bce_b))
                    seed_deltas["fidelity"].append(float(np.mean(np.abs(pa - fprob)) - np.mean(np.abs(pb - fprob))))
                    seed_deltas["agreement"].append(
                        float(np.mean(pred_a == flabel) - np.mean(pred_b == flabel))
                    )
                for metric in deltas:
                    deltas[metric].append(float(np.mean(seed_deltas[metric])))
            for metric, vals in deltas.items():
                m, lo, hi = _percentile_ci(vals)
                pair_rows.append(
                    {
                        "dataset": dataset,
                        "budget": int(budget),
                        "method_a": ma,
                        "method_b": mb,
                        "metric": metric,
                        "delta_mean": m,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "n_bootstrap": int(n_boot_test),
                        "aggregation": "mean_over_seeds",
                    }
                )
    _write_csv(
        out_paired_test_bootstrap,
        [
            "dataset",
            "budget",
            "method_a",
            "method_b",
            "metric",
            "delta_mean",
            "ci95_low",
            "ci95_high",
            "n_bootstrap",
            "aggregation",
        ],
        pair_rows,
    )

    # C) L1-LR stability comparison (budgets 25/50/100)
    # Build global meta-clusters on quantile signatures for tau
    q = np.linspace(0.0, 1.0, num=500, dtype=np.float64)
    sig_rows: list[np.ndarray] = []
    sig_keys: list[tuple[int, int]] = []
    for seed in seeds:
        xtr = np.asarray(cache[int(seed)]["x_train"], dtype=np.float64)
        for r in range(xtr.shape[1]):
            sig_rows.append(np.quantile(xtr[:, r], q))
            sig_keys.append((int(seed), int(r)))
    sig = np.vstack(sig_rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(sig, rowvar=True)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(np.abs(c), 0.0, 1.0)
    np.fill_diagonal(c, 1.0)
    d = 1.0 - c
    d = np.clip(0.5 * (d + d.T), 0.0, 1.0)
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="complete")
    labels = fcluster(z, t=float(1.0 - tau), criterion="distance").astype(np.int64)
    meta_map: dict[tuple[int, int], int] = {}
    for i, k in enumerate(sig_keys):
        meta_map[k] = int(labels[i])

    l1_rows: list[dict[str, object]] = []
    for budget in budgets:
        # gather selected ids per seed/method
        sets: dict[str, list[tuple[int, set[str], set[int], float]]] = {
            "budget_prune": [],
            "loro_bce": [],
            "c_prune_loro_bce": [],
            "l1_lr_top_coef": [],
        }
        for seed in seeds:
            # existing methods from fixed selections
            rule_key = np.asarray(cache[int(seed)]["rule_key"]).astype(str)
            for method in ("budget_prune", "loro_bce", "c_prune_loro_bce"):
                idx_arr = np.asarray(selected_rules[(int(seed), method, int(budget))], dtype=np.int64)
                ids = set(map(int, idx_arr))
                keys = {str(rule_key[r]) for r in idx_arr}
                met = _metrics_from_prob(
                    y_tests[int(seed)],
                    probs[(int(seed), method, int(budget))],
                    full_probs[int(seed)],
                    full_labels[int(seed)],
                )
                sets[method].append((int(seed), keys, ids, float(met["f1"])))

            # l1-lr top-|coef|
            cseed = cache[int(seed)]
            # sklearn>=1.8 deprecates explicit `penalty`; use elastic-net with l1_ratio=1 for pure L1 sparsity.
            l1 = LogisticRegression(solver="saga", l1_ratio=1.0, C=0.1, max_iter=4000, random_state=int(seed))
            l1.fit(np.asarray(cseed["x_train"], dtype=np.float64), np.asarray(cseed["y_train"], dtype=np.int64))
            coef = np.abs(l1.coef_.reshape(-1))
            idx = np.argsort(-coef, kind="stable")[: min(int(budget), coef.size)]
            prob = _eval_subset_prob(
                np.asarray(cseed["x_train"], dtype=np.float64),
                np.asarray(cseed["y_train"], dtype=np.int64),
                np.asarray(cseed["x_test"], dtype=np.float64),
                np.asarray(idx, dtype=np.int64),
            )
            met = _metrics_from_prob(y_tests[int(seed)], prob, full_probs[int(seed)], full_labels[int(seed)])
            idx_set = set(map(int, idx))
            key_set = {str(rule_key[r]) for r in idx}
            sets["l1_lr_top_coef"].append((int(seed), key_set, idx_set, float(met["f1"])))

        # pairwise jaccards + per-method summary
        for method, entries in sets.items():
            f1_mean = float(np.mean([e[3] for e in entries]))
            for (sa, a_keys, a_idx, _), (sb, b_keys, b_idx, _) in combinations(entries, 2):
                ma = {meta_map[(int(sa), int(r))] for r in a_idx}
                mb = {meta_map[(int(sb), int(r))] for r in b_idx}
                l1_rows.append(
                    {
                        "dataset": dataset,
                        "budget": int(budget),
                        "method": method,
                        "corr_threshold": float(tau),
                        "seed_a": int(sa),
                        "seed_b": int(sb),
                        "f1_mean_method": f1_mean,
                        "rule_jaccard": float(_jaccard(a_keys, b_keys)),
                        "meta_cluster_jaccard": float(_jaccard(ma, mb)),
                    }
                )
    _write_csv(
        out_l1lr_stability,
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
        l1_rows,
    )

    # D) local faithfulness: sufficiency + comprehensiveness
    detail_rows: list[dict[str, object]] = []
    for seed in seeds:
        cseed = cache[int(seed)]
        x_test = np.asarray(cseed["x_test"], dtype=np.float64)
        y_test = np.asarray(cseed["y_test"], dtype=np.int64)
        w_full = w_fulls[int(seed)]
        b_full = b_fulls[int(seed)]
        z_full = b_full + x_test @ w_full
        p_full_recon = _sigmoid(z_full)
        pred_full_recon = (p_full_recon >= float(cseed["threshold"])).astype(np.int64)
        full_f1 = float(f1_score(y_test, pred_full_recon))
        full_bce = float(log_loss(y_test, p_full_recon, labels=[0, 1]))
        for budget in budgets:
            for method in ("budget_prune", "loro_bce", "c_prune_loro_bce"):
                idx = np.asarray(selected_rules[(int(seed), method, int(budget))], dtype=np.int64)
                prob = probs[(int(seed), method, int(budget))]
                met = _metrics_from_prob(y_test, prob, full_probs[int(seed)], full_labels[int(seed)])

                z_without = z_full - x_test[:, idx] @ w_full[idx]
                p_without = _sigmoid(z_without)
                pred_without = (p_without >= float(cseed["threshold"])).astype(np.int64)
                bce_without = float(log_loss(y_test, p_without, labels=[0, 1]))
                f1_without = float(f1_score(y_test, pred_without))
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "budget": int(budget),
                        "method": method,
                        "sufficiency_f1": met["f1"],
                        "sufficiency_bce": met["bce"],
                        "fidelity_l1_to_full_prob": met["fidelity_l1_to_full_prob"],
                        "agreement_to_full": met["agreement_to_full"],
                        "deletion_delta_bce": float(bce_without - full_bce),
                        "deletion_delta_f1": float(full_f1 - f1_without),
                        "effective_selected_rules": int(idx.size),
                    }
                )

    # aggregate
    summary_rows: list[dict[str, object]] = []
    for budget in budgets:
        for method in ("budget_prune", "loro_bce", "c_prune_loro_bce"):
            sub = [r for r in detail_rows if int(r["budget"]) == int(budget) and str(r["method"]) == method]
            summary_rows.append(
                {
                    "dataset": dataset,
                    "budget": int(budget),
                    "method": method,
                    "sufficiency_f1_mean": float(np.mean([float(r["sufficiency_f1"]) for r in sub])),
                    "sufficiency_bce_mean": float(np.mean([float(r["sufficiency_bce"]) for r in sub])),
                    "fidelity_l1_to_full_prob_mean": float(
                        np.mean([float(r["fidelity_l1_to_full_prob"]) for r in sub])
                    ),
                    "agreement_to_full_mean": float(np.mean([float(r["agreement_to_full"]) for r in sub])),
                    "deletion_delta_bce_mean": float(np.mean([float(r["deletion_delta_bce"]) for r in sub])),
                    "deletion_delta_f1_mean": float(np.mean([float(r["deletion_delta_f1"]) for r in sub])),
                    "effective_selected_rules_mean": float(np.mean([float(r["effective_selected_rules"]) for r in sub])),
                    "n_seeds": int(len(sub)),
                }
            )
    _write_csv(
        out_faithfulness_detail,
        [
            "dataset",
            "seed",
            "budget",
            "method",
            "sufficiency_f1",
            "sufficiency_bce",
            "fidelity_l1_to_full_prob",
            "agreement_to_full",
            "deletion_delta_bce",
            "deletion_delta_f1",
            "effective_selected_rules",
        ],
        detail_rows,
    )
    _write_csv(
        out_faithfulness_summary,
        [
            "dataset",
            "budget",
            "method",
            "sufficiency_f1_mean",
            "sufficiency_bce_mean",
            "fidelity_l1_to_full_prob_mean",
            "agreement_to_full_mean",
            "deletion_delta_bce_mean",
            "deletion_delta_f1_mean",
            "effective_selected_rules_mean",
            "n_seeds",
        ],
        summary_rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default="covtype_binary_20000")
    ap.add_argument("--seeds", type=str, default="19,23,29")
    ap.add_argument("--budgets", type=str, default="25,50,100")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--n-boot-select", type=int, default=100)
    ap.add_argument("--n-boot-test", type=int, default=1000)
    ap.add_argument("--out-selection-summary", type=Path, default=Path("docs/tables_selection_bootstrap_summary.csv"))
    ap.add_argument("--out-selection-ci", type=Path, default=Path("docs/tables_selection_bootstrap_ci.csv"))
    ap.add_argument(
        "--out-paired-test-bootstrap",
        type=Path,
        default=Path("docs/tables_paired_test_bootstrap_summary.csv"),
    )
    ap.add_argument(
        "--out-l1lr-stability",
        type=Path,
        default=Path("docs/tables_l1_lr_stability_comparison.csv"),
    )
    ap.add_argument(
        "--out-faithfulness-summary",
        type=Path,
        default=Path("docs/tables_local_faithfulness_summary.csv"),
    )
    ap.add_argument(
        "--out-faithfulness-detail",
        type=Path,
        default=Path("docs/tables_local_faithfulness_detail.csv"),
    )
    args = ap.parse_args()

    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    run(
        artifact_dir=Path(args.artifact_dir),
        dataset=str(args.dataset),
        seeds=seeds,
        budgets=budgets,
        tau=float(args.tau),
        n_boot_select=int(args.n_boot_select),
        n_boot_test=int(args.n_boot_test),
        out_selection_summary=Path(args.out_selection_summary),
        out_selection_ci=Path(args.out_selection_ci),
        out_paired_test_bootstrap=Path(args.out_paired_test_bootstrap),
        out_l1lr_stability=Path(args.out_l1lr_stability),
        out_faithfulness_summary=Path(args.out_faithfulness_summary),
        out_faithfulness_detail=Path(args.out_faithfulness_detail),
    )
    print(f"Wrote {args.out_selection_summary}")
    print(f"Wrote {args.out_selection_ci}")
    print(f"Wrote {args.out_paired_test_bootstrap}")
    print(f"Wrote {args.out_l1lr_stability}")
    print(f"Wrote {args.out_faithfulness_summary}")
    print(f"Wrote {args.out_faithfulness_detail}")


if __name__ == "__main__":
    main()
