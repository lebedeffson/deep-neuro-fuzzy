from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _to_binary(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y).reshape(-1) >= 0.5).astype(np.int64)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _reconstruct_linear_head(h_train: np.ndarray, full_train_logits: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(h_train, dtype=np.float64)
    y = np.asarray(full_train_logits, dtype=np.float64).reshape(-1)
    xa = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    beta, *_ = np.linalg.lstsq(xa, y, rcond=None)
    return np.asarray(beta[1:], dtype=np.float64), float(beta[0])


def _fit_surrogate_head_ridge(h_train: np.ndarray, full_train_logits: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    r = Ridge(alpha=float(alpha), fit_intercept=True)
    r.fit(np.asarray(h_train, dtype=np.float64), np.asarray(full_train_logits, dtype=np.float64).reshape(-1))
    return np.asarray(r.coef_, dtype=np.float64).reshape(-1), float(r.intercept_)


def _compute_loro_bce_importance(h_val: np.ndarray, y_val: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    y = _to_binary(y_val)
    z_full = bias + h_val @ weights
    bce_full = float(log_loss(y, _sigmoid(z_full), labels=[0, 1]))
    imp = np.zeros(h_val.shape[1], dtype=np.float64)
    for r in range(h_val.shape[1]):
        z_without = z_full - h_val[:, r] * weights[r]
        imp[r] = float(log_loss(y, _sigmoid(z_without), labels=[0, 1])) - bce_full
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


def _hash_selected(rule_keys: list[str], idx: np.ndarray) -> str:
    keys = sorted(str(rule_keys[int(i)]) for i in np.asarray(idx, dtype=np.int64))
    payload = "\n".join(keys).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _sufficiency_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    full_prob: np.ndarray,
    full_thr: float,
    idx: np.ndarray,
) -> tuple[float, float, float, float]:
    clf = LogisticRegression(solver="liblinear", max_iter=300)
    clf.fit(x_train[:, idx], y_train)
    p = clf.predict_proba(x_test[:, idx])[:, 1]
    yhat = (p >= 0.5).astype(np.int64)
    yhat_full = (np.asarray(full_prob, dtype=np.float64) >= float(full_thr)).astype(np.int64)
    return (
        float(f1_score(y_test, yhat)),
        float(log_loss(y_test, p, labels=[0, 1])),
        float(np.mean(np.abs(p - np.asarray(full_prob, dtype=np.float64)))),
        float(np.mean(yhat == yhat_full)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default="covtype_binary_20000")
    ap.add_argument("--seeds", type=str, default="19,23,29")
    ap.add_argument("--budgets", type=str, default="25,50,100")
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--deletion-head", type=str, default="ridge", choices=["ridge", "lstsq"])
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--out-detail", type=Path, default=Path("docs/tables_local_faithfulness_detail.csv"))
    ap.add_argument("--out-summary", type=Path, default=Path("docs/tables_local_faithfulness_summary.csv"))
    ap.add_argument("--out-sanity", type=Path, default=Path("docs/tables_local_faithfulness_sanity_checks.csv"))
    args = ap.parse_args()

    seeds = tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())

    detail_rows: list[dict[str, object]] = []

    for seed in seeds:
        p = args.artifact_dir / f"v18_h_artifacts_{args.dataset}_seed{seed}.npz"
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")
        z = np.load(p, allow_pickle=True)

        x_train = np.asarray(z["h_train"], dtype=np.float64)
        y_train = _to_binary(np.asarray(z["y_train"]))
        x_test = np.asarray(z["h_test"], dtype=np.float64)
        y_test = _to_binary(np.asarray(z["y_test"]))
        full_test_prob = np.asarray(z["full_test_prob"], dtype=np.float64).reshape(-1)
        threshold = float(np.asarray(z["classification_threshold"]).reshape(-1)[0])
        old_importance = np.asarray(z["importance"], dtype=np.float64).reshape(-1)
        rule_keys = z["rule_key"].astype(str).tolist()

        full_train_logits = np.asarray(z["full_train_logits"], dtype=np.float64)
        if args.deletion_head == "ridge":
            w_full, b_full = _fit_surrogate_head_ridge(x_train, full_train_logits, alpha=float(args.ridge_alpha))
        else:
            w_full, b_full = _reconstruct_linear_head(x_train, full_train_logits)
        z_full = b_full + x_test @ w_full
        p_full = _sigmoid(z_full)
        yhat_full = (p_full >= threshold).astype(np.int64)
        f1_full = float(f1_score(y_test, yhat_full))
        bce_full = float(log_loss(y_test, p_full, labels=[0, 1]))
        surrogate_fidelity_to_full_prob = float(np.mean(np.abs(p_full - full_test_prob)))
        surrogate_agreement_to_full = float(np.mean((p_full >= threshold) == (full_test_prob >= threshold)))

        _, x_val, _, y_val = train_test_split(
            x_train, y_train, test_size=0.25, random_state=int(seed), stratify=y_train
        )
        loro_bce = _compute_loro_bce_importance(x_val, y_val, w_full, b_full)
        abs_corr = _corr_abs_matrix(x_val)

        for budget in budgets:
            k = min(int(budget), old_importance.shape[0])
            idx_budget = np.argsort(-old_importance, kind="stable")[:k]
            idx_loro = np.argsort(
                np.rec.fromarrays((-loro_bce, -old_importance), names=("a", "b")),
                kind="stable",
            )[:k]
            idx_cprune = _cluster_representatives(abs_corr, loro_bce, float(args.tau), int(budget))

            for method, idx in (
                ("budget_prune", idx_budget),
                ("loro_bce", idx_loro),
                ("c_prune_loro_bce", idx_cprune),
            ):
                idx = np.asarray(idx, dtype=np.int64)
                sel_contrib = x_test[:, idx] @ w_full[idx]
                z_without = z_full - sel_contrib
                p_without = _sigmoid(z_without)
                yhat_without = (p_without >= threshold).astype(np.int64)
                f1_without = float(f1_score(y_test, yhat_without))
                bce_without = float(log_loss(y_test, p_without, labels=[0, 1]))
                suff_f1, suff_bce, suff_fid, suff_agr = _sufficiency_metrics(
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    full_prob=full_test_prob,
                    full_thr=threshold,
                    idx=idx,
                )

                detail_rows.append(
                    {
                        "dataset": args.dataset,
                        "seed": int(seed),
                        "budget": int(budget),
                        "method": method,
                        "sufficiency_f1": float(suff_f1),
                        "sufficiency_bce": float(suff_bce),
                        "fidelity_l1_to_full_prob": float(suff_fid),
                        "agreement_to_full": float(suff_agr),
                        "deletion_delta_bce": float(bce_without - bce_full),
                        "deletion_delta_f1": float(f1_full - f1_without),
                        "selected_rules_count": int(idx.size),
                        "selected_rules_hash": _hash_selected(rule_keys, idx),
                        "removed_contribution_l1_mean": float(np.mean(np.abs(sel_contrib))),
                        "bce_full": float(bce_full),
                        "bce_without": float(bce_without),
                        "f1_full": float(f1_full),
                        "f1_without": float(f1_without),
                        "deletion_head_mode": str(args.deletion_head),
                        "deletion_head_ridge_alpha": float(args.ridge_alpha),
                        "surrogate_fidelity_to_full_prob": float(surrogate_fidelity_to_full_prob),
                        "surrogate_agreement_to_full": float(surrogate_agreement_to_full),
                        "effective_selected_rules": int(idx.size),
                    }
                )

    # summary (means over seeds)
    summary_rows: list[dict[str, object]] = []
    for budget in budgets:
        for method in ("budget_prune", "loro_bce", "c_prune_loro_bce"):
            sub = [r for r in detail_rows if int(r["budget"]) == int(budget) and str(r["method"]) == method]
            summary_rows.append(
                {
                    "dataset": args.dataset,
                    "budget": int(budget),
                    "method": method,
                    "deletion_delta_bce_mean": float(np.mean([float(r["deletion_delta_bce"]) for r in sub])),
                    "deletion_delta_f1_mean": float(np.mean([float(r["deletion_delta_f1"]) for r in sub])),
                    "removed_contribution_l1_mean": float(
                        np.mean([float(r["removed_contribution_l1_mean"]) for r in sub])
                    ),
                    "selected_rules_count_mean": float(np.mean([float(r["selected_rules_count"]) for r in sub])),
                    "bce_full_mean": float(np.mean([float(r["bce_full"]) for r in sub])),
                    "bce_without_mean": float(np.mean([float(r["bce_without"]) for r in sub])),
                    "f1_full_mean": float(np.mean([float(r["f1_full"]) for r in sub])),
                    "f1_without_mean": float(np.mean([float(r["f1_without"]) for r in sub])),
                    "surrogate_fidelity_to_full_prob_mean": float(
                        np.mean([float(r["surrogate_fidelity_to_full_prob"]) for r in sub])
                    ),
                    "surrogate_agreement_to_full_mean": float(
                        np.mean([float(r["surrogate_agreement_to_full"]) for r in sub])
                    ),
                    "n_seeds": len(sub),
                }
            )

    # sanity checks
    sanity_rows: list[dict[str, object]] = []
    for seed in seeds:
        # hash differences by budget (per method)
        for method in ("budget_prune", "loro_bce", "c_prune_loro_bce"):
            h = [r["selected_rules_hash"] for r in detail_rows if int(r["seed"]) == seed and r["method"] == method]
            sanity_rows.append(
                {
                    "dataset": args.dataset,
                    "seed": int(seed),
                    "check": f"hash_unique_across_budgets::{method}",
                    "result": str(len(set(h)) == len(h)),
                    "value": "|".join(map(str, h)),
                }
            )
        # hash differences by method (per budget)
        for budget in budgets:
            h = [
                r["selected_rules_hash"]
                for r in detail_rows
                if int(r["seed"]) == seed and int(r["budget"]) == int(budget)
            ]
            sanity_rows.append(
                {
                    "dataset": args.dataset,
                    "seed": int(seed),
                    "check": f"hash_unique_across_methods::B{budget}",
                    "result": str(len(set(h)) == len(h)),
                    "value": "|".join(map(str, h)),
                }
            )
        # monotonic trend rough check for deletion_delta_bce in c_prune
        vals = []
        for budget in budgets:
            sub = [
                float(r["deletion_delta_bce"])
                for r in detail_rows
                if int(r["seed"]) == seed and int(r["budget"]) == int(budget) and r["method"] == "c_prune_loro_bce"
            ]
            vals.append(float(np.mean(sub)))
        sanity_rows.append(
            {
                "dataset": args.dataset,
                "seed": int(seed),
                "check": "deletion_delta_bce_changes_with_budget::c_prune",
                "result": str(len(set(round(v, 12) for v in vals)) > 1),
                "value": ",".join(f"{b}:{v:.6f}" for b, v in zip(budgets, vals)),
            }
        )

    _write_csv(
        args.out_detail,
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
            "selected_rules_count",
            "selected_rules_hash",
            "removed_contribution_l1_mean",
            "bce_full",
            "bce_without",
            "f1_full",
            "f1_without",
            "deletion_head_mode",
            "deletion_head_ridge_alpha",
            "surrogate_fidelity_to_full_prob",
            "surrogate_agreement_to_full",
            "effective_selected_rules",
        ],
        detail_rows,
    )
    _write_csv(
        args.out_summary,
        [
            "dataset",
            "budget",
            "method",
            "deletion_delta_bce_mean",
            "deletion_delta_f1_mean",
            "removed_contribution_l1_mean",
            "selected_rules_count_mean",
            "bce_full_mean",
            "bce_without_mean",
            "f1_full_mean",
            "f1_without_mean",
            "surrogate_fidelity_to_full_prob_mean",
            "surrogate_agreement_to_full_mean",
            "n_seeds",
        ],
        summary_rows,
    )
    _write_csv(args.out_sanity, ["dataset", "seed", "check", "result", "value"], sanity_rows)

    print(f"Wrote {args.out_detail}")
    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.out_sanity}")


if __name__ == "__main__":
    main()
