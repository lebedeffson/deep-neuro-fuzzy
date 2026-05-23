from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def _to_binary(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y).reshape(-1) >= 0.5).astype(np.int64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


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
    from sklearn.metrics import log_loss

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default="covtype_binary_20000")
    ap.add_argument("--seeds", type=str, default="19,23,29")
    ap.add_argument("--budgets", type=str, default="25,50,100")
    ap.add_argument("--taus", type=str, default="0.75")
    ap.add_argument("--out-csv", type=Path, default=Path("docs/tables_c_prune_runtime.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("docs/tables_c_prune_runtime.md"))
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    taus = [float(x.strip()) for x in args.taus.split(",") if x.strip()]
    b_ref = max(budgets)

    rows: list[dict[str, object]] = []
    for seed in seeds:
        z = np.load(args.artifact_dir / f"v18_h_artifacts_{args.dataset}_seed{seed}.npz", allow_pickle=True)
        x_train = np.asarray(z["h_train"], dtype=np.float64)
        y_train = _to_binary(np.asarray(z["y_train"]))
        x_test = np.asarray(z["h_test"], dtype=np.float64)
        y_test = _to_binary(np.asarray(z["y_test"]))
        old_imp = np.asarray(z["importance"], dtype=np.float64).reshape(-1)
        w_full, b_full = _reconstruct_linear_head(x_train, np.asarray(z["full_train_logits"], dtype=np.float64))
        _, x_val, _, y_val = train_test_split(
            x_train, y_train, test_size=0.25, random_state=int(seed), stratify=y_train
        )

        t0 = time.perf_counter()
        loro_imp = _compute_loro_bce_importance(x_val, y_val, w_full, b_full)
        t_loro = time.perf_counter() - t0

        t0 = time.perf_counter()
        abs_corr = _corr_abs_matrix(x_val)
        t_corr = time.perf_counter() - t0

        for tau in taus:
            t0 = time.perf_counter()
            idx_c = _cluster_representatives(abs_corr, loro_imp, tau, b_ref)
            t_cluster = time.perf_counter() - t0

            t0 = time.perf_counter()
            idx_b = np.argsort(-old_imp, kind="stable")[: min(b_ref, old_imp.shape[0])]
            idx_l = np.argsort(np.rec.fromarrays((-loro_imp, -old_imp), names=("a", "b")), kind="stable")[
                : min(b_ref, old_imp.shape[0])
            ]
            t_select_base = time.perf_counter() - t0

            # Optional contextual refit/eval at B_ref
            def _fit_eval(idx: np.ndarray) -> float:
                clf = LogisticRegression(solver="liblinear", max_iter=300)
                clf.fit(x_train[:, idx], y_train)
                prob = clf.predict_proba(x_test[:, idx])[:, 1]
                pred = (prob >= 0.5).astype(np.int64)
                return float(f1_score(y_test, pred))

            t0 = time.perf_counter()
            f1_b = _fit_eval(idx_b)
            f1_l = _fit_eval(idx_l)
            f1_c = _fit_eval(idx_c)
            t_refit_eval = time.perf_counter() - t0

            rows.append(
                {
                    "dataset": args.dataset,
                    "seed": int(seed),
                    "tau": float(tau),
                    "budget_ref": int(b_ref),
                    "n_rules": int(x_train.shape[1]),
                    "n_val": int(x_val.shape[0]),
                    "loro_importance_sec": float(t_loro),
                    "corr_matrix_sec": float(t_corr),
                    "cluster_select_sec": float(t_cluster),
                    "baseline_select_sec": float(t_select_base),
                    "total_selection_sec": float(t_loro + t_corr + t_cluster),
                    "refit_eval_3methods_sec": float(t_refit_eval),
                    "f1_budget_prune_ref": float(f1_b),
                    "f1_loro_bce_ref": float(f1_l),
                    "f1_c_prune_ref": float(f1_c),
                    "effective_rules_c_prune_ref": int(idx_c.size),
                }
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # mean summary in markdown
    import math

    def _m(k: str) -> float:
        vals = [float(r[k]) for r in rows]
        return float(sum(vals) / max(1, len(vals)))

    md = [
        "# Cluster-Prune Runtime (Covtype)",
        "",
        f"- seeds: {seeds}",
        f"- tau: {taus}",
        f"- budget_ref: {b_ref}",
        "",
        "| Metric | Mean seconds |",
        "|---|---:|",
        f"| LORO importance | {_m('loro_importance_sec'):.4f} |",
        f"| Corr matrix | {_m('corr_matrix_sec'):.4f} |",
        f"| Cluster select (complete linkage) | {_m('cluster_select_sec'):.4f} |",
        f"| Total selection (LORO+corr+cluster) | {_m('total_selection_sec'):.4f} |",
        f"| Refit+eval (3 methods, B={b_ref}) | {_m('refit_eval_3methods_sec'):.4f} |",
        "",
        "Note: runtime is workflow context, not strict apples-to-apples training-speed comparison.",
    ]
    args.out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
