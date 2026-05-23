from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import log_loss
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
            "full_train_logits": z["full_train_logits"].reshape(-1),
            "importance": z["importance"].astype(np.float64).reshape(-1),
            "rule_key": z["rule_key"].astype(str),
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
    c = np.corrcoef(np.asarray(x, dtype=np.float64), rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.clip(np.abs(c), 0.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def _corr_abs_matrix_rows(x: np.ndarray) -> np.ndarray:
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
    n_rules = int(importance.shape[0])
    dist = np.clip(0.5 * ((1.0 - abs_corr) + (1.0 - abs_corr).T), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(z, t=float(1.0 - corr_threshold), criterion="distance").astype(np.int64)

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
    k = min(int(budget), reps_arr.size)
    return reps_arr[order[:k]]


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _build_global_meta_labels(
    artifacts: dict[int, dict[str, object]],
    seeds: list[int],
    n_ref: int,
    corr_threshold: float,
) -> dict[tuple[int, int], int]:
    # Distributional signature: quantiles of rule activations.
    # This does not require row-alignment of samples across seeds.
    q = np.linspace(0.0, 1.0, num=max(8, int(n_ref)), dtype=np.float64)
    signatures: list[np.ndarray] = []
    key_of_row: list[tuple[int, int]] = []
    for seed in seeds:
        h_train = np.asarray(artifacts[seed]["h_train"], dtype=np.float64)
        n_rules = h_train.shape[1]
        for r in range(n_rules):
            col = h_train[:, r]
            signatures.append(np.quantile(col, q).astype(np.float64, copy=False))
            key_of_row.append((int(seed), int(r)))
    sig = np.vstack(signatures)

    abs_corr = _corr_abs_matrix_rows(sig)
    dist = np.clip(0.5 * ((1.0 - abs_corr) + (1.0 - abs_corr).T), 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(z, t=float(1.0 - corr_threshold), criterion="distance").astype(np.int64)

    out: dict[tuple[int, int], int] = {}
    for i, key in enumerate(key_of_row):
        out[key] = int(labels[i])
    return out


def run(
    *,
    artifact_dir: Path,
    dataset: str,
    budgets: tuple[int, ...],
    corr_thresholds: tuple[float, ...],
    n_ref: int,
    out_meta_jaccard: Path,
) -> None:
    artifacts = _load_artifacts(artifact_dir, dataset)
    seeds = sorted(artifacts)
    if len(seeds) < 2:
        raise ValueError(f"Need >=2 seeds, found {len(seeds)} in {artifact_dir}")

    selected_sets: dict[tuple[int, str, str], list[tuple[int, set[int], set[str]]]] = {}

    for seed in seeds:
        art = artifacts[seed]
        x_train_full = np.asarray(art["h_train"], dtype=np.float64)
        y_train_full = _to_binary(np.asarray(art["y_train"]))
        old_importance = np.asarray(art["importance"], dtype=np.float64)
        rule_key = [str(x) for x in art["rule_key"]]
        w_full, b_full = _reconstruct_linear_head(x_train_full, np.asarray(art["full_train_logits"]))

        _, x_val, _, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.25,
            random_state=int(seed),
            stratify=y_train_full,
        )
        loro_bce = _compute_loro_bce_importance(x_val, y_val, w_full, b_full)
        abs_corr_val = _corr_abs_matrix_cols(x_val)

        for budget in budgets:
            bp_idx = np.argsort(-old_importance, kind="stable")[: min(int(budget), old_importance.size)]
            lb_idx = np.argsort(
                np.rec.fromarrays((-loro_bce, -old_importance), names=("a", "b")),
                kind="stable",
            )[: min(int(budget), old_importance.size)]
            selected_sets.setdefault((int(budget), "budget_prune", "na"), []).append(
                (int(seed), set(map(int, bp_idx)), {rule_key[int(i)] for i in bp_idx})
            )
            selected_sets.setdefault((int(budget), "loro_bce", "na"), []).append(
                (int(seed), set(map(int, lb_idx)), {rule_key[int(i)] for i in lb_idx})
            )

            for thr in corr_thresholds:
                cp_idx = _cluster_representatives(abs_corr_val, loro_bce, float(thr), int(budget))
                selected_sets.setdefault((int(budget), "c_prune_loro_bce", str(float(thr))), []).append(
                    (int(seed), set(map(int, cp_idx)), {rule_key[int(i)] for i in cp_idx})
                )

    rows: list[dict[str, object]] = []
    for thr in corr_thresholds:
        meta_map = _build_global_meta_labels(artifacts, seeds, n_ref=int(n_ref), corr_threshold=float(thr))

        for budget in budgets:
            # baselines: selection fixed (na), meta-cluster threshold varies
            for method in ("budget_prune", "loro_bce"):
                sets = selected_sets[(int(budget), method, "na")]
                for (sa, a_ids, a_keys), (sb, b_ids, b_keys) in combinations(sets, 2):
                    ma = {meta_map[(int(sa), int(r))] for r in a_ids}
                    mb = {meta_map[(int(sb), int(r))] for r in b_ids}
                    rows.append(
                        {
                            "dataset": dataset,
                            "budget": int(budget),
                            "method": method,
                            "corr_threshold": float(thr),
                            "seed_a": int(sa),
                            "seed_b": int(sb),
                            "rule_jaccard": float(_jaccard(a_keys, b_keys)),
                            "meta_cluster_jaccard": float(_jaccard(ma, mb)),
                        }
                    )

            # c-prune: use matching threshold both for selection and for global meta-clusters
            key = (int(budget), "c_prune_loro_bce", str(float(thr)))
            sets = selected_sets[key]
            for (sa, a_ids, a_keys), (sb, b_ids, b_keys) in combinations(sets, 2):
                ma = {meta_map[(int(sa), int(r))] for r in a_ids}
                mb = {meta_map[(int(sb), int(r))] for r in b_ids}
                rows.append(
                    {
                        "dataset": dataset,
                        "budget": int(budget),
                        "method": "c_prune_loro_bce",
                        "corr_threshold": float(thr),
                        "seed_a": int(sa),
                        "seed_b": int(sb),
                        "rule_jaccard": float(_jaccard(a_keys, b_keys)),
                        "meta_cluster_jaccard": float(_jaccard(ma, mb)),
                    }
                )

    _write_csv(
        out_meta_jaccard,
        [
            "dataset",
            "budget",
            "method",
            "corr_threshold",
            "seed_a",
            "seed_b",
            "rule_jaccard",
            "meta_cluster_jaccard",
        ],
        rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, required=True)
    ap.add_argument("--dataset", type=str, default="covtype_binary_20000")
    ap.add_argument("--budgets", type=str, default="25,50,100")
    ap.add_argument("--corr-thresholds", type=str, default="0.75,0.85,0.90")
    ap.add_argument("--n-ref", type=int, default=500)
    ap.add_argument(
        "--out-meta-jaccard",
        type=Path,
        default=Path("docs/tables_c_prune_meta_cluster_jaccard.csv"),
    )
    args = ap.parse_args()

    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    thresholds = tuple(float(x.strip()) for x in args.corr_thresholds.split(",") if x.strip())
    run(
        artifact_dir=args.artifact_dir,
        dataset=args.dataset,
        budgets=budgets,
        corr_thresholds=thresholds,
        n_ref=int(args.n_ref),
        out_meta_jaccard=args.out_meta_jaccard,
    )
    print(f"Wrote {args.out_meta_jaccard}")


if __name__ == "__main__":
    main()
