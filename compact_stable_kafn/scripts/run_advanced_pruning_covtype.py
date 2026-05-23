from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ruanfis.lofo_f1_prune import (
    bootstrap_lofo_f1_scores,
    lofo_f1_scores,
    top_bootstrap_lofo_f1_indices,
    top_lofo_f1_indices,
)
from ruanfis.stable_budget_prune import stable_budget_scores, top_budget_indices


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
            "full_train_logits": z["full_train_logits"].reshape(-1) if "full_train_logits" in z.files else None,
            "importance": z["importance"].astype(np.float64).reshape(-1),
            "rule_key": z["rule_key"].astype(str),
        }
    return out


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _to_binary(y: np.ndarray) -> np.ndarray:
    return (np.asarray(y).reshape(-1) >= 0.5).astype(np.int64)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _pairwise_mean_jaccard(sets: list[set[str]]) -> float:
    pairs = list(combinations(sets, 2))
    if not pairs:
        return 1.0
    return float(mean(_jaccard(a, b) for a, b in pairs))


def _fit_eval_lr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    return (
        float(f1_score(y_test, pred)),
        float(roc_auc_score(y_test, prob)),
        float(average_precision_score(y_test, prob)),
    )


def _budget_prune_indices(importance: np.ndarray, budget: int) -> np.ndarray:
    k = min(int(budget), importance.shape[0])
    return np.argsort(-importance, kind="stable")[:k]


def _reconstruct_linear_head(h_train: np.ndarray, full_train_logits: np.ndarray | None, y_train: np.ndarray) -> tuple[np.ndarray, float]:
    # Prefer reconstructing full KAFN linear head from saved full logits.
    if full_train_logits is not None:
        x = np.asarray(h_train, dtype=np.float64)
        y = np.asarray(full_train_logits, dtype=np.float64).reshape(-1)
        xa = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
        beta, *_ = np.linalg.lstsq(xa, y, rcond=None)
        bias = float(beta[0])
        w = np.asarray(beta[1:], dtype=np.float64)
        return w, bias
    # Fallback: LR surrogate if full logits are unavailable.
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(h_train, y_train)
    w = np.asarray(clf.coef_).reshape(-1).astype(np.float64)
    b = float(clf.intercept_.reshape(-1)[0])
    return w, b


def _stable_indices_for_heldout(
    artifacts: dict[int, dict[str, object]],
    heldout_seed: int,
    budget: int,
    stability_top_k_multiplier: float,
    std_penalty: float,
) -> np.ndarray:
    seeds = sorted(artifacts)
    heldout = artifacts[heldout_seed]
    universe = [str(x) for x in heldout["rule_key"]]
    key_to_col = {k: i for i, k in enumerate(universe)}
    runs: list[np.ndarray] = []
    for s in seeds:
        if s == heldout_seed:
            continue
        a = artifacts[s]
        vals = np.asarray(a["importance"], dtype=np.float64)
        keys = [str(x) for x in a["rule_key"]]
        mapping = {keys[i]: vals[i] for i in range(len(keys))}
        runs.append(np.asarray([mapping.get(k, 0.0) for k in universe], dtype=np.float64))
    top_k = min(len(universe), max(int(budget), int(round(float(budget) * float(stability_top_k_multiplier)))))
    scores = stable_budget_scores(runs, budget=budget, stability_top_k=top_k, std_penalty=std_penalty)
    idx = top_budget_indices(scores.score, budget)
    # explicit re-map to local columns for safety
    return np.asarray([key_to_col[universe[int(i)]] for i in idx], dtype=np.int64)


def _stable_bootstrap_indices(
    x_train: np.ndarray,
    y_train: np.ndarray,
    importance: np.ndarray,
    budget: int,
    seed: int,
    n_bootstrap: int,
    stability_top_k_multiplier: float,
    candidate_pool: int,
) -> np.ndarray:
    n_rules = x_train.shape[1]
    k_pool = min(int(candidate_pool), n_rules)
    pool = np.argsort(-importance, kind="stable")[:k_pool]
    top_k = min(k_pool, max(int(budget), int(round(float(budget) * float(stability_top_k_multiplier)))))
    freq = np.zeros(k_pool, dtype=np.float64)
    rng = np.random.default_rng(int(seed))

    # Bootstrap sparse selector over H; frequency of top-k hits.
    for b in range(max(1, int(n_bootstrap))):
        idx = rng.integers(0, x_train.shape[0], size=x_train.shape[0], endpoint=False)
        xb = x_train[idx][:, pool]
        yb = y_train[idx]
        clf = LogisticRegression(
            solver="liblinear",
            penalty="l1",
            C=0.3,
            max_iter=1000,
            random_state=int(seed + b),
        )
        clf.fit(xb, yb)
        w = np.abs(clf.coef_).reshape(-1)
        hit = np.argsort(-w, kind="stable")[:top_k]
        freq[hit] += 1.0

    freq = freq / float(max(1, int(n_bootstrap)))
    imp_pool = np.asarray(importance[pool], dtype=np.float64)
    imp_norm = imp_pool / (np.max(imp_pool) + 1e-12)
    score = freq * imp_norm
    k = min(int(budget), k_pool)
    chosen_local = np.argsort(-score, kind="stable")[:k]
    return np.asarray(pool[chosen_local], dtype=np.int64)


def _gate_l1_topk_proxy_indices(x_train: np.ndarray, y_train: np.ndarray, budget: int, seed: int) -> np.ndarray:
    # Proxy for "Gate-L1 + top-k": sparse LR coefficients used as gate magnitudes.
    clf = LogisticRegression(solver="liblinear", penalty="l1", C=0.3, max_iter=2000, random_state=seed)
    clf.fit(x_train, y_train)
    weights = np.abs(clf.coef_).reshape(-1)
    k = min(int(budget), weights.shape[0])
    return np.argsort(-weights, kind="stable")[:k]


def _forward_f1_indices(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    importance: np.ndarray,
    budget: int,
    candidate_pool: int,
    shortlist_per_step: int,
    max_steps: int | None,
    train_max_rows: int | None,
    seed: int,
) -> np.ndarray:
    # Practical greedy: per step we shortlist by |corr(feature, residual)|, then pick best by val F1.
    n_rules = x_train.shape[1]
    k_pool = min(int(candidate_pool), n_rules)
    candidates = list(np.argsort(-importance, kind="stable")[:k_pool])
    selected: list[int] = []
    yv = y_val.astype(np.float64)
    max_sel = min(int(budget), n_rules)
    if max_steps is not None:
        max_sel = min(max_sel, int(max_steps))

    # Optional speed cap: use a stratified subset of train for candidate fits.
    if train_max_rows is not None and int(train_max_rows) > 0 and x_train.shape[0] > int(train_max_rows):
        x_train, _, y_train, _ = train_test_split(
            x_train,
            y_train,
            train_size=int(train_max_rows),
            random_state=int(seed),
            stratify=y_train,
        )

    current_prob = np.full_like(yv, fill_value=np.mean(y_train), dtype=np.float64)
    eps = 1e-9
    for _ in range(max_sel):
        remaining = [c for c in candidates if c not in selected]
        if not remaining:
            break
        residual = yv - current_prob
        # shortlist by absolute correlation with residual
        corr_scores: list[tuple[float, int]] = []
        for c in remaining:
            xv = x_val[:, c]
            xv_std = float(np.std(xv))
            if xv_std < eps:
                corr = 0.0
            else:
                corr = float(abs(np.corrcoef(xv, residual)[0, 1]))
                if not np.isfinite(corr):
                    corr = 0.0
            corr_scores.append((corr, c))
        corr_scores.sort(reverse=True, key=lambda t: t[0])
        shortlist = [c for _, c in corr_scores[: min(int(shortlist_per_step), len(corr_scores))]]

        best = None
        best_prob = None
        cols_base = selected.copy()
        for c in shortlist:
            cols = cols_base + [c]
            clf = LogisticRegression(solver="liblinear", max_iter=120)
            clf.fit(x_train[:, cols], y_train)
            prob = clf.predict_proba(x_val[:, cols])[:, 1]
            pred = (prob >= 0.5).astype(np.int64)
            score = float(f1_score(y_val, pred))
            if best is None or score > best[0]:
                best = (score, c)
                best_prob = prob
        if best is None:
            break
        selected.append(int(best[1]))
        current_prob = np.asarray(best_prob, dtype=np.float64)
    return np.asarray(selected, dtype=np.int64)


def run(
    *,
    artifact_dir: Path,
    dataset: str,
    budgets: tuple[int, ...],
    out_detail: Path,
    out_summary: Path,
    stability_top_k_multiplier: float,
    std_penalty: float,
    forward_candidate_pool: int,
    forward_shortlist_per_step: int,
    forward_max_steps: int | None,
    n_bootstrap: int,
    forward_train_max_rows: int | None,
) -> None:
    artifacts = _load_artifacts(artifact_dir, dataset)
    seeds = sorted(artifacts)
    if len(seeds) < 1:
        raise ValueError(
            "Need >=1 artifact seed (.npz). "
            f"Found {len(seeds)} in {artifact_dir}. "
            "Generate artifacts first via RUN_V18_CONTROLS.md step 1 "
            "(v18_h_artifacts_covtype_binary_20000_seed*.npz)."
        )

    detail_rows: list[dict[str, object]] = []
    selected_sets: dict[tuple[int, str], list[set[str]]] = defaultdict(list)

    for seed in seeds:
        art = artifacts[seed]
        x_train_full = np.asarray(art["h_train"], dtype=np.float64)
        y_train_full = _to_binary(np.asarray(art["y_train"]))
        x_test = np.asarray(art["h_test"], dtype=np.float64)
        y_test = _to_binary(np.asarray(art["y_test"]))
        importance = np.asarray(art["importance"], dtype=np.float64)
        rule_key = [str(x) for x in art["rule_key"]]
        full_train_logits = art["full_train_logits"]

        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=0.25,
            random_state=int(seed),
            stratify=y_train_full,
        )
        w_full, b_full = _reconstruct_linear_head(x_train_full, full_train_logits, y_train_full)

        for budget in budgets:
            bp_idx = _budget_prune_indices(importance, int(budget))
            if int(n_bootstrap) > 0:
                st_idx = _stable_bootstrap_indices(
                    x_train=x_tr,
                    y_train=y_tr,
                    importance=importance,
                    budget=int(budget),
                    seed=int(seed),
                    n_bootstrap=int(n_bootstrap),
                    stability_top_k_multiplier=float(stability_top_k_multiplier),
                    candidate_pool=int(forward_candidate_pool),
                )
            else:
                st_idx = _stable_indices_for_heldout(
                    artifacts=artifacts,
                    heldout_seed=int(seed),
                    budget=int(budget),
                    stability_top_k_multiplier=float(stability_top_k_multiplier),
                    std_penalty=float(std_penalty),
                )
            gt_idx = _gate_l1_topk_proxy_indices(x_tr, y_tr, int(budget), int(seed))
            fw_idx = _forward_f1_indices(
                x_train=x_tr,
                y_train=y_tr,
                x_val=x_val,
                y_val=y_val,
                importance=importance,
                budget=int(budget),
                candidate_pool=int(forward_candidate_pool),
                shortlist_per_step=int(forward_shortlist_per_step),
                max_steps=forward_max_steps,
                train_max_rows=forward_train_max_rows,
                seed=int(seed),
            )
            lofo_scores = lofo_f1_scores(
                h_val=x_val,
                y_val=y_val,
                weights=w_full,
                bias=b_full,
                threshold=0.0,
            )
            lofo_idx = top_lofo_f1_indices(lofo_scores, int(budget))
            lofo_bs_scores = bootstrap_lofo_f1_scores(
                h_val=x_val,
                y_val=y_val,
                weights=w_full,
                bias=b_full,
                threshold=0.0,
                budget=int(budget),
                n_bootstrap=max(1, int(n_bootstrap)),
                seed=int(seed),
            )
            lofo_bs_idx = top_bootstrap_lofo_f1_indices(lofo_bs_scores, int(budget))

            for method, idx in (
                ("budget_prune_h_lr", bp_idx),
                ("stable_budget_prune_h_lr", st_idx),
                ("gate_l1_topk_proxy_h_lr", gt_idx),
                ("forward_f1_h_lr", fw_idx),
                ("lofo_f1_h_lr", lofo_idx),
                ("lofo_f1_bootstrap_h_lr", lofo_bs_idx),
            ):
                idx = np.asarray(idx, dtype=np.int64).reshape(-1)
                if idx.size == 0:
                    continue
                f1, roc_auc, pr_auc = _fit_eval_lr(
                    x_train=x_train_full[:, idx],
                    y_train=y_train_full,
                    x_test=x_test[:, idx],
                    y_test=y_test,
                )
                keys = {rule_key[int(i)] for i in idx}
                selected_sets[(int(budget), method)].append(keys)
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "seed": int(seed),
                        "budget": int(budget),
                        "method": method,
                        "selected_rules": int(idx.size),
                        "f1": f1,
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                    }
                )

    summary_rows: list[dict[str, object]] = []
    bucket: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        bucket[(int(row["budget"]), str(row["method"]))].append(row)
    for (budget, method), rows in sorted(bucket.items()):
        f1_vals = [float(r["f1"]) for r in rows]
        roc_vals = [float(r["roc_auc"]) for r in rows]
        pr_vals = [float(r["pr_auc"]) for r in rows]
        sets = selected_sets[(budget, method)]
        summary_rows.append(
            {
                "dataset": dataset,
                "budget": int(budget),
                "method": method,
                "n_seeds": len(rows),
                "f1_mean": float(mean(f1_vals)),
                "f1_std": float(pstdev(f1_vals)) if len(f1_vals) > 1 else 0.0,
                "roc_auc_mean": float(mean(roc_vals)),
                "roc_auc_std": float(pstdev(roc_vals)) if len(roc_vals) > 1 else 0.0,
                "pr_auc_mean": float(mean(pr_vals)),
                "pr_auc_std": float(pstdev(pr_vals)) if len(pr_vals) > 1 else 0.0,
                "subset_jaccard_pairwise_mean": float(_pairwise_mean_jaccard(sets)),
            }
        )

    _write_csv(
        out_detail,
        ["dataset", "seed", "budget", "method", "selected_rules", "f1", "roc_auc", "pr_auc"],
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
            "subset_jaccard_pairwise_mean",
        ],
        summary_rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", type=Path, default=Path("compact_stable_kafn/paper_tables/v18_h_controls_raw_v2"))
    ap.add_argument("--dataset", type=str, default="covtype_binary_20000")
    ap.add_argument("--budgets", type=str, default="100,200,400")
    ap.add_argument("--out-detail", type=Path, default=Path("docs/advanced_pruning_covtype_detail.csv"))
    ap.add_argument("--out-summary", type=Path, default=Path("docs/advanced_pruning_covtype_summary.csv"))
    ap.add_argument("--stability-top-k-multiplier", type=float, default=3.0)
    ap.add_argument("--std-penalty", type=float, default=0.0)
    ap.add_argument("--forward-candidate-pool", type=int, default=240)
    ap.add_argument("--forward-shortlist-per-step", type=int, default=24)
    ap.add_argument("--forward-max-steps", type=int, default=0, help="0 means full budget")
    ap.add_argument("--forward-train-max-rows", type=int, default=4000, help="0 means full train for forward candidate fits.")
    ap.add_argument("--n-bootstrap", type=int, default=0, help="Bootstrap rounds for stability method (0=old LOO mode).")
    args = ap.parse_args()

    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    fwd_steps = int(args.forward_max_steps)
    fwd_rows = int(args.forward_train_max_rows)
    run(
        artifact_dir=args.artifact_dir,
        dataset=args.dataset,
        budgets=budgets,
        out_detail=args.out_detail,
        out_summary=args.out_summary,
        stability_top_k_multiplier=float(args.stability_top_k_multiplier),
        std_penalty=float(args.std_penalty),
        forward_candidate_pool=int(args.forward_candidate_pool),
        forward_shortlist_per_step=int(args.forward_shortlist_per_step),
        forward_max_steps=(None if fwd_steps <= 0 else fwd_steps),
        n_bootstrap=int(args.n_bootstrap),
        forward_train_max_rows=(None if fwd_rows <= 0 else fwd_rows),
    )
    print(f"Wrote {args.out_detail}")
    print(f"Wrote {args.out_summary}")


if __name__ == "__main__":
    main()
