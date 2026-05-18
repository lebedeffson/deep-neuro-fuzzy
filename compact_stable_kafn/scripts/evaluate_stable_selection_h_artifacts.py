from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ruanfis.stable_budget_prune import stable_budget_scores, top_budget_indices


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_artifacts(artifact_dir: Path, dataset: str) -> dict[int, dict[str, object]]:
    artifacts: dict[int, dict[str, object]] = {}
    pattern = f"v18_h_artifacts_{dataset}_seed*.npz"
    for path in sorted(artifact_dir.glob(pattern)):
        payload = np.load(path)
        seed_text = path.stem.rsplit("seed", maxsplit=1)[-1]
        seed = int(seed_text)
        artifacts[seed] = {
            "path": path,
            "h_train": payload["h_train"],
            "h_test": payload["h_test"],
            "y_train": payload["y_train"],
            "y_test": payload["y_test"],
            "full_test_prob": payload["full_test_prob"] if "full_test_prob" in payload else _sigmoid(payload["full_test_logits"]),
            "rule_key": payload["rule_key"].astype(str),
            "importance": payload["importance"].astype(np.float64),
            "threshold": float(payload["classification_threshold"][0]) if "classification_threshold" in payload else 0.5,
        }
    return artifacts


def _importance_over_universe(artifact: dict[str, object], universe: list[str]) -> np.ndarray:
    keys = [str(x) for x in artifact["rule_key"]]
    values = np.asarray(artifact["importance"], dtype=np.float64)
    mapping = {key: float(values[i]) for i, key in enumerate(keys)}
    return np.asarray([mapping.get(key, 0.0) for key in universe], dtype=np.float64)


def _budget_prune_keys(artifact: dict[str, object], budget: int) -> list[str]:
    keys = [str(x) for x in artifact["rule_key"]]
    importance = np.asarray(artifact["importance"], dtype=np.float64)
    order = np.argsort(-importance, kind="stable")[:budget]
    return [keys[int(i)] for i in order]


def _columns_for_keys(artifact: dict[str, object], selected_keys: list[str], budget: int) -> tuple[list[int], int]:
    keys = [str(x) for x in artifact["rule_key"]]
    mapping = {key: i for i, key in enumerate(keys)}
    columns: list[int] = []
    missing = 0
    for key in selected_keys:
        if key in mapping:
            columns.append(int(mapping[key]))
        else:
            missing += 1
    if len(columns) < budget:
        importance = np.asarray(artifact["importance"], dtype=np.float64)
        selected = set(columns)
        for index in np.argsort(-importance, kind="stable"):
            idx = int(index)
            if idx in selected:
                continue
            columns.append(idx)
            selected.add(idx)
            if len(columns) >= budget:
                break
    return columns[:budget], missing


def _evaluate_lr_selection(
    artifact: dict[str, object],
    selected_keys: list[str],
    budget: int,
) -> tuple[dict[str, float], set[str], int]:
    cols, missing = _columns_for_keys(artifact, selected_keys, budget)
    h_train = np.asarray(artifact["h_train"])[:, cols]
    h_test = np.asarray(artifact["h_test"])[:, cols]
    y_train = (np.asarray(artifact["y_train"]).reshape(-1) >= 0.5).astype(np.int64)
    y_test = (np.asarray(artifact["y_test"]).reshape(-1) >= 0.5).astype(np.int64)

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(h_train, y_train)
    prob = model.predict_proba(h_test)[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    full_prob = np.asarray(artifact["full_test_prob"], dtype=np.float64).reshape(-1)
    full_label = (full_prob >= float(artifact["threshold"])).astype(np.int64)
    metrics = {
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "fidelity_l1_to_full_prob": float(np.mean(np.abs(prob - full_prob))),
        "agreement_to_full": float(np.mean(pred == full_label)),
    }
    keys = [str(x) for x in artifact["rule_key"]]
    selected_present = {keys[int(i)] for i in cols}
    return metrics, selected_present, missing


def run(
    *,
    artifact_dir: Path,
    dataset: str,
    budgets: tuple[int, ...],
    out_detail: Path,
    out_summary: Path,
    stability_top_k_multiplier: float,
    std_penalty: float,
) -> None:
    artifacts = _load_artifacts(artifact_dir, dataset)
    if len(artifacts) < 2:
        raise ValueError(f"Need at least two H artifacts for {dataset}; found {len(artifacts)} in {artifact_dir}.")
    seeds = sorted(artifacts)
    universe = sorted({str(key) for artifact in artifacts.values() for key in artifact["rule_key"]})

    detail_rows: list[dict[str, object]] = []
    for budget in budgets:
        stability_top_k = min(
            len(universe),
            max(int(budget), int(round(float(budget) * float(stability_top_k_multiplier)))),
        )
        for heldout_seed in seeds:
            heldout = artifacts[heldout_seed]
            train_runs = [
                _importance_over_universe(artifacts[seed], universe)
                for seed in seeds
                if seed != heldout_seed
            ]
            scores = stable_budget_scores(
                train_runs,
                budget=int(budget),
                stability_top_k=stability_top_k,
                std_penalty=float(std_penalty),
            )
            stable_keys = [universe[int(index)] for index in top_budget_indices(scores.score, int(budget))]
            budget_prune_keys = _budget_prune_keys(heldout, int(budget))

            for method, selected_keys in (
                ("budget_prune_h_lr", budget_prune_keys),
                ("stable_budget_prune_h_lr", stable_keys),
            ):
                metrics, selected_present, missing = _evaluate_lr_selection(heldout, selected_keys, int(budget))
                bp_set = set(budget_prune_keys)
                jaccard_to_bp = len(selected_present & bp_set) / max(1, len(selected_present | bp_set))
                detail_rows.append(
                    {
                        "dataset": dataset,
                        "budget": int(budget),
                        "heldout_seed": int(heldout_seed),
                        "method": method,
                        "n_features": int(budget),
                        "missing_selected_rules": int(missing),
                        "f1": metrics["f1"],
                        "roc_auc": metrics["roc_auc"],
                        "pr_auc": metrics["pr_auc"],
                        "fidelity_l1_to_full_prob": metrics["fidelity_l1_to_full_prob"],
                        "agreement_to_full": metrics["agreement_to_full"],
                        "jaccard_to_budget_prune": float(jaccard_to_bp),
                        "stability_top_k": int(stability_top_k),
                        "std_penalty": float(std_penalty),
                    }
                )

    summary_bucket: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        summary_bucket[(int(row["budget"]), str(row["method"]))].append(row)
    summary_rows: list[dict[str, object]] = []
    for (budget, method), rows in sorted(summary_bucket.items()):
        summary: dict[str, object] = {
            "dataset": dataset,
            "budget": int(budget),
            "method": method,
            "n_seeds": len(rows),
        }
        for metric in ("f1", "roc_auc", "pr_auc", "fidelity_l1_to_full_prob", "agreement_to_full", "jaccard_to_budget_prune"):
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = float(mean(values))
            summary[f"{metric}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
        summary["missing_selected_rules_total"] = int(sum(int(row["missing_selected_rules"]) for row in rows))
        summary_rows.append(summary)

    _write_csv(
        out_detail,
        [
            "dataset",
            "budget",
            "heldout_seed",
            "method",
            "n_features",
            "missing_selected_rules",
            "f1",
            "roc_auc",
            "pr_auc",
            "fidelity_l1_to_full_prob",
            "agreement_to_full",
            "jaccard_to_budget_prune",
            "stability_top_k",
            "std_penalty",
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
            "fidelity_l1_to_full_prob_mean",
            "fidelity_l1_to_full_prob_std",
            "agreement_to_full_mean",
            "agreement_to_full_std",
            "jaccard_to_budget_prune_mean",
            "jaccard_to_budget_prune_std",
            "missing_selected_rules_total",
        ],
        summary_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("compact_stable_kafn/paper_tables/v18_controls_raw"))
    parser.add_argument("--dataset", type=str, default="covtype_binary_20000")
    parser.add_argument("--budgets", type=str, default="400")
    parser.add_argument("--out-detail", type=Path, default=Path("docs/tables_stable_h_selection_detail.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("docs/tables_stable_h_selection_summary.csv"))
    parser.add_argument("--stability-top-k-multiplier", type=float, default=3.0)
    parser.add_argument("--std-penalty", type=float, default=0.0)
    args = parser.parse_args()
    budgets = tuple(int(token.strip()) for token in args.budgets.split(",") if token.strip())
    run(
        artifact_dir=args.artifact_dir,
        dataset=args.dataset,
        budgets=budgets,
        out_detail=args.out_detail,
        out_summary=args.out_summary,
        stability_top_k_multiplier=float(args.stability_top_k_multiplier),
        std_penalty=float(args.std_penalty),
    )
    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.out_detail}")


if __name__ == "__main__":
    main()
