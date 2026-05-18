from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from statistics import mean

import numpy as np

from ruanfis.stable_budget_prune import stable_budget_scores, top_budget_indices


def _read_importance(path: Path) -> dict[int, dict[str, float]]:
    by_seed: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seed = int(row["seed"])
            rule_key = f'{row["source"]}::{row["rule_name"]}'
            by_seed.setdefault(seed, {})[rule_key] = float(row["importance"])
    return by_seed


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _top_rules(values: dict[str, float], budget: int) -> set[str]:
    return {
        key
        for key, _ in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:budget]
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _mean_pairwise_jaccard(sets: list[set[str]]) -> float:
    pairs = list(combinations(sets, 2))
    if not pairs:
        return 1.0
    return float(mean(_jaccard(a, b) for a, b in pairs))


def _importance_matrix(by_seed: dict[int, dict[str, float]], seeds: list[int], rule_universe: list[str]) -> np.ndarray:
    matrix = np.zeros((len(seeds), len(rule_universe)), dtype=np.float64)
    for i, seed in enumerate(seeds):
        values = by_seed[seed]
        for j, rule in enumerate(rule_universe):
            matrix[i, j] = values.get(rule, 0.0)
    return matrix


def run_smoke(
    *,
    importance_csv: Path,
    out_detail: Path,
    out_summary: Path,
    budgets: tuple[int, ...],
    std_penalty: float,
    stability_top_k_multiplier: float,
) -> None:
    by_seed = _read_importance(importance_csv)
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        raise ValueError("Need at least two seeds for stable-selection smoke.")
    rule_universe = sorted({rule for values in by_seed.values() for rule in values})
    full_matrix = _importance_matrix(by_seed, seeds, rule_universe)

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for budget in budgets:
        stability_top_k = min(
            len(rule_universe),
            max(budget, int(round(float(budget) * float(stability_top_k_multiplier)))),
        )
        current_sets = [_top_rules(by_seed[seed], budget) for seed in seeds]
        current_jaccard = _mean_pairwise_jaccard(current_sets)

        stable_sets: list[set[str]] = []
        retention_values: list[float] = []
        agreement_values: list[float] = []
        for heldout_seed in seeds:
            train_seeds = [seed for seed in seeds if seed != heldout_seed]
            train_matrix = _importance_matrix(by_seed, train_seeds, rule_universe)
            scores = stable_budget_scores(
                train_matrix,
                budget=budget,
                stability_top_k=stability_top_k,
                std_penalty=std_penalty,
            )
            stable_indices = top_budget_indices(scores.score, budget)
            stable_rules = {rule_universe[index] for index in stable_indices}
            stable_sets.append(stable_rules)

            heldout_values = by_seed[heldout_seed]
            heldout_top = _top_rules(heldout_values, budget)
            stable_sum = sum(heldout_values.get(rule, 0.0) for rule in stable_rules)
            oracle_sum = sum(heldout_values.get(rule, 0.0) for rule in heldout_top)
            retention = stable_sum / oracle_sum if oracle_sum > 0.0 else 0.0
            agreement = _jaccard(stable_rules, heldout_top)
            retention_values.append(float(retention))
            agreement_values.append(float(agreement))
            detail_rows.append(
                {
                    "budget": int(budget),
                    "heldout_seed": int(heldout_seed),
                    "method": "stable_budget_prune_loo",
                    "stability_top_k": int(stability_top_k),
                    "std_penalty": float(std_penalty),
                    "stability_top_k_multiplier": float(stability_top_k_multiplier),
                    "importance_retention_to_heldout_topb": float(retention),
                    "jaccard_to_heldout_topb": float(agreement),
                    "selected_rules": int(len(stable_rules)),
                }
            )

        stable_jaccard = _mean_pairwise_jaccard(stable_sets)
        global_scores = stable_budget_scores(
            full_matrix,
            budget=budget,
            stability_top_k=stability_top_k,
            std_penalty=std_penalty,
        )
        global_rules = {rule_universe[index] for index in top_budget_indices(global_scores.score, budget)}
        global_retention = mean(
            sum(by_seed[seed].get(rule, 0.0) for rule in global_rules)
            / max(1e-12, sum(by_seed[seed].get(rule, 0.0) for rule in _top_rules(by_seed[seed], budget)))
            for seed in seeds
        )
        summary_rows.append(
            {
                "budget": int(budget),
                "n_seeds": int(len(seeds)),
                "n_rule_universe": int(len(rule_universe)),
                "stability_top_k": int(stability_top_k),
                "current_budget_prune_pairwise_jaccard": float(current_jaccard),
                "stable_loo_pairwise_jaccard": float(stable_jaccard),
                "stable_loo_importance_retention_mean": float(mean(retention_values)),
                "stable_loo_jaccard_to_heldout_topb_mean": float(mean(agreement_values)),
                "stable_global_importance_retention_mean": float(global_retention),
                "std_penalty": float(std_penalty),
                "stability_top_k_multiplier": float(stability_top_k_multiplier),
            }
        )

    _write_csv(
        out_detail,
        [
            "budget",
            "heldout_seed",
            "method",
            "stability_top_k",
            "std_penalty",
            "stability_top_k_multiplier",
            "importance_retention_to_heldout_topb",
            "jaccard_to_heldout_topb",
            "selected_rules",
        ],
        detail_rows,
    )
    _write_csv(
        out_summary,
        [
            "budget",
            "n_seeds",
            "n_rule_universe",
            "stability_top_k",
            "current_budget_prune_pairwise_jaccard",
            "stable_loo_pairwise_jaccard",
            "stable_loo_importance_retention_mean",
            "stable_loo_jaccard_to_heldout_topb_mean",
            "stable_global_importance_retention_mean",
            "std_penalty",
            "stability_top_k_multiplier",
        ],
        summary_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance-csv", type=Path, default=Path("docs/tables_importance_profile.csv"))
    parser.add_argument("--out-detail", type=Path, default=Path("docs/q1_stable_selection_smoke_detail.csv"))
    parser.add_argument("--out-summary", type=Path, default=Path("docs/q1_stable_selection_smoke_summary.csv"))
    parser.add_argument("--budgets", type=str, default="100,200,400")
    parser.add_argument("--std-penalty", type=float, default=0.0)
    parser.add_argument("--stability-top-k-multiplier", type=float, default=1.0)
    args = parser.parse_args()
    budgets = tuple(int(token.strip()) for token in args.budgets.split(",") if token.strip())
    run_smoke(
        importance_csv=args.importance_csv,
        out_detail=args.out_detail,
        out_summary=args.out_summary,
        budgets=budgets,
        std_penalty=float(args.std_penalty),
        stability_top_k_multiplier=float(args.stability_top_k_multiplier),
    )
    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.out_detail}")


if __name__ == "__main__":
    main()
