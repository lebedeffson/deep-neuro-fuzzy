#!/usr/bin/env python3
"""Generate article-ready statistical reports from benchmark *_report.json files.

Outputs:
1) machine-readable JSON report
2) human-readable Markdown report

Statistics:
- Friedman test
- Pairwise Wilcoxon signed-rank tests
- Holm correction
- 95% CI for average rank (t-interval)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


HIGHER_IS_BETTER_METRICS = {
    "f1",
    "accuracy",
    "precision",
    "recall",
    "roc_auc",
    "r2",
}

LOWER_IS_BETTER_METRICS = {
    "rmse",
    "mse",
    "mae",
    "log_loss",
    "brier",
}


@dataclass(frozen=True)
class Block:
    block_id: str
    dataset: str
    metric_name: str
    higher_is_better: bool
    values: dict[str, float]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Defaults to <input_stem>_stats.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Defaults to <input_stem>_stats.md",
    )
    parser.add_argument(
        "--model-family",
        type=str,
        default="ruanfis",
        choices=("ruanfis", "sklearn", "all"),
        help="Which model family to include in tests.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated explicit model list.",
    )
    parser.add_argument(
        "--analysis-level",
        type=str,
        default="both",
        choices=("dataset", "block", "both"),
        help="dataset: aggregated means; block: dataset x seed; both: both analyses.",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def _detect_primary_metric_name(metric_map: dict[str, Any]) -> str:
    for candidate in ("f1", "rmse", "accuracy", "r2", "mse", "mae"):
        if candidate in metric_map:
            return candidate
    return next(iter(metric_map.keys()))


def _is_higher_better(metric_name: str) -> bool:
    if metric_name in HIGHER_IS_BETTER_METRICS:
        return True
    if metric_name in LOWER_IS_BETTER_METRICS:
        return False
    # Conservative fallback for unknown metrics.
    return True


def _collect_dataset_level_blocks(report: dict[str, Any]) -> list[Block]:
    blocks: list[Block] = []
    dataset_results = report["dataset_results"]
    for dataset_name, dataset_payload in dataset_results.items():
        values: dict[str, float] = {}
        metric_name: str | None = None
        higher_is_better: bool | None = None
        for model_payload in dataset_payload["aggregated_results"]:
            test_metrics = model_payload["test_metrics"]
            model_metric_name = _detect_primary_metric_name(test_metrics)
            metric_value = float(test_metrics[model_metric_name]["mean"])
            values[model_payload["model_name"]] = metric_value
            if metric_name is None:
                metric_name = model_metric_name
                higher_is_better = _is_higher_better(metric_name)
        if not values:
            continue
        assert metric_name is not None and higher_is_better is not None
        blocks.append(
            Block(
                block_id=f"{dataset_name}",
                dataset=dataset_name,
                metric_name=metric_name,
                higher_is_better=higher_is_better,
                values=values,
            )
        )
    return blocks


def _collect_block_level_blocks(report: dict[str, Any]) -> list[Block]:
    blocks: list[Block] = []
    dataset_results = report["dataset_results"]
    for dataset_name, dataset_payload in dataset_results.items():
        for seed_payload in dataset_payload["per_seed_results"]:
            seed = int(seed_payload["seed"])
            values: dict[str, float] = {}
            metric_name: str | None = None
            higher_is_better: bool | None = None
            for result_payload in seed_payload["results"]:
                test_metrics = result_payload["test_metrics"]
                model_metric_name = _detect_primary_metric_name(test_metrics)
                metric_value = float(test_metrics[model_metric_name])
                values[result_payload["model_name"]] = metric_value
                if metric_name is None:
                    metric_name = model_metric_name
                    higher_is_better = _is_higher_better(metric_name)
            if not values:
                continue
            assert metric_name is not None and higher_is_better is not None
            blocks.append(
                Block(
                    block_id=f"{dataset_name}/seed={seed}",
                    dataset=dataset_name,
                    metric_name=metric_name,
                    higher_is_better=higher_is_better,
                    values=values,
                )
            )
    return blocks


def _collect_model_families(report: dict[str, Any]) -> dict[str, str]:
    families: dict[str, str] = {}
    for dataset_payload in report["dataset_results"].values():
        for model_payload in dataset_payload["aggregated_results"]:
            families[model_payload["model_name"]] = model_payload["family"]
    return families


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks with ties, rank 1 is best (smallest value)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros_like(values, dtype=float)
    i = 0
    n = values.size
    while i < n:
        j = i + 1
        while j < n and np.isclose(values[order[j]], values[order[i]], rtol=0.0, atol=1e-12):
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _rank_values(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    model_names = list(values.keys())
    arr = np.array([values[m] for m in model_names], dtype=float)
    arr = -arr if higher_is_better else arr
    ranks = _average_ranks(arr)
    return {model: float(rank) for model, rank in zip(model_names, ranks, strict=True)}


def _bootstrap_ci_mean(
    values: np.ndarray,
    confidence: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = values.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        v = float(values[0])
        return v, v
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    alpha_half = (1.0 - confidence) / 2.0
    lo = float(np.quantile(means, alpha_half))
    hi = float(np.quantile(means, 1.0 - alpha_half))
    return lo, hi


def _friedman_statistic(score_matrix: np.ndarray) -> float:
    """Friedman chi-square statistic with tie correction."""
    n, k = score_matrix.shape
    rank_matrix = np.zeros_like(score_matrix, dtype=float)
    tie_term_sum = 0.0
    for i in range(n):
        ranks = _average_ranks(score_matrix[i])
        rank_matrix[i] = ranks
        # tie correction component
        uniq, counts = np.unique(score_matrix[i], return_counts=True)
        for t in counts:
            if t > 1:
                tie_term_sum += float(t**3 - t)
    rank_sums = rank_matrix.sum(axis=0)
    q = (12.0 / (n * k * (k + 1.0))) * float(np.sum(rank_sums**2)) - 3.0 * n * (k + 1.0)
    denom = n * (k**3 - k)
    if denom > 0 and tie_term_sum > 0:
        c = 1.0 - (tie_term_sum / denom)
        if c > 1e-12:
            q /= c
    return float(q)


def _friedman_permutation_p_value(
    score_matrix: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = _friedman_statistic(score_matrix)
    n, k = score_matrix.shape
    exceed = 0
    for _ in range(n_permutations):
        permuted = np.empty_like(score_matrix)
        for i in range(n):
            permuted[i] = score_matrix[i, rng.permutation(k)]
        stat = _friedman_statistic(permuted)
        if stat >= observed - 1e-12:
            exceed += 1
    p_value = (exceed + 1.0) / (n_permutations + 1.0)
    return observed, float(p_value)


def _wilcoxon_signed_rank_statistic(left: np.ndarray, right: np.ndarray) -> tuple[float, np.ndarray]:
    diffs = left - right
    mask = np.abs(diffs) > 1e-12
    diffs = diffs[mask]
    if diffs.size == 0:
        return 0.0, np.array([], dtype=float)
    abs_diffs = np.abs(diffs)
    ranks = _average_ranks(abs_diffs)
    w_plus = float(ranks[diffs > 0].sum())
    w_minus = float(ranks[diffs < 0].sum())
    w_stat = min(w_plus, w_minus)
    signs = np.where(diffs > 0, 1.0, -1.0)
    signed_ranks = ranks * signs
    return w_stat, signed_ranks


def _wilcoxon_permutation_p_value(
    left: np.ndarray,
    right: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    observed, signed_ranks = _wilcoxon_signed_rank_statistic(left, right)
    n = signed_ranks.size
    if n == 0:
        return 0.0, 1.0, 0
    abs_ranks = np.abs(signed_ranks)
    total = float(abs_ranks.sum())

    def w_from_signs(sign_bits: np.ndarray) -> float:
        w_plus = float(abs_ranks[sign_bits > 0].sum())
        return min(w_plus, total - w_plus)

    # Exact enumeration for moderate n.
    if n <= 20:
        m = 1 << n
        extreme = 0
        for mask in range(m):
            signs = np.ones(n, dtype=float)
            for bit in range(n):
                if (mask >> bit) & 1:
                    signs[bit] = -1.0
            if w_from_signs(signs) <= observed + 1e-12:
                extreme += 1
        p_val = extreme / m
        return observed, float(p_val), int(n)

    extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=n, replace=True)
        if w_from_signs(signs) <= observed + 1e-12:
            extreme += 1
    p_val = (extreme + 1.0) / (n_permutations + 1.0)
    return observed, float(p_val), int(n)


def _holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted_sorted = [0.0] * m
    prev = 0.0
    for rank, original_idx in enumerate(order, start=1):
        raw = p_values[original_idx]
        adj = min(1.0, raw * (m - rank + 1))
        adj = max(adj, prev)
        adjusted_sorted[rank - 1] = adj
        prev = adj
    adjusted = [0.0] * m
    for sorted_pos, original_idx in enumerate(order):
        adjusted[original_idx] = adjusted_sorted[sorted_pos]
    return adjusted


def _run_analysis(
    blocks: list[Block],
    selected_models: list[str],
    confidence: float,
    alpha: float,
    n_permutations: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    # Keep only complete blocks where all selected models are present.
    complete_blocks = [b for b in blocks if all(m in b.values for m in selected_models)]
    if len(complete_blocks) < 2 or len(selected_models) < 3:
        raise ValueError("Need at least 2 complete blocks and 3 models for Friedman analysis.")

    oriented_scores_by_model: dict[str, list[float]] = {m: [] for m in selected_models}
    ranks_by_model: dict[str, list[float]] = {m: [] for m in selected_models}

    per_block_rank_maps: list[dict[str, float]] = []
    for block in complete_blocks:
        rank_map = _rank_values(
            {m: block.values[m] for m in selected_models},
            higher_is_better=block.higher_is_better,
        )
        per_block_rank_maps.append(rank_map)
        for model_name in selected_models:
            raw_value = float(block.values[model_name])
            oriented = raw_value if block.higher_is_better else -raw_value
            oriented_scores_by_model[model_name].append(oriented)
            ranks_by_model[model_name].append(rank_map[model_name])

    score_matrix = np.column_stack(
        [np.asarray(oriented_scores_by_model[m], dtype=float) for m in selected_models]
    )
    friedman_stat, friedman_p = _friedman_permutation_p_value(
        score_matrix=score_matrix,
        n_permutations=n_permutations,
        rng=rng,
    )

    rank_summary: list[dict[str, Any]] = []
    for model_name in selected_models:
        rank_values = np.asarray(ranks_by_model[model_name], dtype=float)
        avg_rank = float(rank_values.mean())
        std_rank = float(rank_values.std(ddof=1)) if rank_values.size > 1 else 0.0
        ci_low, ci_high = _bootstrap_ci_mean(
            rank_values,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        wins = int(sum(1 for rank_map in per_block_rank_maps if abs(rank_map[model_name] - 1.0) < 1e-12))
        rank_summary.append(
            {
                "model": model_name,
                "avg_rank": avg_rank,
                "std_rank": std_rank,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wins": wins,
            }
        )
    rank_summary.sort(key=lambda item: item["avg_rank"])

    pair_records: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    pairs = list(itertools.combinations(selected_models, 2))
    for model_a, model_b in pairs:
        a = np.asarray(ranks_by_model[model_a], dtype=float)
        b = np.asarray(ranks_by_model[model_b], dtype=float)
        diffs = a - b
        nonzero = np.count_nonzero(np.abs(diffs) > 1e-12)
        if nonzero == 0:
            p_val = 1.0
            stat_val = 0.0
        else:
            stat_val, p_val, n_nonzero = _wilcoxon_permutation_p_value(
                a,
                b,
                n_permutations=n_permutations,
                rng=rng,
            )
        raw_p_values.append(p_val)
        mean_rank_a = float(a.mean())
        mean_rank_b = float(b.mean())
        better = model_a if mean_rank_a < mean_rank_b else model_b
        pair_records.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_blocks": int(a.size),
                "wilcoxon_stat": stat_val,
                "p_value": p_val,
                "n_nonzero_pairs": int(nonzero),
                "mean_rank_a": mean_rank_a,
                "mean_rank_b": mean_rank_b,
                "better_model_by_avg_rank": better,
            }
        )

    holm_p = _holm_adjust(raw_p_values)
    for rec, p_adj in zip(pair_records, holm_p, strict=True):
        rec["p_value_holm"] = float(p_adj)
        rec["reject_h0_alpha"] = bool(p_adj < alpha)

    # Helpful totals
    dataset_names = sorted({b.dataset for b in complete_blocks})
    metric_map = {b.block_id: b.metric_name for b in complete_blocks}

    return {
        "n_blocks": len(complete_blocks),
        "n_datasets": len(dataset_names),
        "n_models": len(selected_models),
        "datasets": dataset_names,
        "block_metric_map": metric_map,
        "friedman": {
            "chi2": friedman_stat,
            "p_value": friedman_p,
            "alpha": alpha,
            "reject_h0_alpha": bool(friedman_p < alpha),
            "p_value_method": f"permutation({n_permutations})",
        },
        "rank_summary": rank_summary,
        "pairwise_wilcoxon_holm": pair_records,
    }


def _render_rank_table(rank_summary: list[dict[str, Any]]) -> str:
    lines = [
        "| model | avg_rank | std_rank | ci_low | ci_high | wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rank_summary:
        lines.append(
            "| {model} | {avg_rank:.4f} | {std_rank:.4f} | {ci_low:.4f} | {ci_high:.4f} | {wins} |".format(
                **row
            )
        )
    return "\n".join(lines)


def _render_pairwise_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| model_a | model_b | n_blocks | wilcoxon_stat | p_value | p_holm | reject@0.05 | better_by_rank |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for rec in records:
        lines.append(
            "| {model_a} | {model_b} | {n_blocks} | {wilcoxon_stat:.4f} | {p_value:.6f} | {p_value_holm:.6f} | {reject_h0_alpha} | {better_model_by_avg_rank} |".format(
                **rec
            )
        )
    return "\n".join(lines)


def _render_markdown(
    *,
    input_json: Path,
    selected_models: list[str],
    family_filter: str,
    confidence: float,
    alpha: float,
    n_permutations: int,
    n_bootstrap: int,
    random_seed: int,
    dataset_level: dict[str, Any] | None,
    block_level: dict[str, Any] | None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = []
    lines.append("# Statistical Report From Benchmark JSON")
    lines.append("")
    lines.append(f"- generated_utc: `{now}`")
    lines.append(f"- source: `{input_json}`")
    lines.append(f"- model_family_filter: `{family_filter}`")
    lines.append(f"- selected_models: `{', '.join(selected_models)}`")
    lines.append(f"- confidence: `{confidence}`")
    lines.append(f"- alpha: `{alpha}`")
    lines.append(f"- p_value_method: `permutation`")
    lines.append(f"- n_permutations: `{n_permutations}`")
    lines.append(f"- bootstrap_samples_for_CI: `{n_bootstrap}`")
    lines.append(f"- random_seed: `{random_seed}`")
    lines.append("")

    def append_section(title: str, payload: dict[str, Any]) -> None:
        fr = payload["friedman"]
        lines.append(f"## {title}")
        lines.append("")
        lines.append(
            "- blocks: `{n_blocks}`; datasets: `{n_datasets}`; models: `{n_models}`".format(
                n_blocks=payload["n_blocks"],
                n_datasets=payload["n_datasets"],
                n_models=payload["n_models"],
            )
        )
        lines.append(
            "- Friedman: `chi2={chi2:.6f}`, `p={p_value:.6f}`, `reject@{alpha}={reject}`".format(
                chi2=fr["chi2"],
                p_value=fr["p_value"],
                alpha=fr["alpha"],
                reject=fr["reject_h0_alpha"],
            )
        )
        lines.append("")
        lines.append("### Rank Summary (with CI)")
        lines.append("")
        lines.append(_render_rank_table(payload["rank_summary"]))
        lines.append("")
        lines.append("### Pairwise Wilcoxon + Holm")
        lines.append("")
        lines.append(_render_pairwise_table(payload["pairwise_wilcoxon_holm"]))
        lines.append("")

    if dataset_level is not None:
        append_section("Dataset-Level Analysis (aggregated means)", dataset_level)
    if block_level is not None:
        append_section("Block-Level Analysis (dataset x seed)", block_level)

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.random_seed)
    with args.input_json.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    output_json = args.output_json or args.input_json.with_name(f"{args.input_json.stem}_stats.json")
    output_md = args.output_md or args.input_json.with_name(f"{args.input_json.stem}_stats.md")

    model_families = _collect_model_families(report)
    explicit_models = [item.strip() for item in args.models.split(",") if item.strip()]

    dataset_blocks = _collect_dataset_level_blocks(report)
    block_blocks = _collect_block_level_blocks(report)

    if not dataset_blocks and not block_blocks:
        raise RuntimeError("No blocks extracted from input JSON.")

    # Build candidate model set from all blocks.
    all_models = set()
    for block in [*dataset_blocks, *block_blocks]:
        all_models.update(block.values.keys())

    if args.model_family != "all":
        all_models = {m for m in all_models if model_families.get(m) == args.model_family}

    if explicit_models:
        explicit_set = set(explicit_models)
        missing = sorted(explicit_set - all_models)
        if missing:
            raise ValueError(f"Explicit models not available after filtering: {missing}")
        selected_models = [m for m in explicit_models if m in explicit_set]
    else:
        selected_models = sorted(all_models)

    if len(selected_models) < 3:
        raise ValueError(
            f"Need at least 3 models for Friedman analysis, got {len(selected_models)}: {selected_models}"
        )

    dataset_level_payload: dict[str, Any] | None = None
    block_level_payload: dict[str, Any] | None = None

    if args.analysis_level in {"dataset", "both"}:
        dataset_level_payload = _run_analysis(
            dataset_blocks,
            selected_models=selected_models,
            confidence=args.confidence,
            alpha=args.alpha,
            n_permutations=args.n_permutations,
            n_bootstrap=args.bootstrap_samples,
            rng=rng,
        )

    if args.analysis_level in {"block", "both"}:
        block_level_payload = _run_analysis(
            block_blocks,
            selected_models=selected_models,
            confidence=args.confidence,
            alpha=args.alpha,
            n_permutations=args.n_permutations,
            n_bootstrap=args.bootstrap_samples,
            rng=rng,
        )

    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_report": str(args.input_json),
        "model_family_filter": args.model_family,
        "selected_models": selected_models,
        "confidence": args.confidence,
        "alpha": args.alpha,
        "n_permutations": args.n_permutations,
        "bootstrap_samples": args.bootstrap_samples,
        "random_seed": args.random_seed,
        "dataset_level": dataset_level_payload,
        "block_level": block_level_payload,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    markdown = _render_markdown(
        input_json=args.input_json,
        selected_models=selected_models,
        family_filter=args.model_family,
        confidence=args.confidence,
        alpha=args.alpha,
        n_permutations=args.n_permutations,
        n_bootstrap=args.bootstrap_samples,
        random_seed=args.random_seed,
        dataset_level=dataset_level_payload,
        block_level=block_level_payload,
    )
    output_md.write_text(markdown, encoding="utf-8")

    print(f"[ok] source={args.input_json}")
    print(f"[ok] json={output_json}")
    print(f"[ok] md={output_md}")


if __name__ == "__main__":
    main()
