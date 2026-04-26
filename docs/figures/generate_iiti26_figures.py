#!/usr/bin/env python3
"""Generate paper-grade figures for IITI'26 article from canonical article metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
ARTICLE_METRICS_JSON = ROOT / "article_current" / "article_metrics.json"
OUT_DIR = Path(__file__).resolve().parent


MODELS = [
    "ruanfis_shallow",
    "ruanfis_stacked_anfis",
    "ruanfis_hierarchical_anfis",
    "ruanfis_refined_deep",
]

MODEL_LABELS = {
    "ruanfis_shallow": "Shallow Fuzzy",
    "ruanfis_stacked_anfis": "Stacked ANFIS",
    "ruanfis_hierarchical_anfis": "Hierarchical ANFIS",
    "ruanfis_refined_deep": "DFFL",
}

MODEL_COLORS = {
    "ruanfis_shallow": "#4E79A7",
    "ruanfis_stacked_anfis": "#F28E2B",
    "ruanfis_hierarchical_anfis": "#59A14F",
    "ruanfis_refined_deep": "#E15759",
}

REG_DATASETS = ["diabetes", "linnerud_weight"]
CLF_DATASETS = ["breast_cancer", "wine_binary", "digits_binary"]


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_metrics(metrics: dict) -> None:
    if "main_block_quality" not in metrics or "main_block_complexity_stability" not in metrics:
        raise KeyError("Metrics JSON must contain 'main_block_quality' and 'main_block_complexity_stability'.")

    quality = metrics["main_block_quality"]
    complexity = metrics["main_block_complexity_stability"]
    required_datasets = REG_DATASETS + CLF_DATASETS

    for model in MODELS:
        if model not in quality:
            raise KeyError(f"Missing model '{model}' in main_block_quality.")
        if model not in complexity:
            raise KeyError(f"Missing model '{model}' in main_block_complexity_stability.")
        for dataset in required_datasets:
            if dataset not in quality[model]:
                raise KeyError(f"Missing dataset '{dataset}' for model '{model}' in main_block_quality.")
        for key in ("avg_rules", "active_rule_jaccard"):
            if key not in complexity[model]:
                raise KeyError(f"Missing key '{key}' for model '{model}' in main_block_complexity_stability.")


def extract_quality(metrics: dict) -> dict[str, dict[str, float]]:
    quality: dict[str, dict[str, float]] = {m: {} for m in MODELS}
    block = metrics["main_block_quality"]
    for m in MODELS:
        for d in REG_DATASETS + CLF_DATASETS:
            quality[m][d] = float(block[m][d])
    return quality


def extract_complexity_stability(metrics: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    block = metrics["main_block_complexity_stability"]
    for m in MODELS:
        out[m] = {
            "avg_total_rules": float(block[m]["avg_rules"]),
            "avg_active_rule_jaccard": float(block[m]["active_rule_jaccard"]),
        }
    return out


def _ranks_with_ties(values: list[float], higher_is_better: bool) -> list[float]:
    paired = list(enumerate(values))
    paired.sort(key=lambda x: x[1], reverse=higher_is_better)
    ranks = [0.0] * len(values)

    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][1] == paired[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[paired[k][0]] = avg_rank
        i = j + 1
    return ranks


def compute_avg_rank_and_wins(quality: dict[str, dict[str, float]]) -> tuple[dict[str, float], dict[str, int]]:
    ranks: dict[str, list[float]] = {m: [] for m in MODELS}
    wins: dict[str, int] = {m: 0 for m in MODELS}
    all_datasets = REG_DATASETS + CLF_DATASETS

    for d in all_datasets:
        values = [quality[m][d] for m in MODELS]
        higher_is_better = d in CLF_DATASETS
        r = _ranks_with_ties(values, higher_is_better=higher_is_better)
        for idx, m in enumerate(MODELS):
            ranks[m].append(r[idx])

        if higher_is_better:
            best_value = max(values)
        else:
            best_value = min(values)
        tied_winners = [m for m in MODELS if quality[m][d] == best_value]
        # Keep integer "wins": count one win for the first tied model in canonical order.
        wins[tied_winners[0]] += 1

    avg_rank = {m: float(np.mean(v)) for m, v in ranks.items()}
    return avg_rank, wins


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_fig(fig: plt.Figure, filename: str, out_dir: Path) -> None:
    out = out_dir / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    plt.close(fig)


def draw_box_column(ax: plt.Axes, x_center: float, title: str, blocks: list[str]) -> None:
    ax.text(x_center, 0.90, title, ha="center", va="bottom", fontsize=14, fontweight="semibold")
    y_positions = np.linspace(0.80, 0.14, len(blocks))
    width, height = 0.21, 0.10
    for i, (txt, y) in enumerate(zip(blocks, y_positions)):
        rect = patches.FancyBboxPatch(
            (x_center - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=1.2,
            edgecolor="#4A4A4A",
            facecolor="#F5F7FA",
        )
        ax.add_patch(rect)
        ax.text(x_center, y, txt, ha="center", va="center", fontsize=12.5)
        if i < len(blocks) - 1:
            y_next = y_positions[i + 1]
            ax.annotate(
                "",
                xy=(x_center, y_next + height / 2 + 0.004),
                xytext=(x_center, y - height / 2 - 0.004),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#3C3C3C"),
            )


def make_fig1_architectures(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box_column(ax, 0.17, "Stacked ANFIS", ["Input", "ANFIS L1", "ANFIS L2", "ANFIS L3", "Output"])
    draw_box_column(ax, 0.50, "Hierarchical ANFIS", ["Input", "Local ANFIS Blocks", "Aggregation ANFIS", "Decision", "Output"])
    draw_box_column(ax, 0.83, "DFFL", ["Input", "Fuzzy Feature Layer", "Fuzzy Representation Layer", "Sugeno Decision", "Output"])

    ax.set_title("Compared Architecture Families", pad=12)
    fig.tight_layout()
    save_fig(fig, "fig1_architectures.png", out_dir)


def make_fig2_quality(quality: dict[str, dict[str, float]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=False)

    x_reg = np.arange(len(REG_DATASETS))
    x_clf = np.arange(len(CLF_DATASETS))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(MODELS))

    for off, m in zip(offsets, MODELS):
        axes[0].bar(
            x_reg + off,
            [quality[m][d] for d in REG_DATASETS],
            width=width,
            color=MODEL_COLORS[m],
            label=MODEL_LABELS[m],
        )
        axes[1].bar(
            x_clf + off,
            [quality[m][d] for d in CLF_DATASETS],
            width=width,
            color=MODEL_COLORS[m],
            label=MODEL_LABELS[m],
        )

    axes[0].set_title("Regression")
    axes[0].set_ylabel("RMSE (lower is better)")
    axes[0].set_xticks(x_reg)
    axes[0].set_xticklabels(REG_DATASETS, rotation=12)

    axes[1].set_title("Binary Classification")
    axes[1].set_ylabel("F1 (higher is better)")
    axes[1].set_xticks(x_clf)
    axes[1].set_xticklabels(CLF_DATASETS, rotation=12)
    axes[1].set_ylim(0.84, 1.01)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Predictive Quality by Dataset (Main Comparison Block)", y=1.08, fontsize=16)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    save_fig(fig, "fig2_quality_by_dataset.png", out_dir)


def make_fig3_rank_wins(avg_rank: dict[str, float], wins: dict[str, int], out_dir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10.2, 4.8))
    x = np.arange(len(MODELS))
    labels = [MODEL_LABELS[m] for m in MODELS]
    ranks = [avg_rank[m] for m in MODELS]
    wins_values = [wins[m] for m in MODELS]
    colors = [MODEL_COLORS[m] for m in MODELS]

    bars = ax1.bar(x, ranks, color=colors, alpha=0.88, label="Avg Rank")
    ax1.set_ylabel("Avg Rank (lower is better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=9)
    ax1.set_ylim(0, 3.15)
    ax1.set_title("Integrated Quality: Average Rank and Dataset Wins")

    for b, v in zip(bars, ranks):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5)

    ax2 = ax1.twinx()
    ax2.plot(x, wins_values, color="#212121", marker="o", linewidth=2.2, label="Wins")
    ax2.set_ylabel("Wins (number of datasets)")
    ax2.set_ylim(0, max(wins_values) + 1)

    ax1.grid(True, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig3_rank_and_wins.png", out_dir)


def make_fig4_complexity_stability(stats: dict[str, dict[str, float]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=False)

    x = np.arange(len(MODELS))
    labels = [MODEL_LABELS[m] for m in MODELS]
    colors = [MODEL_COLORS[m] for m in MODELS]

    total_rules = [stats[m]["avg_total_rules"] for m in MODELS]
    jaccard = [stats[m]["avg_active_rule_jaccard"] for m in MODELS]

    axes[0].bar(x, total_rules, width=0.58, color="#6BA6C9", label="Avg Rules")
    axes[0].set_title("Structural Complexity")
    axes[0].set_ylabel("Number of Rules")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    for i, v in enumerate(total_rules):
        axes[0].text(i, v + 1.2, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    bars = axes[1].bar(x, jaccard, color=colors)
    axes[1].set_title("Interpretation Stability")
    axes[1].set_ylabel("Avg Active Rule Jaccard")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10)
    axes[1].set_ylim(0, max(jaccard) * 1.35)
    for b, v in zip(bars, jaccard):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10.5)

    fig.suptitle("Rule-Base Complexity and Stability", y=1.03, fontsize=16)
    fig.tight_layout()
    save_fig(fig, "fig4_complexity_stability.png", out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=ARTICLE_METRICS_JSON,
        help="Path to canonical article metrics JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    metrics = load_metrics(args.metrics_json)
    validate_metrics(metrics)
    quality = extract_quality(metrics)
    complexity_stability = extract_complexity_stability(metrics)
    avg_rank, wins = compute_avg_rank_and_wins(quality)

    make_fig1_architectures(args.out_dir)
    make_fig2_quality(quality, args.out_dir)
    make_fig3_rank_wins(avg_rank, wins, args.out_dir)
    make_fig4_complexity_stability(complexity_stability, args.out_dir)
    print("Generated:", ", ".join(str(p.name) for p in sorted(args.out_dir.glob("fig*.png"))))


if __name__ == "__main__":
    main()
