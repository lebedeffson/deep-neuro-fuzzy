#!/usr/bin/env python3
"""Generate paper-grade figures for IITI'26 article from benchmark JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_JSON = ROOT / "artifacts" / "ruanfis_paper5x3_report.json"
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


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def get_dataset_model_record(report: dict, dataset: str, model: str) -> dict:
    for rec in report["dataset_results"][dataset]["aggregated_results"]:
        if rec["model_name"] == model:
            return rec
    raise KeyError(f"Model '{model}' not found for dataset '{dataset}'")


def extract_quality(report: dict) -> dict[str, dict[str, float]]:
    quality: dict[str, dict[str, float]] = {m: {} for m in MODELS}
    for d in report["datasets"]:
        for m in MODELS:
            rec = get_dataset_model_record(report, d, m)
            if d in REG_DATASETS:
                quality[m][d] = float(rec["test_metrics"]["rmse"]["mean"])
            else:
                quality[m][d] = float(rec["test_metrics"]["f1"]["mean"])
    return quality


def extract_complexity_stability(report: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for m in MODELS:
        totals = []
        actives = []
        jaccards = []
        for d in report["datasets"]:
            rec = get_dataset_model_record(report, d, m)
            totals.append(float(rec["structural_metrics"]["total_rules"]["mean"]))
            actives.append(float(rec["structural_metrics"]["active_rules"]["mean"]))
            jaccards.append(float(rec["stability_metrics"]["active_rule_jaccard"]))
        out[m] = {
            "avg_total_rules": float(np.mean(totals)),
            "avg_active_rules": float(np.mean(actives)),
            "avg_active_rule_jaccard": float(np.mean(jaccards)),
        }
    return out


def compute_avg_rank_and_wins(quality: dict[str, dict[str, float]]) -> tuple[dict[str, float], dict[str, int]]:
    ranks: dict[str, list[int]] = {m: [] for m in MODELS}
    wins: dict[str, int] = {m: 0 for m in MODELS}
    all_datasets = REG_DATASETS + CLF_DATASETS

    for d in all_datasets:
        if d in REG_DATASETS:
            ordered = sorted(MODELS, key=lambda m: quality[m][d])
        else:
            ordered = sorted(MODELS, key=lambda m: quality[m][d], reverse=True)
        for idx, m in enumerate(ordered, start=1):
            ranks[m].append(idx)
        wins[ordered[0]] += 1

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


def save_fig(fig: plt.Figure, filename: str) -> None:
    out = OUT_DIR / filename
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


def make_fig1_architectures() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box_column(ax, 0.17, "Stacked ANFIS", ["Input", "ANFIS L1", "ANFIS L2", "ANFIS L3", "Output"])
    draw_box_column(ax, 0.50, "Hierarchical ANFIS", ["Input", "Local ANFIS Blocks", "Aggregation ANFIS", "Decision", "Output"])
    draw_box_column(ax, 0.83, "DFFL", ["Input", "Fuzzy Feature Layer", "Fuzzy Representation Layer", "Sugeno Decision", "Output"])

    ax.set_title("Сравниваемые архитектурные линии", pad=12)
    fig.tight_layout()
    save_fig(fig, "fig1_architectures.png")


def make_fig2_quality(quality: dict[str, dict[str, float]]) -> None:
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

    axes[0].set_title("Регрессия")
    axes[0].set_ylabel("RMSE (меньше лучше)")
    axes[0].set_xticks(x_reg)
    axes[0].set_xticklabels(REG_DATASETS, rotation=12)

    axes[1].set_title("Бинарная классификация")
    axes[1].set_ylabel("F1 (больше лучше)")
    axes[1].set_xticks(x_clf)
    axes[1].set_xticklabels(CLF_DATASETS, rotation=12)
    axes[1].set_ylim(0.84, 1.01)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Качество моделей по датасетам (paper-grade run)", y=1.08, fontsize=16)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    save_fig(fig, "fig2_quality_by_dataset.png")


def make_fig3_rank_wins(avg_rank: dict[str, float], wins: dict[str, int]) -> None:
    fig, ax1 = plt.subplots(figsize=(10.2, 4.8))
    x = np.arange(len(MODELS))
    labels = [MODEL_LABELS[m] for m in MODELS]
    ranks = [avg_rank[m] for m in MODELS]
    wins_values = [wins[m] for m in MODELS]
    colors = [MODEL_COLORS[m] for m in MODELS]

    bars = ax1.bar(x, ranks, color=colors, alpha=0.88, label="Avg Rank")
    ax1.set_ylabel("Avg Rank (меньше лучше)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=9)
    ax1.set_ylim(0, 3.15)
    ax1.set_title("Интегральное качество: средний ранг и число побед")

    for b, v in zip(bars, ranks):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=10.5)

    ax2 = ax1.twinx()
    ax2.plot(x, wins_values, color="#212121", marker="o", linewidth=2.2, label="Wins")
    ax2.set_ylabel("Wins (число датасетов)")
    ax2.set_ylim(0, max(wins_values) + 1)

    ax1.grid(True, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig3_rank_and_wins.png")


def make_fig4_complexity_stability(stats: dict[str, dict[str, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=False)

    x = np.arange(len(MODELS))
    labels = [MODEL_LABELS[m] for m in MODELS]
    colors = [MODEL_COLORS[m] for m in MODELS]

    total_rules = [stats[m]["avg_total_rules"] for m in MODELS]
    active_rules = [stats[m]["avg_active_rules"] for m in MODELS]
    jaccard = [stats[m]["avg_active_rule_jaccard"] for m in MODELS]

    w = 0.35
    axes[0].bar(x - w / 2, total_rules, width=w, color="#6BA6C9", label="Avg Total Rules")
    axes[0].bar(x + w / 2, active_rules, width=w, color="#F98D3C", label="Avg Active Rules")
    axes[0].set_title("Структурная сложность")
    axes[0].set_ylabel("Количество правил")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=10)
    axes[0].legend(frameon=False)

    bars = axes[1].bar(x, jaccard, color=colors)
    axes[1].set_title("Стабильность интерпретации")
    axes[1].set_ylabel("Avg Active Rule Jaccard")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10)
    axes[1].set_ylim(0, max(jaccard) * 1.35)
    for b, v in zip(bars, jaccard):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10.5)

    fig.suptitle("Сложность и стабильность rule base", y=1.03, fontsize=16)
    fig.tight_layout()
    save_fig(fig, "fig4_complexity_stability.png")


def main() -> None:
    setup_style()
    report = load_report(ARTIFACT_JSON)
    quality = extract_quality(report)
    complexity_stability = extract_complexity_stability(report)
    avg_rank, wins = compute_avg_rank_and_wins(quality)

    make_fig1_architectures()
    make_fig2_quality(quality)
    make_fig3_rank_wins(avg_rank, wins)
    make_fig4_complexity_stability(complexity_stability)
    print("Generated:", ", ".join(str(p.name) for p in sorted(OUT_DIR.glob("fig*.png"))))


if __name__ == "__main__":
    main()
