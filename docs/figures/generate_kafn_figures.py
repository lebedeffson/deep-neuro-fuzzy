#!/usr/bin/env python3
"""Generate Routed KAFN manuscript figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches


OUT_DIR = Path(__file__).resolve().parent


def _box(ax, xy, width, height, text, face="#F7F9FC", edge="#334155", fontsize=10.5):
    rect = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, start, end, color="#334155", lw=1.4):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=4, shrinkB=4),
    )


def make_kafn_architecture(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Routed Kolmogorov--Arnold Fuzzy Network", ha="center", va="center", fontsize=16, weight="semibold")

    _box(ax, (0.04, 0.58), 0.14, 0.16, "Input\nfeatures", "#E0F2FE")
    _box(ax, (0.24, 0.70), 0.18, 0.13, "Teacher-guided\nrouting", "#ECFDF5")
    _box(ax, (0.24, 0.49), 0.18, 0.13, "Binary-aware\nterm dictionary", "#ECFDF5")

    _box(ax, (0.49, 0.70), 0.19, 0.13, "One-dimensional\nfuzzy predicates", "#FFF7ED")
    _box(ax, (0.49, 0.49), 0.19, 0.13, "Sparse projection\npursuit tokens", "#FFF7ED")
    _box(ax, (0.49, 0.27), 0.19, 0.13, "LOW / MEDIUM / HIGH\nabsent / present", "#FFF7ED", fontsize=10)

    _box(ax, (0.75, 0.68), 0.18, 0.13, "KA concept layer 1\nadditive sums", "#F1F5F9")
    _box(ax, (0.75, 0.47), 0.18, 0.13, "KA concept layer 2\nfuzzy superposition", "#F1F5F9")
    _box(ax, (0.75, 0.26), 0.18, 0.13, "Sparse Sugeno\ndecision", "#F1F5F9")

    _box(ax, (0.43, 0.06), 0.25, 0.11, "Stable rule dictionary\nand local explanation", "#F8FAFC", fontsize=10)
    _box(ax, (0.82, 0.06), 0.12, 0.11, "Prediction", "#DBEAFE")

    _arrow(ax, (0.18, 0.66), (0.24, 0.77))
    _arrow(ax, (0.18, 0.66), (0.24, 0.56))
    _arrow(ax, (0.42, 0.77), (0.49, 0.77))
    _arrow(ax, (0.42, 0.56), (0.49, 0.56))
    _arrow(ax, (0.58, 0.70), (0.75, 0.75))
    _arrow(ax, (0.58, 0.49), (0.75, 0.54))
    _arrow(ax, (0.84, 0.68), (0.84, 0.60))
    _arrow(ax, (0.84, 0.47), (0.84, 0.39))
    _arrow(ax, (0.84, 0.26), (0.88, 0.17))
    _arrow(ax, (0.84, 0.26), (0.62, 0.17))

    ax.text(0.585, 0.84, r"$c_q^{(1)}=\sum_{p,t}\alpha_{qpt}\mu_{pt}(x_p)$", ha="center", va="center", fontsize=12)
    ax.text(0.84, 0.63, r"$c_k^{(2)}=\sum_{q,t}\beta_{kqt}\nu_{qt}(c_q^{(1)})$", ha="center", va="center", fontsize=12)

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_kafn_architecture.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def make_kafn_explanation_flow(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.92, "Routed KAFN Explanation Flow", ha="center", va="center", fontsize=16, weight="semibold")

    steps = [
        ("Input object", "original feature values"),
        ("Membership terms", "LOW / MEDIUM / HIGH\nabsent / present"),
        ("Top predicates", "feature IS term\npursuit index IS term"),
        ("Concept evidence", "positive and negative\nconcept contributions"),
        ("Decision trace", "top decision rules\nand final score"),
    ]
    xs = [0.05, 0.25, 0.45, 0.65, 0.83]
    widths = [0.14, 0.15, 0.15, 0.15, 0.14]
    for i, ((title, body), x, w) in enumerate(zip(steps, xs, widths)):
        _box(ax, (x, 0.40), w, 0.25, f"{title}\n\n{body}", "#EEF2FF" if i % 2 == 0 else "#F0FDFA", fontsize=10)
        if i < len(steps) - 1:
            _arrow(ax, (x + w, 0.525), (xs[i + 1], 0.525))

    _box(ax, (0.18, 0.12), 0.25, 0.13, "Global view:\nstable active-rule sets and Jaccard", "#F8FAFC", fontsize=10)
    _box(ax, (0.57, 0.12), 0.25, 0.13, "Local view:\ntop rules for one decision", "#F8FAFC", fontsize=10)
    _arrow(ax, (0.525, 0.40), (0.31, 0.25), lw=1.2)
    _arrow(ax, (0.725, 0.40), (0.70, 0.25), lw=1.2)

    fig.tight_layout()
    fig.savefig(out_dir / "fig6_kafn_explanation_flow.png", dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_kafn_architecture(OUT_DIR)
    make_kafn_explanation_flow(OUT_DIR)
    print("Generated fig5_kafn_architecture.png and fig6_kafn_explanation_flow.png")


if __name__ == "__main__":
    main()
