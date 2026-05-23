from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run(*, report_csv: Path, out_meta_line: Path, out_rule_meta_scatter: Path) -> None:
    rows = _read_csv(report_csv)

    # Plot 1: meta-cluster jaccard vs budget
    series: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        method = r["method"]
        thr = r["corr_threshold"]
        budget = int(r["budget"])
        mj = float(r["meta_cluster_jaccard_mean"])
        label = f"{method}@{thr}"
        series.setdefault(label, []).append((budget, mj))

    plt.figure(figsize=(8, 5))
    for label, pts in sorted(series.items()):
        pts = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lw = 2.5 if label.startswith("c_prune_loro_bce@0.75") else 1.6
        alpha = 1.0 if label.startswith("c_prune_loro_bce") else 0.8
        plt.plot(xs, ys, marker="o", linewidth=lw, alpha=alpha, label=label)
    plt.xlabel("Budget (rules)")
    plt.ylabel("Meta-cluster Jaccard (mean)")
    plt.title("Meta-cluster Stability vs Budget")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, ncol=2, frameon=False)
    out_meta_line.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_meta_line, dpi=180)
    plt.close()

    # Plot 2: rule vs meta scatter
    color_map = {
        "budget_prune": "#1f77b4",
        "loro_bce": "#ff7f0e",
        "c_prune_loro_bce": "#2ca02c",
    }
    marker_map = {25: "o", 50: "s", 100: "^"}

    plt.figure(figsize=(7, 5))
    for r in rows:
        method = r["method"]
        budget = int(r["budget"])
        rj = float(r["rule_jaccard_mean"])
        mj = float(r["meta_cluster_jaccard_mean"])
        plt.scatter(
            rj,
            mj,
            color=color_map.get(method, "#555555"),
            marker=marker_map.get(budget, "o"),
            s=65,
            alpha=0.9,
        )
    plt.xlabel("Rule-level Jaccard (mean)")
    plt.ylabel("Meta-cluster Jaccard (mean)")
    plt.title("Rule vs Meta Stability")
    plt.grid(alpha=0.25)

    # Compact legend
    from matplotlib.lines import Line2D

    method_legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=m)
        for m, c in color_map.items()
    ]
    budget_legend = [
        Line2D([0], [0], marker=mk, color="#666666", linestyle="", markersize=8, label=f"B={b}")
        for b, mk in marker_map.items()
    ]
    plt.legend(handles=method_legend + budget_legend, fontsize=8, frameon=False, ncol=2)
    out_rule_meta_scatter.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_rule_meta_scatter, dpi=180)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-csv", type=Path, default=Path("docs/tables_c_prune_unified_report.csv"))
    ap.add_argument("--out-meta-line", type=Path, default=Path("docs/fig_c_prune_meta_vs_budget.png"))
    ap.add_argument("--out-rule-meta-scatter", type=Path, default=Path("docs/fig_c_prune_rule_vs_meta.png"))
    args = ap.parse_args()

    run(
        report_csv=args.report_csv,
        out_meta_line=args.out_meta_line,
        out_rule_meta_scatter=args.out_rule_meta_scatter,
    )
    print(f"Wrote {args.out_meta_line}")
    print(f"Wrote {args.out_rule_meta_scatter}")


if __name__ == "__main__":
    main()
