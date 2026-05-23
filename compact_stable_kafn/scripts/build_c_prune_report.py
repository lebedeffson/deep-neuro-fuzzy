from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _as_float(x: str) -> float:
    return float(x)


def _group_mean(rows: list[dict[str, str]], keys: tuple[str, ...], value_col: str) -> dict[tuple[str, ...], float]:
    g: defaultdict[tuple[str, ...], list[float]] = defaultdict(list)
    for r in rows:
        k = tuple(r[c] for c in keys)
        g[k].append(_as_float(r[value_col]))
    return {k: mean(v) for k, v in g.items()}


def _pick_metric(
    f1: dict[tuple[str, str, str, str], float],
    red: dict[tuple[str, str, str, str], float],
    dataset: str,
    budget: str,
    method: str,
    corr_threshold: str,
) -> tuple[float, float]:
    if method in ("budget_prune", "loro_bce"):
        k = (dataset, budget, method, "na")
    else:
        k = (dataset, budget, method, corr_threshold)
    return f1.get(k, float("nan")), red.get(k, float("nan"))


def run(
    *,
    summary_csv: Path,
    redundancy_csv: Path,
    meta_jaccard_csv: Path,
    out_csv: Path,
    out_md: Path,
) -> None:
    summary = _read_csv(summary_csv)
    red = _read_csv(redundancy_csv)
    meta = _read_csv(meta_jaccard_csv)

    f1_mean = _group_mean(summary, ("dataset", "budget", "method", "corr_threshold"), "f1")
    red_mean = _group_mean(
        red,
        ("dataset", "budget", "method", "corr_threshold"),
        "mean_abs_pairwise_corr_selected",
    )
    rule_j_mean = _group_mean(meta, ("dataset", "budget", "method", "corr_threshold"), "rule_jaccard")
    meta_j_mean = _group_mean(meta, ("dataset", "budget", "method", "corr_threshold"), "meta_cluster_jaccard")

    keys = sorted(meta_j_mean.keys(), key=lambda x: (x[0], int(x[1]), x[2], float(x[3])))
    rows: list[dict[str, object]] = []
    for dataset, budget, method, thr in keys:
        f1v, redv = _pick_metric(f1_mean, red_mean, dataset, budget, method, thr)
        rj = rule_j_mean[(dataset, budget, method, thr)]
        mj = meta_j_mean[(dataset, budget, method, thr)]
        rows.append(
            {
                "dataset": dataset,
                "budget": int(budget),
                "method": method,
                "corr_threshold": thr,
                "f1_mean": f1v,
                "redundancy_mean_abs_corr": redv,
                "rule_jaccard_mean": rj,
                "meta_cluster_jaccard_mean": mj,
                "meta_minus_rule_jaccard": mj - rj,
            }
        )

    _write_csv(
        out_csv,
        [
            "dataset",
            "budget",
            "method",
            "corr_threshold",
            "f1_mean",
            "redundancy_mean_abs_corr",
            "rule_jaccard_mean",
            "meta_cluster_jaccard_mean",
            "meta_minus_rule_jaccard",
        ],
        rows,
    )

    # Short markdown summary for paper drafting.
    by_budget: defaultdict[int, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_budget[int(r["budget"])].append(r)

    lines: list[str] = []
    lines.append("# C-Prune Stability Report")
    lines.append("")
    lines.append("Means are aggregated over 3 seeds (F1/redundancy) and 3 seed-pairs (Jaccard).")
    lines.append("")

    for b in sorted(by_budget):
        cur = by_budget[b]
        cpr = [r for r in cur if str(r["method"]) == "c_prune_loro_bce"]
        best_meta = max(cpr, key=lambda r: float(r["meta_cluster_jaccard_mean"]))
        best_f1 = max(cpr, key=lambda r: float(r["f1_mean"]))
        lines.append(f"## Budget {b}")
        lines.append(
            f"- Best C-Prune by meta_jaccard: thr={best_meta['corr_threshold']}, "
            f"meta_j={float(best_meta['meta_cluster_jaccard_mean']):.4f}, "
            f"rule_j={float(best_meta['rule_jaccard_mean']):.4f}"
        )
        lines.append(
            f"- Best C-Prune by F1: thr={best_f1['corr_threshold']}, "
            f"F1={float(best_f1['f1_mean']):.4f}, "
            f"meta_j={float(best_f1['meta_cluster_jaccard_mean']):.4f}"
        )
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", type=Path, default=Path("docs/tables_c_prune_summary.csv"))
    ap.add_argument("--redundancy-csv", type=Path, default=Path("docs/tables_c_prune_redundancy.csv"))
    ap.add_argument(
        "--meta-jaccard-csv",
        type=Path,
        default=Path("docs/tables_c_prune_meta_cluster_jaccard.csv"),
    )
    ap.add_argument("--out-csv", type=Path, default=Path("docs/tables_c_prune_unified_report.csv"))
    ap.add_argument("--out-md", type=Path, default=Path("docs/tables_c_prune_unified_report.md"))
    args = ap.parse_args()

    run(
        summary_csv=args.summary_csv,
        redundancy_csv=args.redundancy_csv,
        meta_jaccard_csv=args.meta_jaccard_csv,
        out_csv=args.out_csv,
        out_md=args.out_md,
    )
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
