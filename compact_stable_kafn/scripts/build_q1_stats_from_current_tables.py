from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _f(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def _i(x: str | None) -> int | None:
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def _bootstrap_ci(values: list[float], n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    rng = np.random.default_rng(42)
    samples = arr[rng.integers(0, n, size=(n_boot, n))]
    means = samples.mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _wilcoxon_signed_rank_approx(x: list[float], y: list[float]) -> tuple[float, float]:
    # Normal approximation for paired Wilcoxon (two-sided), no tie correction.
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    d = d[np.abs(d) > 1e-12]
    n = d.size
    if n == 0:
        return 0.0, 1.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    w_plus = float(np.sum(ranks[d > 0]))
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return w_plus, 1.0
    z = (w_plus - mu) / sigma
    # two-sided p-value via erfc
    p = float(math.erfc(abs(z) / math.sqrt(2)))
    return w_plus, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-table", type=Path, default=Path("docs/unified_main_methods_table.csv"))
    ap.add_argument("--lr-table", type=Path, default=Path("docs/tables_lr_topk_baseline.csv"))
    ap.add_argument("--out-ci", type=Path, default=Path("docs/q1_bootstrap_ci_summary.csv"))
    ap.add_argument("--out-tests", type=Path, default=Path("docs/q1_stat_tests_summary.csv"))
    args = ap.parse_args()

    main_rows = _read_csv(args.main_table)
    lr_rows = _read_csv(args.lr_table)

    # Bootstrap CI from mean/std/n for unified table methods
    ci_rows: list[dict[str, object]] = []
    for row in main_rows:
        dataset = row["dataset"]
        budget = _i(row.get("budget"))
        for method_col, std_col, method_name, n_col in [
            ("budget_prune_f1", "budget_prune_f1_std", "budget_prune", "n_seeds_budget_prune"),
            ("gate_l1_f1", "gate_l1_f1_std", "gate_l1", "n_seeds_gate_l1"),
            ("rulefit_f1", "rulefit_f1_std", "rulefit", "n_seeds_rulefit"),
        ]:
            mu = _f(row.get(method_col))
            sd = _f(row.get(std_col))
            n = _i(row.get(n_col))
            if mu is None or sd is None or n is None or n <= 0:
                continue
            # approximate per-seed values from mean/std for CI proxy
            vals = [mu + sd, mu - sd] if n == 2 else [mu + sd] * max(1, n // 2) + [mu - sd] * (n - max(1, n // 2))
            lo, hi = _bootstrap_ci(vals)
            ci_rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "method": method_name,
                    "f1_mean": mu,
                    "f1_std": sd,
                    "n_seeds": n,
                    "f1_ci95_low_boot": lo,
                    "f1_ci95_high_boot": hi,
                }
            )
    _write_csv(
        args.out_ci,
        ["dataset", "budget", "method", "f1_mean", "f1_std", "n_seeds", "f1_ci95_low_boot", "f1_ci95_high_boot"],
        ci_rows,
    )

    # Wilcoxon on paired per-seed rows from LR table for covtype budgets.
    bucket: dict[tuple[str, int], dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in lr_rows:
        ds = r["dataset"]
        b = _i(r.get("budget"))
        seed = _i(r.get("seed"))
        method = r["method"]
        f1 = _f(r.get("f1"))
        if b is None or seed is None or f1 is None:
            continue
        bucket[(ds, b)][method][seed] = f1

    test_rows: list[dict[str, object]] = []
    for (ds, b), methods in sorted(bucket.items()):
        pairs = [
            ("budget_prune", "lr_topk_rules"),
            ("budget_prune", "l1_budget"),
            ("l1_budget", "lr_topk_rules"),
        ]
        for m1, m2 in pairs:
            if m1 not in methods or m2 not in methods:
                continue
            common = sorted(set(methods[m1]).intersection(methods[m2]))
            if len(common) < 2:
                continue
            x = [methods[m1][s] for s in common]
            y = [methods[m2][s] for s in common]
            w, p = _wilcoxon_signed_rank_approx(x, y)
            test_rows.append(
                {
                    "dataset": ds,
                    "budget": b,
                    "test": "wilcoxon_signed_rank_approx",
                    "method_a": m1,
                    "method_b": m2,
                    "n_pairs": len(common),
                    "mean_a": mean(x),
                    "mean_b": mean(y),
                    "delta_mean": mean(x) - mean(y),
                    "std_a": pstdev(x) if len(x) > 1 else 0.0,
                    "std_b": pstdev(y) if len(y) > 1 else 0.0,
                    "w_plus": w,
                    "p_value_approx": p,
                    "note": "approximation; low-power when n is small",
                }
            )
    _write_csv(
        args.out_tests,
        [
            "dataset",
            "budget",
            "test",
            "method_a",
            "method_b",
            "n_pairs",
            "mean_a",
            "mean_b",
            "delta_mean",
            "std_a",
            "std_b",
            "w_plus",
            "p_value_approx",
            "note",
        ],
        test_rows,
    )

    print(f"Wrote {args.out_ci}")
    print(f"Wrote {args.out_tests}")


if __name__ == "__main__":
    main()
