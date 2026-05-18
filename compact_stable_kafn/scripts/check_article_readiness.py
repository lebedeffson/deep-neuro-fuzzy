from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _has_value(row: dict[str, str], key: str) -> bool:
    value = str(row.get(key, "")).strip()
    return value != "" and value.lower() not in {"nan", "none"}


def check_main_table(path: Path) -> list[tuple[str, str]]:
    rows = _read_csv(path)
    issues: list[tuple[str, str]] = []
    if not rows:
        return [("FAIL", f"missing or empty main table: {path}")]

    for row in rows:
        dataset = row.get("dataset", "")
        budget = row.get("budget", "")
        has_kafn = _has_value(row, "budget_prune_f1") or _has_value(row, "gate_l1_f1")
        has_rulefit = _has_value(row, "rulefit_f1")
        if dataset == "susy_binary_200000" and has_rulefit and not has_kafn:
            issues.append(
                (
                    "WARN",
                    f"SUSY budget={budget} has RuleFit but no KAFN quality row; keep out of main win/loss table.",
                )
            )
        if dataset == "covtype_binary_20000" and not has_rulefit:
            issues.append(
                (
                    "WARN",
                    f"Covtype budget={budget} has no RuleFit row; mention external baseline is not run on main Covtype.",
                )
            )
    return issues


def check_stable_proxy(path: Path) -> list[tuple[str, str]]:
    rows = _read_csv(path)
    issues: list[tuple[str, str]] = []
    if not rows:
        return [("WARN", f"missing Stable Budget-Prune proxy table: {path}")]
    for row in rows:
        budget = int(row["budget"])
        retention = float(row["stable_loo_importance_retention_mean"])
        if budget < 400 and retention < 0.8:
            issues.append(
                (
                    "WARN",
                    f"Stable proxy budget={budget} retention={retention:.4f}; do not use as main quality claim.",
                )
            )
        if budget == 400 and retention >= 0.9:
            issues.append(
                (
                    "OK",
                    f"Stable proxy budget=400 retention={retention:.4f}; good candidate for H-based validation.",
                )
            )
    return issues


def check_stable_h_summary(path: Path) -> list[tuple[str, str]]:
    rows = _read_csv(path)
    if not rows:
        return [
            (
                "WARN",
                "missing H-based Stable Budget-Prune summary; Stable remains proxy-only until this is produced.",
            )
        ]
    by_method = {(row["budget"], row["method"]): row for row in rows}
    issues: list[tuple[str, str]] = []
    for budget in sorted({row["budget"] for row in rows}):
        bp = by_method.get((budget, "budget_prune_h_lr"))
        stable = by_method.get((budget, "stable_budget_prune_h_lr"))
        if bp is None or stable is None:
            issues.append(("WARN", f"H-based stable summary budget={budget} lacks paired BP/Stable rows."))
            continue
        f1_drop = float(bp["f1_mean"]) - float(stable["f1_mean"])
        stable_jaccard = float(stable["jaccard_to_budget_prune_mean"])
        if f1_drop <= 0.01:
            issues.append(
                (
                    "OK",
                    f"H-based stable budget={budget}: f1_drop={f1_drop:.4f}, jaccard_to_bp={stable_jaccard:.4f}.",
                )
            )
        else:
            issues.append(
                (
                    "WARN",
                    f"H-based stable budget={budget}: f1_drop={f1_drop:.4f}; keep Stable as extension.",
                )
            )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    docs = args.docs_dir
    checks = []
    checks.extend(check_main_table(docs / "unified_main_methods_table.csv"))
    checks.extend(check_stable_proxy(docs / "q1_stable_selection_smoke_summary.csv"))
    checks.extend(check_stable_h_summary(docs / "tables_stable_h_selection_summary.csv"))

    fail_count = sum(1 for level, _msg in checks if level == "FAIL")
    warn_count = sum(1 for level, _msg in checks if level == "WARN")
    for level, msg in checks:
        print(f"[{level}] {msg}")
    print(f"summary: fails={fail_count} warnings={warn_count}")
    raise SystemExit(1 if fail_count else 0)


if __name__ == "__main__":
    main()
