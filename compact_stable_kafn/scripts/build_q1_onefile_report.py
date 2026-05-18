from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(x: str | None, default: int = 0) -> int:
    if x is None or x == "":
        return default
    try:
        return int(float(x))
    except Exception:
        return default


def to_float(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def git_origin(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "remote", "get-url", "origin"],
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/q1_all_in_one_report.md"),
    )
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    docs = repo / "docs"

    main_rows = read_csv(docs / "unified_main_methods_table.csv")
    control_rows = read_csv(docs / "unified_control_checks_table.csv")
    stable_h = read_csv(docs / "tables_stable_h_selection_summary.csv")

    cov_rows = [r for r in main_rows if r.get("dataset") == "covtype_binary_20000"]
    cov_bp_seeds = [to_int(r.get("n_seeds_budget_prune")) for r in cov_rows if r.get("n_seeds_budget_prune")]
    cov_l1_seeds = [to_int(r.get("n_seeds_gate_l1")) for r in cov_rows if r.get("n_seeds_gate_l1")]
    cov_seed_min = min(cov_bp_seeds + cov_l1_seeds) if (cov_bp_seeds or cov_l1_seeds) else 0

    stable_proxy_budgets = sorted(
        {
            to_int(r.get("budget"))
            for r in control_rows
            if r.get("block") == "stable_budget_prune_proxy"
        }
    )
    stable_h_budgets = sorted(
        {
            to_int(r.get("budget"))
            for r in control_rows
            if r.get("block") == "stable_budget_prune_h_validation"
        }
    )

    stats_csv = docs / "q1_stat_tests_summary.csv"
    stats_ci_csv = docs / "q1_bootstrap_ci_summary.csv"
    stats_rows = read_csv(stats_csv)
    ci_rows = read_csv(stats_ci_csv)
    has_stats = len(stats_rows) > 0 and len(ci_rows) > 0
    stats_hits: list[str] = []
    if stats_csv.exists():
        stats_hits.append(str(stats_csv.relative_to(repo)))
    if stats_ci_csv.exists():
        stats_hits.append(str(stats_ci_csv.relative_to(repo)))

    runtime_anchor = next(
        (
            r
            for r in control_rows
            if r.get("block") == "runtime_budget_prune_anchor"
            and r.get("dataset") == "susy_binary_200000"
            and str(r.get("budget")) == "200"
        ),
        None,
    )
    full_anchor = next(
        (
            r
            for r in control_rows
            if r.get("block") == "runtime_full_reference_anchor"
            and r.get("dataset") == "susy_binary_200000"
        ),
        None,
    )

    has_runtime_decompose = any(
        r.get("metric") in {
            "full_train_sec",
            "h_build_sec",
            "selection_sec",
            "compact_refit_sec",
            "inference_ms_per_sample",
            "memory_mb",
        }
        for r in control_rows
    )
    runtime_stage_csv = docs / "tables_runtime_stagewise_context.csv"
    runtime_stage_rows = read_csv(runtime_stage_csv)
    runtime_stage_partial = len(runtime_stage_rows) > 0
    runtime_stage_has_na = any(
        any(str(r.get(col, "")).strip().upper() == "N/A" for col in ("full_kafn", "budget_prune", "gate_l1"))
        for r in runtime_stage_rows
    )

    # RuleFit coverage
    cov_rulefit = [
        r
        for r in main_rows
        if r.get("dataset") == "covtype_binary_20000" and to_float(r.get("rulefit_f1")) is not None
    ]
    susy_kafn = [
        r
        for r in main_rows
        if r.get("dataset") == "susy_binary_200000" and to_float(r.get("budget_prune_f1")) is not None
    ]

    repo_url = git_origin(repo)

    pass_cov_seeds = cov_seed_min >= 6
    pass_stats = has_stats
    pass_runtime = has_runtime_decompose
    pass_repo = bool(repo_url)
    pass_stable = len(stable_h_budgets) >= 3

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = args.out if args.out.is_absolute() else repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Q1 All-in-One Report")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Repo: `{repo}`")
    lines.append(f"- Origin: `{repo_url or 'N/A'}`")
    lines.append("")
    lines.append("## Current Core Results")
    lines.append("")
    if runtime_anchor:
        lines.append(f"- SUSY B=200 runtime anchor: {runtime_anchor.get('support', '').strip()}")
    if full_anchor:
        lines.append(f"- SUSY full no-prune runtime: {full_anchor.get('support', '').strip()}")
    lines.append(f"- Stable proxy budgets available: {stable_proxy_budgets or 'none'}")
    lines.append(f"- Stable end-to-end H-validation budgets available: {stable_h_budgets or 'none'}")
    lines.append("")
    lines.append("## Q1 Blocker Status")
    lines.append("")
    lines.append("| Check | Status | Evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Covtype seeds >= 6 for stability claims | {'PASS' if pass_cov_seeds else 'FAIL'} | min seeds on Covtype = {cov_seed_min} |"
    )
    lines.append(
        f"| Statistical tests (Wilcoxon/Friedman/p-values) | {'PASS' if pass_stats else 'FAIL'} | files: {', '.join(stats_hits) if stats_hits else 'not found'} |"
    )
    lines.append(
        f"| Runtime decomposition (train/H/select/refit/infer/memory) | {'PASS' if pass_runtime else 'FAIL'} | "
        + (
            "stage-wise metrics found in unified control table"
            if pass_runtime
            else (
                f"stage table present={runtime_stage_partial}, has_NA={runtime_stage_has_na}; unified control table lacks explicit stage metrics"
            )
        )
        + " |"
    )
    lines.append(
        f"| Public reproducibility link present | {'PASS' if pass_repo else 'FAIL'} | origin remote = {repo_url or 'N/A'} |"
    )
    lines.append(
        f"| Stable Budget-Prune end-to-end across >=3 budgets | {'PASS' if pass_stable else 'FAIL'} | budgets with H-validation = {stable_h_budgets or 'none'} |"
    )
    lines.append("")
    lines.append("## Coverage Gaps")
    lines.append("")
    lines.append(f"- Covtype RuleFit rows present: {len(cov_rulefit)}")
    lines.append(f"- SUSY KAFN quality rows present: {len(susy_kafn)}")
    lines.append("- Interpretation: Covtype still has no RuleFit rows; SUSY still lacks matched KAFN quality rows in unified main comparison.")
    lines.append("")
    lines.append("## Immediate Next Runs")
    lines.append("")
    next_steps: list[str] = []
    if not pass_cov_seeds:
        next_steps.append("Covtype: raise seeds to 6-10 for Budget-Prune/Gate-L1/Stable.")
    if not pass_stable:
        next_steps.append("Stable end-to-end: run at B=100/200/400 (then full budget grid if resources allow).")
    if not pass_stats:
        next_steps.append("Add stats outputs (Wilcoxon/Friedman + effect size + CI) into docs.")
    if not pass_runtime:
        next_steps.append("Build runtime stage table: full-train, H-build, selection, refit, inference, memory.")
    if not next_steps:
        next_steps.append("No critical blockers left in this checklist. Prepare final manuscript cleanup and submission package.")
    for idx, step in enumerate(next_steps, start=1):
        lines.append(f"{idx}. {step}")
    lines.append("")
    lines.append("## Stats Artifacts")
    lines.append("")
    lines.append(f"- Wilcoxon/paired tests rows: {len(stats_rows)}")
    lines.append(f"- Bootstrap CI rows: {len(ci_rows)}")
    if stats_hits:
        lines.append(f"- Files: {', '.join(stats_hits)}")
    if runtime_stage_partial:
        lines.append(f"- Runtime stage table: `{runtime_stage_csv.relative_to(repo)}`")
    lines.append("")
    lines.append("## Source Files Used")
    lines.append("")
    lines.append("- `docs/unified_main_methods_table.csv`")
    lines.append("- `docs/unified_control_checks_table.csv`")
    lines.append("- `docs/tables_stable_h_selection_summary.csv`")
    if stats_csv.exists():
        lines.append("- `docs/q1_stat_tests_summary.csv`")
    if stats_ci_csv.exists():
        lines.append("- `docs/q1_bootstrap_ci_summary.csv`")
    if runtime_stage_csv.exists():
        lines.append("- `docs/tables_runtime_stagewise_context.csv`")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
