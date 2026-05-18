from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _winner(values: dict[str, float | None]) -> str:
    present = {k: v for k, v in values.items() if v is not None}
    if not present:
        return ""
    return max(present.items(), key=lambda item: item[1])[0]


def build_main_table(methods_csv: Path, out_csv: Path) -> list[dict[str, object]]:
    rows = _read_csv(methods_csv)
    buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    buckets_std: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    n_seeds: dict[tuple[str, str, str], str] = {}
    for row in rows:
        budget = str(row.get("budget", ""))
        if budget not in {"100", "200", "400"}:
            continue
        method = str(row.get("method", ""))
        if method not in {"budget_prune", "gate_l1", "random_b", "rulefit"}:
            continue
        key = (str(row["dataset"]), budget)
        buckets[key][method] = float(row["f1_mean"])
        f1_std_raw = str(row.get("f1_std", "")).strip()
        if f1_std_raw:
            try:
                buckets_std[key][method] = float(f1_std_raw)
            except ValueError:
                pass
        n_seeds[(key[0], key[1], method)] = str(row.get("n_seeds", ""))

    out_rows: list[dict[str, object]] = []
    for dataset, budget in sorted(buckets, key=lambda item: (item[0], int(item[1]))):
        values = buckets[(dataset, budget)]
        compact_values = {
            "budget_prune": values.get("budget_prune"),
            "gate_l1": values.get("gate_l1"),
        }
        compact_winner = _winner(compact_values)
        all_winner = _winner(values)
        out_rows.append(
            {
                "dataset": dataset,
                "budget": int(budget),
                "budget_prune_f1": _fmt(values.get("budget_prune")),
                "budget_prune_f1_std": _fmt(buckets_std[(dataset, budget)].get("budget_prune")),
                "gate_l1_f1": _fmt(values.get("gate_l1")),
                "gate_l1_f1_std": _fmt(buckets_std[(dataset, budget)].get("gate_l1")),
                "random_b_f1": _fmt(values.get("random_b")),
                "random_b_f1_std": _fmt(buckets_std[(dataset, budget)].get("random_b")),
                "rulefit_f1": _fmt(values.get("rulefit")),
                "rulefit_f1_std": _fmt(buckets_std[(dataset, budget)].get("rulefit")),
                "best_compact_kafn": compact_winner,
                "winner_available_methods": all_winner,
                "n_seeds_budget_prune": n_seeds.get((dataset, budget, "budget_prune"), ""),
                "n_seeds_gate_l1": n_seeds.get((dataset, budget, "gate_l1"), ""),
                "n_seeds_rulefit": n_seeds.get((dataset, budget, "rulefit"), ""),
            }
        )
    _write_csv(
        out_csv,
        [
            "dataset",
            "budget",
            "budget_prune_f1",
            "budget_prune_f1_std",
            "gate_l1_f1",
            "gate_l1_f1_std",
            "random_b_f1",
            "random_b_f1_std",
            "rulefit_f1",
            "rulefit_f1_std",
            "best_compact_kafn",
            "winner_available_methods",
            "n_seeds_budget_prune",
            "n_seeds_gate_l1",
            "n_seeds_rulefit",
        ],
        out_rows,
    )
    return out_rows


def build_control_table(
    lr_csv: Path,
    stable_csv: Path,
    stable_h_csv: Path,
    runtime_csv: Path,
    rulefit_csv: Path,
    out_csv: Path,
) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []

    lr_rows = _read_csv(lr_csv)
    lr_bucket: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in lr_rows:
        lr_bucket[(row["dataset"], row["budget"], row["method"])].append(float(row["f1"]))
    for dataset, budget, _method in sorted(
        {key for key in lr_bucket if key[2] == "budget_prune"},
        key=lambda key: (key[0], int(key[1])),
    ):
        bp = mean(lr_bucket[(dataset, budget, "budget_prune")])
        lr = mean(lr_bucket[(dataset, budget, "lr_topk_rules")])
        out_rows.append(
            {
                "block": "lr_topk_control",
                "dataset": dataset,
                "budget": int(budget),
                "metric": "f1_delta_budget_prune_minus_lr_topk",
                "value": _fmt(bp - lr),
                "support": f"budget_prune={bp:.4f}; lr_topk={lr:.4f}",
            }
        )

    for row in _read_csv(stable_csv):
        out_rows.append(
            {
                "block": "stable_budget_prune_proxy",
                "dataset": "covtype_binary_20000",
                "budget": int(row["budget"]),
                "metric": "jaccard_gain_and_importance_retention",
                "value": _fmt(float(row["stable_loo_pairwise_jaccard"]) - float(row["current_budget_prune_pairwise_jaccard"])),
                "support": (
                    f"current_jaccard={float(row['current_budget_prune_pairwise_jaccard']):.4f}; "
                    f"stable_jaccard={float(row['stable_loo_pairwise_jaccard']):.4f}; "
                    f"retention={float(row['stable_loo_importance_retention_mean']):.4f}"
                ),
            }
        )

    stable_h_rows = _read_csv(stable_h_csv)
    by_h = {(row["budget"], row["method"]): row for row in stable_h_rows}
    for budget in sorted({row["budget"] for row in stable_h_rows}, key=int):
        bp = by_h.get((budget, "budget_prune_h_lr"))
        stable = by_h.get((budget, "stable_budget_prune_h_lr"))
        if bp is None or stable is None:
            continue
        f1_drop = float(bp["f1_mean"]) - float(stable["f1_mean"])
        fidelity_delta = float(stable["fidelity_l1_to_full_prob_mean"]) - float(bp["fidelity_l1_to_full_prob_mean"])
        out_rows.append(
            {
                "block": "stable_budget_prune_h_validation",
                "dataset": stable["dataset"],
                "budget": int(budget),
                "metric": "f1_drop_fidelity_delta_jaccard_to_bp",
                "value": _fmt(f1_drop),
                "support": (
                    f"bp_f1={float(bp['f1_mean']):.4f}; stable_f1={float(stable['f1_mean']):.4f}; "
                    f"fidelity_delta={fidelity_delta:.4f}; "
                    f"agreement_stable={float(stable['agreement_to_full_mean']):.4f}; "
                    f"jaccard_to_bp={float(stable['jaccard_to_budget_prune_mean']):.4f}"
                ),
            }
        )

    runtime = {
        (row["dataset"], row["method"], row["budget"]): (
            float(row["elapsed_sec_mean"]),
            float(row.get("elapsed_sec_std", 0.0)),
        )
        for row in _read_csv(runtime_csv)
    }
    rulefit = {
        (row["dataset"], row["rule_budget"]): float(row["elapsed_sec_mean"])
        for row in _read_csv(rulefit_csv)
    }
    for dataset in {"breast_cancer", "susy_binary_200000"}:
        for budget in ("100", "200", "400"):
            kafn = runtime.get((dataset, "budget_prune", budget)) or runtime.get((dataset, "l1_budget", budget))
            rf = rulefit.get((dataset, budget))
            if kafn is None or rf is None:
                continue
            kafn_mean, kafn_std = kafn
            out_rows.append(
                {
                    "block": "runtime_context",
                    "dataset": dataset,
                    "budget": int(budget),
                    "metric": "runtime_context_not_strict_speed_claim",
                    "value": _fmt(rf / max(kafn_mean, 1e-12)),
                    "support": f"compact_kafn_pipeline={kafn_mean:.1f}s (+/-{kafn_std:.1f}); rulefit={rf:.1f}s",
                }
            )
    susy_b200 = runtime.get(("susy_binary_200000", "budget_prune", "200"))
    if susy_b200 is not None:
        susy_mean, susy_std = susy_b200
        out_rows.append(
            {
                "block": "runtime_budget_prune_anchor",
                "dataset": "susy_binary_200000",
                "budget": 200,
                "metric": "budget_prune_end_to_end_runtime_sec",
                "value": _fmt(susy_mean),
                "support": (
                    f"budget_prune_runtime={susy_mean:.2f}s (+/-{susy_std:.2f}); "
                    "selection-after-full-dictionary context (not strict train-speed race)."
                ),
            }
        )
    full_ref_paths = sorted(
        glob.glob(
            "compact_stable_kafn/runs/compact_stable_kafn_susy200k_q1/seed_*/method_full_reference/repeat_0/no_prune/reproducibility_manifest.json"
        )
    )
    full_secs: list[float] = []
    for p in full_ref_paths:
        try:
            j = json.loads(Path(p).read_text(encoding="utf-8"))
            full_secs.append(float(j["dataset_specific_protocols"]["susy_binary_200000"]["runtime_seconds"]))
        except Exception:
            continue
    if full_secs:
        full_mean = mean(full_secs)
        full_std = pstdev(full_secs) if len(full_secs) > 1 else 0.0
        out_rows.append(
            {
                "block": "runtime_full_reference_anchor",
                "dataset": "susy_binary_200000",
                "budget": "full",
                "metric": "full_dictionary_no_prune_runtime_sec",
                "value": _fmt(full_mean),
                "support": (
                    f"full_no_prune_runtime={full_mean:.2f}s (+/-{full_std:.2f}); "
                    f"n={len(full_secs)}; includes training+evaluation with full active dictionary."
                ),
            }
        )

    _write_csv(out_csv, ["block", "dataset", "budget", "metric", "value", "support"], out_rows)
    return out_rows


def write_takeaways(path: Path, main_rows: list[dict[str, object]], control_rows: list[dict[str, object]]) -> None:
    cov_400 = next((r for r in main_rows if r["dataset"] == "covtype_binary_20000" and r["budget"] == 400), None)
    breast_400 = next((r for r in main_rows if r["dataset"] == "breast_cancer" and r["budget"] == 400), None)
    stable_400 = next(
        (r for r in control_rows if r["block"] == "stable_budget_prune_proxy" and r["budget"] == 400),
        None,
    )
    lines = [
        "# Unified Q1-Oriented Result Package",
        "",
        "## What We Can Use Now",
        "",
        "- Main method story: Budget-Prune is the reliable structural pruning method; Gate-L1 is a learned sparsification ablation that can win on small data.",
        "- External baseline story: RuleFit is included as an interpretable competitor, not as a toy baseline.",
        "- Control story: LR on the same top-K rules is consistently weaker than Budget-Prune on Covtype.",
    ]
    if cov_400:
        lines.append(
            f"- Covtype B=400: Budget-Prune F1={cov_400['budget_prune_f1']}, Gate-L1 F1={cov_400['gate_l1_f1']}, Random-B F1={cov_400['random_b_f1']}."
        )
    if breast_400:
        lines.append(
            f"- Breast Cancer B=400: Gate-L1 F1={breast_400['gate_l1_f1']}, Budget-Prune F1={breast_400['budget_prune_f1']}, RuleFit F1={breast_400['rulefit_f1']}."
        )
    if stable_400:
        lines.append(f"- Stable Budget-Prune proxy at B=400: {stable_400['support']}.")
    stable_h_400 = next(
        (r for r in control_rows if r["block"] == "stable_budget_prune_h_validation" and r["budget"] == 400),
        None,
    )
    if stable_h_400:
        lines.append(f"- Stable Budget-Prune H-based validation at B=400: {stable_h_400['support']}.")
    runtime_anchor = next(
        (
            r
            for r in control_rows
            if r["block"] == "runtime_budget_prune_anchor"
            and r["dataset"] == "susy_binary_200000"
            and r["budget"] == 200
        ),
        None,
    )
    if runtime_anchor:
        lines.append(f"- Runtime anchor (SUSY B=200): {str(runtime_anchor['support']).rstrip('.')}.")
    runtime_full = next(
        (r for r in control_rows if r["block"] == "runtime_full_reference_anchor"),
        None,
    )
    if runtime_full:
        lines.append(f"- Full KAFN reference (SUSY no-prune): {str(runtime_full['support']).rstrip('.')}.")
    lines.extend(
        [
            "",
            "## How To Combine Into One Paper",
            "",
            "Use Budget-Prune as the main submitted method, Gate-L1 as ablation, RuleFit as external baseline, and Stable Budget-Prune as a validated stability-aware extension at B=400.",
            "",
            "Important: Stable Budget-Prune improves stability with a small F1 drop in the H-based check; do not claim it improves every metric.",
            "",
            "## Remaining Practical Gap",
            "",
            "Covtype still has no RuleFit row, and SUSY still lacks matched KAFN quality rows in the unified main comparison. Keep those claims separate unless we run/recover them.",
            "",
            "## Metric Notes",
            "",
            "Importance retention in Stable proxy tables means retained heldout importance mass: sum(importance of selected subset under heldout seed) / sum(importance of heldout top-B).",
            "Runtime numbers are context metrics for compactization workflow; they are not a strict apples-to-apples full training speed comparison against RuleFit.",
            "SUSY B=200 runtime anchor reports the compact budgeted KAFN pipeline under that budget setting.",
            "SUSY no-prune full reference reports full active-dictionary KAFN runtime over available seeds and is shown only as context, not as a strict speed race against RuleFit.",
            "The work does not claim full end-to-end KAFN training speed superiority over RuleFit; the focus is selection quality and subset stability.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    docs = args.docs_dir
    main_rows = build_main_table(
        docs / "tables_methods_comparison.csv",
        docs / "unified_main_methods_table.csv",
    )
    control_rows = build_control_table(
        docs / "tables_lr_topk_baseline.csv",
        docs / "q1_stable_selection_smoke_summary.csv",
        docs / "tables_stable_h_selection_summary.csv",
        docs / "tables_compact_kafn_runtime_minimal.csv",
        Path("compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_summary.csv"),
        docs / "unified_control_checks_table.csv",
    )
    write_takeaways(docs / "unified_q1_result_package.md", main_rows, control_rows)
    print(f"Wrote {docs / 'unified_main_methods_table.csv'}")
    print(f"Wrote {docs / 'unified_control_checks_table.csv'}")
    print(f"Wrote {docs / 'unified_q1_result_package.md'}")


if __name__ == "__main__":
    main()
