from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path


STAGE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("full_train_sec", "train full dictionary (sec)"),
    ("h_build_sec", "H-build/export (sec)"),
    ("selection_sec", "selection (sec)"),
    ("refit_head_sec", "refit head (sec)"),
    ("inference_ms_per_sample", "inference (ms/sample)"),
    ("memory_peak_mb", "memory peak (MB)"),
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Runtime Stagewise Context",
        "",
        "| Stage | Full KAFN | Budget-Prune | Gate-L1 | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['full_kafn']} | {row['budget_prune']} | {row['gate_l1']} | {row['notes']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _combine_stage_summaries(summaries: list[tuple[float, float, int]]) -> tuple[float, float, int] | None:
    if not summaries:
        return None
    total_n = sum(n for _, _, n in summaries)
    if total_n <= 0:
        return None
    weighted_mean = sum(mu * n for mu, _, n in summaries) / float(total_n)
    # Pooled population variance from group means/variances.
    variance_num = 0.0
    for mu, sd, n in summaries:
        if n <= 0:
            continue
        variance_num += n * (sd * sd + (mu - weighted_mean) ** 2)
    pooled_sd = math.sqrt(max(0.0, variance_num / float(total_n)))
    return float(weighted_mean), float(pooled_sd), int(total_n)


def _fmt(summary: tuple[float, float, int] | None, *, decimals: int = 2) -> str:
    if summary is None:
        return "N/A"
    mu, sd, n = summary
    return f"{mu:.{decimals}f} +/- {sd:.{decimals}f} (n={n})"


def _to_float_or_none(v: object) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _detect_method(manifest: dict[str, object], *, budget: int | None) -> str | None:
    protocol = manifest.get("protocol", {})
    execution = protocol.get("execution", {}) if isinstance(protocol, dict) else {}
    if not isinstance(execution, dict):
        return None
    fuzzy_models = execution.get("fuzzy_models")
    if not isinstance(fuzzy_models, list) or "ruanfis_kanfis" not in fuzzy_models:
        return None
    prune_rules = execution.get("kanfis_prune_rules")
    train_rule_gates = bool(execution.get("kanfis_train_rule_gates", False))
    if train_rule_gates:
        if budget is not None and _to_float_or_none(prune_rules) not in (None, float(budget)):
            return None
        return "gate_l1"
    prune_value = _to_float_or_none(prune_rules)
    if prune_value is None or int(prune_value) <= 0:
        return "full_kafn"
    if budget is not None and int(prune_value) != int(budget):
        return None
    return "budget_prune"


def _collect_stage_values(
    manifest_paths: list[Path],
    *,
    dataset: str,
    budget: int | None,
) -> tuple[dict[str, dict[str, list[tuple[float, float, int]]]], int]:
    values: dict[str, dict[str, list[tuple[float, float, int]]]] = {
        "full_kafn": {key: [] for key, _ in STAGE_COLUMNS},
        "budget_prune": {key: [] for key, _ in STAGE_COLUMNS},
        "gate_l1": {key: [] for key, _ in STAGE_COLUMNS},
    }
    used_manifests = 0
    for path in manifest_paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        method = _detect_method(manifest, budget=budget)
        if method is None:
            continue
        dataset_protocols = manifest.get("dataset_specific_protocols", {})
        if not isinstance(dataset_protocols, dict) or dataset not in dataset_protocols:
            continue
        protocol = dataset_protocols.get(dataset, {})
        if not isinstance(protocol, dict):
            continue
        runtime_summary = protocol.get("runtime_stages_summary", {})
        if not isinstance(runtime_summary, dict):
            continue
        used_manifests += 1
        for stage_key, _ in STAGE_COLUMNS:
            entry = runtime_summary.get(stage_key)
            if not isinstance(entry, dict):
                continue
            mu = _to_float_or_none(entry.get("mean"))
            sd = _to_float_or_none(entry.get("std"))
            n = _to_float_or_none(entry.get("n"))
            if mu is None or sd is None or n is None:
                continue
            n_int = int(n)
            if n_int <= 0:
                continue
            values[method][stage_key].append((float(mu), float(sd), n_int))
    return values, used_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=str, default="susy_binary_200000")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument(
        "--manifest-glob",
        type=str,
        default="compact_stable_kafn/runs/**/reproducibility_manifest.json",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("docs/tables_runtime_stagewise_context.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("docs/tables_runtime_stagewise_context.md"))
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    manifest_paths = [Path(p) for p in glob.glob(str(repo / args.manifest_glob), recursive=True)]
    values, used_manifests = _collect_stage_values(
        manifest_paths,
        dataset=str(args.dataset),
        budget=int(args.budget) if args.budget is not None else None,
    )

    rows: list[dict[str, object]] = []
    for stage_key, stage_label in STAGE_COLUMNS:
        decimals = 4 if stage_key == "inference_ms_per_sample" else 2
        rows.append(
            {
                "stage": stage_label,
                "full_kafn": _fmt(_combine_stage_summaries(values["full_kafn"][stage_key]), decimals=decimals),
                "budget_prune": _fmt(
                    _combine_stage_summaries(values["budget_prune"][stage_key]),
                    decimals=decimals,
                ),
                "gate_l1": _fmt(_combine_stage_summaries(values["gate_l1"][stage_key]), decimals=decimals),
                "notes": f"dataset={args.dataset}; budget={args.budget}; manifests_used={used_manifests}",
            }
        )

    _write_csv(
        repo / args.out_csv,
        ["stage", "full_kafn", "budget_prune", "gate_l1", "notes"],
        rows,
    )
    _write_markdown(repo / args.out_md, rows)
    print(f"Wrote {repo / args.out_csv}")
    print(f"Wrote {repo / args.out_md}")


if __name__ == "__main__":
    main()
