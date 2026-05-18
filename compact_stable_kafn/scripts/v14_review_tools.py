#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_dataset_key(dataset_name: str) -> str:
    return str(dataset_name).strip().lower()


def _extract_metric(metrics: dict[str, Any], primary: str, fallback: str | None = None) -> float:
    if primary in metrics:
        return _safe_float(metrics[primary])
    if fallback and fallback in metrics:
        return _safe_float(metrics[fallback])
    return 0.0


@dataclass(frozen=True)
class KafnRunRecord:
    path: Path
    method: str
    nominal_budget: int
    importance_split: str
    payload: dict[str, Any]


def _detect_method(payload: dict[str, Any]) -> str:
    gate_l1 = _safe_float(payload.get("kanfis_skip_gate_l1_weight", 0.0))
    if gate_l1 > 0.0:
        return "gate_l1"
    if int(payload.get("kanfis_prune_rules", 0)) > 0:
        return "budget_prune"
    return "unknown"


def _extract_dataset_results(payload: dict[str, Any]) -> dict[str, Any]:
    if "dataset_results" in payload and isinstance(payload["dataset_results"], dict):
        return payload["dataset_results"]
    return {}


def _load_kafn_run_record(path: Path) -> KafnRunRecord:
    payload = _read_json(path)
    method = _detect_method(payload)
    nominal_budget = int(payload.get("kanfis_prune_rules", 0))
    importance_split = str(payload.get("kanfis_importance_split", "train"))
    return KafnRunRecord(
        path=path,
        method=method,
        nominal_budget=nominal_budget,
        importance_split=importance_split,
        payload=payload,
    )


def _iter_kafn_seed_rows(record: KafnRunRecord, dataset_filter: str) -> list[dict[str, Any]]:
    dataset_results = _extract_dataset_results(record.payload)
    target = _normalize_dataset_key(dataset_filter)
    rows: list[dict[str, Any]] = []
    for dataset_name, dataset_payload in dataset_results.items():
        if target not in _normalize_dataset_key(dataset_name):
            continue
        per_seed_results = dataset_payload.get("per_seed_results", [])
        for seed_entry in per_seed_results:
            seed = int(seed_entry.get("seed", 0))
            for result in seed_entry.get("results", []):
                if str(result.get("model_name")) != "ruanfis_kanfis":
                    continue
                structural = result.get("structural_metrics", {}) or {}
                test_metrics = result.get("test_metrics", {}) or {}
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "method": record.method,
                        "nominal_budget": record.nominal_budget,
                        "importance_split": record.importance_split,
                        "actual_active_rules": _safe_float(structural.get("active_rules", 0.0)),
                        "f1": _extract_metric(test_metrics, "f1", fallback="f1_score"),
                        "roc_auc": _extract_metric(test_metrics, "roc_auc"),
                        "pr_auc": _extract_metric(test_metrics, "pr_auc"),
                    }
                )
    return rows


def _matched_budget_label_for_gate(nominal_budget: int, actual_active_rules: float) -> str:
    mapping = {
        25: "6/7",
        50: "6/7",
        100: "8/10",
        200: "12",
        400: "16/17",
    }
    if nominal_budget in mapping:
        return mapping[nominal_budget]
    rounded = max(1, int(round(actual_active_rules)))
    return str(rounded)


def command_rulefit_config(args: argparse.Namespace) -> None:
    rows = _read_csv(Path(args.per_run_csv))
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("model", "")).strip().lower() != "rulefit":
            continue
        out_rows.append(
            {
                "dataset": row["dataset"],
                "budget": row["rule_budget"],
                "seed": row["seed"],
                "n_estimators": "default",
                "tree_size_or_max_depth": "default",
                "max_rules": row["rule_budget"],
                "alpha_or_regularization": "default",
                "train_size": 0.6,
                "val_size": 0.2,
                "test_size": 0.2,
                "random_state": row["seed"],
                "elapsed_sec": row.get("elapsed_sec", ""),
                "status": "done",
            }
        )
    out_rows.sort(key=lambda r: (r["dataset"], int(r["budget"]), int(r["seed"])))
    _write_csv(
        Path(args.out_csv),
        out_rows,
        [
            "dataset",
            "budget",
            "seed",
            "n_estimators",
            "tree_size_or_max_depth",
            "max_rules",
            "alpha_or_regularization",
            "train_size",
            "val_size",
            "test_size",
            "random_state",
            "elapsed_sec",
            "status",
        ],
    )


def command_fair_active_rules(args: argparse.Namespace) -> None:
    records = [_load_kafn_run_record(Path(path)) for path in args.json_inputs]
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.method not in {"gate_l1", "budget_prune"}:
            continue
        for row in _iter_kafn_seed_rows(record, args.dataset):
            matched_budget = (
                _matched_budget_label_for_gate(int(row["nominal_budget"]), float(row["actual_active_rules"]))
                if row["method"] == "gate_l1"
                else str(int(row["nominal_budget"]))
            )
            rows.append(
                {
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "method": row["method"],
                    "nominal_budget": row["nominal_budget"],
                    "actual_active_rules": row["actual_active_rules"],
                    "matched_budget": matched_budget,
                    "f1": row["f1"],
                    "roc_auc": row["roc_auc"],
                    "pr_auc": row["pr_auc"],
                }
            )
    rows.sort(key=lambda r: (r["dataset"], r["method"], int(r["seed"]), int(r["nominal_budget"])))
    _write_csv(
        Path(args.out_csv),
        rows,
        [
            "dataset",
            "seed",
            "method",
            "nominal_budget",
            "actual_active_rules",
            "matched_budget",
            "f1",
            "roc_auc",
            "pr_auc",
        ],
    )


def command_importance_ablation(args: argparse.Namespace) -> None:
    records = [_load_kafn_run_record(Path(path)) for path in args.json_inputs]
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.method not in {"gate_l1", "budget_prune"}:
            continue
        for row in _iter_kafn_seed_rows(record, args.dataset):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "budget": row["nominal_budget"],
                    "seed": row["seed"],
                    "importance_split": row["importance_split"],
                    "f1": row["f1"],
                    "roc_auc": row["roc_auc"],
                    "pr_auc": row["pr_auc"],
                    "selected_rules": row["actual_active_rules"],
                }
            )
    rows.sort(key=lambda r: (r["dataset"], int(r["budget"]), r["importance_split"], int(r["seed"])))
    _write_csv(
        Path(args.out_csv),
        rows,
        [
            "dataset",
            "budget",
            "seed",
            "importance_split",
            "f1",
            "roc_auc",
            "pr_auc",
            "selected_rules",
        ],
    )


def command_runtime_table(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []

    for path_str in args.json_inputs or []:
        record = _load_kafn_run_record(Path(path_str))
        dataset_protocols = (
            record.payload.get("reproducibility_manifest", {})
            .get("dataset_specific_protocols", {})
        )
        for dataset_name, protocol in dataset_protocols.items():
            if args.dataset and _normalize_dataset_key(args.dataset) not in _normalize_dataset_key(dataset_name):
                continue
            runtime_seconds = _safe_float(protocol.get("runtime_seconds", 0.0))
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": record.method,
                    "budget": record.nominal_budget,
                    "seed": "all_seeds",
                    "train_full_dictionary_sec": runtime_seconds if runtime_seconds > 0 else "",
                    "selection_sec": "",
                    "compact_refit_sec": "",
                    "test_inference_ms_per_sample": "",
                    "explanation_ms_per_sample": "",
                    "total_pipeline_sec": runtime_seconds if runtime_seconds > 0 else "",
                }
            )

    if args.rulefit_summary_csv:
        rulefit_rows = _read_csv(Path(args.rulefit_summary_csv))
        for row in rulefit_rows:
            if str(row.get("model", "")).strip().lower() != "rulefit":
                continue
            if args.dataset and _normalize_dataset_key(args.dataset) not in _normalize_dataset_key(row["dataset"]):
                continue
            elapsed = _safe_float(row.get("elapsed_sec_mean", 0.0))
            rows.append(
                {
                    "dataset": row["dataset"],
                    "method": "rulefit",
                    "budget": row["rule_budget"],
                    "seed": f"mean_over_{row['n_seeds']}_seeds",
                    "train_full_dictionary_sec": elapsed if elapsed > 0 else "",
                    "selection_sec": "",
                    "compact_refit_sec": "",
                    "test_inference_ms_per_sample": "",
                    "explanation_ms_per_sample": "",
                    "total_pipeline_sec": elapsed if elapsed > 0 else "",
                }
            )

    rows.sort(key=lambda r: (r["dataset"], str(r["method"]), str(r["budget"]), str(r["seed"])))
    _write_csv(
        Path(args.out_csv),
        rows,
        [
            "dataset",
            "method",
            "budget",
            "seed",
            "train_full_dictionary_sec",
            "selection_sec",
            "compact_refit_sec",
            "test_inference_ms_per_sample",
            "explanation_ms_per_sample",
            "total_pipeline_sec",
        ],
    )


def command_covtype_concepts(args: argparse.Namespace) -> None:
    rows = _read_csv(Path(args.input_csv))
    concept_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        concept = str(row.get("concept_name", "")).strip()
        if not concept:
            continue
        concept_rows.setdefault(concept, []).append(row)

    dictionary_rows: list[dict[str, Any]] = []
    for concept, c_rows in sorted(concept_rows.items()):
        predicates: list[str] = []
        for key in ("top_predicate_1", "top_predicate_2", "top_predicate_3"):
            value = str(c_rows[0].get(key, "")).strip()
            if value and value not in predicates:
                predicates.append(value)
        dictionary_rows.append(
            {
                "concept_id": concept,
                "human_readable_rule": " ; ".join(predicates),
            }
        )

    _write_csv(
        Path(args.out_dictionary_csv),
        dictionary_rows,
        ["concept_id", "human_readable_rule"],
    )

    examples_path = Path(args.out_examples_md) if args.out_examples_md else None
    if examples_path is not None:
        examples_path.parent.mkdir(parents=True, exist_ok=True)
        # Prefer concepts with strongest absolute signed contribution if provided.
        scored: list[tuple[float, str]] = []
        for concept, c_rows in concept_rows.items():
            signed = _safe_float(c_rows[0].get("signed_contribution", 0.0))
            scored.append((abs(signed), concept))
        scored.sort(reverse=True)
        picked = [concept for _, concept in scored[:2]]
        if len(picked) < 2:
            picked = [row["concept_id"] for row in dictionary_rows[:2]]
        dataset_label = str(args.dataset_label).strip() if str(args.dataset_label).strip() else "Dataset"
        lines = [f"# {dataset_label} Concept Examples", ""]
        for concept_id in picked:
            matched = next((row for row in dictionary_rows if row["concept_id"] == concept_id), None)
            if matched is None:
                continue
            lines.append(f"- `{concept_id}`: {matched['human_readable_rule']}")
        examples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v14 review tables from existing experiment artifacts.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rulefit = sub.add_parser("rulefit-config", help="Export RuleFit run configuration table.")
    p_rulefit.add_argument(
        "--per-run-csv",
        type=Path,
        default=Path("compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_per_run.csv"),
    )
    p_rulefit.add_argument(
        "--out-csv",
        type=Path,
        default=Path("docs/rulefit_config_table.csv"),
    )
    p_rulefit.set_defaults(func=command_rulefit_config)

    p_fair = sub.add_parser("fair-active-rules", help="Build fair active-rule comparison table.")
    p_fair.add_argument("--json-inputs", type=str, nargs="+", required=True)
    p_fair.add_argument("--dataset", type=str, default="covtype")
    p_fair.add_argument("--out-csv", type=Path, default=Path("docs/tables_fair_active_rules_comparison.csv"))
    p_fair.set_defaults(func=command_fair_active_rules)

    p_importance = sub.add_parser("importance-ablation", help="Build train/val importance split ablation table.")
    p_importance.add_argument("--json-inputs", type=str, nargs="+", required=True)
    p_importance.add_argument("--dataset", type=str, default="covtype")
    p_importance.add_argument("--out-csv", type=Path, default=Path("docs/tables_importance_split_ablation.csv"))
    p_importance.set_defaults(func=command_importance_ablation)

    p_runtime = sub.add_parser("runtime-table", help="Build compact KAFN runtime table.")
    p_runtime.add_argument("--json-inputs", type=str, nargs="*", default=[])
    p_runtime.add_argument("--dataset", type=str, default="")
    p_runtime.add_argument(
        "--rulefit-summary-csv",
        type=Path,
        default=Path("compact_stable_kafn/paper_tables/interpretable_baselines_q1/interpretable_baselines_summary.csv"),
    )
    p_runtime.add_argument("--out-csv", type=Path, default=Path("docs/tables_rulefit_runtime.csv"))
    p_runtime.set_defaults(func=command_runtime_table)

    p_covtype = sub.add_parser("covtype-concepts", help="Build concept_dictionary_covtype.csv from local concept export.")
    p_covtype.add_argument("--input-csv", type=Path, required=True)
    p_covtype.add_argument("--out-dictionary-csv", type=Path, default=Path("docs/concept_dictionary_covtype.csv"))
    p_covtype.add_argument("--out-examples-md", type=Path, default=Path("docs/v14_covtype_examples.md"))
    p_covtype.add_argument("--dataset-label", type=str, default="Covtype")
    p_covtype.set_defaults(func=command_covtype_concepts)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
