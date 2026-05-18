#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    fair_path = DOCS / "tables_fair_active_rules_comparison.csv"
    importance_path = DOCS / "tables_importance_split_ablation.csv"
    rulefit_cfg_path = DOCS / "rulefit_config_table.csv"
    covtype_examples_path = DOCS / "v14_covtype_examples.md"
    rulefit_summary_doc_path = DOCS / "rulefit_q1_partial_summary_2026-05-17.md"

    checks.append(("fair_active_exists", fair_path.exists(), str(fair_path)))
    checks.append(("importance_ablation_exists", importance_path.exists(), str(importance_path)))

    # 1) No old Gate-L1 ultra-low active-rules artifact in this package.
    gate_l1_ok = True
    gate_l1_reason = "ok"
    if fair_path.exists():
        fair_rows = _read_csv(fair_path)
        for row in fair_rows:
            if row.get("method") != "gate_l1":
                continue
            nominal = int(float(row.get("nominal_budget", "0")))
            active = float(row.get("actual_active_rules", "0"))
            if nominal >= 25 and active < 20:
                gate_l1_ok = False
                gate_l1_reason = f"found nominal={nominal} active={active}"
                break
    checks.append(("gate_l1_no_old_6_3_8_3_pattern", gate_l1_ok, gate_l1_reason))

    # 2) SUSY RuleFit B=400 policy:
    # - if partial (<6), it must be outside main table;
    # - if complete (>=6), either placement is acceptable.
    susy_main_ok = False
    susy_reason = "missing summary doc"
    if rulefit_summary_doc_path.exists():
        text = rulefit_summary_doc_path.read_text(encoding="utf-8")
        summary_csv = DOCS.parent / "compact_stable_kafn" / "paper_tables" / "interpretable_baselines_q1" / "interpretable_baselines_summary.csv"
        seeds_400 = None
        if summary_csv.exists():
            for row in _read_csv(summary_csv):
                if (
                    row.get("dataset") == "susy_binary_200000"
                    and row.get("model", "rulefit").strip().lower() == "rulefit"
                    and int(float(row.get("rule_budget", "0"))) == 400
                ):
                    seeds_400 = int(float(row.get("n_seeds", "0")))
                    break

        main_part = text.split("## Partial/Appendix Only", 1)[0]
        has_susy_400_in_main = "| susy_binary_200000 | 400 |" in main_part
        has_keep_out_marker = "keep out of main table" in text.lower()
        if seeds_400 is None:
            susy_main_ok = False
            susy_reason = "could not read SUSY@400 seeds from summary csv"
        elif seeds_400 < 6:
            susy_main_ok = (not has_susy_400_in_main) and has_keep_out_marker
            susy_reason = "ok" if susy_main_ok else "partial SUSY@400 policy not enforced"
        else:
            susy_main_ok = True
            susy_reason = "ok (SUSY@400 complete)"
    checks.append(("susy_rulefit_400_not_in_main", susy_main_ok, susy_reason))

    # 3) Covtype examples should look like Covtype features, not Breast Cancer labels.
    covtype_ok = False
    covtype_reason = "missing covtype examples"
    if covtype_examples_path.exists():
        text = covtype_examples_path.read_text(encoding="utf-8")
        has_covtype_tokens = any(
            token in text
            for token in (
                "Horizontal_Distance",
                "Soil_Type",
                "Wilderness_Area",
                "Elevation",
                "Slope",
            )
        )
        has_breast_tokens = any(
            token in text.lower()
            for token in (
                "mean radius",
                "worst smoothness",
                "fractal dimension",
                "radius error",
            )
        )
        covtype_ok = has_covtype_tokens and not has_breast_tokens
        covtype_reason = "ok" if covtype_ok else "feature tokens mismatch"
    checks.append(("covtype_examples_feature_domain_check", covtype_ok, covtype_reason))

    # 4) RuleFit config table should not contain empty key parameter fields.
    cfg_ok = True
    cfg_reason = "ok"
    if rulefit_cfg_path.exists():
        cfg_rows = _read_csv(rulefit_cfg_path)
        required_fields = ("n_estimators", "tree_size_or_max_depth", "max_rules", "alpha_or_regularization")
        for row in cfg_rows:
            if any(not str(row.get(field, "")).strip() for field in required_fields):
                cfg_ok = False
                cfg_reason = "empty key field in rulefit_config_table.csv"
                break
    else:
        cfg_ok = False
        cfg_reason = "missing rulefit_config_table.csv"
    checks.append(("rulefit_config_non_empty_keys", cfg_ok, cfg_reason))

    # 5) NO-LEAKAGE style check for importance split labels.
    leak_ok = True
    leak_reason = "ok"
    if importance_path.exists():
        imp_rows = _read_csv(importance_path)
        allowed = {"train", "val"}
        for row in imp_rows:
            split = str(row.get("importance_split", "")).strip().lower()
            if split not in allowed:
                leak_ok = False
                leak_reason = f"invalid split: {split}"
                break
    else:
        leak_ok = False
        leak_reason = "missing importance ablation file"
    checks.append(("no_leakage_split_labels", leak_ok, leak_reason))

    print("v16 sanity checks")
    print("================")
    failures = 0
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"- {name}: {status} ({detail})")
        if not ok:
            failures += 1

    if failures == 0:
        print("NO-LEAKAGE CHECK: OK")
        return 0
    print(f"NO-LEAKAGE CHECK: FAIL ({failures} failed checks)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
