#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "artifacts" / "article_support_tables_2026-04-27.md"

MODEL_NAME_MAP = {
    "ruanfis_stacked_anfis": "stacked",
    "ruanfis_hierarchical_anfis": "hierarchical",
    "ruanfis_refined_deep": "dffl",
    "ruanfis_shallow": "shallow",
}


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pm(metric: dict) -> str:
    return f"{metric['mean']:.4f} +/- {metric['std']:.4f}"


def _load_report_rows(report_json: Path) -> dict[str, dict]:
    data = _load_json(report_json)
    rows = {}
    for row in data["aggregated_results"]:
        rows[row["model_name"]] = row
    return rows


def _capacity_table(capacity_dir: Path) -> str:
    p = capacity_dir / "covtype_binary_full_report.json"
    if not p.exists():
        return "## Capacity-matched (Covtype full)\n\nmissing run bundle\n"
    rows = _load_report_rows(p)
    order = ["ruanfis_stacked_anfis", "ruanfis_hierarchical_anfis", "ruanfis_refined_deep"]
    out = [
        "## Capacity-matched (Covtype full)",
        "",
        "| Model | F1 | ROC-AUC | PR-AUC | Total rules | Active rules | Active-rule Jaccard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in order:
        if m not in rows:
            continue
        r = rows[m]
        t = r["test_metrics"]
        s = r["structural_metrics"]
        st = r["stability_metrics"]
        out.append(
            "| {model} | {f1} | {roc} | {pr} | {rules_t} | {rules_a} | {j:.4f} |".format(
                model=MODEL_NAME_MAP.get(m, m),
                f1=_fmt_pm(t["f1"]),
                roc=_fmt_pm(t["roc_auc"]),
                pr=_fmt_pm(t["pr_auc"]),
                rules_t=_fmt_pm(s["total_rules"]),
                rules_a=_fmt_pm(s["active_rules"]),
                j=float(st.get("active_rule_jaccard", float("nan"))),
            )
        )
    out.append("")
    return "\n".join(out)


def _tau_table(tau_dirs: list[tuple[str, Path]]) -> str:
    lines = [
        "## Tau Sensitivity (paper_main, fuzzy-only)",
        "",
        "| tau | Model | Mean active-rule Jaccard across datasets |",
        "| --- | --- | ---: |",
    ]
    has_any = False
    for tau_label, tau_dir in tau_dirs:
        if not tau_dir.exists():
            continue
        vals_by_model: dict[str, list[float]] = {}
        for report in sorted(tau_dir.glob("*_report.json")):
            rows = _load_report_rows(report)
            for model_name, row in rows.items():
                model = MODEL_NAME_MAP.get(model_name, model_name)
                val = row["stability_metrics"].get("active_rule_jaccard")
                if val is not None:
                    vals_by_model.setdefault(model, []).append(float(val))
        for model in ("shallow", "stacked", "hierarchical", "dffl"):
            vals = vals_by_model.get(model, [])
            if not vals:
                continue
            has_any = True
            lines.append(f"| {tau_label} | {model} | {mean(vals):.4f} |")
    if not has_any:
        lines.append("| n/a | n/a | n/a |")
    lines.append("")
    return "\n".join(lines)


def _extract_model_times(log_path: Path) -> dict[str, list[float]]:
    if not log_path.exists():
        return {}
    txt = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: dict[str, list[float]] = {}
    for line in txt:
        m = re.search(r"model=([a-zA-Z0-9_]+): done in ([0-9.]+)s", line)
        if m:
            out.setdefault(m.group(1), []).append(float(m.group(2)))
    return out


def _extract_eval_times(log_path: Path) -> list[float]:
    if not log_path.exists():
        return []
    txt = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[float] = []
    for line in txt:
        m = re.search(r"fuzzy_eval: done in ([0-9.]+)s", line)
        if m:
            out.append(float(m.group(1)))
    return out


def _runtime_table(runtime_dirs: dict[str, Path], capacity_log: Path) -> str:
    lines = [
        "## Runtime Table (Covtype full)",
        "",
        "| Model | Train time (s, mean over seeds) | Inference+eval (s, single-model seed23) |",
        "| --- | ---: | ---: |",
    ]
    train_all = _extract_model_times(capacity_log)
    for model_key, model_label in [
        ("stacked", "stacked"),
        ("hierarchical_anfis", "hierarchical"),
        ("dffl", "dffl"),
    ]:
        # training times from capacity run
        times = train_all.get(model_key, [])
        train_str = f"{mean(times):.2f}" if times else "n/a"

        # inference time from single-model runtime run
        run_dir = runtime_dirs.get(model_label)
        infer_str = "n/a"
        if run_dir is not None:
            eval_times = _extract_eval_times(run_dir / "run.status.log")
            if eval_times:
                infer_str = f"{mean(eval_times):.2f}"
        lines.append(f"| {model_label} | {train_str} | {infer_str} |")
    lines.append("")
    return "\n".join(lines)


def _appendix_hparams(manifest_paths: list[Path]) -> str:
    lines = [
        "## Reproducibility Appendix (Hyperparameters)",
        "",
        "| Run | Dataset | Seeds | Max epochs | Batch | Patience | Device | Rule threshold |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    any_row = False
    for mpath in manifest_paths:
        if not mpath.exists():
            continue
        d = _load_json(mpath)
        p = d.get("protocol", {})
        b = p.get("training_budgets", {})
        e = p.get("evaluation", {})
        ex = p.get("execution", {})
        seeds = p.get("seeds", [])
        datasets = p.get("datasets", [])
        lines.append(
            "| {run} | {ds} | {seeds} | {ep} | {bs} | {pat} | {dev} | {thr} |".format(
                run=mpath.parent.name,
                ds=",".join(datasets),
                seeds=",".join(str(s) for s in seeds),
                ep=int(b.get("max_epochs", -1)),
                bs=int(b.get("batch_size", -1)),
                pat=int(b.get("patience", -1)),
                dev=ex.get("device", "n/a"),
                thr=float(e.get("rule_probability_threshold", float("nan"))),
            )
        )
        any_row = True
    if not any_row:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    capacity_dir = RUNS / "covtypefull_capacity_matched_fuzzy3_3s_2026-04-27"
    tau_dirs = [
        ("0.3", RUNS / "paper_main_fuzzy_tau03_3s_2026-04-27"),
        ("0.5", RUNS / "paper_main_fuzzy_tau05_3s_2026-04-27"),
        ("0.7", RUNS / "paper_main_fuzzy_tau07_3s_2026-04-27"),
    ]
    runtime_dirs = {
        "stacked": RUNS / "runtime_covtypefull_stacked_capacity_s23_2026-04-27",
        "hierarchical": RUNS / "runtime_covtypefull_hierarchical_capacity_s23_2026-04-27",
        "dffl": RUNS / "runtime_covtypefull_dffl_capacity_s23_2026-04-27",
    }
    manifest_paths = [
        capacity_dir / "reproducibility_manifest.json",
        RUNS / "paper_main_fuzzy_tau03_3s_2026-04-27" / "reproducibility_manifest.json",
        RUNS / "paper_main_fuzzy_tau05_3s_2026-04-27" / "reproducibility_manifest.json",
        RUNS / "paper_main_fuzzy_tau07_3s_2026-04-27" / "reproducibility_manifest.json",
    ]

    blocks = [
        "# Article Support Tables (2026-04-27)",
        "",
        _capacity_table(capacity_dir),
        _tau_table(tau_dirs),
        _runtime_table(runtime_dirs, capacity_dir / "run.status.log"),
        _appendix_hparams(manifest_paths),
    ]
    OUT.write_text("\n".join(blocks), encoding="utf-8")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
