from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_xor_example_runs_and_writes_report(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "xor_report.txt"
    bundle_path = tmp_path / "xor_bundle.pt"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "examples" / "train_xor_with_rule_report.py"),
            "--max-epochs",
            "120",
            "--output",
            str(output_path),
            "--bundle-output",
            str(bundle_path),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    report_text = output_path.read_text(encoding="utf-8")
    assert "XOR DEEP FUZZY FEATURE LEARNING DEMO" in report_text
    assert "MODEL REPORT BEFORE TRAINING" in report_text
    assert "MODEL REPORT AFTER TRAINING AND PRUNING" in report_text
    assert "PRUNING REPORT" in report_text
    assert "TOP-1 CONCEPT FLOW" in report_text
    assert "VALIDATION METRICS" in report_text
    assert bundle_path.exists()
    assert result.stdout.strip()


def test_hierarchical_regression_example_runs_and_writes_report(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "hierarchical_report.txt"
    bundle_path = tmp_path / "hierarchical_bundle.pt"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "examples" / "train_hierarchical_regression.py"),
            "--max-epochs",
            "80",
            "--output",
            str(output_path),
            "--bundle-output",
            str(bundle_path),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    report_text = output_path.read_text(encoding="utf-8")
    assert "HIERARCHICAL REGRESSION DEEP FUZZY DEMO" in report_text
    assert "MODEL CONFIG REPORT" in report_text
    assert "Flat full-rule count:" in report_text
    assert "Hierarchical generated-rule count:" in report_text
    assert "PRUNING REPORT" in report_text
    assert "GLOBAL CONCEPT FLOW" in report_text
    assert "PATH-BASED HIDDEN FLOW" in report_text
    assert "MODEL REPORT" in report_text
    assert bundle_path.exists()
    assert result.stdout.strip()


def test_regression_benchmark_example_runs_and_writes_report(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "benchmark_report.txt"
    json_path = tmp_path / "benchmark_report.json"
    table_path = tmp_path / "benchmark_report.md"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "examples" / "run_regression_benchmark.py"),
            "--seeds",
            "19,23",
            "--train-size",
            "160",
            "--validation-size",
            "64",
            "--test-size",
            "64",
            "--pretrain-epochs",
            "10",
            "--decision-pretrain-epochs",
            "8",
            "--max-epochs",
            "40",
            "--refinement-cycles",
            "2",
            "--output",
            str(output_path),
            "--json-output",
            str(json_path),
            "--table-output",
            str(table_path),
        ],
        cwd=project_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    report_text = output_path.read_text(encoding="utf-8")
    table_text = table_path.read_text(encoding="utf-8")
    assert "MULTI-SEED TABULAR BENCHMARK" in report_text
    assert "AGGREGATED TABULAR BENCHMARK" in report_text
    assert "linear_regression [sklearn]" in report_text
    assert "ruanfis_shallow [ruanfis]" in report_text
    assert "ruanfis_refined_deep [ruanfis]" in report_text
    assert "PAPER-READY SUMMARY TABLE" in report_text
    assert "INTERPRETATION STABILITY" in report_text
    assert "| model | family |" in table_text
    assert "test:rmse" in table_text
    assert "rules:total_rules" in table_text
    assert "stab:active_rule_jaccard" in table_text
    assert json_path.exists()
    assert result.stdout.strip()
