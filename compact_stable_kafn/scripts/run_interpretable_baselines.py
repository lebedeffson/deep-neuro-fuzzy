#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.datasets import fetch_covtype, load_breast_cancer, load_wine
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings(
    "ignore",
    message=".*'penalty' was deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Inconsistent values: penalty=l1 with l1_ratio=0.0.*",
    category=UserWarning,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Any


def _load_covtype_binary_20000() -> tuple[np.ndarray, np.ndarray]:
    x, y = fetch_covtype(return_X_y=True)
    yb = (y == 2).astype(np.int32)
    xs, _, ys, _ = train_test_split(
        x,
        yb,
        train_size=20000,
        random_state=42,
        stratify=yb,
    )
    return xs.astype(np.float32), ys.astype(np.int32)


def _load_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    d = load_breast_cancer()
    return d.data.astype(np.float32), d.target.astype(np.int32)


def _load_wine_binary() -> tuple[np.ndarray, np.ndarray]:
    d = load_wine()
    y = (d.target == 0).astype(np.int32)
    return d.data.astype(np.float32), y


def _ensure_susy_csv_gz() -> Path:
    cache_dir = Path(__file__).resolve().parents[2] / "data" / "external" / "susy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_gz_path = cache_dir / "SUSY.csv.gz"
    if csv_gz_path.exists():
        return csv_gz_path
    zip_path = cache_dir / "susy.zip"
    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        zip_path.unlink()
    if not zip_path.exists():
        urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/279/susy.zip", zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [name for name in archive.namelist() if name.endswith("SUSY.csv.gz")]
        if not csv_members:
            raise RuntimeError("SUSY.csv.gz not found in susy.zip")
        archive.extract(csv_members[0], cache_dir)
        extracted_path = cache_dir / csv_members[0]
        if extracted_path != csv_gz_path:
            extracted_path.replace(csv_gz_path)
    return csv_gz_path


def _load_susy_binary_200000() -> tuple[np.ndarray, np.ndarray]:
    csv_gz_path = _ensure_susy_csv_gz()
    data = np.loadtxt(csv_gz_path, delimiter=",", dtype=np.float32, max_rows=200_000)
    target = data[:, 0].astype(np.int32)
    features = data[:, 1:].astype(np.float32)
    return features, target


SPECS: dict[str, DatasetSpec] = {
    "covtype_binary_20000": DatasetSpec("covtype_binary_20000", _load_covtype_binary_20000),
    "breast_cancer": DatasetSpec("breast_cancer", _load_breast_cancer),
    "wine_binary": DatasetSpec("wine_binary", _load_wine_binary),
    "susy_binary_200000": DatasetSpec("susy_binary_200000", _load_susy_binary_200000),
}


def _safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(pstdev(values))


def _split_scale(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=int(seed),
        stratify=y,
    )
    rel_val = float(val_size / (1.0 - test_size))
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=rel_val,
        random_state=int(seed) + 1,
        stratify=y_train_val,
    )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    return x_train, x_val, x_test, y_train, y_val, y_test


def _train_rulefit(seed: int, x_train: np.ndarray, y_train: np.ndarray, *, max_rules: int) -> Any:
    from imodels import RuleFitClassifier

    model = RuleFitClassifier(
        random_state=int(seed),
        max_rules=int(max_rules),
    )
    model.fit(x_train, y_train)
    return model


def _train_ebm(seed: int, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    from interpret.glassbox import ExplainableBoostingClassifier

    model = ExplainableBoostingClassifier(
        random_state=int(seed),
        interactions=0,
    )
    model.fit(x_train, y_train)
    return model


def _proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(x)
        if p.ndim == 2 and p.shape[1] >= 2:
            return p[:, 1].astype(np.float64)
    pred = model.predict(x)
    return np.asarray(pred, dtype=np.float64)


def _metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(np.int32)
    return {
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _task_key(dataset: str, seed: int, model: str, rule_budget: Any) -> tuple[str, int, str, str]:
    return (str(dataset), int(seed), str(model), str(rule_budget))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _save_outputs(
    out_dir: Path,
    per_run_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    dataset_names: list[str],
    seeds: list[int],
    rulefit_budgets: list[int],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in per_run_rows:
        key = (str(row["dataset"]), str(row["model"]), str(row.get("rule_budget", "na")))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset_name, model_name, rule_budget), bucket in sorted(grouped.items()):
        f1_values = [_safe_float(item["f1"]) for item in bucket]
        roc_values = [_safe_float(item["roc_auc"]) for item in bucket]
        pr_values = [_safe_float(item["pr_auc"]) for item in bucket]
        t_values = [_safe_float(item["elapsed_sec"]) for item in bucket]
        summary_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "rule_budget": rule_budget,
                "n_seeds": len(bucket),
                "f1_mean": mean(f1_values),
                "f1_std": _safe_std(f1_values),
                "roc_auc_mean": mean(roc_values),
                "roc_auc_std": _safe_std(roc_values),
                "pr_auc_mean": mean(pr_values),
                "pr_auc_std": _safe_std(pr_values),
                "elapsed_sec_mean": mean(t_values),
                "elapsed_sec_std": _safe_std(t_values),
            }
        )

    _write_csv(
        out_dir / "interpretable_baselines_per_run.csv",
        per_run_rows,
        ["dataset", "seed", "model", "rule_budget", "f1", "roc_auc", "pr_auc", "elapsed_sec"],
    )
    _write_csv(
        out_dir / "interpretable_baselines_summary.csv",
        summary_rows,
        [
            "dataset",
            "model",
            "rule_budget",
            "n_seeds",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
            "pr_auc_mean",
            "pr_auc_std",
            "elapsed_sec_mean",
            "elapsed_sec_std",
        ],
    )
    _write_csv(out_dir / "interpretable_baselines_status.csv", status_rows, ["dataset", "model", "status"])

    manifest = {
        "datasets": dataset_names,
        "seeds": seeds,
        "rulefit_budgets": rulefit_budgets,
        "rows_per_run": len(per_run_rows),
        "rows_summary": len(summary_rows),
        "rows_status": len(status_rows),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "interpretable_baselines_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run interpretable tabular baselines (RuleFit, EBM).")
    parser.add_argument(
        "--datasets",
        type=str,
        default="covtype_binary_20000,breast_cancer,wine_binary,susy_binary_200000",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="19,23,29",
        help="Comma-separated seed list.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("compact_stable_kafn/paper_tables/interpretable_baselines"),
    )
    parser.add_argument(
        "--rulefit-budgets",
        type=str,
        default="100,200,400",
        help="Comma-separated max_rules values for RuleFit.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="rulefit,ebm",
        help="Comma-separated models to run: rulefit,ebm",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing interpretable_baselines_per_run.csv if present.",
    )
    parser.add_argument(
        "--checkpoint-every-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per_run/summary/manifest after each completed run.",
    )
    parser.add_argument(
        "--progress-log-name",
        type=str,
        default="interpretable_baselines_progress.csv",
        help="Progress log filename inside out-dir (with ETA columns).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    rulefit_budgets = [int(item.strip()) for item in args.rulefit_budgets.split(",") if item.strip()]
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    for m in selected_models:
        if m not in ("rulefit", "ebm"):
            raise ValueError(f"Unsupported model in --models: {m}")

    available_rulefit = True
    available_ebm = True
    try:
        from imodels import RuleFitClassifier  # noqa: F401
    except Exception:
        available_rulefit = False
    try:
        from interpret.glassbox import ExplainableBoostingClassifier  # noqa: F401
    except Exception:
        available_ebm = False

    per_run_path = out_dir / "interpretable_baselines_per_run.csv"
    existing_rows = _read_csv(per_run_path) if args.resume else []
    per_run_rows: list[dict[str, Any]] = []
    done_keys: set[tuple[str, int, str, str]] = set()
    for row in existing_rows:
        try:
            key = _task_key(
                row["dataset"],
                int(row["seed"]),
                row["model"],
                row.get("rule_budget", "na"),
            )
        except Exception:
            continue
        done_keys.add(key)
        per_run_rows.append(
            {
                "dataset": row["dataset"],
                "seed": int(row["seed"]),
                "model": row["model"],
                "rule_budget": row.get("rule_budget", "na"),
                "f1": _safe_float(row.get("f1", 0.0)),
                "roc_auc": _safe_float(row.get("roc_auc", 0.0)),
                "pr_auc": _safe_float(row.get("pr_auc", 0.0)),
                "elapsed_sec": _safe_float(row.get("elapsed_sec", 0.0)),
            }
        )

    progress_log_path = out_dir / args.progress_log_name
    if not progress_log_path.exists():
        _write_csv(
            progress_log_path,
            [],
            [
                "ts",
                "dataset",
                "seed",
                "model",
                "rule_budget",
                "elapsed_sec",
                "done",
                "total",
                "remaining",
                "eta_sec",
                "eta_at",
            ],
        )
    progress_rows = _read_csv(progress_log_path)

    status_rows: list[dict[str, Any]] = []
    all_tasks: list[tuple[str, int, str, Any]] = []
    for dataset_name in dataset_names:
        for seed in seeds:
            for model_name in selected_models:
                budgets = rulefit_budgets if model_name == "rulefit" else [0]
                for budget in budgets:
                    all_tasks.append((dataset_name, int(seed), model_name, int(budget) if model_name == "rulefit" else "na"))
    total_tasks = len(all_tasks)

    for dataset_name in dataset_names:
        spec = SPECS.get(dataset_name)
        if spec is None:
            status_rows.append({"dataset": dataset_name, "model": "-", "status": "unsupported_dataset"})
            continue
        x, y = spec.loader()
        for seed in seeds:
            x_train, _x_val, x_test, y_train, _y_val, y_test = _split_scale(x, y, seed=seed)
            for model_name in selected_models:
                if model_name == "rulefit" and not available_rulefit:
                    status_rows.append({"dataset": dataset_name, "model": model_name, "status": "package_not_installed"})
                    continue
                if model_name == "ebm" and not available_ebm:
                    status_rows.append({"dataset": dataset_name, "model": model_name, "status": "package_not_installed"})
                    continue
                budgets = rulefit_budgets if model_name == "rulefit" else [0]
                for budget in budgets:
                    budget_key = int(budget) if model_name == "rulefit" else "na"
                    key = _task_key(dataset_name, seed, model_name, budget_key)
                    if key in done_keys:
                        print(f"[baseline] dataset={dataset_name} seed={seed} model={model_name} budget={budget_key}: skip (resume)")
                        continue
                    started = time.time()
                    label = f"b{budget}" if model_name == "rulefit" else "na"
                    print(f"[baseline] dataset={dataset_name} seed={seed} model={model_name} budget={label}: start")
                    if model_name == "rulefit":
                        model = _train_rulefit(seed, x_train, y_train, max_rules=int(budget))
                    else:
                        model = _train_ebm(seed, x_train, y_train)
                    elapsed = time.time() - started
                    p = _proba(model, x_test)
                    metrics = _metrics(y_test, p)
                    per_run_rows.append(
                        {
                            "dataset": dataset_name,
                            "seed": int(seed),
                            "model": model_name,
                            "rule_budget": budget_key,
                            "f1": metrics["f1"],
                            "roc_auc": metrics["roc_auc"],
                            "pr_auc": metrics["pr_auc"],
                            "elapsed_sec": float(elapsed),
                        }
                    )
                    done_keys.add(key)
                    done = len(done_keys)
                    remaining = max(0, total_tasks - done)
                    elapsed_vals = [_safe_float(r["elapsed_sec"]) for r in per_run_rows if _safe_float(r["elapsed_sec"]) > 0]
                    avg_elapsed = mean(elapsed_vals) if elapsed_vals else 0.0
                    eta_sec = float(avg_elapsed * remaining) if avg_elapsed > 0 else 0.0
                    eta_at = (datetime.now() + timedelta(seconds=eta_sec)).isoformat(timespec="seconds") if eta_sec > 0 else ""
                    progress_rows.append(
                        {
                            "ts": datetime.now().isoformat(timespec="seconds"),
                            "dataset": dataset_name,
                            "seed": int(seed),
                            "model": model_name,
                            "rule_budget": budget_key,
                            "elapsed_sec": float(elapsed),
                            "done": done,
                            "total": total_tasks,
                            "remaining": remaining,
                            "eta_sec": round(eta_sec, 2),
                            "eta_at": eta_at,
                        }
                    )
                    _write_csv(
                        progress_log_path,
                        progress_rows,
                        [
                            "ts",
                            "dataset",
                            "seed",
                            "model",
                            "rule_budget",
                            "elapsed_sec",
                            "done",
                            "total",
                            "remaining",
                            "eta_sec",
                            "eta_at",
                        ],
                    )
                    if args.checkpoint_every_run:
                        _save_outputs(
                            out_dir=out_dir,
                            per_run_rows=per_run_rows,
                            status_rows=status_rows,
                            dataset_names=dataset_names,
                            seeds=seeds,
                            rulefit_budgets=rulefit_budgets,
                        )
                    print(
                        f"[baseline] dataset={dataset_name} seed={seed} model={model_name} budget={label}: "
                        f"done ({elapsed:.2f}s), progress {done}/{total_tasks}, eta~{eta_sec/60.0:.1f}m"
                    )

    summary_rows = _save_outputs(
        out_dir=out_dir,
        per_run_rows=per_run_rows,
        status_rows=status_rows,
        dataset_names=dataset_names,
        seeds=seeds,
        rulefit_budgets=rulefit_budgets,
    )
    print(f"written: {out_dir}")


if __name__ == "__main__":
    main()
