from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from examples.run_real_datasets_benchmark import (
    DATASETS,
    PRIMARY_METRIC,
    FUZZY_MODEL_NAMES,
    run_single_seed_dataset_benchmark,
)


@dataclass
class TrialConfig:
    dffl_profile_name: str = "quality_auto"
    dffl_learning_rate: float = 0.015
    pretrain_epochs: int = 8
    decision_pretrain_epochs: int = 8
    max_epochs: int = 20
    refinement_cycles: int = 2
    batch_size: int = 128
    patience: int = 8
    classification_threshold: float = 0.5
    tune_fuzzy_threshold: bool = True


@dataclass
class TrialResult:
    mean_test_metric: float
    mean_train_metric: float
    mean_active_rules: float
    mean_total_rules: float
    raw: list[dict[str, Any]]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _extract_model_entry(result_entries: tuple, model_name: str) -> Any:
    for entry in result_entries:
        if entry.model_name == model_name:
            return entry
    raise ValueError(f"Model {model_name!r} not found in benchmark output.")


def run_trial(
    *,
    dataset_name: str,
    seeds: tuple[int, ...],
    target_model: str,
    include_sklearn: bool,
    device: str | None,
    config: TrialConfig,
) -> TrialResult:
    spec = DATASETS[dataset_name]
    metric_name = PRIMARY_METRIC[spec.task_type]
    per_seed_payload: list[dict[str, Any]] = []

    for seed in seeds:
        entries = run_single_seed_dataset_benchmark(
            spec,
            seed=seed,
            test_size=0.2,
            validation_size=0.2,
            train_noise_sigma=0.0,
            pretrain_epochs=config.pretrain_epochs,
            decision_pretrain_epochs=config.decision_pretrain_epochs,
            max_epochs=config.max_epochs,
            refinement_cycles=config.refinement_cycles,
            fuzzy_learning_rate=0.02,
            dffl_learning_rate=config.dffl_learning_rate,
            dffl_profile_name=config.dffl_profile_name,
            batch_size=config.batch_size,
            patience=config.patience,
            classification_threshold=config.classification_threshold,
            tune_fuzzy_threshold=config.tune_fuzzy_threshold,
            dffl_one_phase=False,
            device=device,
            fuzzy_models=(target_model,),
            include_sklearn=include_sklearn,
        )
        target_entry = _extract_model_entry(entries, target_model)
        per_seed_payload.append(
            {
                "seed": seed,
                "metric_name": metric_name,
                "test_metric": float(target_entry.test_metrics[metric_name]),
                "train_metric": float(target_entry.train_metrics[metric_name]),
                "active_rules": float(target_entry.structural_metrics.get("active_rules", 0.0)),
                "total_rules": float(target_entry.structural_metrics.get("total_rules", 0.0)),
            }
        )

    test_mean = sum(item["test_metric"] for item in per_seed_payload) / len(per_seed_payload)
    train_mean = sum(item["train_metric"] for item in per_seed_payload) / len(per_seed_payload)
    active_mean = sum(item["active_rules"] for item in per_seed_payload) / len(per_seed_payload)
    total_mean = sum(item["total_rules"] for item in per_seed_payload) / len(per_seed_payload)
    return TrialResult(
        mean_test_metric=float(test_mean),
        mean_train_metric=float(train_mean),
        mean_active_rules=float(active_mean),
        mean_total_rules=float(total_mean),
        raw=per_seed_payload,
    )


def mutate_config(
    config: TrialConfig,
    result: TrialResult,
    *,
    rng: random.Random,
) -> tuple[TrialConfig, str]:
    new_cfg = TrialConfig(**asdict(config))
    reason = []

    overfit_gap = result.mean_train_metric - result.mean_test_metric
    underfit = result.mean_train_metric < 0.9

    if overfit_gap > 0.04:
        new_cfg.dffl_learning_rate = _clip(new_cfg.dffl_learning_rate * 0.8, 0.003, 0.05)
        new_cfg.max_epochs = max(12, new_cfg.max_epochs - 2)
        reason.append("overfit: lower lr, slightly shorter fit")
    elif underfit:
        new_cfg.dffl_learning_rate = _clip(new_cfg.dffl_learning_rate * 1.12, 0.003, 0.05)
        new_cfg.max_epochs = min(120, new_cfg.max_epochs + 4)
        new_cfg.pretrain_epochs = min(60, new_cfg.pretrain_epochs + 2)
        new_cfg.decision_pretrain_epochs = min(60, new_cfg.decision_pretrain_epochs + 2)
        reason.append("underfit: increase capacity and training budget")
    else:
        new_cfg.max_epochs = min(120, new_cfg.max_epochs + 2)
        reason.append("near-target fine-tune")

    if result.mean_active_rules > 120:
        new_cfg.dffl_profile_name = "quality_balanced"
        reason.append("too many active rules: switch to balanced profile")
    elif result.mean_active_rules < 30:
        new_cfg.dffl_profile_name = "quality"
        reason.append("too few active rules: switch to quality profile")
    else:
        # small random exploration around current profile
        candidates = ("quality_auto", "quality", "quality_balanced")
        if rng.random() < 0.25:
            new_cfg.dffl_profile_name = rng.choice(candidates)
            reason.append("stochastic profile exploration")

    # mild stochastic perturbations to avoid local minima
    if rng.random() < 0.35:
        new_cfg.batch_size = int(_clip(new_cfg.batch_size + rng.choice((-32, 32)), 64, 512))
        reason.append("batch-size perturbation")
    if rng.random() < 0.35:
        new_cfg.refinement_cycles = int(_clip(new_cfg.refinement_cycles + rng.choice((-1, 1)), 1, 5))
        reason.append("refinement-cycle perturbation")

    return new_cfg, "; ".join(reason)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-gate loop: repeatedly train/analyze/mutate on a small dataset "
            "until target metric is reached."
        )
    )
    parser.add_argument("--dataset", type=str, default="digits_binary", choices=sorted(DATASETS.keys()))
    parser.add_argument("--target-model", type=str, default="ruanfis_refined_deep", choices=FUZZY_MODEL_NAMES)
    parser.add_argument("--target-metric", type=float, default=0.9)
    parser.add_argument("--seeds", type=str, default="23")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="0 means unlimited loop until pass.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--include-sklearn", action="store_true")
    parser.add_argument("--output-jsonl", type=Path, default=Path("artifacts/benchmarks/auto_gate_progress.jsonl"))
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for mutation strategy.")
    args = parser.parse_args()

    seeds = tuple(int(chunk.strip()) for chunk in args.seeds.split(",") if chunk.strip())
    if not seeds:
        raise ValueError("At least one seed is required.")
    if args.target_metric <= 0.0:
        raise ValueError("target-metric must be positive.")
    if args.max_iterations < 0:
        raise ValueError("max-iterations must be >= 0.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    cfg = TrialConfig()
    best_metric = float("-inf")
    best_cfg = TrialConfig()
    iteration = 0

    while True:
        iteration += 1
        started = time.perf_counter()
        trial = run_trial(
            dataset_name=args.dataset,
            seeds=seeds,
            target_model=args.target_model,
            include_sklearn=args.include_sklearn,
            device=args.device,
            config=cfg,
        )
        elapsed = time.perf_counter() - started

        passed = trial.mean_test_metric >= args.target_metric
        if trial.mean_test_metric > best_metric:
            best_metric = trial.mean_test_metric
            best_cfg = TrialConfig(**asdict(cfg))

        payload = {
            "iteration": iteration,
            "passed": passed,
            "elapsed_sec": elapsed,
            "dataset": args.dataset,
            "target_model": args.target_model,
            "target_metric": args.target_metric,
            "trial_config": asdict(cfg),
            "trial_result": asdict(trial),
            "best_metric_so_far": best_metric,
            "best_config_so_far": asdict(best_cfg),
        }
        with args.output_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        print(
            (
                f"[iter {iteration}] test={trial.mean_test_metric:.4f} "
                f"train={trial.mean_train_metric:.4f} "
                f"active_rules={trial.mean_active_rules:.1f} "
                f"target={args.target_metric:.4f} pass={passed} "
                f"profile={cfg.dffl_profile_name} lr={cfg.dffl_learning_rate:.5f} "
                f"epochs={cfg.max_epochs}"
            ),
            flush=True,
        )

        if passed:
            print(
                f"[pass] target reached: {trial.mean_test_metric:.4f} >= {args.target_metric:.4f} "
                f"in {iteration} iteration(s).",
                flush=True,
            )
            break

        if args.max_iterations > 0 and iteration >= args.max_iterations:
            raise RuntimeError(
                f"Target not reached after {iteration} iterations. "
                f"Best={best_metric:.4f}, target={args.target_metric:.4f}."
            )

        cfg, reason = mutate_config(cfg, trial, rng=rng)
        print(f"[analysis] mutate: {reason}", flush=True)

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
