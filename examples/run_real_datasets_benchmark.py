from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_linnerud, load_wine
from sklearn.datasets import fetch_california_housing, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ruanfis import (  # noqa: E402
    BootstrapConfig,
    DecisionLayerConfig,
    FuzzyVariable,
    GaussianMembership,
    HierarchicalAnfisBlockConfig,
    HierarchicalAnfisModelConfig,
    HierarchicalAnfisStageConfig,
    HierarchicalModelConfig,
    MultiSeedBenchmarkResult,
    RefinementLoopConfig,
    ShallowFuzzyModelConfig,
    StackedAnfisLayerConfig,
    StackedAnfisModelConfig,
    StageConfig,
    StagewisePretrainingConfig,
    TrainingConfig,
    TransparentBlockConfig,
    build_bootstrapped_shallow_model,
    build_hierarchical_anfis_model,
    build_refined_hierarchical_model,
    build_stacked_anfis_model,
    aggregate_benchmark_results,
    evaluate_trained_model,
    format_aggregated_benchmark_results,
    format_benchmark_results,
    format_paper_benchmark_markdown_table,
    run_multi_seed_benchmark,
    run_tabular_benchmark,
    save_multi_seed_benchmark_results_json,
    save_paper_benchmark_markdown_table,
    serialize_multi_seed_benchmark_result,
)
from ruanfis.metrics import TaskType, compute_metrics
from ruanfis.trainer import FuzzyTrainer


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task_type: TaskType
    loader: Callable[[], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class DatasetSplit:
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    validation_inputs: torch.Tensor
    validation_targets: torch.Tensor
    test_inputs: torch.Tensor
    test_targets: torch.Tensor
    n_samples: int
    input_dim: int


@dataclass(frozen=True)
class DfflProfile:
    name: str
    local_concepts: int
    local_max_rules: int
    aggregate_max_rules: int
    decision_max_rules: int
    stage2_width_min: int
    stage2_width_max: int
    local_rule_generation_mode: str
    aggregate_rule_generation_mode: str
    decision_rule_generation_mode: str
    learning_rate_scale_regression: float = 1.0
    learning_rate_scale_classification: float = 1.0
    refinement_cycle_floor: int = 1


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def progress_log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[progress {timestamp}] {message}", flush=True)


def parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


def parse_dataset_names(raw: str) -> tuple[str, ...]:
    names = tuple(chunk.strip() for chunk in raw.split(",") if chunk.strip())
    if not names:
        raise ValueError("At least one dataset name must be provided.")
    return names


def var3(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.2, 0.5, 0.8], [0.18, 0.18, 0.18], term_names=["low", "mid", "high"]),
    )


def var2(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.3, 0.7], [0.22, 0.22], term_names=["low", "high"]),
    )


def _load_diabetes() -> tuple[np.ndarray, np.ndarray]:
    data = load_diabetes()
    return data.data.astype(np.float32), data.target.astype(np.float32)


def _load_linnerud_weight() -> tuple[np.ndarray, np.ndarray]:
    data = load_linnerud()
    return data.data.astype(np.float32), data.target[:, 0].astype(np.float32)


def _load_breast_cancer_binary() -> tuple[np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    return data.data.astype(np.float32), data.target.astype(np.float32)


def _load_wine_binary() -> tuple[np.ndarray, np.ndarray]:
    data = load_wine()
    target = (data.target == 0).astype(np.float32)
    return data.data.astype(np.float32), target


def _load_digits_binary() -> tuple[np.ndarray, np.ndarray]:
    data = load_digits()
    target = (data.target == 0).astype(np.float32)
    return data.data.astype(np.float32), target


def _load_california_housing() -> tuple[np.ndarray, np.ndarray]:
    data = fetch_california_housing()
    return data.data.astype(np.float32), data.target.astype(np.float32)


def _load_covtype_binary_20000() -> tuple[np.ndarray, np.ndarray]:
    features, target = fetch_covtype(return_X_y=True)
    target_binary = (target == 2).astype(np.float32)
    features_subset, _, target_subset, _ = train_test_split(
        features,
        target_binary,
        train_size=20_000,
        random_state=42,
        stratify=target_binary,
    )
    return features_subset.astype(np.float32), target_subset.astype(np.float32)


def _load_covtype_binary_8000() -> tuple[np.ndarray, np.ndarray]:
    features, target = fetch_covtype(return_X_y=True)
    target_binary = (target == 2).astype(np.float32)
    features_subset, _, target_subset, _ = train_test_split(
        features,
        target_binary,
        train_size=8_000,
        random_state=42,
        stratify=target_binary,
    )
    return features_subset.astype(np.float32), target_subset.astype(np.float32)


DATASETS: dict[str, DatasetSpec] = {
    "diabetes": DatasetSpec(name="diabetes", task_type="regression", loader=_load_diabetes),
    "linnerud_weight": DatasetSpec(name="linnerud_weight", task_type="regression", loader=_load_linnerud_weight),
    "breast_cancer": DatasetSpec(
        name="breast_cancer",
        task_type="binary_classification",
        loader=_load_breast_cancer_binary,
    ),
    "wine_binary": DatasetSpec(name="wine_binary", task_type="binary_classification", loader=_load_wine_binary),
    "digits_binary": DatasetSpec(name="digits_binary", task_type="binary_classification", loader=_load_digits_binary),
    "california_housing": DatasetSpec(
        name="california_housing",
        task_type="regression",
        loader=_load_california_housing,
    ),
    "covtype_binary_20000": DatasetSpec(
        name="covtype_binary_20000",
        task_type="binary_classification",
        loader=_load_covtype_binary_20000,
    ),
    "covtype_binary_8000": DatasetSpec(
        name="covtype_binary_8000",
        task_type="binary_classification",
        loader=_load_covtype_binary_8000,
    ),
}


PRIMARY_METRIC: dict[TaskType, str] = {
    "regression": "rmse",
    "binary_classification": "f1",
}


DFFL_PROFILES: dict[str, DfflProfile] = {
    "baseline": DfflProfile(
        name="baseline",
        local_concepts=2,
        local_max_rules=8,
        aggregate_max_rules=12,
        decision_max_rules=10,
        stage2_width_min=3,
        stage2_width_max=6,
        local_rule_generation_mode="prototype",
        aggregate_rule_generation_mode="prototype",
        decision_rule_generation_mode="prototype",
        learning_rate_scale_regression=1.0,
        learning_rate_scale_classification=1.0,
        refinement_cycle_floor=1,
    ),
    "quality": DfflProfile(
        name="quality",
        local_concepts=3,
        local_max_rules=12,
        aggregate_max_rules=20,
        decision_max_rules=14,
        stage2_width_min=4,
        stage2_width_max=8,
        local_rule_generation_mode="enumerate",
        aggregate_rule_generation_mode="prototype",
        decision_rule_generation_mode="prototype",
        learning_rate_scale_regression=1.0,
        learning_rate_scale_classification=0.8,
        refinement_cycle_floor=2,
    ),
    "quality_balanced": DfflProfile(
        name="quality_balanced",
        local_concepts=3,
        local_max_rules=10,
        aggregate_max_rules=16,
        decision_max_rules=12,
        stage2_width_min=4,
        stage2_width_max=8,
        local_rule_generation_mode="prototype",
        aggregate_rule_generation_mode="prototype",
        decision_rule_generation_mode="prototype",
        learning_rate_scale_regression=1.0,
        learning_rate_scale_classification=0.9,
        refinement_cycle_floor=2,
    ),
    "quality_large_cls": DfflProfile(
        name="quality_large_cls",
        local_concepts=2,
        local_max_rules=6,
        aggregate_max_rules=14,
        decision_max_rules=10,
        stage2_width_min=4,
        stage2_width_max=7,
        local_rule_generation_mode="prototype",
        aggregate_rule_generation_mode="prototype",
        decision_rule_generation_mode="prototype",
        learning_rate_scale_regression=1.0,
        learning_rate_scale_classification=1.05,
        refinement_cycle_floor=2,
    ),
}


def resolve_dffl_profile(
    *,
    profile_name: str,
    task_type: TaskType,
    n_samples: int,
) -> DfflProfile:
    if profile_name == "quality_auto":
        # Large binary datasets benefit from a tighter, less explosive rule budget.
        if task_type == "binary_classification" and n_samples >= 8_000:
            return DFFL_PROFILES["quality_large_cls"]
        if task_type == "binary_classification" and n_samples >= 500:
            return DFFL_PROFILES["quality"]
        return DFFL_PROFILES["quality"]
    return DFFL_PROFILES[profile_name]


def _make_groups(total_dim: int, group_size: int) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    for start in range(0, total_dim, group_size):
        groups.append(tuple(range(start, min(start + group_size, total_dim))))
    return tuple(groups)


def _concept_width(input_dim: int) -> int:
    return min(8, max(4, int(round(input_dim**0.5))))


def _hidden_width(input_dim: int) -> int:
    return min(6, max(3, _concept_width(input_dim) // 2 + 1))


def build_shallow_config(input_dim: int) -> ShallowFuzzyModelConfig:
    concept_dim = _concept_width(input_dim)
    return ShallowFuzzyModelConfig(
        input_dim=input_dim,
        feature_block=TransparentBlockConfig(
            name="shallow_block",
            input_indices=tuple(range(input_dim)),
            variables=tuple(var3(f"x{i}") for i in range(input_dim)),
            n_concepts=concept_dim,
            concept_names=tuple(f"c{i}" for i in range(concept_dim)),
            max_rule_arity=2,
            max_rules=24,
            rule_generation_mode="prototype",
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=tuple(var2(f"c{i}") for i in range(concept_dim)),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=12,
            rule_generation_mode="prototype",
        ),
    )


def build_stacked_config(input_dim: int) -> StackedAnfisModelConfig:
    hidden_1 = _concept_width(input_dim)
    hidden_2 = _hidden_width(input_dim)
    return StackedAnfisModelConfig(
        input_dim=input_dim,
        layers=(
            StackedAnfisLayerConfig(
                name="stacked_layer_1",
                variables=tuple(var3(f"x{i}") for i in range(input_dim)),
                output_dim=hidden_1,
                output_names=tuple(f"h1_{i}" for i in range(hidden_1)),
                max_rule_arity=2,
                max_rules=24,
                rule_generation_mode="prototype",
            ),
            StackedAnfisLayerConfig(
                name="stacked_layer_2",
                variables=tuple(var3(f"h1_{i}") for i in range(hidden_1)),
                output_dim=hidden_2,
                output_names=tuple(f"h2_{i}" for i in range(hidden_2)),
                max_rule_arity=2,
                max_rules=16,
                rule_generation_mode="prototype",
            ),
            StackedAnfisLayerConfig(
                name="stacked_layer_3",
                variables=tuple(var3(f"h2_{i}") for i in range(hidden_2)),
                output_dim=1,
                output_names=("target",),
                max_rule_arity=2,
                max_rules=10,
                rule_generation_mode="prototype",
            ),
        ),
    )


def build_hierarchical_anfis_config(input_dim: int) -> HierarchicalAnfisModelConfig:
    groups = _make_groups(input_dim, group_size=4)
    stage_1_blocks = []
    for block_index, indices in enumerate(groups):
        stage_1_blocks.append(
            HierarchicalAnfisBlockConfig(
                name=f"anfis_local_{block_index}",
                input_indices=indices,
                variables=tuple(var3(f"x{idx}") for idx in indices),
                output_dim=2,
                output_names=(f"s1_{block_index}_0", f"s1_{block_index}_1"),
                max_rule_arity=min(2, len(indices)),
                max_rules=8,
                rule_generation_mode="prototype",
            )
        )

    stage_1_width = 2 * len(groups)
    stage_2_width = min(6, max(3, len(groups) // 2 + 2))

    return HierarchicalAnfisModelConfig(
        input_dim=input_dim,
        stages=(
            HierarchicalAnfisStageConfig(
                name="anfis_stage_1_local",
                blocks=tuple(stage_1_blocks),
            ),
            HierarchicalAnfisStageConfig(
                name="anfis_stage_2_aggregate",
                blocks=(
                    HierarchicalAnfisBlockConfig(
                        name="anfis_aggregate",
                        input_indices=tuple(range(stage_1_width)),
                        variables=tuple(var3(f"s1_{i}") for i in range(stage_1_width)),
                        output_dim=stage_2_width,
                        output_names=tuple(f"s2_{i}" for i in range(stage_2_width)),
                        max_rule_arity=2,
                        max_rules=16,
                        rule_generation_mode="prototype",
                    ),
                ),
            ),
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=tuple(var2(f"s2_{i}") for i in range(stage_2_width)),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=10,
            rule_generation_mode="prototype",
        ),
    )


def build_dffl_config(input_dim: int, profile: DfflProfile) -> HierarchicalModelConfig:
    groups = _make_groups(input_dim, group_size=4)
    stage_1_blocks = []
    for block_index, indices in enumerate(groups):
        stage_1_blocks.append(
            TransparentBlockConfig(
                name=f"dffl_local_{block_index}",
                input_indices=indices,
                variables=tuple(var3(f"x{idx}") for idx in indices),
                n_concepts=profile.local_concepts,
                concept_names=tuple(
                    f"s1_{block_index}_{concept_index}" for concept_index in range(profile.local_concepts)
                ),
                max_rule_arity=min(2, len(indices)),
                max_rules=profile.local_max_rules,
                rule_generation_mode=profile.local_rule_generation_mode,
            )
        )

    stage_1_width = profile.local_concepts * len(groups)
    stage_2_width = min(
        profile.stage2_width_max,
        max(profile.stage2_width_min, len(groups) + 1),
    )

    return HierarchicalModelConfig(
        input_dim=input_dim,
        stages=(
            StageConfig(
                name="dffl_stage_1_local",
                blocks=tuple(stage_1_blocks),
            ),
            StageConfig(
                name="dffl_stage_2_aggregate",
                blocks=(
                    TransparentBlockConfig(
                        name="dffl_aggregate",
                        input_indices=tuple(range(stage_1_width)),
                        variables=tuple(var3(f"s1_{i}") for i in range(stage_1_width)),
                        n_concepts=stage_2_width,
                        concept_names=tuple(f"s2_{i}" for i in range(stage_2_width)),
                        max_rule_arity=2,
                        max_rules=profile.aggregate_max_rules,
                        rule_generation_mode=profile.aggregate_rule_generation_mode,
                    ),
                ),
            ),
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=tuple(var2(f"s2_{i}") for i in range(stage_2_width)),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=profile.decision_max_rules,
            rule_generation_mode=profile.decision_rule_generation_mode,
        ),
    )


def _scale_regression_targets(
    train_targets: np.ndarray,
    validation_targets: np.ndarray,
    test_targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_targets.reshape(-1, 1)).astype(np.float32)
    validation_scaled = scaler.transform(validation_targets.reshape(-1, 1)).astype(np.float32)
    test_scaled = scaler.transform(test_targets.reshape(-1, 1)).astype(np.float32)
    return train_scaled, validation_scaled, test_scaled


def _choose_best_classification_threshold(
    model: torch.nn.Module,
    *,
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    default_threshold: float,
) -> float:
    if validation_inputs.numel() == 0 or validation_targets.numel() == 0:
        return default_threshold

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        validation_logits = model(validation_inputs.to(device=device, dtype=torch.float32)).detach().cpu()

    best_threshold = float(default_threshold)
    best_f1 = float("-inf")
    validation_targets_cpu = validation_targets.detach().cpu()
    for threshold in np.linspace(0.30, 0.70, 21):
        metrics = compute_metrics(
            "binary_classification",
            validation_logits,
            validation_targets_cpu,
            classification_threshold=float(threshold),
        )
        current_f1 = float(metrics["f1"])
        if current_f1 > best_f1 + 1e-12:
            best_f1 = current_f1
            best_threshold = float(threshold)
            continue
        if abs(current_f1 - best_f1) <= 1e-12 and abs(float(threshold) - default_threshold) < abs(
            best_threshold - default_threshold
        ):
            best_threshold = float(threshold)

    return best_threshold


def prepare_dataset_split(
    spec: DatasetSpec,
    *,
    seed: int,
    test_size: float,
    validation_size: float,
) -> DatasetSplit:
    features, targets = spec.loader()
    if features.ndim != 2:
        raise ValueError(f"Dataset '{spec.name}' must provide a 2D feature matrix.")

    stratify = targets if spec.task_type == "binary_classification" else None
    train_features, test_features, train_targets, test_targets = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    if spec.task_type == "binary_classification":
        stratify_inner = train_targets
    else:
        stratify_inner = None

    inner_val_fraction = validation_size / (1.0 - test_size)
    train_features, validation_features, train_targets, validation_targets = train_test_split(
        train_features,
        train_targets,
        test_size=inner_val_fraction,
        random_state=seed + 1,
        stratify=stratify_inner,
    )

    feature_scaler = MinMaxScaler()
    train_features_scaled = feature_scaler.fit_transform(train_features).astype(np.float32)
    validation_features_scaled = feature_scaler.transform(validation_features).astype(np.float32)
    test_features_scaled = feature_scaler.transform(test_features).astype(np.float32)

    train_features_scaled = np.clip(train_features_scaled, 0.0, 1.0)
    validation_features_scaled = np.clip(validation_features_scaled, 0.0, 1.0)
    test_features_scaled = np.clip(test_features_scaled, 0.0, 1.0)

    if spec.task_type == "regression":
        train_targets_scaled, validation_targets_scaled, test_targets_scaled = _scale_regression_targets(
            train_targets,
            validation_targets,
            test_targets,
        )
    else:
        train_targets_scaled = train_targets.reshape(-1, 1).astype(np.float32)
        validation_targets_scaled = validation_targets.reshape(-1, 1).astype(np.float32)
        test_targets_scaled = test_targets.reshape(-1, 1).astype(np.float32)

    return DatasetSplit(
        train_inputs=torch.from_numpy(train_features_scaled),
        train_targets=torch.from_numpy(train_targets_scaled),
        validation_inputs=torch.from_numpy(validation_features_scaled),
        validation_targets=torch.from_numpy(validation_targets_scaled),
        test_inputs=torch.from_numpy(test_features_scaled),
        test_targets=torch.from_numpy(test_targets_scaled),
        n_samples=int(features.shape[0]),
        input_dim=int(features.shape[1]),
    )


def run_single_seed_dataset_benchmark(
    spec: DatasetSpec,
    *,
    seed: int,
    test_size: float,
    validation_size: float,
    train_noise_sigma: float,
    pretrain_epochs: int,
    decision_pretrain_epochs: int,
    max_epochs: int,
    refinement_cycles: int,
    fuzzy_learning_rate: float,
    dffl_learning_rate: float,
    dffl_profile_name: str,
    batch_size: int,
    patience: int,
    classification_threshold: float,
    tune_fuzzy_threshold: bool,
    device: str | None,
):
    phase_started_at = time.perf_counter()
    set_seed(seed)
    progress_log(f"seed={seed} split: start")
    split = prepare_dataset_split(
        spec,
        seed=seed,
        test_size=test_size,
        validation_size=validation_size,
    )
    progress_log(
        "seed={seed} split: done in {elapsed:.2f}s (n={n_samples}, d={input_dim})".format(
            seed=seed,
            elapsed=time.perf_counter() - phase_started_at,
            n_samples=split.n_samples,
            input_dim=split.input_dim,
        )
    )

    train_inputs = split.train_inputs
    train_targets = split.train_targets
    validation_inputs = split.validation_inputs
    validation_targets = split.validation_targets
    test_inputs = split.test_inputs
    test_targets = split.test_targets

    if train_noise_sigma < 0.0:
        raise ValueError("train_noise_sigma must be non-negative.")
    if train_noise_sigma > 0.0 and spec.task_type == "regression":
        train_targets = train_targets + train_noise_sigma * torch.randn_like(train_targets)

    dffl_profile = resolve_dffl_profile(
        profile_name=dffl_profile_name,
        task_type=spec.task_type,
        n_samples=split.n_samples,
    )
    progress_log(
        f"seed={seed} profile: dffl={dffl_profile.name}, task={spec.task_type}, device={device or 'default'}"
    )

    dffl_learning_rate_effective = (
        dffl_learning_rate * dffl_profile.learning_rate_scale_classification
        if spec.task_type == "binary_classification"
        else dffl_learning_rate * dffl_profile.learning_rate_scale_regression
    )
    dffl_refinement_cycles = max(refinement_cycles, dffl_profile.refinement_cycle_floor)

    phase_started_at = time.perf_counter()
    progress_log(f"seed={seed} model=dffl: start")
    dffl_result = build_refined_hierarchical_model(
        build_dffl_config(split.input_dim, profile=dffl_profile),
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        bootstrap_config=BootstrapConfig(decision_task_type=spec.task_type),
        pretraining_config=StagewisePretrainingConfig(
            task_type=spec.task_type,
            epochs_per_stage=pretrain_epochs,
            decision_epochs=decision_pretrain_epochs,
            learning_rate=dffl_learning_rate_effective,
            batch_size=batch_size,
            shuffle=True,
        ),
        training_config=TrainingConfig(
            task_type=spec.task_type,
            max_epochs=max_epochs,
            learning_rate=dffl_learning_rate_effective,
            patience=min(patience, max_epochs),
            batch_size=batch_size,
            shuffle=True,
            classification_threshold=classification_threshold,
            device=device,
        ),
        refinement_loop_config=RefinementLoopConfig(
            max_cycles=dffl_refinement_cycles,
            patience=1,
            min_delta=1e-4,
        ),
    )
    progress_log(f"seed={seed} model=dffl: done in {time.perf_counter() - phase_started_at:.2f}s")

    phase_started_at = time.perf_counter()
    progress_log(f"seed={seed} model=shallow: bootstrap+train start")
    shallow_model = build_bootstrapped_shallow_model(
        build_shallow_config(split.input_dim),
        sample_inputs=train_inputs,
        sample_targets=train_targets,
        bootstrap_config=BootstrapConfig(decision_task_type=spec.task_type),
        device=device,
    )
    shallow_trainer = FuzzyTrainer(
        shallow_model,
        TrainingConfig(
            task_type=spec.task_type,
            max_epochs=max_epochs,
            learning_rate=fuzzy_learning_rate,
            patience=min(patience, max_epochs),
            batch_size=batch_size,
            shuffle=True,
            classification_threshold=classification_threshold,
            device=device,
        ),
    )
    shallow_trainer.fit(train_inputs, train_targets, validation_inputs, validation_targets)
    progress_log(f"seed={seed} model=shallow: done in {time.perf_counter() - phase_started_at:.2f}s")

    phase_started_at = time.perf_counter()
    progress_log(f"seed={seed} model=stacked: build+train start")
    stacked_model = build_stacked_anfis_model(
        build_stacked_config(split.input_dim),
        sample_inputs=train_inputs,
    )
    stacked_trainer = FuzzyTrainer(
        stacked_model,
        TrainingConfig(
            task_type=spec.task_type,
            max_epochs=max_epochs,
            learning_rate=fuzzy_learning_rate,
            patience=min(patience, max_epochs),
            batch_size=batch_size,
            shuffle=True,
            classification_threshold=classification_threshold,
            device=device,
        ),
    )
    stacked_trainer.fit(train_inputs, train_targets, validation_inputs, validation_targets)
    progress_log(f"seed={seed} model=stacked: done in {time.perf_counter() - phase_started_at:.2f}s")

    phase_started_at = time.perf_counter()
    progress_log(f"seed={seed} model=hierarchical_anfis: build+train start")
    hierarchical_anfis_model = build_hierarchical_anfis_model(
        build_hierarchical_anfis_config(split.input_dim),
        sample_inputs=train_inputs,
    )
    hierarchical_anfis_trainer = FuzzyTrainer(
        hierarchical_anfis_model,
        TrainingConfig(
            task_type=spec.task_type,
            max_epochs=max_epochs,
            learning_rate=fuzzy_learning_rate,
            patience=min(patience, max_epochs),
            batch_size=batch_size,
            shuffle=True,
            classification_threshold=classification_threshold,
            device=device,
        ),
    )
    hierarchical_anfis_trainer.fit(train_inputs, train_targets, validation_inputs, validation_targets)
    progress_log(
        f"seed={seed} model=hierarchical_anfis: done in {time.perf_counter() - phase_started_at:.2f}s"
    )

    phase_started_at = time.perf_counter()
    progress_log(f"seed={seed} sklearn: start")
    sklearn_results = run_tabular_benchmark(
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        task_type=spec.task_type,
        random_state=seed,
        classification_threshold=classification_threshold,
    )
    progress_log(f"seed={seed} sklearn: done in {time.perf_counter() - phase_started_at:.2f}s")

    phase_started_at = time.perf_counter()
    progress_log(f"seed={seed} fuzzy_eval: start")
    model_thresholds = {
        "ruanfis_shallow": classification_threshold,
        "ruanfis_stacked_anfis": classification_threshold,
        "ruanfis_hierarchical_anfis": classification_threshold,
        "ruanfis_refined_deep": classification_threshold,
    }
    if tune_fuzzy_threshold and spec.task_type == "binary_classification":
        model_thresholds["ruanfis_shallow"] = _choose_best_classification_threshold(
            shallow_model,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            default_threshold=classification_threshold,
        )
        model_thresholds["ruanfis_stacked_anfis"] = _choose_best_classification_threshold(
            stacked_model,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            default_threshold=classification_threshold,
        )
        model_thresholds["ruanfis_hierarchical_anfis"] = _choose_best_classification_threshold(
            hierarchical_anfis_model,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            default_threshold=classification_threshold,
        )
        model_thresholds["ruanfis_refined_deep"] = _choose_best_classification_threshold(
            dffl_result.model,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            default_threshold=classification_threshold,
        )
        progress_log(
            "seed={seed} threshold_tuning: shallow={shallow:.2f}, stacked={stacked:.2f}, "
            "hierarchical={hierarchical:.2f}, dffl={dffl:.2f}".format(
                seed=seed,
                shallow=model_thresholds["ruanfis_shallow"],
                stacked=model_thresholds["ruanfis_stacked_anfis"],
                hierarchical=model_thresholds["ruanfis_hierarchical_anfis"],
                dffl=model_thresholds["ruanfis_refined_deep"],
            )
        )

    fuzzy_results = (
        evaluate_trained_model(
            "ruanfis_shallow",
            shallow_model,
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            classification_threshold=model_thresholds["ruanfis_shallow"],
        ),
        evaluate_trained_model(
            "ruanfis_stacked_anfis",
            stacked_model,
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            classification_threshold=model_thresholds["ruanfis_stacked_anfis"],
        ),
        evaluate_trained_model(
            "ruanfis_hierarchical_anfis",
            hierarchical_anfis_model,
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            classification_threshold=model_thresholds["ruanfis_hierarchical_anfis"],
        ),
        evaluate_trained_model(
            "ruanfis_refined_deep",
            dffl_result.model,
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            classification_threshold=model_thresholds["ruanfis_refined_deep"],
        ),
    )
    progress_log(f"seed={seed} fuzzy_eval: done in {time.perf_counter() - phase_started_at:.2f}s")
    progress_log(f"seed={seed} all models: complete")

    return (*sklearn_results, *fuzzy_results)


def render_dataset_multi_seed_report(spec: DatasetSpec, result: MultiSeedBenchmarkResult) -> str:
    lines = [
        f"DATASET: {spec.name}",
        f"task: {spec.task_type}",
        f"seeds: {', '.join(str(seed) for seed in result.seeds)}",
        "",
    ]

    for seed, seed_results in zip(result.seeds, result.per_seed_results, strict=True):
        lines.append(f"SEED {seed}")
        lines.append(format_benchmark_results(seed_results))
        lines.append("")

    lines.append(format_aggregated_benchmark_results(result.aggregated_results))
    lines.append("")
    lines.append("PAPER-READY SUMMARY TABLE")
    lines.append(format_paper_benchmark_markdown_table(result.aggregated_results))
    return "\n".join(lines)


def _primary_score(entry, task_type: TaskType) -> float:
    metric_name = PRIMARY_METRIC[task_type]
    return float(entry.test_metrics[metric_name].mean)


def _rank_scores(values: dict[str, float], higher_is_better: bool) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=higher_is_better)
    return {model_name: float(index + 1) for index, (model_name, _) in enumerate(ordered)}


def build_cross_dataset_summary(
    dataset_results: dict[str, MultiSeedBenchmarkResult],
    specs: dict[str, DatasetSpec],
) -> str:
    fuzzy_model_names = (
        "ruanfis_shallow",
        "ruanfis_stacked_anfis",
        "ruanfis_hierarchical_anfis",
        "ruanfis_refined_deep",
    )

    per_dataset_scores: dict[str, dict[str, float]] = {}
    per_dataset_ranks: dict[str, dict[str, float]] = {}

    for dataset_name, multi_seed in dataset_results.items():
        spec = specs[dataset_name]
        aggregated = {
            entry.model_name: entry
            for entry in multi_seed.aggregated_results
            if entry.model_name in fuzzy_model_names
        }
        scores = {
            model_name: _primary_score(aggregated[model_name], spec.task_type)
            for model_name in fuzzy_model_names
        }
        per_dataset_scores[dataset_name] = scores
        per_dataset_ranks[dataset_name] = _rank_scores(
            scores,
            higher_is_better=(spec.task_type == "binary_classification"),
        )

    lines = [
        "| model | avg_rank | wins | avg_rules | avg_active_rule_jaccard | "
        + " | ".join(
            f"{dataset_name}:{PRIMARY_METRIC[specs[dataset_name].task_type]}"
            for dataset_name in dataset_results.keys()
        )
        + " |",
        "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in dataset_results.keys()) + " |",
    ]

    for model_name in fuzzy_model_names:
        ranks = [per_dataset_ranks[dataset_name][model_name] for dataset_name in dataset_results.keys()]
        wins = sum(1 for rank in ranks if abs(rank - 1.0) < 1e-9)

        total_rules_values: list[float] = []
        stability_values: list[float] = []
        dataset_metric_cells: list[str] = []

        for dataset_name, multi_seed in dataset_results.items():
            entry_map = {entry.model_name: entry for entry in multi_seed.aggregated_results}
            entry = entry_map[model_name]
            if "total_rules" in entry.structural_metrics:
                total_rules_values.append(float(entry.structural_metrics["total_rules"].mean))
            if "active_rule_jaccard" in entry.stability_metrics:
                stability_values.append(float(entry.stability_metrics["active_rule_jaccard"]))

            score = per_dataset_scores[dataset_name][model_name]
            dataset_metric_cells.append(f"{score:.4f}")

        avg_rank = float(sum(ranks) / len(ranks))
        avg_rules = float(sum(total_rules_values) / len(total_rules_values)) if total_rules_values else float("nan")
        avg_stability = float(sum(stability_values) / len(stability_values)) if stability_values else float("nan")

        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    f"{avg_rank:.3f}",
                    str(wins),
                    f"{avg_rules:.2f}",
                    f"{avg_stability:.4f}",
                    *dataset_metric_cells,
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def _find_model_entry(seed_results, model_name: str):
    for entry in seed_results:
        if entry.model_name == model_name:
            return entry
    return None


def build_interpretability_report(
    dataset_results: dict[str, MultiSeedBenchmarkResult],
    specs: dict[str, DatasetSpec],
    *,
    top_rules: int = 8,
) -> str:
    fuzzy_model_names = (
        "ruanfis_shallow",
        "ruanfis_stacked_anfis",
        "ruanfis_hierarchical_anfis",
        "ruanfis_refined_deep",
    )

    lines: list[str] = [
        "# Interpretable Rule Structure Report",
        "",
        "The report summarizes rule-level artifacts collected from repeated runs.",
        "It provides compact, human-readable evidence of hidden and decision rule activation patterns.",
        "",
    ]

    for dataset_name, multi_seed in dataset_results.items():
        lines.append(f"## Dataset: {dataset_name}")
        lines.append(f"task: {specs[dataset_name].task_type}")
        lines.append(f"seeds: {', '.join(str(seed) for seed in multi_seed.seeds)}")
        lines.append("")

        aggregated_map = {entry.model_name: entry for entry in multi_seed.aggregated_results}
        for model_name in fuzzy_model_names:
            if model_name not in aggregated_map:
                continue

            aggregated = aggregated_map[model_name]
            active_rule_jaccard = aggregated.stability_metrics.get("active_rule_jaccard", float("nan"))
            decision_jaccard = aggregated.stability_metrics.get("decision_active_rule_jaccard", float("nan"))
            total_rules = aggregated.structural_metrics.get("total_rules")
            active_rules = aggregated.structural_metrics.get("active_rules")
            total_rules_mean = float(total_rules.mean) if total_rules is not None else float("nan")
            active_rules_mean = float(active_rules.mean) if active_rules is not None else float("nan")

            decision_counter: Counter[str] = Counter()
            hidden_counter: Counter[str] = Counter()
            for seed_results in multi_seed.per_seed_results:
                entry = _find_model_entry(seed_results, model_name)
                if entry is None:
                    continue
                decision_counter.update(entry.stability_artifacts.get("active:decision", ()))
                hidden_counter.update(entry.stability_artifacts.get("active:hidden", ()))

            seed_count = len(multi_seed.seeds)
            lines.append(f"### {model_name}")
            lines.append(
                "structure: total_rules={:.2f}, active_rules={:.2f}; stability: active_jaccard={:.4f}, decision_jaccard={:.4f}".format(
                    total_rules_mean,
                    active_rules_mean,
                    active_rule_jaccard,
                    decision_jaccard,
                )
            )
            lines.append("")

            lines.append("Top decision rules by activation frequency:")
            if decision_counter:
                for rule, count in decision_counter.most_common(top_rules):
                    lines.append(f"- [{count}/{seed_count}] {rule}")
            else:
                lines.append("- (no decision-level active rules captured)")
            lines.append("")

            lines.append("Top hidden rules by activation frequency:")
            if hidden_counter:
                for rule, count in hidden_counter.most_common(top_rules):
                    lines.append(f"- [{count}/{seed_count}] {rule}")
            else:
                lines.append("- (no hidden-level active rules captured)")
            lines.append("")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a unified multi-seed benchmark on 3-5 real tabular datasets for stacked/hierarchical/DFFL models."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="diabetes,linnerud_weight,breast_cancer,wine_binary,digits_binary",
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--seeds", type=str, default="19,23,29")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--train-noise-sigma", type=float, default=0.0)
    parser.add_argument("--pretrain-epochs", type=int, default=18)
    parser.add_argument("--decision-pretrain-epochs", type=int, default=14)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--refinement-cycles", type=int, default=2)
    parser.add_argument("--fuzzy-learning-rate", type=float, default=0.02)
    parser.add_argument("--dffl-learning-rate", type=float, default=0.015)
    parser.add_argument(
        "--dffl-profile",
        type=str,
        default="quality_auto",
        choices=tuple(DFFL_PROFILES.keys()) + ("quality_auto",),
        help="DFFL tuning profile: fixed baseline/quality/quality_balanced or adaptive quality_auto.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument(
        "--tune-fuzzy-threshold",
        action="store_true",
        help="Tune binary-classification threshold per fuzzy model on validation split (max F1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device for fuzzy models, e.g. 'cpu' or 'cuda'. If omitted, use model default.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-table-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--interpretability-report-output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset_names = parse_dataset_names(args.datasets)
    unknown = [name for name in dataset_names if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset names: {', '.join(unknown)}. Available: {', '.join(DATASETS.keys())}")

    seeds = parse_seeds(args.seeds)

    dataset_results: dict[str, MultiSeedBenchmarkResult] = {}
    dataset_reports: dict[str, str] = {}

    for dataset_name in dataset_names:
        spec = DATASETS[dataset_name]
        dataset_features, _ = spec.loader()
        resolved_profile = resolve_dffl_profile(
            profile_name=args.dffl_profile,
            task_type=spec.task_type,
            n_samples=int(dataset_features.shape[0]),
        )
        per_seed_results = []
        total_seeds = len(seeds)
        for seed_index, seed in enumerate(seeds, start=1):
            print(
                f"[progress] dataset={dataset_name} seed={seed} start ({seed_index}/{total_seeds})",
                flush=True,
            )
            seed_results = run_single_seed_dataset_benchmark(
                spec,
                seed=seed,
                test_size=args.test_size,
                validation_size=args.validation_size,
                train_noise_sigma=args.train_noise_sigma,
                pretrain_epochs=args.pretrain_epochs,
                decision_pretrain_epochs=args.decision_pretrain_epochs,
                max_epochs=args.max_epochs,
                refinement_cycles=args.refinement_cycles,
                fuzzy_learning_rate=args.fuzzy_learning_rate,
                dffl_learning_rate=args.dffl_learning_rate,
                dffl_profile_name=args.dffl_profile,
                batch_size=args.batch_size,
                patience=args.patience,
                classification_threshold=args.classification_threshold,
                tune_fuzzy_threshold=args.tune_fuzzy_threshold,
                device=args.device,
            )
            per_seed_results.append(tuple(seed_results))
            print(
                f"[progress] dataset={dataset_name} seed={seed} done ({seed_index}/{total_seeds})",
                flush=True,
            )

        benchmark = MultiSeedBenchmarkResult(
            seeds=seeds,
            per_seed_results=tuple(per_seed_results),
            aggregated_results=aggregate_benchmark_results(per_seed_results),
        )
        dataset_results[dataset_name] = benchmark
        dataset_reports[dataset_name] = (
            f"DFFL PROFILE USED: {resolved_profile.name}\n"
            + render_dataset_multi_seed_report(spec, benchmark)
        )

        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            dataset_report_path = args.output_dir / f"{dataset_name}_report.txt"
            dataset_json_path = args.output_dir / f"{dataset_name}_report.json"
            dataset_table_path = args.output_dir / f"{dataset_name}_paper_table.md"

            dataset_report_path.write_text(dataset_reports[dataset_name], encoding="utf-8")
            save_multi_seed_benchmark_results_json(benchmark, dataset_json_path)
            save_paper_benchmark_markdown_table(benchmark.aggregated_results, dataset_table_path)

    summary_table = build_cross_dataset_summary(dataset_results, {name: DATASETS[name] for name in dataset_names})
    interpretability_report = build_interpretability_report(
        dataset_results,
        {name: DATASETS[name] for name in dataset_names},
    )

    full_report_sections = [
        "UNIFIED REAL-DATASET BENCHMARK",
        f"datasets: {', '.join(dataset_names)}",
        f"seeds: {', '.join(str(seed) for seed in seeds)}",
        f"train_noise_sigma (regression only): {args.train_noise_sigma:.4f}",
        f"dffl_profile: {args.dffl_profile}",
        "",
    ]
    for dataset_name in dataset_names:
        full_report_sections.append(dataset_reports[dataset_name])
        full_report_sections.append("")
    full_report_sections.append("CROSS-DATASET FUZZY SUMMARY")
    full_report_sections.append(summary_table)
    full_report = "\n".join(full_report_sections)

    print(full_report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(full_report, encoding="utf-8")

    if args.summary_table_output is not None:
        args.summary_table_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_table_output.write_text(summary_table, encoding="utf-8")

    if args.interpretability_report_output is not None:
        args.interpretability_report_output.parent.mkdir(parents=True, exist_ok=True)
        args.interpretability_report_output.write_text(interpretability_report, encoding="utf-8")

    if args.json_output is not None:
        payload = {
            "datasets": list(dataset_names),
            "seeds": list(seeds),
            "train_noise_sigma": float(args.train_noise_sigma),
            "dffl_profile": args.dffl_profile,
            "dataset_results": {
                dataset_name: serialize_multi_seed_benchmark_result(result)
                for dataset_name, result in dataset_results.items()
            },
            "cross_dataset_summary_markdown": summary_table,
            "interpretability_report_markdown": interpretability_report,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.output_dir is not None:
        interpretability_path = args.output_dir / "interpretability_report.md"
        interpretability_path.write_text(interpretability_report, encoding="utf-8")


if __name__ == "__main__":
    main()
