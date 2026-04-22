from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
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
    build_bootstrapped_hierarchical_model,
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
    aggregate_block_count: int = 1
    aggregate_overlap: int = 0
    aggregate_global_context_dim: int = 0
    aggregate_global_max_rules: int = 0
    local_max_rule_arity: int = 2
    aggregate_max_rule_arity: int = 2
    decision_max_rule_arity: int = 2
    decision_three_terms: bool = False
    learning_rate_scale_regression: float = 1.0
    learning_rate_scale_classification: float = 1.0
    refinement_cycle_floor: int = 1
    pretrain_refinement_rounds: int = 1
    top_k_rules: int | None = None
    input_group_size: int = 4
    input_group_strategy: str = "contiguous"
    input_group_correlation_weight: float = 0.7
    local_prototype_term_limit: int = 2
    aggregate_prototype_term_limit: int = 2
    decision_prototype_term_limit: int = 2
    local_prototype_scoring_mode: str = "max"
    aggregate_prototype_scoring_mode: str = "max"
    decision_prototype_scoring_mode: str = "max"
    local_prototype_variable_pool_size: int | None = None
    aggregate_prototype_variable_pool_size: int | None = None
    decision_prototype_variable_pool_size: int | None = None
    local_prototype_sample_size: int | None = 256
    aggregate_prototype_sample_size: int | None = 256
    decision_prototype_sample_size: int | None = 256
    binary_auto_pos_weight: bool = False
    binary_soft_f1_weight: float = 0.0
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None
    rule_sparsity_weight: float = 0.0
    rule_length_weight: float = 0.0
    decision_usage_balance_weight: float = 0.0
    block_gates_enabled: bool = False
    block_gate_init_logit: float = 5.0
    block_gate_l1_weight: float = 0.0
    local_consequent_mode: str = "constant"
    aggregate_consequent_mode: str = "constant"
    bridge_enabled: bool = False
    bridge_top_pairs: int = 0
    bridge_pair_score_alpha: float = 0.5
    bridge_score_interaction_weight: float = 0.25
    bridge_score_stability_weight: float = 0.15
    bridge_max_pairs_per_feature: int = 2
    bridge_max_pairs_per_group_pair: int = 1
    bridge_concepts: int = 1
    bridge_concepts_max: int = 2
    bridge_concepts_adaptive: bool = True
    bridge_concepts_high_share_threshold: float = 0.82
    bridge_max_rules: int = 6
    bridge_max_rule_arity: int = 2
    bridge_rule_generation_mode: str = "prototype"
    bridge_prototype_term_limit: int = 2
    bridge_prototype_scoring_mode: str = "max"
    bridge_prototype_variable_pool_size: int | None = None
    bridge_prototype_sample_size: int | None = 256
    bridge_token_enabled: bool = True
    bridge_token_concepts: int = 1
    bridge_token_max_rules: int = 3
    bridge_token_max_rule_arity: int = 2
    stagewise_rule_swap_ratio: float = 0.0
    stagewise_rule_swap_min_keep: int = 0
    stage1_fixed_rule_budget: bool = True
    stage1_min_rules_per_block: int = 6
    total_rule_budget: int = 0
    block_agreement_weight: float = 0.0
    block_agreement_target_corr: float = 0.2
    block_agreement_stage_limit: int = 1


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


def parse_fuzzy_model_names(raw: str) -> tuple[str, ...]:
    chunks = tuple(chunk.strip() for chunk in raw.split(",") if chunk.strip())
    if not chunks:
        raise ValueError("At least one fuzzy model must be provided.")
    if len(chunks) == 1 and chunks[0].lower() == "all":
        return FUZZY_MODEL_NAMES

    aliases = {
        "shallow": "ruanfis_shallow",
        "stacked": "ruanfis_stacked_anfis",
        "hierarchical": "ruanfis_hierarchical_anfis",
        "hier": "ruanfis_hierarchical_anfis",
        "dffl": "ruanfis_refined_deep",
        "refined_deep": "ruanfis_refined_deep",
    }
    allowed = set(FUZZY_MODEL_NAMES)
    resolved: list[str] = []
    for chunk in chunks:
        model_name = aliases.get(chunk.lower(), chunk)
        if model_name not in allowed:
            raise ValueError(
                f"Unknown fuzzy model name: {chunk}. Allowed: all, {', '.join(FUZZY_MODEL_NAMES)}"
            )
        if model_name not in resolved:
            resolved.append(model_name)
    return tuple(resolved)


def _apply_gpu_only_mode(args: argparse.Namespace) -> None:
    if not bool(args.gpu_only):
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--gpu-only requires CUDA, but torch.cuda.is_available() is False. "
            "Use CUDA environment or run without --gpu-only."
        )
    # Enforce GPU fuzzy pipeline and disable CPU-only sklearn baselines.
    args.device = "cuda"
    args.skip_sklearn = True


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


def var_binary(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.05, 0.95], [0.12, 0.12], term_names=["off", "on"]),
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

FEATURE_GEOMETRIES: tuple[str, ...] = ("euclidean", "hyperbolic", "auto")
BINARY_HEAVY_GROUPING_MODES: tuple[str, ...] = ("binary_aware", "contiguous")

FUZZY_MODEL_NAMES: tuple[str, ...] = (
    "ruanfis_shallow",
    "ruanfis_stacked_anfis",
    "ruanfis_hierarchical_anfis",
    "ruanfis_refined_deep",
)


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
        local_max_rules=14,
        aggregate_max_rules=20,
        decision_max_rules=16,
        stage2_width_min=4,
        stage2_width_max=9,
        local_rule_generation_mode="prototype",
        aggregate_rule_generation_mode="prototype",
        decision_rule_generation_mode="prototype",
        aggregate_max_rule_arity=2,
        aggregate_global_context_dim=2,
        aggregate_global_max_rules=8,
        decision_three_terms=True,
        learning_rate_scale_regression=1.0,
        learning_rate_scale_classification=1.0,
        refinement_cycle_floor=2,
        local_prototype_term_limit=3,
        local_prototype_variable_pool_size=4,
        aggregate_prototype_term_limit=3,
        aggregate_prototype_variable_pool_size=8,
        decision_prototype_term_limit=3,
        decision_prototype_variable_pool_size=6,
        local_prototype_sample_size=512,
        aggregate_prototype_sample_size=512,
        decision_prototype_sample_size=512,
        bridge_enabled=True,
        bridge_top_pairs=4,
        bridge_pair_score_alpha=0.5,
        bridge_concepts=1,
        bridge_max_rules=6,
        bridge_prototype_term_limit=2,
        stagewise_rule_swap_ratio=0.25,
        stagewise_rule_swap_min_keep=6,
        block_agreement_weight=1e-3,
        block_agreement_target_corr=0.2,
        block_agreement_stage_limit=1,
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
        aggregate_max_rule_arity=2,
        learning_rate_scale_regression=1.0,
        learning_rate_scale_classification=1.0,
        refinement_cycle_floor=2,
        local_prototype_term_limit=3,
        stagewise_rule_swap_ratio=0.2,
        stagewise_rule_swap_min_keep=6,
        block_agreement_weight=8e-4,
        block_agreement_target_corr=0.18,
        block_agreement_stage_limit=1,
    ),
    "quality_tiny_reg": DfflProfile(
        name="quality_tiny_reg",
        local_concepts=3,
        local_max_rules=10,
        aggregate_max_rules=16,
        decision_max_rules=12,
        stage2_width_min=4,
        stage2_width_max=8,
        local_rule_generation_mode="prototype",
        aggregate_rule_generation_mode="prototype",
        decision_rule_generation_mode="prototype",
        aggregate_max_rule_arity=2,
        # Keep architecture compact, but raise effective LR for tiny regression tasks.
        learning_rate_scale_regression=1.3333333333333333,
        learning_rate_scale_classification=1.0,
        refinement_cycle_floor=2,
        local_prototype_term_limit=3,
        bridge_enabled=True,
        bridge_top_pairs=2,
        bridge_concepts=1,
        bridge_max_rules=4,
        stagewise_rule_swap_ratio=0.15,
        stagewise_rule_swap_min_keep=4,
        block_agreement_weight=8e-4,
        block_agreement_target_corr=0.18,
        block_agreement_stage_limit=1,
    ),
    "quality_large_cls": DfflProfile(
        name="quality_large_cls",
        local_concepts=2,
        local_max_rules=10,
        aggregate_max_rules=18,
        decision_max_rules=12,
        stage2_width_min=4,
        stage2_width_max=9,
        aggregate_global_context_dim=3,
        aggregate_global_max_rules=12,
        local_rule_generation_mode="prototype",
        aggregate_rule_generation_mode="prototype",
        decision_rule_generation_mode="prototype",
        aggregate_max_rule_arity=3,
        decision_max_rule_arity=3,
        decision_three_terms=False,
        local_max_rule_arity=3,
        learning_rate_scale_regression=1.0,
        learning_rate_scale_classification=1.0,
        refinement_cycle_floor=2,
        pretrain_refinement_rounds=2,
        input_group_strategy="target_corr",
        input_group_correlation_weight=0.75,
        local_prototype_term_limit=3,
        local_prototype_scoring_mode="max",
        local_prototype_variable_pool_size=4,
        aggregate_prototype_term_limit=3,
        aggregate_prototype_scoring_mode="max",
        aggregate_prototype_variable_pool_size=10,
        decision_prototype_term_limit=3,
        decision_prototype_scoring_mode="max",
        decision_prototype_variable_pool_size=8,
        local_prototype_sample_size=1024,
        aggregate_prototype_sample_size=1024,
        decision_prototype_sample_size=1024,
        top_k_rules=24,
        binary_auto_pos_weight=True,
        binary_soft_f1_weight=0.2,
        weight_decay=1e-5,
        gradient_clip_norm=5.0,
        rule_sparsity_weight=3e-5,
        rule_length_weight=1e-5,
        decision_usage_balance_weight=3e-3,
        block_gates_enabled=True,
        block_gate_init_logit=4.0,
        block_gate_l1_weight=5e-5,
        local_consequent_mode="affine_sigmoid",
        aggregate_consequent_mode="affine_sigmoid",
        bridge_enabled=True,
        bridge_top_pairs=8,
        bridge_pair_score_alpha=0.65,
        bridge_concepts=1,
        bridge_max_rules=6,
        bridge_prototype_term_limit=2,
        bridge_prototype_scoring_mode="hybrid",
        bridge_prototype_sample_size=1024,
        stagewise_rule_swap_ratio=0.3,
        stagewise_rule_swap_min_keep=8,
        block_agreement_weight=1.5e-3,
        block_agreement_target_corr=0.22,
        block_agreement_stage_limit=1,
    ),
    "quality_large_cls_plus": DfflProfile(
        name="quality_large_cls_plus",
        local_concepts=2,
        local_max_rules=10,
        aggregate_max_rules=28,
        decision_max_rules=14,
        stage2_width_min=5,
        stage2_width_max=12,
        aggregate_block_count=2,
        aggregate_overlap=1,
        aggregate_global_context_dim=4,
        aggregate_global_max_rules=14,
        local_max_rule_arity=3,
        local_rule_generation_mode="prototype",
        aggregate_rule_generation_mode="prototype",
        decision_rule_generation_mode="prototype",
        aggregate_max_rule_arity=3,
        decision_max_rule_arity=3,
        decision_three_terms=False,
        learning_rate_scale_regression=1.0,
        learning_rate_scale_classification=1.05,
        refinement_cycle_floor=2,
        pretrain_refinement_rounds=2,
        top_k_rules=24,
        input_group_size=4,
        input_group_strategy="target_corr",
        input_group_correlation_weight=0.8,
        local_prototype_term_limit=3,
        local_prototype_scoring_mode="max",
        local_prototype_variable_pool_size=4,
        aggregate_prototype_term_limit=3,
        aggregate_prototype_scoring_mode="hybrid",
        aggregate_prototype_variable_pool_size=16,
        decision_prototype_term_limit=3,
        decision_prototype_scoring_mode="hybrid",
        decision_prototype_variable_pool_size=10,
        local_prototype_sample_size=1024,
        aggregate_prototype_sample_size=1024,
        decision_prototype_sample_size=1024,
        binary_auto_pos_weight=True,
        binary_soft_f1_weight=0.25,
        weight_decay=1e-5,
        gradient_clip_norm=5.0,
        rule_sparsity_weight=3e-5,
        rule_length_weight=1e-5,
        decision_usage_balance_weight=3e-3,
        block_gates_enabled=True,
        block_gate_init_logit=4.0,
        block_gate_l1_weight=5e-5,
        local_consequent_mode="affine_sigmoid",
        aggregate_consequent_mode="affine_sigmoid",
        bridge_enabled=True,
        bridge_top_pairs=12,
        bridge_pair_score_alpha=0.7,
        bridge_concepts=1,
        bridge_max_rules=8,
        bridge_prototype_term_limit=2,
        bridge_prototype_scoring_mode="hybrid",
        bridge_prototype_sample_size=1024,
        stagewise_rule_swap_ratio=0.35,
        stagewise_rule_swap_min_keep=10,
        block_agreement_weight=1.5e-3,
        block_agreement_target_corr=0.22,
        block_agreement_stage_limit=1,
    ),
}


def resolve_dffl_profile(
    *,
    profile_name: str,
    task_type: TaskType,
    n_samples: int,
    input_dim: int | None = None,
) -> DfflProfile:
    if profile_name == "quality_auto":
        # Large binary datasets need slightly larger rule budgets than compact large-cls.
        if task_type == "binary_classification" and n_samples >= 8_000:
            return DFFL_PROFILES["quality_large_cls_plus"]
        # Large high-dimensional binary datasets can benefit from tighter large-cls profile.
        if (
            task_type == "binary_classification"
            and input_dim is not None
            and n_samples >= 1_500
            and input_dim >= 40
        ):
            return DFFL_PROFILES["quality_large_cls"]
        # Very small low-dimensional binary datasets (e.g., wine) benefit from richer concept profile.
        if task_type == "binary_classification" and n_samples <= 250:
            if input_dim is not None and input_dim <= 20:
                return DFFL_PROFILES["quality"]
            return DFFL_PROFILES["quality_balanced"]
        # Tiny regression datasets are highly sensitive to over-parameterized DFFL variants.
        if task_type == "regression" and n_samples <= 50:
            return DFFL_PROFILES["quality_tiny_reg"]
        if task_type == "binary_classification" and n_samples >= 500:
            return DFFL_PROFILES["quality"]
        return DFFL_PROFILES["quality"]
    return DFFL_PROFILES[profile_name]


def apply_dffl_profile_overrides(
    profile: DfflProfile,
    *,
    rule_swap_ratio: float | None = None,
    rule_swap_min_keep: int | None = None,
    total_rule_budget: int | None = None,
    bridge_score_interaction_weight: float | None = None,
    bridge_score_stability_weight: float | None = None,
) -> DfflProfile:
    updates: dict[str, object] = {}
    if rule_swap_ratio is not None:
        if not 0.0 <= float(rule_swap_ratio) <= 1.0:
            raise ValueError("dffl_rule_swap_ratio must be in [0, 1].")
        updates["stagewise_rule_swap_ratio"] = float(rule_swap_ratio)
    if rule_swap_min_keep is not None:
        if int(rule_swap_min_keep) < 0:
            raise ValueError("dffl_rule_swap_min_keep must be non-negative.")
        updates["stagewise_rule_swap_min_keep"] = int(rule_swap_min_keep)
    if total_rule_budget is not None:
        if int(total_rule_budget) < 0:
            raise ValueError("dffl_total_rule_budget must be non-negative.")
        updates["total_rule_budget"] = int(total_rule_budget)
    if bridge_score_interaction_weight is not None:
        if float(bridge_score_interaction_weight) < 0.0:
            raise ValueError("dffl_bridge_score_interaction_weight must be non-negative.")
        updates["bridge_score_interaction_weight"] = float(bridge_score_interaction_weight)
    if bridge_score_stability_weight is not None:
        if float(bridge_score_stability_weight) < 0.0:
            raise ValueError("dffl_bridge_score_stability_weight must be non-negative.")
        updates["bridge_score_stability_weight"] = float(bridge_score_stability_weight)
    if not updates:
        return profile
    return replace(profile, **updates)


def _make_groups(total_dim: int, group_size: int) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    for start in range(0, total_dim, group_size):
        groups.append(tuple(range(start, min(start + group_size, total_dim))))
    return tuple(groups)


def _feature_target_relevance(inputs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    features = inputs.detach().to(dtype=torch.float32)
    labels = targets.detach().reshape(-1).to(device=features.device, dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got {features.ndim}D.")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D targets, got {labels.ndim}D.")
    if features.size(0) != labels.size(0):
        raise ValueError("Features and targets must have the same number of samples.")

    centered_x = features - features.mean(dim=0, keepdim=True)
    centered_y = labels - labels.mean()
    denom_y = torch.linalg.vector_norm(centered_y).clamp_min(1e-12)
    denom_x = torch.linalg.vector_norm(centered_x, dim=0)
    denominator = denom_x * denom_y
    numerator = torch.abs(centered_x.transpose(0, 1) @ centered_y)
    relevance = torch.where(denominator > 1e-12, numerator / denominator, torch.zeros_like(numerator))
    relevance = relevance.clamp(0.0, 1.0)
    return relevance.detach().cpu().numpy().astype(np.float64, copy=False)


def _feature_feature_correlation(inputs: torch.Tensor) -> np.ndarray:
    features = inputs.detach().to(dtype=torch.float32)
    centered = features - features.mean(dim=0, keepdim=True)
    gram = centered.transpose(0, 1) @ centered
    norms = torch.sqrt(torch.clamp(torch.diag(gram), min=0.0))
    denominator = norms[:, None] * norms[None, :]
    corr = torch.where(denominator > 1e-12, gram / denominator, torch.zeros_like(gram))
    corr = torch.abs(corr)
    corr.fill_diagonal_(1.0)
    corr = corr.clamp(0.0, 1.0)
    return corr.detach().cpu().numpy().astype(np.float64, copy=False)


def _validate_feature_geometry(feature_geometry: str) -> str:
    normalized = str(feature_geometry).strip().lower()
    if normalized not in FEATURE_GEOMETRIES:
        raise ValueError(
            f"Unsupported feature_geometry={feature_geometry!r}. "
            f"Expected one of: {', '.join(FEATURE_GEOMETRIES)}."
        )
    return normalized


def _validate_binary_heavy_grouping_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in BINARY_HEAVY_GROUPING_MODES:
        raise ValueError(
            f"Unsupported binary_heavy_grouping_mode={mode!r}. "
            f"Expected one of: {', '.join(BINARY_HEAVY_GROUPING_MODES)}."
        )
    return normalized


def _binary_feature_ratio(inputs: torch.Tensor) -> float:
    if inputs.ndim != 2 or int(inputs.shape[1]) <= 0:
        return 0.0
    return float(len(_detect_binary_feature_indices(inputs)) / float(inputs.shape[1]))


def _feature_poincare_embedding(inputs: torch.Tensor) -> np.ndarray:
    features = inputs.detach().to(dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D inputs for hyperbolic embedding, got shape {tuple(features.shape)}.")
    if int(features.shape[1]) < 2:
        return np.zeros((int(features.shape[1]), 2), dtype=np.float64)

    # Feature points: each original feature is represented by its values across samples.
    # Shape: [n_features, n_samples]
    points = features.transpose(0, 1)
    points = points - points.mean(dim=1, keepdim=True)
    norms = torch.linalg.vector_norm(points, dim=1, keepdim=True).clamp_min(1e-12)
    points = points / norms

    # Deterministic PCA-like projection to 2D.
    u, s, _ = torch.linalg.svd(points, full_matrices=False)
    embedding = u[:, :2] * s[:2].unsqueeze(0)
    if int(embedding.shape[1]) < 2:
        embedding = torch.nn.functional.pad(embedding, (0, 2 - int(embedding.shape[1]), 0, 0))

    # Map to Poincare ball with norm strictly below 1.
    radii = torch.linalg.vector_norm(embedding, dim=1, keepdim=True)
    scale = torch.quantile(radii.reshape(-1), 0.90).clamp_min(1e-12)
    normalized_radii = torch.tanh(radii / scale) * 0.95
    direction = embedding / radii.clamp_min(1e-12)
    mapped = normalized_radii * direction
    return mapped.detach().cpu().numpy().astype(np.float64, copy=False)


def _pairwise_poincare_distance(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2:
        raise ValueError(f"Expected point matrix [n, dim], got shape {points.shape}.")
    n = int(points.shape[0])
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float64)

    squared_norms = np.sum(points * points, axis=1)
    left = points[:, None, :]
    right = points[None, :, :]
    squared_diff = np.sum((left - right) ** 2, axis=2)
    denom = (1.0 - squared_norms[:, None]) * (1.0 - squared_norms[None, :])
    argument = 1.0 + (2.0 * squared_diff / np.maximum(denom, 1e-12))
    argument = np.maximum(argument, 1.0 + 1e-12)
    distances = np.arccosh(argument)
    np.fill_diagonal(distances, 0.0)
    return distances


def _detect_binary_feature_indices(inputs: torch.Tensor, eps: float = 1e-6) -> tuple[int, ...]:
    if inputs.ndim != 2:
        raise ValueError(f"Expected 2D inputs, got shape {tuple(inputs.shape)}.")
    clamped_binary_mask = ((inputs <= eps) | (inputs >= (1.0 - eps))).all(dim=0)
    return tuple(int(index) for index in torch.where(clamped_binary_mask)[0].detach().cpu().tolist())


def _make_target_corr_groups(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    group_size: int,
    correlation_weight: float,
) -> tuple[tuple[int, ...], ...]:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    input_dim = int(inputs.shape[1])
    if input_dim <= group_size or inputs.shape[0] < 4:
        return _make_groups(input_dim, group_size=group_size)

    corr_weight = min(1.0, max(0.0, float(correlation_weight)))
    relevance = _feature_target_relevance(inputs, targets)
    pairwise_corr = _feature_feature_correlation(inputs)

    priority = np.argsort(-relevance, kind="stable").tolist()
    remaining = set(range(input_dim))
    groups: list[tuple[int, ...]] = []

    while remaining:
        seed = next((idx for idx in priority if idx in remaining), min(remaining))
        current_group = [int(seed)]
        remaining.remove(seed)

        while len(current_group) < group_size and remaining:
            best_candidate = None
            best_score = float("-inf")
            for candidate in remaining:
                cohesion = float(np.mean(pairwise_corr[candidate, current_group]))
                score = corr_weight * cohesion + (1.0 - corr_weight) * float(relevance[candidate])
                if score > best_score + 1e-12:
                    best_score = score
                    best_candidate = int(candidate)
                    continue
                if abs(score - best_score) <= 1e-12 and best_candidate is not None and candidate < best_candidate:
                    best_candidate = int(candidate)
            if best_candidate is None:
                break
            current_group.append(best_candidate)
            remaining.remove(best_candidate)

        groups.append(tuple(current_group))

    return tuple(groups)


def _make_binary_aware_target_corr_groups(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    group_size: int,
    correlation_weight: float,
) -> tuple[tuple[int, ...], ...]:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    input_dim = int(inputs.shape[1])
    if input_dim <= group_size or inputs.shape[0] < 4:
        return _make_groups(input_dim, group_size=group_size)

    corr_weight = min(1.0, max(0.0, float(correlation_weight)))
    relevance = _feature_target_relevance(inputs, targets)
    pairwise_corr = _feature_feature_correlation(inputs)

    binary_set = set(_detect_binary_feature_indices(inputs))
    non_binary = [idx for idx in range(input_dim) if idx not in binary_set]
    if len(non_binary) < 2:
        return _make_target_corr_groups(
            inputs,
            targets,
            group_size=group_size,
            correlation_weight=correlation_weight,
        )

    n_groups = int(np.ceil(input_dim / float(group_size)))
    non_binary_priority = sorted(non_binary, key=lambda idx: (-relevance[idx], idx))
    anchors = non_binary_priority[: max(1, min(n_groups, len(non_binary_priority)))]
    groups: list[list[int]] = [[int(anchor)] for anchor in anchors]
    assigned = set(anchors)

    # Ensure deterministic group count even when non-binary anchors are fewer than n_groups.
    while len(groups) < n_groups:
        groups.append([])

    all_priority = sorted(range(input_dim), key=lambda idx: (-relevance[idx], idx))
    for candidate in all_priority:
        if candidate in assigned:
            continue
        best_group = None
        best_score = float("-inf")
        candidate_is_binary = candidate in binary_set
        for group_index, group in enumerate(groups):
            if len(group) >= group_size:
                continue
            if not group:
                score = float(relevance[candidate]) + 0.02
            else:
                cohesion = float(np.mean(pairwise_corr[candidate, group]))
                score = corr_weight * cohesion + (1.0 - corr_weight) * float(relevance[candidate])
                group_binary_count = sum(1 for idx in group if idx in binary_set)
                group_non_binary_count = len(group) - group_binary_count
                # Prefer mixed groups to preserve cross-type interactions in binary-heavy settings.
                if candidate_is_binary and group_non_binary_count > 0:
                    score += 0.03
                if (not candidate_is_binary) and group_binary_count > 0:
                    score += 0.03
                if candidate_is_binary and group_binary_count == len(group):
                    score -= 0.02

            if score > best_score + 1e-12:
                best_score = score
                best_group = group_index
                continue
            if (
                abs(score - best_score) <= 1e-12
                and best_group is not None
                and len(groups[group_index]) < len(groups[best_group])
            ):
                best_group = group_index

        if best_group is None:
            for group_index, group in enumerate(groups):
                if len(group) < group_size:
                    best_group = group_index
                    break
        if best_group is None:
            best_group = 0
        groups[best_group].append(int(candidate))
        assigned.add(int(candidate))

    # Flatten and restore deterministic sorted tuples.
    normalized = [tuple(sorted(group)) for group in groups if group]
    covered = set(idx for group in normalized for idx in group)
    if covered != set(range(input_dim)):
        missing = sorted(set(range(input_dim)) - covered)
        for idx in missing:
            normalized.append((int(idx),))
    return tuple(normalized)


def _make_hyperbolic_graph_groups(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    group_size: int,
) -> tuple[tuple[int, ...], ...]:
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    input_dim = int(inputs.shape[1])
    if input_dim <= group_size:
        return _make_groups(input_dim, group_size=group_size)

    relevance = _feature_target_relevance(inputs, targets)
    pairwise_corr = _feature_feature_correlation(inputs)
    points = _feature_poincare_embedding(inputs)
    distances = _pairwise_poincare_distance(points)
    max_distance = float(np.max(distances)) + 1e-12
    proximity = 1.0 - (distances / max_distance)
    np.fill_diagonal(proximity, 1.0)

    remaining = set(range(input_dim))
    priority = np.argsort(-relevance, kind="stable").tolist()
    groups: list[tuple[int, ...]] = []

    while remaining:
        seed = next((idx for idx in priority if idx in remaining), min(remaining))
        current_group = [int(seed)]
        remaining.remove(seed)

        while len(current_group) < group_size and remaining:
            best_candidate = None
            best_score = float("-inf")
            for candidate in remaining:
                hyp_cohesion = float(np.mean(proximity[candidate, current_group]))
                corr_cohesion = float(np.mean(pairwise_corr[candidate, current_group]))
                cohesion = 0.55 * hyp_cohesion + 0.45 * corr_cohesion
                score = 0.7 * cohesion + 0.3 * float(relevance[candidate])
                if score > best_score + 1e-12:
                    best_score = score
                    best_candidate = int(candidate)
                    continue
                if abs(score - best_score) <= 1e-12 and best_candidate is not None and candidate < best_candidate:
                    best_candidate = int(candidate)
            if best_candidate is None:
                break
            current_group.append(best_candidate)
            remaining.remove(best_candidate)

        groups.append(tuple(current_group))
    return tuple(groups)


def _resolve_effective_feature_geometry(
    *,
    requested_geometry: str,
    profile: DfflProfile,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
) -> tuple[str, str]:
    requested = _validate_feature_geometry(requested_geometry)
    if requested in {"euclidean", "hyperbolic"}:
        return requested, "explicit"

    input_dim = int(train_inputs.shape[1])
    n_samples = int(train_inputs.shape[0])
    binary_ratio = _binary_feature_ratio(train_inputs)
    eu_groups = make_dffl_input_groups(
        profile,
        train_inputs=train_inputs,
        train_targets=train_targets,
        feature_geometry="euclidean",
    )
    intergroup_share = estimate_intergroup_interaction_share(
        train_inputs=train_inputs,
        train_targets=train_targets,
        input_groups=eu_groups,
    )

    # Conservative policy: hyperbolic only for large, richly inter-group, non-binary-dominant settings.
    if input_dim >= 80 and n_samples >= 4000 and intergroup_share >= 0.80 and binary_ratio <= 0.45:
        return "hyperbolic", "auto_large_rich_intergroup"
    return "euclidean", "auto_default"


def make_dffl_input_groups(
    profile: DfflProfile,
    *,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    feature_geometry: str = "euclidean",
    binary_heavy_grouping_mode: str = "binary_aware",
) -> tuple[tuple[int, ...], ...]:
    geometry = _validate_feature_geometry(feature_geometry)
    binary_mode = _validate_binary_heavy_grouping_mode(binary_heavy_grouping_mode)
    if geometry == "hyperbolic":
        return _make_hyperbolic_graph_groups(
            train_inputs,
            train_targets,
            group_size=profile.input_group_size,
        )
    if profile.input_group_strategy == "contiguous":
        return _make_groups(int(train_inputs.shape[1]), group_size=profile.input_group_size)
    if profile.input_group_strategy == "target_corr":
        binary_indices = _detect_binary_feature_indices(train_inputs)
        binary_ratio = len(binary_indices) / float(train_inputs.shape[1])
        # For binary-heavy datasets (e.g., one-hot dominant), use binary-aware grouping
        # to preserve informative cross-type interactions without exploding complexity.
        if binary_ratio >= 0.6:
            if binary_mode == "contiguous":
                return _make_groups(int(train_inputs.shape[1]), group_size=profile.input_group_size)
            return _make_binary_aware_target_corr_groups(
                train_inputs,
                train_targets,
                group_size=profile.input_group_size,
                correlation_weight=profile.input_group_correlation_weight,
            )
        return _make_target_corr_groups(
            train_inputs,
            train_targets,
            group_size=profile.input_group_size,
            correlation_weight=profile.input_group_correlation_weight,
        )
    raise ValueError(
        f"Unsupported input_group_strategy={profile.input_group_strategy!r}. "
        "Expected 'contiguous' or 'target_corr'."
    )


def _build_feature_to_group_index(
    input_dim: int,
    input_groups: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    feature_to_group = np.full(input_dim, fill_value=-1, dtype=np.int64)
    for group_index, group in enumerate(input_groups):
        for feature_index in group:
            feature_to_group[int(feature_index)] = group_index
    return feature_to_group


def estimate_intergroup_interaction_share(
    *,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    input_groups: tuple[tuple[int, ...], ...],
) -> float:
    input_dim = int(train_inputs.shape[1])
    if input_dim <= 1 or train_inputs.shape[0] < 8:
        return 0.0

    feature_to_group = _build_feature_to_group_index(input_dim, input_groups)
    pair_strength = _pair_interaction_relevance(train_inputs, train_targets).astype(np.float64, copy=False)
    upper_triangle = np.triu(np.ones((input_dim, input_dim), dtype=bool), k=1)
    total_strength = float(np.sum(pair_strength[upper_triangle]))
    if total_strength <= 1e-12:
        return 0.0
    left_groups = feature_to_group[:, None]
    right_groups = feature_to_group[None, :]
    intergroup_mask = (
        upper_triangle
        & (left_groups >= 0)
        & (right_groups >= 0)
        & (left_groups != right_groups)
    )
    intergroup_strength = float(np.sum(pair_strength[intergroup_mask]))
    return float(intergroup_strength / (total_strength + 1e-12))


def _pair_interaction_relevance(inputs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    x = inputs.detach().to(dtype=torch.float32)
    y = targets.detach().reshape(-1).to(device=x.device, dtype=torch.float32)
    input_dim = int(x.shape[1])
    if input_dim <= 1 or int(x.shape[0]) < 8:
        return np.zeros((input_dim, input_dim), dtype=np.float64)

    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean()
    y_norm = torch.linalg.vector_norm(y_centered).clamp_min(1e-12)

    interactions = x_centered[:, :, None] * x_centered[:, None, :]
    interactions_centered = interactions - interactions.mean(dim=0, keepdim=True)
    numerator = torch.abs(torch.sum(interactions_centered * y_centered[:, None, None], dim=0))
    interaction_norm = torch.linalg.vector_norm(interactions_centered, dim=0).clamp_min(1e-12)
    scores = torch.where(interaction_norm > 1e-12, numerator / (interaction_norm * y_norm), torch.zeros_like(numerator))
    scores.fill_diagonal_(0.0)
    max_score = torch.max(scores)
    if float(max_score.item()) > 1e-12:
        scores = scores / max_score
    scores = scores.clamp(0.0, 1.0)
    return scores.detach().cpu().numpy().astype(np.float64, copy=False)


def _feature_relevance_stability_proxy(inputs: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    inputs_tensor = inputs.detach()
    targets_tensor = targets.detach()
    n_samples = int(inputs_tensor.shape[0])
    input_dim = int(inputs_tensor.shape[1])
    if n_samples < 12:
        return np.ones(input_dim, dtype=np.float64)

    first_idx = torch.arange(0, n_samples, 2, device=inputs_tensor.device)
    second_idx = torch.arange(1, n_samples, 2, device=inputs_tensor.device)
    if int(second_idx.numel()) == 0:
        return np.ones(input_dim, dtype=np.float64)

    rel_first = _feature_target_relevance(inputs_tensor[first_idx], targets_tensor[first_idx])
    rel_second = _feature_target_relevance(inputs_tensor[second_idx], targets_tensor[second_idx])
    stability = 1.0 - np.abs(rel_first - rel_second)
    return np.clip(stability, 0.0, 1.0)


def _resolve_dffl_effective_rule_budgets(
    *,
    profile: DfflProfile,
    input_dim: int,
    n_local_groups: int,
    n_bridge_pairs: int,
    intergroup_interaction_share: float,
    adaptive_budget_enabled: bool = True,
) -> dict[str, int | str]:
    local_cap = int(profile.local_max_rules)
    if n_local_groups > 4:
        stage1_target_total_rules = min(160, max(96, 10 * n_local_groups))
        if input_dim >= 40 and n_local_groups >= 12:
            stage1_target_total_rules = min(stage1_target_total_rules, 128)
        local_cap = min(local_cap, max(8, stage1_target_total_rules // n_local_groups))

    bridge_cap = min(profile.bridge_max_rules, max(4, local_cap - 2))
    if n_bridge_pairs <= 0:
        bridge_cap = 0

    decision_cap = int(profile.decision_max_rules)
    if n_local_groups >= 12 and input_dim >= 40:
        decision_cap = min(decision_cap, 12)

    aggregate_cap_total = int(profile.aggregate_max_rules)
    if profile.aggregate_global_context_dim > 0:
        global_max_rules = (
            int(profile.aggregate_global_max_rules)
            if profile.aggregate_global_max_rules > 0
            else max(4, min(profile.aggregate_max_rules // 2, 12))
        )
        aggregate_cap_total += int(global_max_rules)

    local_cap_total = int(n_local_groups * local_cap)
    bridge_cap_total = int(n_bridge_pairs * bridge_cap)
    cap_total = int(local_cap_total + bridge_cap_total + aggregate_cap_total + decision_cap)
    if cap_total <= 0:
        return {
            "allocation_policy": "zero_cap",
            "local_max_rules_effective": 0,
            "bridge_max_rules_effective": 0,
            "aggregate_max_rules_effective": 0,
            "decision_max_rules_effective": 0,
            "local_total_budget_effective": 0,
            "bridge_total_budget_effective": 0,
            "aggregate_total_budget_effective": 0,
            "decision_total_budget_effective": 0,
            "total_rule_budget_effective": 0,
        }

    if int(profile.total_rule_budget) > 0:
        budget_total = min(cap_total, int(profile.total_rule_budget))
        allocation_policy = "explicit_total_rule_budget"
    elif not adaptive_budget_enabled:
        budget_total = cap_total
        allocation_policy = "cap_only_no_allocator"
    else:
        base_budget = int(8 * n_local_groups + 24)
        if input_dim >= 40:
            base_budget += 12
        if intergroup_interaction_share >= 0.65:
            base_budget += 8
        if n_bridge_pairs >= 6:
            base_budget += 6
        budget_total = min(cap_total, max(64, base_budget))
        allocation_policy = "adaptive_capped_budget"

    local_min_per_block = max(2, min(4, int(profile.stage1_min_rules_per_block)))
    local_min_total = min(local_cap_total, n_local_groups * local_min_per_block)
    bridge_min_total = min(bridge_cap_total, n_bridge_pairs * 2)
    aggregate_min_total = min(aggregate_cap_total, 4 if aggregate_cap_total > 0 else 0)
    decision_min_total = min(decision_cap, 4 if decision_cap > 0 else 0)

    min_total = int(local_min_total + bridge_min_total + aggregate_min_total + decision_min_total)
    if budget_total < min_total:
        budget_total = min(cap_total, min_total)
        allocation_policy = f"{allocation_policy}_raised_to_min"

    share = float(np.clip(intergroup_interaction_share, 0.0, 1.0))
    utilities = {
        "local": 0.95 + 0.35 * (1.0 - share),
        "bridge": (0.20 + 1.10 * share) if bridge_cap_total > 0 else 0.0,
        "aggregate": 0.90 + 0.45 * share,
        "decision": 0.80 + (0.15 if input_dim >= 20 else 0.0),
    }
    caps = {
        "local": int(local_cap_total),
        "bridge": int(bridge_cap_total),
        "aggregate": int(aggregate_cap_total),
        "decision": int(decision_cap),
    }
    budgets = {
        "local": int(local_min_total),
        "bridge": int(bridge_min_total),
        "aggregate": int(aggregate_min_total),
        "decision": int(decision_min_total),
    }

    remaining = int(budget_total - sum(budgets.values()))
    while remaining > 0:
        rooms = {name: caps[name] - budgets[name] for name in caps}
        weighted_rooms = {
            name: max(0.0, float(rooms[name])) * float(utilities[name])
            for name in caps
            if rooms[name] > 0
        }
        if not weighted_rooms:
            break
        target_name = max(weighted_rooms.items(), key=lambda item: (item[1], item[0]))[0]
        budgets[target_name] += 1
        remaining -= 1

    local_max_rules_effective = max(local_min_per_block, int(budgets["local"] // max(1, n_local_groups)))
    local_max_rules_effective = min(local_cap, local_max_rules_effective)
    if n_bridge_pairs > 0 and bridge_cap > 0:
        bridge_max_rules_effective = max(2, int(budgets["bridge"] // n_bridge_pairs))
        bridge_max_rules_effective = min(bridge_cap, bridge_max_rules_effective)
    else:
        bridge_max_rules_effective = 0
    aggregate_max_rules_effective = max(1, min(aggregate_cap_total, int(budgets["aggregate"])))
    decision_max_rules_effective = max(1, min(decision_cap, int(budgets["decision"])))

    total_effective = int(
        local_max_rules_effective * n_local_groups
        + bridge_max_rules_effective * n_bridge_pairs
        + aggregate_max_rules_effective
        + decision_max_rules_effective
    )
    return {
        "allocation_policy": allocation_policy,
        "local_max_rules_effective": int(local_max_rules_effective),
        "bridge_max_rules_effective": int(bridge_max_rules_effective),
        "aggregate_max_rules_effective": int(aggregate_max_rules_effective),
        "decision_max_rules_effective": int(decision_max_rules_effective),
        "local_total_budget_effective": int(local_max_rules_effective * n_local_groups),
        "bridge_total_budget_effective": int(bridge_max_rules_effective * n_bridge_pairs),
        "aggregate_total_budget_effective": int(aggregate_max_rules_effective),
        "decision_total_budget_effective": int(decision_max_rules_effective),
        "total_rule_budget_effective": int(total_effective),
    }


def _resolve_dffl_block_agreement_params(
    *,
    profile: DfflProfile,
    intergroup_interaction_share: float,
    n_bridge_pairs: int,
) -> tuple[float, float]:
    weight = float(profile.block_agreement_weight)
    target_corr = float(profile.block_agreement_target_corr)
    if weight <= 0.0:
        return weight, target_corr

    share = float(np.clip(intergroup_interaction_share, 0.0, 1.0))
    if n_bridge_pairs > 0:
        boost = 1.0 + 0.8 * max(0.0, share - 0.45)
        weight *= boost
        target_corr = min(0.35, target_corr + 0.08 * max(0.0, share - 0.50))
    return float(weight), float(np.clip(target_corr, -1.0, 1.0))


def _resolve_effective_bridge_concepts(
    *,
    profile: DfflProfile,
    intergroup_interaction_share: float,
    input_dim: int,
    n_bridge_pairs: int,
) -> int:
    base = max(1, int(profile.bridge_concepts))
    cap = max(base, int(profile.bridge_concepts_max))
    if (
        (not profile.bridge_concepts_adaptive)
        or n_bridge_pairs <= 0
        or input_dim < 20
    ):
        return base
    share = float(np.clip(intergroup_interaction_share, 0.0, 1.0))
    if share >= float(profile.bridge_concepts_high_share_threshold) and n_bridge_pairs <= 8:
        return min(cap, max(base, 2))
    return base


def resolve_adaptive_rule_swap_ratio(
    *,
    base_ratio: float,
    input_dim: int,
    intergroup_interaction_share: float,
) -> tuple[float, str]:
    ratio = float(base_ratio)
    if ratio <= 0.0:
        return 0.0, "base_zero"
    share = float(np.clip(intergroup_interaction_share, 0.0, 1.0))

    # Quality guardrail: rich inter-group structure should not be perturbed by aggressive swapping.
    if input_dim >= 40 and share >= 0.65:
        return 0.0, "high_dim_high_intergroup_disable"
    if share >= 0.75:
        return 0.0, "very_high_intergroup_disable"
    if share >= 0.60:
        return max(0.0, ratio * 0.5), "high_intergroup_half"
    if share <= 0.35:
        return min(0.5, ratio * 1.15), "low_intergroup_boost"
    return ratio, "mid_intergroup_keep"


def make_dffl_bridge_pairs(
    profile: DfflProfile,
    *,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    input_groups: tuple[tuple[int, ...], ...],
    feature_geometry: str = "euclidean",
    adaptive_budget_enabled: bool = True,
) -> tuple[tuple[int, int], ...]:
    geometry = _validate_feature_geometry(feature_geometry)
    if (not profile.bridge_enabled) or profile.bridge_top_pairs <= 0:
        return ()
    input_dim = int(train_inputs.shape[1])
    if input_dim <= 1 or train_inputs.shape[0] < 8:
        return ()

    feature_to_group = _build_feature_to_group_index(input_dim, input_groups)

    relevance = _feature_target_relevance(train_inputs, train_targets)
    relevance_stability = _feature_relevance_stability_proxy(train_inputs, train_targets)
    pairwise_corr = _feature_feature_correlation(train_inputs)
    pairwise_interaction = _pair_interaction_relevance(train_inputs, train_targets)
    pairwise_hyp_proximity: np.ndarray | None = None
    if geometry == "hyperbolic":
        hyp_points = _feature_poincare_embedding(train_inputs)
        hyp_distances = _pairwise_poincare_distance(hyp_points)
        max_hyp_distance = float(np.max(hyp_distances)) + 1e-12
        pairwise_hyp_proximity = 1.0 - (hyp_distances / max_hyp_distance)
        np.fill_diagonal(pairwise_hyp_proximity, 1.0)
    alpha = min(1.0, max(0.0, float(profile.bridge_pair_score_alpha)))
    interaction_share = estimate_intergroup_interaction_share(
        train_inputs=train_inputs,
        train_targets=train_targets,
        input_groups=input_groups,
    )
    interaction_weight = max(0.0, float(profile.bridge_score_interaction_weight))
    stability_weight = max(0.0, float(profile.bridge_score_stability_weight))
    # Enrich bridge selection beyond correlation:
    # target relevance + interaction strength + stability proxy.
    base_geom_weight = alpha
    base_relevance_weight = 1.0 - alpha
    dynamic_interaction_weight = interaction_weight * (0.75 + 0.5 * float(np.clip(interaction_share, 0.0, 1.0)))
    dynamic_stability_weight = stability_weight
    weight_sum = base_geom_weight + base_relevance_weight + dynamic_interaction_weight + dynamic_stability_weight
    if weight_sum <= 1e-12:
        geom_weight = 0.5
        relevance_weight = 0.5
        interaction_weight_final = 0.0
        stability_weight_final = 0.0
    else:
        geom_weight = base_geom_weight / weight_sum
        relevance_weight = base_relevance_weight / weight_sum
        interaction_weight_final = dynamic_interaction_weight / weight_sum
        stability_weight_final = dynamic_stability_weight / weight_sum

    scored_pairs: list[tuple[float, int, int]] = []
    for left in range(input_dim):
        group_left = int(feature_to_group[left])
        if group_left < 0:
            continue
        for right in range(left + 1, input_dim):
            group_right = int(feature_to_group[right])
            if group_right < 0 or group_right == group_left:
                continue
            rel_score = float(relevance[left] * relevance[right])
            if geometry == "hyperbolic":
                if pairwise_hyp_proximity is None:
                    raise RuntimeError("hyperbolic pairwise proximity is not initialized.")
                geom_score = 0.60 * float(pairwise_hyp_proximity[left, right]) + 0.40 * float(
                    pairwise_corr[left, right]
                )
            else:
                if pairwise_corr is None:
                    raise RuntimeError("euclidean pairwise correlation is not initialized.")
                geom_score = float(pairwise_corr[left, right])
            interaction_score = float(pairwise_interaction[left, right])
            stability_score = 0.5 * float(relevance_stability[left]) + 0.5 * float(relevance_stability[right])
            score = (
                geom_weight * geom_score
                + relevance_weight * rel_score
                + interaction_weight_final * interaction_score
                + stability_weight_final * stability_score
            )
            scored_pairs.append((score, left, right))

    base_top_pairs = int(profile.bridge_top_pairs)
    if interaction_share >= 0.85:
        effective_top_pairs = base_top_pairs
    elif interaction_share >= 0.70:
        effective_top_pairs = max(2, int(np.ceil(base_top_pairs * 0.75)))
    elif interaction_share >= 0.55:
        effective_top_pairs = max(1, int(np.ceil(base_top_pairs * 0.50)))
    else:
        effective_top_pairs = max(1, int(np.ceil(base_top_pairs * 0.33)))
    effective_top_pairs = min(base_top_pairs, effective_top_pairs)
    # Budget-aware pair search: cap bridge pairs before final selection.
    budget_probe = _resolve_dffl_effective_rule_budgets(
        profile=profile,
        input_dim=input_dim,
        n_local_groups=len(input_groups),
        n_bridge_pairs=effective_top_pairs,
        intergroup_interaction_share=interaction_share,
        adaptive_budget_enabled=adaptive_budget_enabled,
    )
    bridge_rules_per_pair_probe = max(1, int(budget_probe["bridge_max_rules_effective"]))
    bridge_total_budget_probe = int(budget_probe["bridge_total_budget_effective"])
    budget_pair_limit = bridge_total_budget_probe // bridge_rules_per_pair_probe
    if budget_pair_limit <= 0:
        return ()
    effective_top_pairs = min(effective_top_pairs, max(1, int(budget_pair_limit)))

    scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    # Diversity-aware bridge selection:
    # avoid overusing the same features and the same group-pair corridor.
    max_pairs_per_feature = max(1, int(profile.bridge_max_pairs_per_feature))
    max_pairs_per_group_pair = max(1, int(profile.bridge_max_pairs_per_group_pair))
    feature_pair_count: dict[int, int] = {}
    group_pair_count: dict[tuple[int, int], int] = {}
    selected: list[tuple[float, int, int]] = []

    for score, left, right in scored_pairs:
        group_left = int(feature_to_group[left])
        group_right = int(feature_to_group[right])
        if group_left < 0 or group_right < 0 or group_left == group_right:
            continue
        group_pair = (group_left, group_right) if group_left < group_right else (group_right, group_left)
        if feature_pair_count.get(left, 0) >= max_pairs_per_feature:
            continue
        if feature_pair_count.get(right, 0) >= max_pairs_per_feature:
            continue
        if group_pair_count.get(group_pair, 0) >= max_pairs_per_group_pair:
            continue
        selected.append((score, left, right))
        feature_pair_count[left] = feature_pair_count.get(left, 0) + 1
        feature_pair_count[right] = feature_pair_count.get(right, 0) + 1
        group_pair_count[group_pair] = group_pair_count.get(group_pair, 0) + 1
        if len(selected) >= effective_top_pairs:
            break

    if len(selected) < effective_top_pairs:
        selected_pairs = {(left, right) for _, left, right in selected}
        for score, left, right in scored_pairs:
            pair = (left, right)
            if pair in selected_pairs:
                continue
            selected.append((score, left, right))
            selected_pairs.add(pair)
            if len(selected) >= effective_top_pairs:
                break

    return tuple((int(left), int(right)) for _, left, right in selected[:effective_top_pairs])


def _concept_width(input_dim: int) -> int:
    return min(8, max(4, int(round(input_dim**0.5))))


def _hidden_width(input_dim: int) -> int:
    return min(6, max(3, _concept_width(input_dim) // 2 + 1))


def _split_even(total: int, parts: int) -> tuple[int, ...]:
    if total <= 0:
        raise ValueError("total must be positive.")
    if parts <= 0:
        raise ValueError("parts must be positive.")
    base = total // parts
    remainder = total % parts
    return tuple(base + (1 if index < remainder else 0) for index in range(parts))


def _build_overlapping_windows(total: int, blocks: int, overlap: int) -> tuple[tuple[int, ...], ...]:
    if total <= 0:
        raise ValueError("total must be positive.")
    if blocks <= 0:
        raise ValueError("blocks must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")
    if blocks == 1:
        return (tuple(range(total)),)

    block_span = int(np.ceil(total / blocks))
    windows: list[tuple[int, ...]] = []
    for block_index in range(blocks):
        start = block_index * block_span
        end = min(total, (block_index + 1) * block_span)
        if block_index > 0:
            start = max(0, start - overlap)
        if block_index < blocks - 1:
            end = min(total, end + overlap)
        if end <= start:
            end = min(total, start + 1)
        windows.append(tuple(range(start, end)))
    return tuple(windows)


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


def build_dffl_config(
    input_dim: int,
    profile: DfflProfile,
    *,
    input_groups: tuple[tuple[int, ...], ...] | None = None,
    binary_feature_indices: tuple[int, ...] | None = None,
    bridge_feature_pairs: tuple[tuple[int, int], ...] | None = None,
    intergroup_interaction_share: float | None = None,
    adaptive_budget_enabled: bool = True,
) -> HierarchicalModelConfig:
    groups = input_groups or _make_groups(input_dim, group_size=profile.input_group_size)
    binary_feature_set = set(binary_feature_indices or ())
    bridge_pairs = tuple(bridge_feature_pairs or ())
    if not profile.bridge_enabled:
        bridge_pairs = ()
    n_local_groups = len(groups)
    share = (
        float(np.clip(intergroup_interaction_share, 0.0, 1.0))
        if intergroup_interaction_share is not None
        else 0.5
    )
    effective_bridge_concepts = _resolve_effective_bridge_concepts(
        profile=profile,
        intergroup_interaction_share=share,
        input_dim=input_dim,
        n_bridge_pairs=len(bridge_pairs),
    )
    effective_budgets = _resolve_dffl_effective_rule_budgets(
        profile=profile,
        input_dim=input_dim,
        n_local_groups=n_local_groups,
        n_bridge_pairs=len(bridge_pairs),
        intergroup_interaction_share=share,
        adaptive_budget_enabled=adaptive_budget_enabled,
    )
    local_max_rules_effective = int(effective_budgets["local_max_rules_effective"])
    bridge_max_rules_effective = int(effective_budgets["bridge_max_rules_effective"])
    decision_max_rules_effective = int(effective_budgets["decision_max_rules_effective"])
    aggregate_max_rules_effective = int(effective_budgets["aggregate_max_rules_effective"])

    stage_1_blocks = []
    bridge_output_indices: list[int] = []
    stage_1_offset = 0
    for block_index, indices in enumerate(groups):
        stage_1_blocks.append(
            TransparentBlockConfig(
                name=f"dffl_local_{block_index}",
                input_indices=indices,
                variables=tuple(
                    (var_binary(f"x{idx}") if idx in binary_feature_set else var3(f"x{idx}"))
                    for idx in indices
                ),
                n_concepts=profile.local_concepts,
                concept_names=tuple(
                    f"s1_{block_index}_{concept_index}" for concept_index in range(profile.local_concepts)
                ),
                max_rule_arity=min(profile.local_max_rule_arity, len(indices)),
                max_rules=local_max_rules_effective,
                rule_generation_mode=profile.local_rule_generation_mode,
                prototype_term_limit=profile.local_prototype_term_limit,
                prototype_scoring_mode=profile.local_prototype_scoring_mode,
                prototype_variable_pool_size=profile.local_prototype_variable_pool_size,
                prototype_sample_size=profile.local_prototype_sample_size,
                consequent_mode=profile.local_consequent_mode,
            )
        )
        stage_1_offset += int(profile.local_concepts)

    if bridge_pairs:
        for pair_index, (left_idx, right_idx) in enumerate(bridge_pairs):
            if left_idx == right_idx:
                continue
            pair = tuple(sorted((int(left_idx), int(right_idx))))
            stage_1_blocks.append(
                TransparentBlockConfig(
                    name=f"dffl_bridge_{pair_index}",
                    input_indices=pair,
                    variables=tuple(
                        (var_binary(f"x{idx}") if idx in binary_feature_set else var3(f"x{idx}"))
                        for idx in pair
                    ),
                    n_concepts=effective_bridge_concepts,
                    concept_names=tuple(f"s1b_{pair_index}_{i}" for i in range(effective_bridge_concepts)),
                    max_rule_arity=min(profile.bridge_max_rule_arity, len(pair)),
                    max_rules=bridge_max_rules_effective,
                    rule_generation_mode=profile.bridge_rule_generation_mode,
                    prototype_term_limit=profile.bridge_prototype_term_limit,
                    prototype_scoring_mode=profile.bridge_prototype_scoring_mode,
                    prototype_variable_pool_size=profile.bridge_prototype_variable_pool_size,
                    prototype_sample_size=profile.bridge_prototype_sample_size,
                    consequent_mode=profile.local_consequent_mode,
                )
            )
            bridge_output_indices.extend(range(stage_1_offset, stage_1_offset + effective_bridge_concepts))
            stage_1_offset += effective_bridge_concepts

    stage_1_width = int(stage_1_offset)
    width_driver = len(groups) + len(bridge_pairs)
    stage_2_width = min(
        profile.stage2_width_max,
        max(profile.stage2_width_min, width_driver + 1),
    )
    aggregate_blocks = max(1, min(profile.aggregate_block_count, stage_2_width))
    aggregate_rule_budget_total = max(1, aggregate_max_rules_effective)
    bridge_token_rules_effective = 0
    bridge_token_concepts_effective = 0
    if profile.bridge_token_enabled and bridge_output_indices and aggregate_rule_budget_total >= 6:
        bridge_token_concepts_effective = max(1, int(profile.bridge_token_concepts))
        bridge_token_rules_effective = min(
            int(profile.bridge_token_max_rules),
            max(1, aggregate_rule_budget_total // 5),
        )
        aggregate_rule_budget_total = max(1, aggregate_rule_budget_total - bridge_token_rules_effective)
    global_max_rules_effective = 0
    if profile.aggregate_global_context_dim > 0:
        global_cap = (
            int(profile.aggregate_global_max_rules)
            if profile.aggregate_global_max_rules > 0
            else max(4, min(profile.aggregate_max_rules // 2, 12))
        )
        global_max_rules_effective = min(global_cap, max(2, aggregate_rule_budget_total // 3))
        aggregate_rule_budget_total = max(1, aggregate_rule_budget_total - global_max_rules_effective)
    aggregate_rule_budget = _split_even(aggregate_rule_budget_total, aggregate_blocks)
    aggregate_output_dims = _split_even(stage_2_width, aggregate_blocks)
    aggregate_input_windows = _build_overlapping_windows(
        stage_1_width,
        aggregate_blocks,
        overlap=profile.aggregate_overlap,
    )

    stage_2_blocks: list[TransparentBlockConfig] = []
    concept_offset = 0
    for block_index in range(aggregate_blocks):
        block_output_dim = aggregate_output_dims[block_index]
        block_concept_names = tuple(f"s2_{concept_offset + i}" for i in range(block_output_dim))
        concept_offset += block_output_dim
        block_input_indices = aggregate_input_windows[block_index]
        stage_2_blocks.append(
            TransparentBlockConfig(
                name=f"dffl_aggregate_{block_index}",
                input_indices=block_input_indices,
                variables=tuple(var3(f"s1_{i}") for i in block_input_indices),
                n_concepts=block_output_dim,
                concept_names=block_concept_names,
                max_rule_arity=min(profile.aggregate_max_rule_arity, len(block_input_indices)),
                max_rules=aggregate_rule_budget[block_index],
                rule_generation_mode=profile.aggregate_rule_generation_mode,
                prototype_term_limit=profile.aggregate_prototype_term_limit,
                prototype_scoring_mode=profile.aggregate_prototype_scoring_mode,
                prototype_variable_pool_size=profile.aggregate_prototype_variable_pool_size,
                prototype_sample_size=profile.aggregate_prototype_sample_size,
                consequent_mode=profile.aggregate_consequent_mode,
            )
        )

    if bridge_token_concepts_effective > 0 and bridge_token_rules_effective > 0 and bridge_output_indices:
        bridge_token_names = tuple(
            f"s2_{concept_offset + i}" for i in range(bridge_token_concepts_effective)
        )
        concept_offset += bridge_token_concepts_effective
        stage_2_blocks.append(
            TransparentBlockConfig(
                name="dffl_aggregate_bridge_token",
                input_indices=tuple(bridge_output_indices),
                variables=tuple(var3(f"s1_{i}") for i in bridge_output_indices),
                n_concepts=bridge_token_concepts_effective,
                concept_names=bridge_token_names,
                max_rule_arity=min(profile.bridge_token_max_rule_arity, len(bridge_output_indices)),
                max_rules=bridge_token_rules_effective,
                rule_generation_mode=profile.aggregate_rule_generation_mode,
                prototype_term_limit=profile.aggregate_prototype_term_limit,
                prototype_scoring_mode=profile.aggregate_prototype_scoring_mode,
                prototype_variable_pool_size=profile.aggregate_prototype_variable_pool_size,
                prototype_sample_size=profile.aggregate_prototype_sample_size,
                consequent_mode=profile.aggregate_consequent_mode,
            )
        )
        stage_2_width += bridge_token_concepts_effective

    if profile.aggregate_global_context_dim > 0:
        global_dim = int(profile.aggregate_global_context_dim)
        global_max_rules = max(1, int(global_max_rules_effective))
        global_names = tuple(f"s2g_{i}" for i in range(global_dim))
        stage_2_blocks.append(
            TransparentBlockConfig(
                name="dffl_aggregate_global",
                input_indices=tuple(range(stage_1_width)),
                variables=tuple(var3(f"s1_{i}") for i in range(stage_1_width)),
                n_concepts=global_dim,
                concept_names=global_names,
                max_rule_arity=min(profile.aggregate_max_rule_arity, stage_1_width),
                max_rules=global_max_rules,
                rule_generation_mode=profile.aggregate_rule_generation_mode,
                prototype_term_limit=profile.aggregate_prototype_term_limit,
                prototype_scoring_mode=profile.aggregate_prototype_scoring_mode,
                prototype_variable_pool_size=profile.aggregate_prototype_variable_pool_size,
                prototype_sample_size=profile.aggregate_prototype_sample_size,
                consequent_mode=profile.aggregate_consequent_mode,
            )
        )
        stage_2_width += global_dim

    return HierarchicalModelConfig(
        input_dim=input_dim,
        stages=(
            StageConfig(
                name="dffl_stage_1_local",
                blocks=tuple(stage_1_blocks),
                enable_block_gates=profile.block_gates_enabled,
                block_gate_init_logit=profile.block_gate_init_logit,
            ),
            StageConfig(
                name="dffl_stage_2_aggregate",
                blocks=tuple(stage_2_blocks),
                enable_block_gates=profile.block_gates_enabled,
                block_gate_init_logit=profile.block_gate_init_logit,
            ),
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=tuple(
                (var3(f"s2_{i}") if profile.decision_three_terms else var2(f"s2_{i}"))
                for i in range(stage_2_width)
            ),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=profile.decision_max_rule_arity,
            max_rules=decision_max_rules_effective,
            rule_generation_mode=profile.decision_rule_generation_mode,
            prototype_term_limit=profile.decision_prototype_term_limit,
            prototype_scoring_mode=profile.decision_prototype_scoring_mode,
            prototype_variable_pool_size=profile.decision_prototype_variable_pool_size,
            prototype_sample_size=profile.decision_prototype_sample_size,
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
    top_k_rules: int | None = None,
) -> float:
    if validation_inputs.numel() == 0 or validation_targets.numel() == 0:
        return default_threshold

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        validation_logits = model(
            validation_inputs.to(device=device, dtype=torch.float32),
            top_k_rules=top_k_rules,
        ).detach().cpu()

    best_threshold = float(default_threshold)
    best_f1 = float("-inf")
    validation_targets_cpu = validation_targets.detach().cpu()
    for threshold in np.linspace(0.05, 0.95, 91):
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


def _prepare_structure_analysis_tensors(
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: str | None,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if device is None:
        return inputs, targets, "cpu_default"
    requested = str(device).strip().lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            return inputs, targets, "cpu_fallback_no_cuda"
        target_device = torch.device(device)
        return (
            inputs.to(device=target_device, dtype=torch.float32, non_blocking=True),
            targets.to(device=target_device, dtype=torch.float32, non_blocking=True),
            str(target_device),
        )
    return inputs.to(dtype=torch.float32), targets.to(dtype=torch.float32), "cpu_explicit"


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
    feature_geometry: str,
    binary_heavy_grouping_mode: str,
    dffl_rule_swap_ratio: float | None,
    dffl_rule_swap_min_keep: int | None,
    dffl_adaptive_rule_swap: bool,
    dffl_total_rule_budget: int | None,
    dffl_bridge_score_interaction_weight: float | None,
    dffl_bridge_score_stability_weight: float | None,
    dffl_adaptive_budget: bool,
    batch_size: int,
    patience: int,
    classification_threshold: float,
    tune_fuzzy_threshold: bool,
    dffl_one_phase: bool,
    device: str | None,
    fuzzy_models: tuple[str, ...] = FUZZY_MODEL_NAMES,
    include_sklearn: bool = True,
):
    feature_geometry_requested = _validate_feature_geometry(feature_geometry)
    binary_heavy_grouping_mode = _validate_binary_heavy_grouping_mode(binary_heavy_grouping_mode)
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

    analysis_train_inputs, analysis_train_targets, analysis_device = _prepare_structure_analysis_tensors(
        inputs=train_inputs,
        targets=train_targets,
        device=device,
    )

    dffl_profile = resolve_dffl_profile(
        profile_name=dffl_profile_name,
        task_type=spec.task_type,
        n_samples=split.n_samples,
        input_dim=split.input_dim,
    )
    dffl_profile = apply_dffl_profile_overrides(
        dffl_profile,
        rule_swap_ratio=dffl_rule_swap_ratio,
        rule_swap_min_keep=dffl_rule_swap_min_keep,
        total_rule_budget=dffl_total_rule_budget,
        bridge_score_interaction_weight=dffl_bridge_score_interaction_weight,
        bridge_score_stability_weight=dffl_bridge_score_stability_weight,
    )
    feature_geometry_effective, feature_geometry_policy = _resolve_effective_feature_geometry(
        requested_geometry=feature_geometry_requested,
        profile=dffl_profile,
        train_inputs=analysis_train_inputs,
        train_targets=analysis_train_targets,
    )
    dffl_input_groups = make_dffl_input_groups(
        dffl_profile,
        train_inputs=analysis_train_inputs,
        train_targets=analysis_train_targets,
        feature_geometry=feature_geometry_effective,
        binary_heavy_grouping_mode=binary_heavy_grouping_mode,
    )
    dffl_bridge_pairs = make_dffl_bridge_pairs(
        dffl_profile,
        train_inputs=analysis_train_inputs,
        train_targets=analysis_train_targets,
        input_groups=dffl_input_groups,
        feature_geometry=feature_geometry_effective,
        adaptive_budget_enabled=dffl_adaptive_budget,
    )
    dffl_intergroup_share = estimate_intergroup_interaction_share(
        train_inputs=analysis_train_inputs,
        train_targets=analysis_train_targets,
        input_groups=dffl_input_groups,
    )
    dffl_bridge_concepts_effective = _resolve_effective_bridge_concepts(
        profile=dffl_profile,
        intergroup_interaction_share=dffl_intergroup_share,
        input_dim=split.input_dim,
        n_bridge_pairs=len(dffl_bridge_pairs),
    )
    dffl_budget_snapshot = _resolve_dffl_effective_rule_budgets(
        profile=dffl_profile,
        input_dim=split.input_dim,
        n_local_groups=len(dffl_input_groups),
        n_bridge_pairs=len(dffl_bridge_pairs),
        intergroup_interaction_share=dffl_intergroup_share,
        adaptive_budget_enabled=dffl_adaptive_budget,
    )
    dffl_rule_swap_ratio_effective = float(dffl_profile.stagewise_rule_swap_ratio)
    dffl_rule_swap_reason = "profile_default"
    if dffl_rule_swap_ratio is not None:
        dffl_rule_swap_reason = "cli_override"
    elif dffl_adaptive_rule_swap:
        dffl_rule_swap_ratio_effective, dffl_rule_swap_reason = resolve_adaptive_rule_swap_ratio(
            base_ratio=dffl_rule_swap_ratio_effective,
            input_dim=split.input_dim,
            intergroup_interaction_share=dffl_intergroup_share,
        )
    dffl_rule_swap_min_keep_effective = (
        int(dffl_profile.stagewise_rule_swap_min_keep) if dffl_rule_swap_ratio_effective > 0.0 else 0
    )
    dffl_binary_feature_indices = _detect_binary_feature_indices(analysis_train_inputs)
    progress_log(
        (
            "seed={seed} profile: dffl={name}, task={task}, device={device}, grouping={grouping}, "
            "analysis_device={analysis_device}, "
            "feature_geometry={feature_geometry}, feature_geometry_policy={feature_geometry_policy}, binary_heavy_grouping={binary_mode}, "
            "groups={groups}, bridges={bridges}, intergroup_share={share:.3f}, "
            "bridge_concepts_effective={bridge_concepts}, "
            "rule_swap={swap:.3f}, swap_keep={keep}, swap_policy={policy}, "
            "budget_total={budget_total}, budget_policy={budget_policy}, adaptive_budget={adaptive_budget}"
        ).format(
            seed=seed,
            name=dffl_profile.name,
            task=spec.task_type,
            device=device or "default",
            analysis_device=analysis_device,
            grouping=dffl_profile.input_group_strategy,
            feature_geometry=feature_geometry_effective,
            feature_geometry_policy=feature_geometry_policy,
            binary_mode=binary_heavy_grouping_mode,
            groups=len(dffl_input_groups),
            bridges=len(dffl_bridge_pairs),
            share=dffl_intergroup_share,
            bridge_concepts=int(dffl_bridge_concepts_effective),
            swap=dffl_rule_swap_ratio_effective,
            keep=dffl_rule_swap_min_keep_effective,
            policy=dffl_rule_swap_reason,
            budget_total=int(dffl_budget_snapshot["total_rule_budget_effective"]),
            budget_policy=str(dffl_budget_snapshot["allocation_policy"]),
            adaptive_budget=dffl_adaptive_budget,
        )
    )

    dffl_learning_rate_effective = (
        dffl_learning_rate * dffl_profile.learning_rate_scale_classification
        if spec.task_type == "binary_classification"
        else dffl_learning_rate * dffl_profile.learning_rate_scale_regression
    )
    dffl_refinement_cycles = max(refinement_cycles, dffl_profile.refinement_cycle_floor)

    trained_fuzzy_models: dict[str, torch.nn.Module] = {}
    model_top_k_rules: dict[str, int | None] = {}

    if "ruanfis_refined_deep" in fuzzy_models:
        phase_started_at = time.perf_counter()
        progress_log(f"seed={seed} model=dffl: start")
        hidden_high = 0.75 if spec.task_type == "binary_classification" else 0.8
        hidden_low = 0.25 if spec.task_type == "binary_classification" else 0.2
        gate_floor = 0.15
        gate_ceiling = 0.9
        # Large binary tasks are sensitive to over-confident bootstrap concepts/rules.
        if (
            dffl_one_phase
            and spec.task_type == "binary_classification"
            and dffl_profile.name in {"quality_large_cls", "quality_large_cls_plus"}
        ):
            hidden_high = 0.65
            hidden_low = 0.35
            gate_floor = 0.2
            gate_ceiling = 0.8
        dffl_bootstrap_config = BootstrapConfig(
            decision_task_type=spec.task_type,
            hidden_high=hidden_high,
            hidden_low=hidden_low,
            gate_floor=gate_floor,
            gate_ceiling=gate_ceiling,
        )
        dffl_config = build_dffl_config(
            split.input_dim,
            profile=dffl_profile,
            input_groups=dffl_input_groups,
            binary_feature_indices=dffl_binary_feature_indices,
            bridge_feature_pairs=dffl_bridge_pairs,
            intergroup_interaction_share=dffl_intergroup_share,
            adaptive_budget_enabled=dffl_adaptive_budget,
        )
        dffl_block_agreement_weight, dffl_block_agreement_target_corr = _resolve_dffl_block_agreement_params(
            profile=dffl_profile,
            intergroup_interaction_share=dffl_intergroup_share,
            n_bridge_pairs=len(dffl_bridge_pairs),
        )
        dffl_training_config = TrainingConfig(
            task_type=spec.task_type,
            max_epochs=max_epochs,
            learning_rate=dffl_learning_rate_effective,
            patience=min(patience, max_epochs),
            batch_size=batch_size,
            shuffle=True,
            classification_threshold=classification_threshold,
            monitor_metric="f1" if spec.task_type == "binary_classification" else None,
            monitor_mode="max" if spec.task_type == "binary_classification" else None,
            top_k_rules=dffl_profile.top_k_rules,
            binary_auto_pos_weight=(
                spec.task_type == "binary_classification" and dffl_profile.binary_auto_pos_weight
            ),
            binary_soft_f1_weight=(
                (
                    dffl_profile.binary_soft_f1_weight
                    if (spec.task_type == "binary_classification" and dffl_one_phase)
                    else 0.0
                )
            ),
            weight_decay=dffl_profile.weight_decay,
            gradient_clip_norm=dffl_profile.gradient_clip_norm,
            rule_sparsity_weight=dffl_profile.rule_sparsity_weight,
            rule_length_weight=dffl_profile.rule_length_weight,
            decision_usage_balance_weight=dffl_profile.decision_usage_balance_weight,
            block_gate_l1_weight=dffl_profile.block_gate_l1_weight,
            block_agreement_weight=dffl_block_agreement_weight,
            block_agreement_target_corr=dffl_block_agreement_target_corr,
            block_agreement_stage_limit=dffl_profile.block_agreement_stage_limit,
            regularization_warmup_epochs=max(1, max_epochs // 3),
            top_k_warmup_epochs=(max(1, max_epochs // 4) if dffl_profile.top_k_rules is not None else 0),
            prune_after_fit=False,
            device=device,
        )
        if dffl_one_phase:
            dffl_model = build_bootstrapped_hierarchical_model(
                dffl_config,
                sample_inputs=train_inputs,
                sample_targets=train_targets,
                bootstrap_config=dffl_bootstrap_config,
                device=device,
            )
            dffl_trainer = FuzzyTrainer(dffl_model, dffl_training_config)
            dffl_trainer.fit(train_inputs, train_targets, validation_inputs, validation_targets)
            trained_fuzzy_models["ruanfis_refined_deep"] = dffl_model
        else:
            dffl_result = build_refined_hierarchical_model(
                dffl_config,
                train_inputs=train_inputs,
                train_targets=train_targets,
                validation_inputs=validation_inputs,
                validation_targets=validation_targets,
                bootstrap_config=dffl_bootstrap_config,
                pretraining_config=StagewisePretrainingConfig(
                    task_type=spec.task_type,
                    epochs_per_stage=pretrain_epochs,
                    decision_epochs=decision_pretrain_epochs,
                    refinement_rounds=dffl_profile.pretrain_refinement_rounds,
                    learning_rate=dffl_learning_rate_effective,
                    batch_size=batch_size,
                    shuffle=True,
                    rule_sparsity_weight=0.0,
                    stage_selection_metric="auto",
                    stage_selection_threshold=classification_threshold,
                    rule_swap_ratio=dffl_rule_swap_ratio_effective,
                    rule_swap_min_keep=dffl_rule_swap_min_keep_effective,
                ),
                training_config=dffl_training_config,
                refinement_loop_config=RefinementLoopConfig(
                    max_cycles=dffl_refinement_cycles,
                    patience=1,
                    min_delta=1e-4,
                ),
            )
            trained_fuzzy_models["ruanfis_refined_deep"] = dffl_result.model
        model_top_k_rules["ruanfis_refined_deep"] = dffl_profile.top_k_rules
        progress_log(f"seed={seed} model=dffl: done in {time.perf_counter() - phase_started_at:.2f}s")

    if "ruanfis_shallow" in fuzzy_models:
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
        trained_fuzzy_models["ruanfis_shallow"] = shallow_model
        model_top_k_rules["ruanfis_shallow"] = None
        progress_log(f"seed={seed} model=shallow: done in {time.perf_counter() - phase_started_at:.2f}s")

    if "ruanfis_stacked_anfis" in fuzzy_models:
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
        trained_fuzzy_models["ruanfis_stacked_anfis"] = stacked_model
        model_top_k_rules["ruanfis_stacked_anfis"] = None
        progress_log(f"seed={seed} model=stacked: done in {time.perf_counter() - phase_started_at:.2f}s")

    if "ruanfis_hierarchical_anfis" in fuzzy_models:
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
        trained_fuzzy_models["ruanfis_hierarchical_anfis"] = hierarchical_anfis_model
        model_top_k_rules["ruanfis_hierarchical_anfis"] = None
        progress_log(
            f"seed={seed} model=hierarchical_anfis: done in {time.perf_counter() - phase_started_at:.2f}s"
        )

    sklearn_results: tuple = ()
    if include_sklearn:
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
    model_thresholds = {model_name: classification_threshold for model_name in trained_fuzzy_models.keys()}
    if tune_fuzzy_threshold and spec.task_type == "binary_classification":
        for model_name, model in trained_fuzzy_models.items():
            model_thresholds[model_name] = _choose_best_classification_threshold(
                model,
                validation_inputs=validation_inputs,
                validation_targets=validation_targets,
                default_threshold=classification_threshold,
                top_k_rules=model_top_k_rules.get(model_name),
            )
        progress_log(
            "seed={seed} threshold_tuning: {pairs}".format(
                seed=seed,
                pairs=", ".join(f"{name}={value:.2f}" for name, value in model_thresholds.items()),
            )
        )

    fuzzy_results = tuple(
        evaluate_trained_model(
            model_name,
            trained_fuzzy_models[model_name],
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            classification_threshold=model_thresholds[model_name],
            top_k_rules=model_top_k_rules.get(model_name),
        )
        for model_name in FUZZY_MODEL_NAMES
        if model_name in trained_fuzzy_models
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


def _format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def build_cross_dataset_summary(
    dataset_results: dict[str, MultiSeedBenchmarkResult],
    specs: dict[str, DatasetSpec],
) -> str:
    if not dataset_results:
        return (
            "| model | avg_rank | wins | avg_rules (mean+/-std) | avg_active_rule_jaccard (mean+/-std) |\n"
            "| --- | --- | --- | --- | --- |"
        )

    per_dataset_available = {
        dataset_name: {entry.model_name for entry in multi_seed.aggregated_results}
        for dataset_name, multi_seed in dataset_results.items()
    }
    fuzzy_model_names = tuple(
        model_name
        for model_name in FUZZY_MODEL_NAMES
        if all(model_name in available for available in per_dataset_available.values())
    )
    if not fuzzy_model_names:
        return (
            "| model | avg_rank | wins | avg_rules (mean+/-std) | avg_active_rule_jaccard (mean+/-std) |\n"
            "| --- | --- | --- | --- | --- |"
        )

    per_dataset_scores: dict[str, dict[str, tuple[float, float]]] = {}
    per_dataset_ranks: dict[str, dict[str, float]] = {}

    for dataset_name, multi_seed in dataset_results.items():
        spec = specs[dataset_name]
        aggregated = {
            entry.model_name: entry
            for entry in multi_seed.aggregated_results
            if entry.model_name in fuzzy_model_names
        }
        metric_name = PRIMARY_METRIC[spec.task_type]
        scores = {
            model_name: (
                float(aggregated[model_name].test_metrics[metric_name].mean),
                float(aggregated[model_name].test_metrics[metric_name].std),
            )
            for model_name in fuzzy_model_names
        }
        per_dataset_scores[dataset_name] = scores
        per_dataset_ranks[dataset_name] = _rank_scores(
            {model_name: score_mean for model_name, (score_mean, _) in scores.items()},
            higher_is_better=(spec.task_type == "binary_classification"),
        )

    lines = [
        "| model | avg_rank | wins | avg_rules (mean+/-std) | avg_active_rule_jaccard (mean+/-std) | "
        + " | ".join(
            f"{dataset_name}:{PRIMARY_METRIC[specs[dataset_name].task_type]} (mean+/-std)"
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

            score_mean, score_std = per_dataset_scores[dataset_name][model_name]
            dataset_metric_cells.append(_format_mean_std(score_mean, score_std, decimals=4))

        avg_rank = float(sum(ranks) / len(ranks))
        avg_rules = float(sum(total_rules_values) / len(total_rules_values)) if total_rules_values else float("nan")
        avg_rules_std = (
            float(np.std(total_rules_values, ddof=0)) if len(total_rules_values) > 1 else 0.0
        )
        avg_stability = float(sum(stability_values) / len(stability_values)) if stability_values else float("nan")
        avg_stability_std = (
            float(np.std(stability_values, ddof=0)) if len(stability_values) > 1 else 0.0
        )

        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    f"{avg_rank:.3f}",
                    str(wins),
                    _format_mean_std(avg_rules, avg_rules_std, decimals=2),
                    _format_mean_std(avg_stability, avg_stability_std, decimals=4),
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
    fuzzy_model_names = FUZZY_MODEL_NAMES

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


def _resolve_dffl_bootstrap_hyperparams(
    *,
    task_type: TaskType,
    dffl_one_phase: bool,
    resolved_profile_name: str,
) -> dict[str, float]:
    hidden_high = 0.75 if task_type == "binary_classification" else 0.8
    hidden_low = 0.25 if task_type == "binary_classification" else 0.2
    gate_floor = 0.15
    gate_ceiling = 0.9
    if (
        dffl_one_phase
        and task_type == "binary_classification"
        and resolved_profile_name in {"quality_large_cls", "quality_large_cls_plus"}
    ):
        hidden_high = 0.65
        hidden_low = 0.35
        gate_floor = 0.2
        gate_ceiling = 0.8
    return {
        "hidden_high": float(hidden_high),
        "hidden_low": float(hidden_low),
        "gate_floor": float(gate_floor),
        "gate_ceiling": float(gate_ceiling),
    }


def _summarize_shallow_architecture(input_dim: int) -> dict[str, object]:
    config = build_shallow_config(input_dim)
    return {
        "input_dim": int(input_dim),
        "feature_block": {
            "variables": len(config.feature_block.variables),
            "n_concepts": int(config.feature_block.n_concepts),
            "max_rule_arity": int(config.feature_block.max_rule_arity),
            "max_rules": int(config.feature_block.max_rules),
            "rule_generation_mode": str(config.feature_block.rule_generation_mode),
        },
        "decision_layer": {
            "variables": len(config.decision_layer.variables),
            "output_dim": int(config.decision_layer.output_dim),
            "max_rule_arity": int(config.decision_layer.max_rule_arity),
            "max_rules": int(config.decision_layer.max_rules),
            "rule_generation_mode": str(config.decision_layer.rule_generation_mode),
        },
    }


def _summarize_stacked_architecture(input_dim: int) -> dict[str, object]:
    config = build_stacked_config(input_dim)
    return {
        "input_dim": int(input_dim),
        "n_layers": len(config.layers),
        "layers": [
            {
                "name": layer.name,
                "variables": len(layer.variables),
                "output_dim": int(layer.output_dim),
                "max_rule_arity": int(layer.max_rule_arity),
                "max_rules": int(layer.max_rules),
                "rule_generation_mode": str(layer.rule_generation_mode),
            }
            for layer in config.layers
        ],
    }


def _summarize_hierarchical_anfis_architecture(input_dim: int) -> dict[str, object]:
    config = build_hierarchical_anfis_config(input_dim)
    stage_summaries = []
    for stage in config.stages:
        stage_summaries.append(
            {
                "name": stage.name,
                "n_blocks": len(stage.blocks),
                "total_max_rules": int(sum(block.max_rules for block in stage.blocks)),
                "blocks": [
                    {
                        "name": block.name,
                        "input_arity": len(block.input_indices),
                        "output_dim": int(block.output_dim),
                        "max_rule_arity": int(block.max_rule_arity),
                        "max_rules": int(block.max_rules),
                        "rule_generation_mode": str(block.rule_generation_mode),
                    }
                    for block in stage.blocks
                ],
            }
        )
    return {
        "input_dim": int(input_dim),
        "stages": stage_summaries,
        "decision_layer": {
            "variables": len(config.decision_layer.variables),
            "output_dim": int(config.decision_layer.output_dim),
            "max_rule_arity": int(config.decision_layer.max_rule_arity),
            "max_rules": int(config.decision_layer.max_rules),
            "rule_generation_mode": str(config.decision_layer.rule_generation_mode),
        },
    }


def _summarize_dffl_architecture(
    *,
    input_dim: int,
    task_type: TaskType,
    profile: DfflProfile,
    input_groups: tuple[tuple[int, ...], ...],
    bridge_pairs: tuple[tuple[int, int], ...] = (),
    intergroup_interaction_share: float | None = None,
    adaptive_budget_enabled: bool = True,
) -> dict[str, object]:
    bridge_pairs = tuple(bridge_pairs) if profile.bridge_enabled else ()
    n_local_groups = len(input_groups)
    share = (
        float(np.clip(intergroup_interaction_share, 0.0, 1.0))
        if intergroup_interaction_share is not None
        else 0.5
    )
    effective_budgets = _resolve_dffl_effective_rule_budgets(
        profile=profile,
        input_dim=input_dim,
        n_local_groups=n_local_groups,
        n_bridge_pairs=len(bridge_pairs),
        intergroup_interaction_share=share,
        adaptive_budget_enabled=adaptive_budget_enabled,
    )
    effective_bridge_concepts = _resolve_effective_bridge_concepts(
        profile=profile,
        intergroup_interaction_share=share,
        input_dim=input_dim,
        n_bridge_pairs=len(bridge_pairs),
    )
    local_max_rules_effective = int(effective_budgets["local_max_rules_effective"])
    bridge_max_rules_effective = int(effective_budgets["bridge_max_rules_effective"])
    decision_max_rules_effective = int(effective_budgets["decision_max_rules_effective"])
    aggregate_max_rules_effective = int(effective_budgets["aggregate_max_rules_effective"])

    bridge_width = int(effective_bridge_concepts * len(bridge_pairs))
    stage_1_width = profile.local_concepts * len(input_groups) + bridge_width
    stage_2_width_base = min(
        profile.stage2_width_max,
        max(profile.stage2_width_min, len(input_groups) + len(bridge_pairs) + 1),
    )
    aggregate_blocks = max(1, min(profile.aggregate_block_count, stage_2_width_base))
    aggregate_budget_for_blocks = max(1, aggregate_max_rules_effective)
    bridge_token_rules_effective = 0
    bridge_token_concepts_effective = 0
    if profile.bridge_token_enabled and bridge_pairs and aggregate_budget_for_blocks >= 6:
        bridge_token_concepts_effective = max(1, int(profile.bridge_token_concepts))
        bridge_token_rules_effective = min(
            int(profile.bridge_token_max_rules),
            max(1, aggregate_budget_for_blocks // 5),
        )
        aggregate_budget_for_blocks = max(1, aggregate_budget_for_blocks - bridge_token_rules_effective)
    aggregate_global_max_rules_effective = 0
    if profile.aggregate_global_context_dim > 0:
        global_cap = (
            int(profile.aggregate_global_max_rules)
            if profile.aggregate_global_max_rules > 0
            else max(4, min(profile.aggregate_max_rules // 2, 12))
        )
        aggregate_global_max_rules_effective = min(global_cap, max(2, aggregate_budget_for_blocks // 3))
        aggregate_budget_for_blocks = max(1, aggregate_budget_for_blocks - aggregate_global_max_rules_effective)
    aggregate_rule_budget = _split_even(aggregate_budget_for_blocks, aggregate_blocks)
    stage_2_width_total = stage_2_width_base + (
        int(profile.aggregate_global_context_dim) if profile.aggregate_global_context_dim > 0 else 0
    )
    stage_2_width_total += int(bridge_token_concepts_effective)

    return {
        "input_dim": int(input_dim),
        "task_type": str(task_type),
        "resolved_profile_name": profile.name,
        "adaptive_budget_enabled": bool(adaptive_budget_enabled),
        "profile_params": asdict(profile),
        "intergroup_interaction_share": float(share),
        "input_grouping": {
            "strategy": profile.input_group_strategy,
            "group_count": len(input_groups),
            "group_sizes": [len(group) for group in input_groups],
        },
        "bridge_intergroup": {
            "enabled": bool(profile.bridge_enabled),
            "pair_count": int(len(bridge_pairs)),
            "pairs": [[int(left), int(right)] for left, right in bridge_pairs],
            "concepts_per_pair": int(effective_bridge_concepts),
            "width": int(bridge_width),
            "token_enabled": bool(profile.bridge_token_enabled),
            "token_concepts_effective": int(bridge_token_concepts_effective),
            "token_max_rules_effective": int(bridge_token_rules_effective),
        },
        "derived_widths": {
            "stage_1_width": int(stage_1_width),
            "stage_2_width_base": int(stage_2_width_base),
            "stage_2_width_total": int(stage_2_width_total),
            "aggregate_blocks": int(aggregate_blocks),
        },
        "rule_budgets": {
            "allocation_policy": str(effective_budgets["allocation_policy"]),
            "total_rule_budget_effective": int(effective_budgets["total_rule_budget_effective"]),
            "local_max_rules_per_block": int(profile.local_max_rules),
            "local_max_rules_per_block_effective": int(local_max_rules_effective),
            "aggregate_max_rules_total": int(aggregate_max_rules_effective),
            "aggregate_max_rules_split": [int(value) for value in aggregate_rule_budget],
            "aggregate_bridge_token_concepts": int(bridge_token_concepts_effective),
            "aggregate_bridge_token_max_rules": int(bridge_token_rules_effective),
            "aggregate_global_context_dim": int(profile.aggregate_global_context_dim),
            "aggregate_global_max_rules": int(aggregate_global_max_rules_effective),
            "bridge_max_rules_per_block": int(profile.bridge_max_rules),
            "bridge_max_rules_per_block_effective": int(bridge_max_rules_effective),
            "decision_max_rules": int(profile.decision_max_rules),
            "decision_max_rules_effective": int(decision_max_rules_effective),
        },
    }


def build_reproducibility_manifest_payload(
    *,
    args: argparse.Namespace,
    dataset_names: tuple[str, ...],
    seeds: tuple[int, ...],
    fuzzy_models: tuple[str, ...],
    dataset_protocols: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "generated_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script": "examples/run_real_datasets_benchmark.py",
        "protocol": {
            "datasets": list(dataset_names),
            "seeds": list(seeds),
            "split": {
                "test_size": float(args.test_size),
                "validation_size": float(args.validation_size),
                "inner_validation_fraction": float(args.validation_size / (1.0 - args.test_size)),
                "split_method": "train_test_split with fixed random_state per seed",
                "stratification": "binary tasks stratified; regression tasks non-stratified",
            },
            "preprocessing": {
                "feature_scaling": "MinMaxScaler to [0,1], followed by clipping to [0,1]",
                "target_scaling_regression": "MinMaxScaler to [0,1]",
                "target_scaling_classification": "none (binary labels as float tensor)",
                "train_noise_sigma_regression_only": float(args.train_noise_sigma),
            },
            "execution": {
                "fuzzy_models": list(fuzzy_models),
                "include_sklearn_baselines": bool(not args.skip_sklearn),
                "gpu_only": bool(args.gpu_only),
                "device": args.device if args.device is not None else "default",
                "feature_geometry_requested": str(args.feature_geometry),
                "binary_heavy_grouping_mode": str(args.binary_heavy_grouping),
                "feature_geometry_auto_policy": (
                    "if d>=80 and n>=4000 and intergroup_share>=0.80 and binary_ratio<=0.45 then hyperbolic else euclidean"
                ),
                "dffl_adaptive_rule_swap": bool(not args.disable_dffl_adaptive_rule_swap),
                "dffl_adaptive_budget": bool(not args.disable_dffl_adaptive_budget),
                "dffl_rule_swap_ratio_override": (
                    float(args.dffl_rule_swap_ratio) if args.dffl_rule_swap_ratio is not None else None
                ),
                "dffl_rule_swap_min_keep_override": (
                    int(args.dffl_rule_swap_min_keep) if args.dffl_rule_swap_min_keep is not None else None
                ),
                "dffl_total_rule_budget_override": (
                    int(args.dffl_total_rule_budget) if args.dffl_total_rule_budget is not None else None
                ),
                "dffl_bridge_score_interaction_weight_override": (
                    float(args.dffl_bridge_score_interaction_weight)
                    if args.dffl_bridge_score_interaction_weight is not None
                    else None
                ),
                "dffl_bridge_score_stability_weight_override": (
                    float(args.dffl_bridge_score_stability_weight)
                    if args.dffl_bridge_score_stability_weight is not None
                    else None
                ),
            },
            "training_budgets": {
                "max_epochs": int(args.max_epochs),
                "pretrain_epochs": int(args.pretrain_epochs),
                "decision_pretrain_epochs": int(args.decision_pretrain_epochs),
                "refinement_cycles_requested": int(args.refinement_cycles),
                "batch_size": int(args.batch_size),
                "patience": int(args.patience),
                "learning_rate_fuzzy": float(args.fuzzy_learning_rate),
                "learning_rate_dffl_base": float(args.dffl_learning_rate),
            },
            "evaluation": {
                "primary_metrics": {"regression": "rmse", "binary_classification": "f1"},
                "classification_threshold_default": float(args.classification_threshold),
                "tune_fuzzy_threshold": bool(args.tune_fuzzy_threshold),
                "threshold_tuning_grid_if_enabled": "0.05..0.95 step=0.01 on validation split",
                "active_rule_criterion": "rule_probability >= 0.5",
                "stability_metrics": [
                    "active_rule_jaccard",
                    "decision_active_rule_jaccard",
                    "layer_active_rule_jaccard",
                ],
            },
        },
        "model_hyperparameters": {
            "ruanfis_shallow": {
                "training": {
                    "max_epochs": int(args.max_epochs),
                    "learning_rate": float(args.fuzzy_learning_rate),
                    "patience": int(min(args.patience, args.max_epochs)),
                    "batch_size": int(args.batch_size),
                    "shuffle": True,
                }
            },
            "ruanfis_stacked_anfis": {
                "training": {
                    "max_epochs": int(args.max_epochs),
                    "learning_rate": float(args.fuzzy_learning_rate),
                    "patience": int(min(args.patience, args.max_epochs)),
                    "batch_size": int(args.batch_size),
                    "shuffle": True,
                }
            },
            "ruanfis_hierarchical_anfis": {
                "training": {
                    "max_epochs": int(args.max_epochs),
                    "learning_rate": float(args.fuzzy_learning_rate),
                    "patience": int(min(args.patience, args.max_epochs)),
                    "batch_size": int(args.batch_size),
                    "shuffle": True,
                }
            },
            "ruanfis_refined_deep": {
                "dffl_profile_requested": str(args.dffl_profile),
                "dffl_one_phase": bool(args.dffl_one_phase),
                "dffl_adaptive_rule_swap": bool(not args.disable_dffl_adaptive_rule_swap),
                "dffl_adaptive_budget": bool(not args.disable_dffl_adaptive_budget),
                "dffl_rule_swap_ratio_override": (
                    float(args.dffl_rule_swap_ratio) if args.dffl_rule_swap_ratio is not None else None
                ),
                "dffl_rule_swap_min_keep_override": (
                    int(args.dffl_rule_swap_min_keep) if args.dffl_rule_swap_min_keep is not None else None
                ),
                "dffl_total_rule_budget_override": (
                    int(args.dffl_total_rule_budget) if args.dffl_total_rule_budget is not None else None
                ),
                "dffl_bridge_score_interaction_weight_override": (
                    float(args.dffl_bridge_score_interaction_weight)
                    if args.dffl_bridge_score_interaction_weight is not None
                    else None
                ),
                "dffl_bridge_score_stability_weight_override": (
                    float(args.dffl_bridge_score_stability_weight)
                    if args.dffl_bridge_score_stability_weight is not None
                    else None
                ),
                "training_base": {
                    "max_epochs": int(args.max_epochs),
                    "learning_rate_base": float(args.dffl_learning_rate),
                    "patience": int(min(args.patience, args.max_epochs)),
                    "batch_size": int(args.batch_size),
                    "shuffle": True,
                    "top_k_rules": "from resolved profile",
                    "regularization": "from resolved profile",
                },
                "pretraining_refinement": {
                    "pretrain_epochs": int(args.pretrain_epochs),
                    "decision_pretrain_epochs": int(args.decision_pretrain_epochs),
                    "refinement_cycles_requested": int(args.refinement_cycles),
                    "stage_selection_metric": "auto",
                    "stage_selection_threshold": float(args.classification_threshold),
                },
            },
        },
        "dataset_specific_protocols": dataset_protocols,
    }


def build_reproducibility_manifest_markdown(payload: dict[str, object]) -> str:
    protocol = payload["protocol"]
    split = protocol["split"]
    preprocessing = protocol["preprocessing"]
    execution = protocol["execution"]
    training = protocol["training_budgets"]
    evaluation = protocol["evaluation"]

    lines: list[str] = []
    lines.append("# Reproducibility Manifest")
    lines.append("")
    lines.append(f"- generated_utc: `{payload['generated_utc']}`")
    lines.append(f"- script: `{payload['script']}`")
    lines.append(f"- datasets: `{', '.join(protocol['datasets'])}`")
    lines.append(f"- seeds: `{', '.join(str(seed) for seed in protocol['seeds'])}`")
    lines.append("")
    lines.append("## Split Protocol")
    lines.append("")
    lines.append(f"- test_size: `{split['test_size']}`")
    lines.append(f"- validation_size: `{split['validation_size']}`")
    lines.append(f"- inner_validation_fraction: `{split['inner_validation_fraction']:.6f}`")
    lines.append(f"- split_method: {split['split_method']}")
    lines.append(f"- stratification: {split['stratification']}")
    lines.append("")
    lines.append("## Preprocessing")
    lines.append("")
    lines.append(f"- feature_scaling: {preprocessing['feature_scaling']}")
    lines.append(f"- target_scaling_regression: {preprocessing['target_scaling_regression']}")
    lines.append(f"- target_scaling_classification: {preprocessing['target_scaling_classification']}")
    lines.append(f"- train_noise_sigma_regression_only: `{preprocessing['train_noise_sigma_regression_only']}`")
    lines.append("")
    lines.append("## Execution")
    lines.append("")
    lines.append(f"- feature_geometry_requested: `{execution['feature_geometry_requested']}`")
    lines.append(f"- binary_heavy_grouping_mode: `{execution['binary_heavy_grouping_mode']}`")
    lines.append(f"- feature_geometry_auto_policy: {execution['feature_geometry_auto_policy']}")
    lines.append(f"- device: `{execution['device']}`")
    lines.append(f"- gpu_only: `{execution['gpu_only']}`")
    lines.append(f"- include_sklearn_baselines: `{execution['include_sklearn_baselines']}`")
    lines.append("")
    lines.append("## Training Budgets")
    lines.append("")
    lines.append(f"- max_epochs: `{training['max_epochs']}`")
    lines.append(f"- pretrain_epochs: `{training['pretrain_epochs']}`")
    lines.append(f"- decision_pretrain_epochs: `{training['decision_pretrain_epochs']}`")
    lines.append(f"- refinement_cycles_requested: `{training['refinement_cycles_requested']}`")
    lines.append(f"- batch_size: `{training['batch_size']}`")
    lines.append(f"- patience: `{training['patience']}`")
    lines.append(f"- learning_rate_fuzzy: `{training['learning_rate_fuzzy']}`")
    lines.append(f"- learning_rate_dffl_base: `{training['learning_rate_dffl_base']}`")
    lines.append("")
    lines.append("## Evaluation Protocol")
    lines.append("")
    lines.append(f"- primary_metrics: `{evaluation['primary_metrics']}`")
    lines.append(f"- classification_threshold_default: `{evaluation['classification_threshold_default']}`")
    lines.append(f"- tune_fuzzy_threshold: `{evaluation['tune_fuzzy_threshold']}`")
    lines.append(f"- threshold_tuning_grid_if_enabled: {evaluation['threshold_tuning_grid_if_enabled']}")
    lines.append(f"- active_rule_criterion: {evaluation['active_rule_criterion']}")
    lines.append(f"- stability_metrics: `{', '.join(evaluation['stability_metrics'])}`")
    lines.append("")
    lines.append("## Dataset-Specific DFFL Resolution")
    lines.append("")
    lines.append(
        "| dataset | task | n_samples | input_dim | profile_resolved | geom_req | geom_eff | one_phase | lr_effective | "
        "groups | stage1_width | stage2_width_total | decision_max_rules |"
    )
    lines.append("| --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for dataset_name, entry in payload["dataset_specific_protocols"].items():
        dffl = entry["dffl"]
        arch = dffl["architecture"]
        lines.append(
            "| {dataset} | {task} | {n_samples} | {input_dim} | {profile} | {geom_req} | {geom_eff} | {one_phase} | {lr:.6f} | "
            "{groups} | {s1} | {s2} | {dec_rules} |".format(
                dataset=dataset_name,
                task=entry["task_type"],
                n_samples=entry["n_samples"],
                input_dim=entry["input_dim"],
                profile=dffl["resolved_profile"],
                geom_req=dffl.get("feature_geometry_requested", "n/a"),
                geom_eff=dffl.get("feature_geometry_effective", "n/a"),
                one_phase=dffl["one_phase"],
                lr=dffl["effective_learning_rate"],
                groups=arch["input_grouping"]["group_count"],
                s1=arch["derived_widths"]["stage_1_width"],
                s2=arch["derived_widths"]["stage_2_width_total"],
                dec_rules=arch["rule_budgets"]["decision_max_rules"],
            )
        )
    lines.append("")
    lines.append("## Model Hyperparameters (Detailed)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(payload["model_hyperparameters"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


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
    parser.add_argument(
        "--feature-geometry",
        type=str,
        default="euclidean",
        choices=FEATURE_GEOMETRIES,
        help="Feature geometry for DFFL grouping/bridges: euclidean (corr) or hyperbolic (Poincare graph).",
    )
    parser.add_argument(
        "--binary-heavy-grouping",
        type=str,
        default="binary_aware",
        choices=BINARY_HEAVY_GROUPING_MODES,
        help="Stage-1 grouping policy when binary feature ratio is high.",
    )
    parser.add_argument(
        "--dffl-rule-swap-ratio",
        type=float,
        default=None,
        help="Optional override for stage-wise rule swap ratio in [0,1].",
    )
    parser.add_argument(
        "--dffl-rule-swap-min-keep",
        type=int,
        default=None,
        help="Optional override for minimum kept rules during stage-wise rule swap.",
    )
    parser.add_argument(
        "--disable-dffl-adaptive-rule-swap",
        action="store_true",
        help="Disable adaptive rule-swap policy (use profile/override value directly).",
    )
    parser.add_argument(
        "--disable-dffl-adaptive-budget",
        action="store_true",
        help="Disable adaptive DFFL rule-budget allocation (use full cap-level budget).",
    )
    parser.add_argument(
        "--dffl-total-rule-budget",
        type=int,
        default=None,
        help="Optional explicit total DFFL rule budget (0 means profile/default behavior).",
    )
    parser.add_argument(
        "--dffl-bridge-score-interaction-weight",
        type=float,
        default=None,
        help="Optional override for bridge pair interaction-score weight.",
    )
    parser.add_argument(
        "--dffl-bridge-score-stability-weight",
        type=float,
        default=None,
        help="Optional override for bridge pair stability-score weight.",
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
        "--dffl-one-phase",
        action="store_true",
        help="Train DFFL in one-phase mode (bootstrap + single joint fit) without stage-wise refinement.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device for fuzzy models, e.g. 'cpu' or 'cuda'. If omitted, use model default.",
    )
    parser.add_argument(
        "--fuzzy-models",
        type=str,
        default="all",
        help=(
            "Comma-separated fuzzy models: all, "
            "ruanfis_shallow, ruanfis_stacked_anfis, ruanfis_hierarchical_anfis, ruanfis_refined_deep "
            "(aliases: shallow, stacked, hierarchical, dffl)."
        ),
    )
    parser.add_argument(
        "--skip-sklearn",
        action="store_true",
        help="Skip sklearn baselines for fast fuzzy-only tuning.",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Force GPU-only fuzzy pipeline: requires CUDA, sets --device=cuda and enables --skip-sklearn.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-table-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--interpretability-report-output", type=Path, default=None)
    parser.add_argument("--reproducibility-manifest-output", type=Path, default=None)
    parser.add_argument("--reproducibility-manifest-json-output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    _apply_gpu_only_mode(args)
    args.feature_geometry = _validate_feature_geometry(args.feature_geometry)
    args.binary_heavy_grouping = _validate_binary_heavy_grouping_mode(args.binary_heavy_grouping)

    dataset_names = parse_dataset_names(args.datasets)
    unknown = [name for name in dataset_names if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset names: {', '.join(unknown)}. Available: {', '.join(DATASETS.keys())}")

    seeds = parse_seeds(args.seeds)
    fuzzy_models = parse_fuzzy_model_names(args.fuzzy_models)

    dataset_results: dict[str, MultiSeedBenchmarkResult] = {}
    dataset_reports: dict[str, str] = {}
    dataset_protocols: dict[str, dict[str, object]] = {}

    for dataset_name in dataset_names:
        spec = DATASETS[dataset_name]
        dataset_features, dataset_targets = spec.loader()
        resolved_profile = resolve_dffl_profile(
            profile_name=args.dffl_profile,
            task_type=spec.task_type,
            n_samples=int(dataset_features.shape[0]),
            input_dim=int(dataset_features.shape[1]),
        )
        resolved_profile = apply_dffl_profile_overrides(
            resolved_profile,
            rule_swap_ratio=args.dffl_rule_swap_ratio,
            rule_swap_min_keep=args.dffl_rule_swap_min_keep,
            total_rule_budget=args.dffl_total_rule_budget,
            bridge_score_interaction_weight=args.dffl_bridge_score_interaction_weight,
            bridge_score_stability_weight=args.dffl_bridge_score_stability_weight,
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
                feature_geometry=args.feature_geometry,
                binary_heavy_grouping_mode=args.binary_heavy_grouping,
                dffl_rule_swap_ratio=args.dffl_rule_swap_ratio,
                dffl_rule_swap_min_keep=args.dffl_rule_swap_min_keep,
                dffl_adaptive_rule_swap=not args.disable_dffl_adaptive_rule_swap,
                dffl_total_rule_budget=args.dffl_total_rule_budget,
                dffl_bridge_score_interaction_weight=args.dffl_bridge_score_interaction_weight,
                dffl_bridge_score_stability_weight=args.dffl_bridge_score_stability_weight,
                dffl_adaptive_budget=not args.disable_dffl_adaptive_budget,
                batch_size=args.batch_size,
                patience=args.patience,
                classification_threshold=args.classification_threshold,
                tune_fuzzy_threshold=args.tune_fuzzy_threshold,
                dffl_one_phase=args.dffl_one_phase,
                device=args.device,
                fuzzy_models=fuzzy_models,
                include_sklearn=not args.skip_sklearn,
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
        input_dim = int(dataset_features.shape[1])
        n_samples = int(dataset_features.shape[0])
        targets_array = np.asarray(dataset_targets).astype(np.float32)
        if targets_array.ndim == 1:
            targets_tensor = torch.from_numpy(targets_array.reshape(-1, 1))
        else:
            targets_tensor = torch.from_numpy(targets_array)
        full_inputs_tensor = torch.from_numpy(np.asarray(dataset_features).astype(np.float32))
        analysis_full_inputs, analysis_full_targets, analysis_full_device = _prepare_structure_analysis_tensors(
            inputs=full_inputs_tensor,
            targets=targets_tensor,
            device=args.device,
        )
        dataset_feature_geometry_effective, dataset_feature_geometry_policy = _resolve_effective_feature_geometry(
            requested_geometry=args.feature_geometry,
            profile=resolved_profile,
            train_inputs=analysis_full_inputs,
            train_targets=analysis_full_targets,
        )
        dffl_groups = make_dffl_input_groups(
            resolved_profile,
            train_inputs=analysis_full_inputs,
            train_targets=analysis_full_targets,
            feature_geometry=dataset_feature_geometry_effective,
            binary_heavy_grouping_mode=args.binary_heavy_grouping,
        )
        dffl_bridge_pairs = make_dffl_bridge_pairs(
            resolved_profile,
            train_inputs=analysis_full_inputs,
            train_targets=analysis_full_targets,
            input_groups=dffl_groups,
            feature_geometry=dataset_feature_geometry_effective,
            adaptive_budget_enabled=not args.disable_dffl_adaptive_budget,
        )
        dffl_intergroup_share = estimate_intergroup_interaction_share(
            train_inputs=analysis_full_inputs,
            train_targets=analysis_full_targets,
            input_groups=dffl_groups,
        )
        dffl_lr_effective = (
            args.dffl_learning_rate * resolved_profile.learning_rate_scale_classification
            if spec.task_type == "binary_classification"
            else args.dffl_learning_rate * resolved_profile.learning_rate_scale_regression
        )
        dataset_protocols[dataset_name] = {
            "task_type": spec.task_type,
            "n_samples": n_samples,
            "input_dim": input_dim,
            "primary_metric": PRIMARY_METRIC[spec.task_type],
            "architectures": {
                "ruanfis_shallow": _summarize_shallow_architecture(input_dim),
                "ruanfis_stacked_anfis": _summarize_stacked_architecture(input_dim),
                "ruanfis_hierarchical_anfis": _summarize_hierarchical_anfis_architecture(input_dim),
            },
            "dffl": {
                "resolved_profile": resolved_profile.name,
                "feature_geometry_requested": str(args.feature_geometry),
                "feature_geometry_effective": dataset_feature_geometry_effective,
                "feature_geometry_policy": dataset_feature_geometry_policy,
                "analysis_device_effective": analysis_full_device,
                "binary_heavy_grouping_mode": str(args.binary_heavy_grouping),
                "one_phase": bool(args.dffl_one_phase),
                "effective_learning_rate": float(dffl_lr_effective),
                "refinement_cycles_effective": int(max(args.refinement_cycles, resolved_profile.refinement_cycle_floor)),
                "bootstrap_hyperparams": _resolve_dffl_bootstrap_hyperparams(
                    task_type=spec.task_type,
                    dffl_one_phase=bool(args.dffl_one_phase),
                    resolved_profile_name=resolved_profile.name,
                ),
                "architecture": _summarize_dffl_architecture(
                    input_dim=input_dim,
                    task_type=spec.task_type,
                    profile=resolved_profile,
                    input_groups=dffl_groups,
                    bridge_pairs=dffl_bridge_pairs,
                    intergroup_interaction_share=dffl_intergroup_share,
                    adaptive_budget_enabled=not args.disable_dffl_adaptive_budget,
                ),
            },
        }

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
    reproducibility_manifest_payload = build_reproducibility_manifest_payload(
        args=args,
        dataset_names=dataset_names,
        seeds=seeds,
        fuzzy_models=fuzzy_models,
        dataset_protocols=dataset_protocols,
    )
    reproducibility_manifest_markdown = build_reproducibility_manifest_markdown(reproducibility_manifest_payload)

    full_report_sections = [
        "UNIFIED REAL-DATASET BENCHMARK",
        f"datasets: {', '.join(dataset_names)}",
        f"seeds: {', '.join(str(seed) for seed in seeds)}",
        f"train_noise_sigma (regression only): {args.train_noise_sigma:.4f}",
        f"dffl_profile: {args.dffl_profile}",
        f"feature_geometry: {args.feature_geometry}",
        f"binary_heavy_grouping: {args.binary_heavy_grouping}",
        f"dffl_rule_swap_ratio_override: {args.dffl_rule_swap_ratio}",
        f"dffl_rule_swap_min_keep_override: {args.dffl_rule_swap_min_keep}",
        f"dffl_adaptive_rule_swap: {not args.disable_dffl_adaptive_rule_swap}",
        f"dffl_adaptive_budget: {not args.disable_dffl_adaptive_budget}",
        f"dffl_total_rule_budget_override: {args.dffl_total_rule_budget}",
        f"dffl_bridge_score_interaction_weight_override: {args.dffl_bridge_score_interaction_weight}",
        f"dffl_bridge_score_stability_weight_override: {args.dffl_bridge_score_stability_weight}",
        f"dffl_one_phase: {args.dffl_one_phase}",
        f"gpu_only: {args.gpu_only}",
        f"fuzzy_models: {', '.join(fuzzy_models)}",
        f"sklearn_baselines: {'off' if args.skip_sklearn else 'on'}",
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

    if args.reproducibility_manifest_output is not None:
        args.reproducibility_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.reproducibility_manifest_output.write_text(reproducibility_manifest_markdown, encoding="utf-8")

    if args.reproducibility_manifest_json_output is not None:
        args.reproducibility_manifest_json_output.parent.mkdir(parents=True, exist_ok=True)
        args.reproducibility_manifest_json_output.write_text(
            json.dumps(reproducibility_manifest_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if args.json_output is not None:
        payload = {
            "datasets": list(dataset_names),
            "seeds": list(seeds),
            "train_noise_sigma": float(args.train_noise_sigma),
            "dffl_profile": args.dffl_profile,
            "feature_geometry": args.feature_geometry,
            "binary_heavy_grouping": args.binary_heavy_grouping,
            "dffl_rule_swap_ratio_override": args.dffl_rule_swap_ratio,
            "dffl_rule_swap_min_keep_override": args.dffl_rule_swap_min_keep,
            "dffl_adaptive_rule_swap": bool(not args.disable_dffl_adaptive_rule_swap),
            "dffl_adaptive_budget": bool(not args.disable_dffl_adaptive_budget),
            "dffl_total_rule_budget_override": args.dffl_total_rule_budget,
            "dffl_bridge_score_interaction_weight_override": args.dffl_bridge_score_interaction_weight,
            "dffl_bridge_score_stability_weight_override": args.dffl_bridge_score_stability_weight,
            "dffl_one_phase": bool(args.dffl_one_phase),
            "gpu_only": bool(args.gpu_only),
            "fuzzy_models": list(fuzzy_models),
            "skip_sklearn": bool(args.skip_sklearn),
            "dataset_results": {
                dataset_name: serialize_multi_seed_benchmark_result(result)
                for dataset_name, result in dataset_results.items()
            },
            "cross_dataset_summary_markdown": summary_table,
            "interpretability_report_markdown": interpretability_report,
            "reproducibility_manifest": reproducibility_manifest_payload,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.output_dir is not None:
        interpretability_path = args.output_dir / "interpretability_report.md"
        interpretability_path.write_text(interpretability_report, encoding="utf-8")
        reproducibility_manifest_path = args.output_dir / "reproducibility_manifest.md"
        reproducibility_manifest_path.write_text(reproducibility_manifest_markdown, encoding="utf-8")
        reproducibility_manifest_json_path = args.output_dir / "reproducibility_manifest.json"
        reproducibility_manifest_json_path.write_text(
            json.dumps(reproducibility_manifest_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
