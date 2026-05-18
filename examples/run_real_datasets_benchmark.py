from __future__ import annotations

import argparse
import csv
import copy
import datetime as dt
import json
import math
import platform
import resource
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.datasets import fetch_kddcup99, load_breast_cancer, load_diabetes, load_digits, load_linnerud, load_wine
from sklearn.datasets import fetch_california_housing, fetch_covtype
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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
    DeepKANFISModel,
    KANFISModel,
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
    build_stagewise_initialized_stacked_anfis_model,
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
from ruanfis.trainer import FuzzyTrainer, predict_with_optional_residual_head


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
    bridge_validation_gain_weight: float = 0.35
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
    residual_enabled: bool = False
    residual_top_features: int = 0
    residual_concepts: int = 1
    residual_max_rules: int = 2
    residual_max_per_group: int = 1
    stagewise_rule_swap_ratio: float = 0.0
    stagewise_rule_swap_min_keep: int = 0
    stage1_fixed_rule_budget: bool = True
    stage1_min_rules_per_block: int = 6
    total_rule_budget: int = 0
    block_agreement_weight: float = 0.0
    block_agreement_target_corr: float = 0.2
    block_agreement_stage_limit: int = 1
    rule_activation_entropy_weight: float = 0.0
    rule_anchor_stability_weight: float = 0.0
    concept_dropout_rate: float = 0.0
    final_skip_input_count: int = 0
    final_skip_input_indices: tuple[int, ...] = ()
    final_skip_mode: str = "target_corr_diverse"
    final_skip_gates_enabled: bool = False
    final_skip_gate_init_logit: float = 2.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def progress_log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[progress {timestamp}] {message}", flush=True)


def _read_current_rss_mb() -> float | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                chunks = line.split()
                if len(chunks) >= 2:
                    return float(chunks[1]) / 1024.0
    except Exception:
        return None
    return None


def _read_peak_rss_mb() -> float | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux ru_maxrss is in KiB.
        return float(usage.ru_maxrss) / 1024.0
    except Exception:
        return None


def _is_covtype_binary_dataset(dataset_name: str) -> bool:
    return str(dataset_name).strip().lower().startswith("covtype_binary")


def detect_git_revision() -> str | None:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return revision or None
    except Exception:
        return None


def collect_runtime_environment(*, requested_device: str | None) -> dict[str, object]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_names: list[str] = []
    if cuda_available:
        for idx in range(int(torch.cuda.device_count())):
            try:
                gpu_names.append(str(torch.cuda.get_device_name(idx)))
            except Exception:
                gpu_names.append(f"cuda:{idx}")
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_compiled_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "cuda_devices": gpu_names,
        "requested_device": requested_device if requested_device is not None else "default",
        "git_revision": detect_git_revision(),
        "command": " ".join(sys.argv),
    }


def parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


DEFAULT_SEED_POOL: tuple[int, ...] = (
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
)


def resolve_seeds(raw: str, seed_count: int | None) -> tuple[int, ...]:
    explicit = parse_seeds(raw)
    if seed_count is None or int(seed_count) <= 0:
        return explicit
    n = int(seed_count)
    if n <= len(DEFAULT_SEED_POOL):
        return DEFAULT_SEED_POOL[:n]
    # Extend deterministically by stepping odd numbers and selecting probable primes.
    extra: list[int] = list(DEFAULT_SEED_POOL)
    candidate = extra[-1] + 2
    while len(extra) < n:
        is_prime = True
        divisor = 3
        while divisor * divisor <= candidate:
            if candidate % divisor == 0:
                is_prime = False
                break
            divisor += 2
        if is_prime:
            extra.append(candidate)
        candidate += 2
    return tuple(extra[:n])


def parse_dataset_names(raw: str) -> tuple[str, ...]:
    names = tuple(chunk.strip() for chunk in raw.split(",") if chunk.strip())
    if not names:
        raise ValueError("At least one dataset name must be provided.")
    return names


def parse_dataset_suite(raw: str | None) -> str:
    source = str(raw or "").strip().lower()
    if not source:
        return "custom"
    return source


def parse_fuzzy_model_names(raw: str) -> tuple[str, ...]:
    chunks = tuple(chunk.strip() for chunk in raw.split(",") if chunk.strip())
    if not chunks:
        raise ValueError("At least one fuzzy model must be provided.")
    if len(chunks) == 1 and chunks[0].lower() in {"none", "off"}:
        return tuple()
    if len(chunks) == 1 and chunks[0].lower() == "all":
        return FUZZY_MODEL_NAMES

    aliases = {
        "shallow": "ruanfis_shallow",
        "stacked": "ruanfis_stacked_anfis",
        "hierarchical": "ruanfis_hierarchical_anfis",
        "hier": "ruanfis_hierarchical_anfis",
        "kanfis": "ruanfis_kanfis",
        "ka_anfis": "ruanfis_kanfis",
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


def parse_optional_fuzzy_model_names(raw: str) -> tuple[str, ...]:
    source = str(raw).strip()
    if not source or source.lower() in {"none", "off"}:
        return tuple()
    return parse_fuzzy_model_names(source)


def parse_dffl_dataset_overrides(raw: str | None) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    source = str(raw).strip()
    if not source:
        return {}

    payload_text = source
    maybe_path = Path(source)
    if maybe_path.exists():
        payload_text = maybe_path.read_text(encoding="utf-8")

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--dffl-dataset-overrides must be a JSON object (inline JSON or path to JSON file)."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("--dffl-dataset-overrides must decode to a JSON object keyed by dataset name.")

    normalized: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Dataset override keys must be non-empty strings.")
        if not isinstance(value, dict):
            raise ValueError(f"Dataset override for {key!r} must be a JSON object.")
        normalized[key.strip().lower()] = dict(value)
    return normalized


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


def _load_covtype_binary_subset(train_size: int | None) -> tuple[np.ndarray, np.ndarray]:
    features, target = fetch_covtype(return_X_y=True)
    target_binary = (target == 2).astype(np.float32)
    if train_size is None or int(train_size) >= int(features.shape[0]):
        return features.astype(np.float32), target_binary.astype(np.float32)
    features_subset, _, target_subset, _ = train_test_split(
        features,
        target_binary,
        train_size=int(train_size),
        random_state=42,
        stratify=target_binary,
    )
    return features_subset.astype(np.float32), target_subset.astype(np.float32)


def _load_covtype_binary_8000() -> tuple[np.ndarray, np.ndarray]:
    return _load_covtype_binary_subset(8_000)


def _load_covtype_binary_20000() -> tuple[np.ndarray, np.ndarray]:
    return _load_covtype_binary_subset(20_000)


def _load_covtype_binary_50000() -> tuple[np.ndarray, np.ndarray]:
    return _load_covtype_binary_subset(50_000)


def _load_covtype_binary_100000() -> tuple[np.ndarray, np.ndarray]:
    return _load_covtype_binary_subset(100_000)


def _load_covtype_binary_200000() -> tuple[np.ndarray, np.ndarray]:
    return _load_covtype_binary_subset(200_000)


def _load_covtype_binary_full() -> tuple[np.ndarray, np.ndarray]:
    return _load_covtype_binary_subset(None)


def _covtype_feature_names() -> list[str]:
    names = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    names.extend([f"Wilderness_Area_{index}" for index in range(1, 5)])
    names.extend([f"Soil_Type_{index}" for index in range(1, 41)])
    return names


def _dataset_feature_names(dataset_name: str, input_dim: int) -> list[str]:
    try:
        if dataset_name == "breast_cancer":
            names = list(load_breast_cancer().feature_names)
        elif dataset_name == "diabetes":
            names = list(load_diabetes().feature_names)
        elif dataset_name == "wine_binary":
            names = list(load_wine().feature_names)
        elif dataset_name == "digits_binary":
            data = load_digits()
            names = list(getattr(data, "feature_names", [])) or [f"pixel_{index}" for index in range(input_dim)]
        elif dataset_name == "california_housing":
            names = list(fetch_california_housing().feature_names)
        elif _is_covtype_binary_dataset(dataset_name):
            names = _covtype_feature_names()
        elif dataset_name == "linnerud_weight":
            names = list(load_linnerud().feature_names)
        else:
            names = []
    except Exception:
        names = []
    if len(names) < input_dim:
        names.extend([f"feature_{index}" for index in range(len(names), input_dim)])
    return names[:input_dim]


def _ensure_susy_csv_gz() -> Path:
    cache_dir = PROJECT_ROOT / "data" / "external" / "susy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_gz_path = cache_dir / "SUSY.csv.gz"
    if csv_gz_path.exists():
        return csv_gz_path

    zip_path = cache_dir / "susy.zip"
    if zip_path.exists() and not zipfile.is_zipfile(zip_path):
        progress_log(f"removing incomplete SUSY archive: {zip_path}")
        zip_path.unlink()
    if not zip_path.exists():
        url = "https://archive.ics.uci.edu/static/public/279/susy.zip"
        progress_log(f"downloading SUSY dataset from {url}")
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        member_names = archive.namelist()
        csv_members = [name for name in member_names if name.endswith("SUSY.csv.gz")]
        if not csv_members:
            raise RuntimeError(f"SUSY.csv.gz not found in {zip_path}; members={member_names[:10]}")
        archive.extract(csv_members[0], cache_dir)
        extracted_path = cache_dir / csv_members[0]
        if extracted_path != csv_gz_path:
            extracted_path.replace(csv_gz_path)
    return csv_gz_path


def _load_susy_binary_subset(sample_size: int | None) -> tuple[np.ndarray, np.ndarray]:
    csv_gz_path = _ensure_susy_csv_gz()
    max_rows = None if sample_size is None else int(sample_size)
    data = np.loadtxt(csv_gz_path, delimiter=",", dtype=np.float32, max_rows=max_rows)
    target = data[:, 0].astype(np.float32)
    features = data[:, 1:].astype(np.float32)
    return features, target


def _load_susy_binary_200000() -> tuple[np.ndarray, np.ndarray]:
    return _load_susy_binary_subset(200_000)


def _load_susy_binary_1000000() -> tuple[np.ndarray, np.ndarray]:
    return _load_susy_binary_subset(1_000_000)


def _load_susy_binary_full() -> tuple[np.ndarray, np.ndarray]:
    return _load_susy_binary_subset(None)


def _load_kddcup99_binary(percent10: bool) -> tuple[np.ndarray, np.ndarray]:
    features_raw, target_raw = fetch_kddcup99(percent10=percent10, return_X_y=True, as_frame=False)
    categorical = features_raw[:, [1, 2, 3]]
    numeric = np.delete(features_raw, [1, 2, 3], axis=1).astype(np.float32)
    encoder = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown="ignore")
    categorical_encoded = encoder.fit_transform(categorical)
    features = np.concatenate([numeric, categorical_encoded], axis=1).astype(np.float32)
    target = (target_raw != b"normal.").astype(np.float32)
    return features, target


def _load_kddcup99_binary_10percent() -> tuple[np.ndarray, np.ndarray]:
    return _load_kddcup99_binary(percent10=True)


def _load_kddcup99_binary_full() -> tuple[np.ndarray, np.ndarray]:
    return _load_kddcup99_binary(percent10=False)


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
    "covtype_binary_50000": DatasetSpec(
        name="covtype_binary_50000",
        task_type="binary_classification",
        loader=_load_covtype_binary_50000,
    ),
    "covtype_binary_100000": DatasetSpec(
        name="covtype_binary_100000",
        task_type="binary_classification",
        loader=_load_covtype_binary_100000,
    ),
    "covtype_binary_200000": DatasetSpec(
        name="covtype_binary_200000",
        task_type="binary_classification",
        loader=_load_covtype_binary_200000,
    ),
    "covtype_binary_full": DatasetSpec(
        name="covtype_binary_full",
        task_type="binary_classification",
        loader=_load_covtype_binary_full,
    ),
    "covtype_binary_8000": DatasetSpec(
        name="covtype_binary_8000",
        task_type="binary_classification",
        loader=_load_covtype_binary_8000,
    ),
    "susy_binary_200000": DatasetSpec(
        name="susy_binary_200000",
        task_type="binary_classification",
        loader=_load_susy_binary_200000,
    ),
    "susy_binary_1000000": DatasetSpec(
        name="susy_binary_1000000",
        task_type="binary_classification",
        loader=_load_susy_binary_1000000,
    ),
    "susy_binary_full": DatasetSpec(
        name="susy_binary_full",
        task_type="binary_classification",
        loader=_load_susy_binary_full,
    ),
    "kddcup99_binary_10percent": DatasetSpec(
        name="kddcup99_binary_10percent",
        task_type="binary_classification",
        loader=_load_kddcup99_binary_10percent,
    ),
    "kddcup99_binary_full": DatasetSpec(
        name="kddcup99_binary_full",
        task_type="binary_classification",
        loader=_load_kddcup99_binary_full,
    ),
}

DATASET_SUITES: dict[str, tuple[str, ...]] = {
    # Historical compact benchmark (main block in paper).
    "paper_main": (
        "diabetes",
        "linnerud_weight",
        "breast_cancer",
        "wine_binary",
        "digits_binary",
    ),
    # Historical extended benchmark (paper extended block).
    "paper_extended": (
        "california_housing",
        "covtype_binary_20000",
    ),
    # Full paper benchmark blocks together.
    "paper_all": (
        "diabetes",
        "linnerud_weight",
        "breast_cancer",
        "wine_binary",
        "digits_binary",
        "california_housing",
        "covtype_binary_20000",
    ),
    # Larger-scale tabular suite for stronger post-review evidence.
    "q1_large": (
        "covtype_binary_full",
        "susy_binary_200000",
        "kddcup99_binary_10percent",
    ),
    # Most expensive stress suite (use GPU-friendly options).
    "q1_full": (
        "covtype_binary_full",
        "susy_binary_1000000",
        "kddcup99_binary_full",
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
    "ruanfis_kanfis",
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
        # Tiny regression tasks are highly sensitive to over-parameterized DFFL variants.
        learning_rate_scale_regression=1.15,
        learning_rate_scale_classification=1.0,
        refinement_cycle_floor=2,
        local_prototype_term_limit=3,
        local_consequent_mode="affine_sigmoid",
        aggregate_consequent_mode="affine_sigmoid",
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
        stagewise_rule_swap_ratio=0.10,
        stagewise_rule_swap_min_keep=12,
        block_agreement_weight=1.5e-3,
        block_agreement_target_corr=0.22,
        block_agreement_stage_limit=1,
        rule_activation_entropy_weight=2e-4,
        rule_anchor_stability_weight=2e-3,
        concept_dropout_rate=0.08,
        final_skip_input_count=16,
        final_skip_gates_enabled=True,
        final_skip_gate_init_logit=2.0,
    ),
    "quality_large_cls_v2": DfflProfile(
        name="quality_large_cls_v2",
        local_concepts=2,
        local_max_rules=10,
        aggregate_max_rules=56,
        decision_max_rules=24,
        stage2_width_min=8,
        stage2_width_max=20,
        aggregate_block_count=4,
        aggregate_overlap=2,
        aggregate_global_context_dim=12,
        aggregate_global_max_rules=32,
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
        top_k_rules=None,
        input_group_size=4,
        input_group_strategy="target_corr",
        input_group_correlation_weight=0.8,
        local_prototype_term_limit=3,
        local_prototype_scoring_mode="max",
        local_prototype_variable_pool_size=4,
        aggregate_prototype_term_limit=3,
        aggregate_prototype_scoring_mode="hybrid",
        aggregate_prototype_variable_pool_size=24,
        decision_prototype_term_limit=3,
        decision_prototype_scoring_mode="hybrid",
        decision_prototype_variable_pool_size=16,
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
        bridge_top_pairs=40,
        bridge_pair_score_alpha=0.7,
        bridge_concepts=3,
        bridge_concepts_max=5,
        bridge_max_rules=16,
        bridge_prototype_term_limit=2,
        bridge_prototype_scoring_mode="hybrid",
        bridge_prototype_sample_size=1024,
        bridge_max_pairs_per_feature=3,
        bridge_max_pairs_per_group_pair=2,
        stagewise_rule_swap_ratio=0.2,
        stagewise_rule_swap_min_keep=8,
        block_agreement_weight=1.5e-3,
        block_agreement_target_corr=0.22,
        block_agreement_stage_limit=1,
        final_skip_input_count=32,
        final_skip_mode="target_corr_diverse",
        final_skip_gates_enabled=True,
        final_skip_gate_init_logit=2.0,
    ),
    "stability_large_cls": DfflProfile(
        name="stability_large_cls",
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
        learning_rate_scale_regression=0.95,
        learning_rate_scale_classification=0.95,
        refinement_cycle_floor=2,
        pretrain_refinement_rounds=2,
        top_k_rules=20,
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
        binary_soft_f1_weight=0.2,
        weight_decay=2e-5,
        gradient_clip_norm=4.0,
        rule_sparsity_weight=5e-5,
        rule_length_weight=2e-5,
        decision_usage_balance_weight=4e-3,
        block_gates_enabled=True,
        block_gate_init_logit=4.0,
        block_gate_l1_weight=8e-5,
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
        stagewise_rule_swap_ratio=0.0,
        stagewise_rule_swap_min_keep=14,
        block_agreement_weight=2e-3,
        block_agreement_target_corr=0.25,
        block_agreement_stage_limit=1,
        rule_activation_entropy_weight=4e-4,
        rule_anchor_stability_weight=4e-3,
        concept_dropout_rate=0.10,
        final_skip_input_count=20,
        final_skip_mode="target_corr_diverse",
        final_skip_gates_enabled=True,
        final_skip_gate_init_logit=2.0,
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
        # Very large high-dimensional binary tasks can benefit from less bottlenecked v2 profile.
        # For mid-size regimes (e.g., covtype_binary_20000), v2 is often too heavy and less stable.
        if (
            task_type == "binary_classification"
            and input_dim is not None
            and input_dim >= 40
            and n_samples >= 50_000
        ):
            return DFFL_PROFILES["quality_large_cls_v2"]
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


def _get_dataset_override_entry(
    *,
    dataset_overrides: Mapping[str, Mapping[str, Any]] | None,
    dataset_name: str,
) -> dict[str, Any]:
    if not dataset_overrides:
        return {}
    normalized_name = str(dataset_name).strip().lower()
    if normalized_name in dataset_overrides:
        return dict(dataset_overrides[normalized_name])
    if "*" in dataset_overrides:
        return dict(dataset_overrides["*"])
    return {}


def _resolve_dataset_profile_override(
    profile: DfflProfile,
    *,
    dataset_name: str,
    dataset_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[DfflProfile, str]:
    entry = _get_dataset_override_entry(dataset_overrides=dataset_overrides, dataset_name=dataset_name)
    if not entry:
        return profile, "none"
    profile_name_raw = entry.get("profile")
    if profile_name_raw is None:
        return profile, "none"
    profile_name = str(profile_name_raw).strip()
    if profile_name not in DFFL_PROFILES:
        raise ValueError(
            f"Dataset override for {dataset_name!r} has unknown profile={profile_name!r}. "
            f"Allowed: {', '.join(sorted(DFFL_PROFILES.keys()))}"
        )
    return DFFL_PROFILES[profile_name], f"profile:{profile_name}"


def _dataset_override_skip_builtin_budget_policy(
    *,
    dataset_name: str,
    dataset_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> bool:
    entry = _get_dataset_override_entry(dataset_overrides=dataset_overrides, dataset_name=dataset_name)
    if not entry:
        return False
    return bool(entry.get("skip_builtin_budget_policy", False))


def _apply_dataset_profile_field_overrides(
    profile: DfflProfile,
    *,
    dataset_name: str,
    dataset_overrides: Mapping[str, Mapping[str, Any]] | None,
    total_budget_locked: bool,
) -> tuple[DfflProfile, str]:
    entry = _get_dataset_override_entry(dataset_overrides=dataset_overrides, dataset_name=dataset_name)
    if not entry:
        return profile, "none"

    allowed_field_names = {field.name for field in fields(DfflProfile)}
    reserved_keys = {
        "profile",
        "skip_builtin_budget_policy",
        "restarts",
        "tiny_restarts",
        "threshold_tuning_strategy",
        "binary_loss_name",
        "binary_focal_gamma",
        "binary_focal_alpha",
        "binary_class_balanced_beta",
    }
    updates: dict[str, Any] = {}
    for key, value in entry.items():
        if key in reserved_keys:
            continue
        if key not in allowed_field_names:
            raise ValueError(
                f"Dataset override for {dataset_name!r} contains unknown DfflProfile field: {key!r}."
            )
        if key == "total_rule_budget" and total_budget_locked:
            continue
        updates[key] = value

    if not updates:
        return profile, "none"
    return replace(profile, **updates), "field_overrides"


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


def _cap_optional_int(value: int | None, cap: int) -> int:
    if value is None:
        return int(cap)
    return int(min(int(value), int(cap)))


def apply_dffl_fast_gpu_profile(
    profile: DfflProfile,
    *,
    task_type: TaskType,
    input_dim: int,
    n_samples: int,
    total_budget_locked: bool,
) -> tuple[DfflProfile, str]:
    # Speed-oriented profile for GPU runs: reduce CPU-heavy prototype/rule generation
    # while preserving core DFFL topology.
    if input_dim < 30:
        return profile, "none"

    updates: dict[str, object] = {}
    if task_type == "binary_classification":
        updates["local_max_rules"] = min(profile.local_max_rules, 9)
        updates["aggregate_max_rules"] = min(profile.aggregate_max_rules, 20)
        updates["decision_max_rules"] = min(profile.decision_max_rules, 12)
        updates["bridge_top_pairs"] = min(profile.bridge_top_pairs, 8)
        updates["bridge_max_rules"] = min(profile.bridge_max_rules, 6)
        updates["aggregate_block_count"] = min(profile.aggregate_block_count, 1)
        updates["aggregate_overlap"] = 0
        updates["pretrain_refinement_rounds"] = min(profile.pretrain_refinement_rounds, 1)
        updates["refinement_cycle_floor"] = min(profile.refinement_cycle_floor, 1)
        updates["stagewise_rule_swap_ratio"] = min(profile.stagewise_rule_swap_ratio, 0.10)
        updates["stagewise_rule_swap_min_keep"] = min(profile.stagewise_rule_swap_min_keep, 4)
        if profile.top_k_rules is not None:
            updates["top_k_rules"] = min(int(profile.top_k_rules), 16)
        if not total_budget_locked:
            target_budget = 120 if n_samples >= 8_000 else 96
            current_budget = int(profile.total_rule_budget)
            updates["total_rule_budget"] = target_budget if current_budget <= 0 else min(current_budget, target_budget)
    else:
        updates["pretrain_refinement_rounds"] = min(profile.pretrain_refinement_rounds, 1)
        updates["refinement_cycle_floor"] = min(profile.refinement_cycle_floor, 1)
        if not total_budget_locked and input_dim >= 40:
            current_budget = int(profile.total_rule_budget)
            updates["total_rule_budget"] = 96 if current_budget <= 0 else min(current_budget, 96)

    updates["local_prototype_sample_size"] = _cap_optional_int(profile.local_prototype_sample_size, 256)
    updates["aggregate_prototype_sample_size"] = _cap_optional_int(profile.aggregate_prototype_sample_size, 256)
    updates["decision_prototype_sample_size"] = _cap_optional_int(profile.decision_prototype_sample_size, 256)
    updates["bridge_prototype_sample_size"] = _cap_optional_int(profile.bridge_prototype_sample_size, 256)
    updates["local_prototype_variable_pool_size"] = _cap_optional_int(profile.local_prototype_variable_pool_size, 3)
    updates["aggregate_prototype_variable_pool_size"] = _cap_optional_int(
        profile.aggregate_prototype_variable_pool_size,
        8,
    )
    updates["decision_prototype_variable_pool_size"] = _cap_optional_int(profile.decision_prototype_variable_pool_size, 6)

    fast_profile = replace(profile, **updates)
    if not str(fast_profile.name).endswith("_fastgpu"):
        fast_profile = replace(fast_profile, name=f"{fast_profile.name}_fastgpu")
    return fast_profile, "fast_gpu_mode"


def apply_dataset_budget_reallocation(
    profile: DfflProfile,
    *,
    dataset_name: str,
    task_type: TaskType,
    input_dim: int,
    total_budget_locked: bool,
) -> tuple[DfflProfile, str]:
    updates: dict[str, object] = {}
    policy = "none"
    name = str(dataset_name).strip().lower()

    if task_type != "binary_classification":
        return profile, policy

    if name == "digits_binary" and input_dim >= 40:
        # Keep high-dim rule growth under control.
        if profile.bridge_top_pairs > 0:
            updates["bridge_top_pairs"] = min(profile.bridge_top_pairs, 6)
        updates["decision_max_rules"] = min(profile.decision_max_rules, 10)
        updates["aggregate_max_rules"] = min(profile.aggregate_max_rules, 16)
        if not total_budget_locked:
            base_budget = int(profile.total_rule_budget) if int(profile.total_rule_budget) > 0 else 150
            updates["total_rule_budget"] = min(base_budget, 150)
        policy = "digits_compact_budget"

    elif name == "breast_cancer" and input_dim <= 40:
        # Breast-cancer-like medium tabular classification benefits from
        # a mildly higher LR and a fixed small stage-wise swap ratio.
        updates["learning_rate_scale_classification"] = max(
            float(profile.learning_rate_scale_classification), 1.1333333333333333
        )
        updates["stagewise_rule_swap_ratio"] = 0.10
        updates["stagewise_rule_swap_min_keep"] = 6
        policy = "breast_lr_swap_tuned"

    elif name == "wine_binary" and input_dim <= 20:
        # Small low-dim classification needs only mild extra capacity.
        updates["bridge_top_pairs"] = max(profile.bridge_top_pairs, 4)
        updates["bridge_max_rules"] = max(profile.bridge_max_rules, 6)
        updates["decision_max_rules"] = max(profile.decision_max_rules, 17)
        # Wine-like tiny classification benefits from a slightly stronger optimization step.
        updates["learning_rate_scale_classification"] = max(
            float(profile.learning_rate_scale_classification), 1.3333333333333333
        )
        updates["stagewise_rule_swap_ratio"] = 0.0
        updates["stagewise_rule_swap_min_keep"] = 0
        updates["bridge_validation_gain_weight"] = 0.0
        if not total_budget_locked:
            updates["total_rule_budget"] = max(int(profile.total_rule_budget), 64)
        policy = "wine_boost_bridge_decision"

    if not updates:
        return profile, "none"
    return replace(profile, **updates), policy


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


def _reorder_input_groups_by_interaction(
    input_groups: tuple[tuple[int, ...], ...],
    *,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
) -> tuple[tuple[int, ...], ...]:
    n_groups = len(input_groups)
    if n_groups <= 2:
        return input_groups
    input_dim = int(train_inputs.shape[1])
    # Conservative guardrail: on high-dimensional settings this reordering can hurt
    # already strong target-corr groupings (e.g., digits-like tasks).
    if input_dim >= 40 or n_groups > 10:
        return input_groups
    intergroup_share = estimate_intergroup_interaction_share(
        train_inputs=train_inputs,
        train_targets=train_targets,
        input_groups=input_groups,
    )
    if intergroup_share >= 0.85:
        return input_groups

    relevance = _feature_target_relevance(train_inputs, train_targets)
    pair_interaction = _pair_interaction_relevance(train_inputs, train_targets)
    group_relevance = np.zeros(n_groups, dtype=np.float64)
    interaction_graph = np.zeros((n_groups, n_groups), dtype=np.float64)

    for i, group_i in enumerate(input_groups):
        idx_i = np.array(group_i, dtype=np.int64)
        if idx_i.size > 0:
            group_relevance[i] = float(np.mean(relevance[idx_i]))
        for j in range(i + 1, n_groups):
            idx_j = np.array(input_groups[j], dtype=np.int64)
            if idx_i.size == 0 or idx_j.size == 0:
                score = 0.0
            else:
                score = float(np.mean(pair_interaction[np.ix_(idx_i, idx_j)]))
            interaction_graph[i, j] = score
            interaction_graph[j, i] = score

    start = int(np.argmax(group_relevance))
    ordered: list[int] = [start]
    remaining = set(range(n_groups))
    remaining.remove(start)

    while remaining:
        prev = ordered[-1]
        next_group = max(
            remaining,
            key=lambda g: (
                float(interaction_graph[prev, g]),
                float(group_relevance[g]),
                -int(g),
            ),
        )
        ordered.append(int(next_group))
        remaining.remove(int(next_group))

    return tuple(input_groups[group_idx] for group_idx in ordered)


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
        base_groups = _make_hyperbolic_graph_groups(
            train_inputs,
            train_targets,
            group_size=profile.input_group_size,
        )
        return _reorder_input_groups_by_interaction(
            base_groups,
            train_inputs=train_inputs,
            train_targets=train_targets,
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
            base_groups = _make_binary_aware_target_corr_groups(
                train_inputs,
                train_targets,
                group_size=profile.input_group_size,
                correlation_weight=profile.input_group_correlation_weight,
            )
        else:
            base_groups = _make_target_corr_groups(
                train_inputs,
                train_targets,
                group_size=profile.input_group_size,
                correlation_weight=profile.input_group_correlation_weight,
            )
        return _reorder_input_groups_by_interaction(
            base_groups,
            train_inputs=train_inputs,
            train_targets=train_targets,
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


def _select_adaptive_high_arity_groups(
    *,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    input_groups: tuple[tuple[int, ...], ...],
    input_dim: int,
) -> tuple[int, ...]:
    n_groups = len(input_groups)
    if n_groups <= 0:
        return ()
    eligible_indices = [idx for idx, group in enumerate(input_groups) if len(group) >= 3]
    if not eligible_indices:
        return ()

    if input_dim >= 40:
        top_k = min(3, max(1, int(np.ceil(0.20 * n_groups))))
    elif input_dim >= 20:
        top_k = min(2, max(1, int(np.ceil(0.25 * n_groups))))
    else:
        top_k = 1 if n_groups >= 3 else 0
    if top_k <= 0:
        return ()

    relevance = _feature_target_relevance(train_inputs, train_targets)
    pair_interaction = _pair_interaction_relevance(train_inputs, train_targets)
    scored: list[tuple[float, int]] = []
    for group_index in eligible_indices:
        indices = np.array(input_groups[group_index], dtype=np.int64)
        group_relevance = float(np.mean(relevance[indices])) if indices.size > 0 else 0.0
        if indices.size <= 1:
            interaction = 0.0
        else:
            block = pair_interaction[np.ix_(indices, indices)]
            upper = np.triu(np.ones_like(block, dtype=bool), k=1)
            interaction = float(np.mean(block[upper])) if int(np.count_nonzero(upper)) > 0 else 0.0
        score = 0.55 * group_relevance + 0.45 * interaction
        scored.append((score, int(group_index)))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [group_index for _, group_index in scored[:top_k]]
    selected.sort()
    return tuple(selected)


def _resolve_adaptive_local_arity_policy(
    *,
    enabled: bool,
    task_type: TaskType,
    n_samples: int,
    input_dim: int,
) -> tuple[bool, str]:
    if not enabled:
        return False, "cli_disabled"
    if task_type != "binary_classification":
        return False, "non_classification_guard"
    # Keep this policy only for tiny low-dimensional classification where it helped (wine-like settings).
    if input_dim <= 20 and n_samples <= 300:
        return True, "tiny_cls_enabled"
    return False, "non_tiny_guard"


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


def _estimate_dffl_component_marginal_gains(
    *,
    task_type: TaskType,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    validation_inputs: torch.Tensor | None,
    validation_targets: torch.Tensor | None,
    input_groups: tuple[tuple[int, ...], ...],
    bridge_pairs: tuple[tuple[int, int], ...],
) -> dict[str, float]:
    input_dim = int(train_inputs.shape[1])
    n_train = int(train_inputs.shape[0])
    if input_dim <= 1:
        return {"local": 1.0, "bridge": 1.0, "aggregate": 1.0, "decision": 1.0}
    if task_type == "binary_classification":
        # Avoid noisy reallocation on tiny/compact classification datasets.
        if n_train < 700 and input_dim < 40:
            return {"local": 1.0, "bridge": 1.0, "aggregate": 1.0, "decision": 1.0}
    elif n_train < 140 and input_dim < 24:
        # Small-sample settings are too noisy for reliable marginal-gain estimation.
        return {"local": 1.0, "bridge": 1.0, "aggregate": 1.0, "decision": 1.0}

    if (
        validation_inputs is None
        or validation_targets is None
        or int(validation_inputs.shape[0]) < 8
        or int(validation_inputs.shape[1]) != input_dim
    ):
        validation_inputs = train_inputs
        validation_targets = train_targets

    train_pair = _pair_interaction_relevance(train_inputs, train_targets).astype(np.float64, copy=False)
    val_pair = _pair_interaction_relevance(validation_inputs, validation_targets).astype(np.float64, copy=False)
    feature_to_group = _build_feature_to_group_index(input_dim, input_groups)
    upper = np.triu(np.ones((input_dim, input_dim), dtype=bool), k=1)
    left_groups = feature_to_group[:, None]
    right_groups = feature_to_group[None, :]
    within_mask = upper & (left_groups >= 0) & (right_groups >= 0) & (left_groups == right_groups)
    inter_mask = upper & (left_groups >= 0) & (right_groups >= 0) & (left_groups != right_groups)
    bridge_mask = np.zeros((input_dim, input_dim), dtype=bool)
    for left, right in bridge_pairs:
        li = int(left)
        ri = int(right)
        if 0 <= li < input_dim and 0 <= ri < input_dim and li != ri:
            lo = min(li, ri)
            hi = max(li, ri)
            bridge_mask[lo, hi] = True
    bridge_mask = bridge_mask & upper
    inter_non_bridge_mask = inter_mask & (~bridge_mask)

    def _mean_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
        if int(np.count_nonzero(mask)) == 0:
            return 0.0
        return float(np.mean(values[mask]))

    local_train = _mean_or_zero(train_pair, within_mask)
    local_val = _mean_or_zero(val_pair, within_mask)
    bridge_train = _mean_or_zero(train_pair, bridge_mask) if len(bridge_pairs) > 0 else 0.0
    bridge_val = _mean_or_zero(val_pair, bridge_mask) if len(bridge_pairs) > 0 else 0.0
    aggregate_train = _mean_or_zero(train_pair, inter_non_bridge_mask)
    aggregate_val = _mean_or_zero(val_pair, inter_non_bridge_mask)
    if int(np.count_nonzero(inter_non_bridge_mask)) == 0:
        aggregate_train = _mean_or_zero(train_pair, inter_mask)
        aggregate_val = _mean_or_zero(val_pair, inter_mask)

    def _mass_or_zero(values: np.ndarray, mask: np.ndarray) -> float:
        if int(np.count_nonzero(mask)) == 0:
            return 0.0
        return float(np.sum(values[mask]))

    within_val_mass = _mass_or_zero(val_pair, within_mask)
    inter_val_mass = _mass_or_zero(val_pair, inter_mask)
    bridge_val_mass = _mass_or_zero(val_pair, bridge_mask) if len(bridge_pairs) > 0 else 0.0
    total_val_mass = within_val_mass + inter_val_mass
    inter_structural_share = float(np.clip(inter_val_mass / (total_val_mass + 1e-12), 0.0, 1.0))
    bridge_coverage = float(np.clip(bridge_val_mass / (inter_val_mass + 1e-12), 0.0, 1.0))

    if task_type == "binary_classification":
        y_train = train_targets.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
        y_val = validation_targets.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
        p_train = float(np.clip(np.mean(y_train), 1e-6, 1.0 - 1e-6))
        p_val = float(np.clip(np.mean(y_val), 1e-6, 1.0 - 1e-6))
        decision_train = float(-(p_train * np.log2(p_train) + (1.0 - p_train) * np.log2(1.0 - p_train)))
        decision_val = float(-(p_val * np.log2(p_val) + (1.0 - p_val) * np.log2(1.0 - p_val)))
    else:
        y_train = train_targets.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
        y_val = validation_targets.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
        std_train = float(np.std(y_train))
        std_val = float(np.std(y_val))
        ratio = std_val / (std_train + 1e-9)
        decision_train = 0.5
        decision_val = float(np.clip(ratio, 0.0, 2.0) / 2.0)

    def _gain(train_strength: float, val_strength: float) -> float:
        t = float(np.clip(train_strength, 0.0, 1.0))
        v = float(np.clip(val_strength, 0.0, 1.0))
        stability = float(np.clip(1.0 - abs(v - t), 0.0, 1.0))
        return float(max(0.05, 0.70 * v + 0.30 * stability))

    if task_type == "binary_classification":
        local_boost = 0.85 + 0.45 * (1.0 - inter_structural_share)
        bridge_boost = 0.70 + 0.90 * inter_structural_share + 0.60 * bridge_coverage
        aggregate_need = max(0.0, inter_structural_share - bridge_coverage)
        aggregate_boost = 0.85 + 0.95 * aggregate_need
        decision_boost = 0.85 + 0.35 * float(np.clip(decision_val, 0.0, 1.0))
        raw = {
            "local": _gain(local_train, local_val) * local_boost,
            "bridge": (_gain(bridge_train, bridge_val) if len(bridge_pairs) > 0 else 0.20)
            * (bridge_boost if len(bridge_pairs) > 0 else 1.0),
            "aggregate": _gain(aggregate_train, aggregate_val) * aggregate_boost,
            "decision": _gain(decision_train, decision_val) * decision_boost,
        }
    else:
        raw = {
            "local": _gain(local_train, local_val),
            "bridge": _gain(bridge_train, bridge_val) if len(bridge_pairs) > 0 else 0.25,
            "aggregate": _gain(aggregate_train, aggregate_val),
            "decision": _gain(decision_train, decision_val),
        }
    mean_gain = float(np.mean(list(raw.values()))) if raw else 1.0
    if mean_gain <= 1e-9:
        return {"local": 1.0, "bridge": 1.0, "aggregate": 1.0, "decision": 1.0}
    if task_type == "binary_classification":
        if input_dim >= 40:
            lo_clip = 0.70
            hi_clip = 1.40
        else:
            lo_clip = 0.35
            hi_clip = 2.60
    else:
        lo_clip = 0.25
        hi_clip = 3.00
    normalized = {
        name: float(np.clip(value / mean_gain, lo_clip, hi_clip))
        for name, value in raw.items()
    }
    return normalized


def _resolve_dffl_effective_rule_budgets(
    *,
    profile: DfflProfile,
    input_dim: int,
    n_local_groups: int,
    n_bridge_pairs: int,
    intergroup_interaction_share: float,
    adaptive_budget_enabled: bool = True,
    component_marginal_gains: dict[str, float] | None = None,
) -> dict[str, int | str | float]:
    high_dim_compact = input_dim >= 40 and n_local_groups >= 12
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
        # Keep decision layer controlled on high-dimensional tasks, but
        # allow a wider cap than legacy settings to reduce late-stage bottlenecking.
        decision_cap = min(decision_cap, 20)

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
    if high_dim_compact:
        high_dim_soft_cap = min(cap_total, max(176, min(236, 11 * n_local_groups + 24)))
        if budget_total > high_dim_soft_cap:
            budget_total = high_dim_soft_cap
            allocation_policy = f"{allocation_policy}+highdim_cap{int(high_dim_soft_cap)}"

    share = float(np.clip(intergroup_interaction_share, 0.0, 1.0))
    utilities = {
        "local": 0.95 + 0.35 * (1.0 - share),
        "bridge": (0.20 + 1.10 * share) if bridge_cap_total > 0 else 0.0,
        "aggregate": 0.90 + 0.45 * share,
        "decision": 0.80 + (0.15 if input_dim >= 20 else 0.0),
    }
    if high_dim_compact:
        utilities["local"] = max(0.85, float(utilities["local"]))
        utilities["bridge"] = min(1.35, float(utilities["bridge"]))
        utilities["aggregate"] = max(0.95, float(utilities["aggregate"]))
        utilities["decision"] = max(0.85, float(utilities["decision"]))
    if component_marginal_gains:
        for name in ("local", "bridge", "aggregate", "decision"):
            raw_gain = float(component_marginal_gains.get(name, 1.0))
            gain = float(np.clip(raw_gain, 0.25, 3.0))
            utilities[name] *= gain
        allocation_policy = f"{allocation_policy}+marginal_gain"
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
        "utility_local": float(utilities["local"]),
        "utility_bridge": float(utilities["bridge"]),
        "utility_aggregate": float(utilities["aggregate"]),
        "utility_decision": float(utilities["decision"]),
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


def resolve_dffl_shuffle_policy(
    *,
    task_type: TaskType,
    input_dim: int,
    n_samples: int,
    intergroup_interaction_share: float,
) -> tuple[bool, str]:
    if (
        task_type == "binary_classification"
        and input_dim >= 40
        and n_samples >= 1500
        and float(np.clip(intergroup_interaction_share, 0.0, 1.0)) >= 0.60
    ):
        return False, "high_dim_stability_disable"
    return True, "default_enable"


def make_dffl_bridge_pairs(
    profile: DfflProfile,
    *,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    validation_inputs: torch.Tensor | None = None,
    validation_targets: torch.Tensor | None = None,
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
    pairwise_validation_gain: np.ndarray | None = None
    if (
        validation_inputs is not None
        and validation_targets is not None
        and int(validation_inputs.shape[0]) >= 8
        and int(validation_inputs.shape[1]) == input_dim
    ):
        pairwise_validation_gain = _pair_interaction_relevance(validation_inputs, validation_targets)
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
    validation_gain_weight = (
        max(0.0, float(profile.bridge_validation_gain_weight)) if pairwise_validation_gain is not None else 0.0
    )
    if validation_gain_weight > 0.0 and validation_inputs is not None:
        n_validation = int(validation_inputs.shape[0])
        # Guard against noisy reranking on tiny validation splits (e.g., wine).
        if n_validation < 40:
            validation_gain_weight = 0.0
        elif n_validation < 80:
            validation_gain_weight *= 0.5
    # Enrich bridge selection beyond correlation:
    # target relevance + interaction strength + stability proxy.
    base_geom_weight = alpha
    base_relevance_weight = 1.0 - alpha
    dynamic_interaction_weight = interaction_weight * (0.75 + 0.5 * float(np.clip(interaction_share, 0.0, 1.0)))
    dynamic_stability_weight = stability_weight
    weight_sum = (
        base_geom_weight
        + base_relevance_weight
        + dynamic_interaction_weight
        + dynamic_stability_weight
        + validation_gain_weight
    )
    if weight_sum <= 1e-12:
        geom_weight = 0.5
        relevance_weight = 0.5
        interaction_weight_final = 0.0
        stability_weight_final = 0.0
        validation_gain_weight_final = 0.0
    else:
        geom_weight = base_geom_weight / weight_sum
        relevance_weight = base_relevance_weight / weight_sum
        interaction_weight_final = dynamic_interaction_weight / weight_sum
        stability_weight_final = dynamic_stability_weight / weight_sum
        validation_gain_weight_final = validation_gain_weight / weight_sum

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
            validation_gain_score = (
                float(pairwise_validation_gain[left, right]) if pairwise_validation_gain is not None else 0.0
            )
            score = (
                geom_weight * geom_score
                + relevance_weight * rel_score
                + interaction_weight_final * interaction_score
                + stability_weight_final * stability_score
                + validation_gain_weight_final * validation_gain_score
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
    if input_dim >= 40 and interaction_share >= 0.55:
        max_pairs_per_group_pair = max(max_pairs_per_group_pair, 2)
    if input_dim >= 80 and interaction_share >= 0.65:
        max_pairs_per_group_pair = max(max_pairs_per_group_pair, 3)
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

    # Score-aware bridge pruning for small/medium tabular settings:
    # keep complex cross-group links, drop low-value bridge tails that mostly add noise/rules.
    if selected and input_dim <= 24 and int(train_inputs.shape[0]) <= 1200 and len(selected) > 1:
        top_score = float(selected[0][0])
        relative_floor = 0.45 * top_score
        absolute_floor = 0.12
        keep_floor = max(absolute_floor, relative_floor)
        min_keep = 2 if interaction_share >= 0.70 else 1
        pruned = [item for item in selected if float(item[0]) >= keep_floor]
        if len(pruned) >= min_keep:
            selected = pruned
        else:
            selected = selected[:min_keep]

    return tuple((int(left), int(right)) for _, left, right in selected[:effective_top_pairs])


def make_dffl_residual_features(
    profile: DfflProfile,
    *,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    input_groups: tuple[tuple[int, ...], ...],
    validation_inputs: torch.Tensor | None = None,
    validation_targets: torch.Tensor | None = None,
) -> tuple[int, ...]:
    if (not profile.residual_enabled) or int(profile.residual_top_features) <= 0:
        return ()
    input_dim = int(train_inputs.shape[1])
    if input_dim <= 1 or int(train_inputs.shape[0]) < 8:
        return ()

    relevance = _feature_target_relevance(train_inputs, train_targets)
    stability = _feature_relevance_stability_proxy(train_inputs, train_targets)
    score = relevance * (0.75 + 0.25 * stability)

    if (
        validation_inputs is not None
        and validation_targets is not None
        and int(validation_inputs.shape[0]) >= 8
        and int(validation_inputs.shape[1]) == input_dim
    ):
        validation_relevance = _feature_target_relevance(validation_inputs, validation_targets)
        agreement = 1.0 - np.abs(relevance - validation_relevance)
        score = score * np.clip(agreement, 0.20, 1.00)

    feature_to_group = _build_feature_to_group_index(input_dim, input_groups)
    max_per_group = max(1, int(profile.residual_max_per_group))
    target_count = min(input_dim, max(1, int(profile.residual_top_features)))

    selected: list[int] = []
    group_counts: dict[int, int] = {}
    ordered = np.argsort(-score, kind="stable")
    for feature_idx in ordered.tolist():
        idx = int(feature_idx)
        group_idx = int(feature_to_group[idx])
        if group_idx >= 0 and group_counts.get(group_idx, 0) >= max_per_group:
            continue
        selected.append(idx)
        if group_idx >= 0:
            group_counts[group_idx] = group_counts.get(group_idx, 0) + 1
        if len(selected) >= target_count:
            break

    return tuple(selected)


def _concept_width(input_dim: int) -> int:
    return min(8, max(4, int(round(input_dim**0.5))))


def _hidden_width(input_dim: int) -> int:
    return min(6, max(3, _concept_width(input_dim) // 2 + 1))


def _scaled_int(value: int, scale: float, *, minimum: int = 1) -> int:
    if scale <= 0.0:
        raise ValueError("scale must be positive.")
    return max(minimum, int(round(value * scale)))


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


def build_stacked_config(
    input_dim: int,
    *,
    width_scale: float = 1.0,
    rule_scale: float = 1.0,
    prototype_scoring_mode: str = "max",
    hidden_output_activation: str | None = None,
    final_skip_input_count: int = 0,
    final_skip_input_indices: tuple[int, ...] | None = None,
    final_skip_gates_enabled: bool = False,
    final_skip_gate_init_logit: float = 2.0,
) -> StackedAnfisModelConfig:
    hidden_1 = _scaled_int(_concept_width(input_dim), width_scale, minimum=2)
    hidden_2 = _scaled_int(_hidden_width(input_dim), width_scale, minimum=2)
    if final_skip_input_indices is None:
        skip_indices = tuple(range(min(max(0, int(final_skip_input_count)), input_dim)))
    else:
        skip_indices = tuple(int(index) for index in final_skip_input_indices if 0 <= int(index) < input_dim)
    final_variables = tuple(var3(f"h2_{i}") for i in range(hidden_2)) + tuple(
        var3(f"x{index}") for index in skip_indices
    )
    return StackedAnfisModelConfig(
        input_dim=input_dim,
        final_skip_input_indices=skip_indices,
        final_skip_gates_enabled=final_skip_gates_enabled,
        final_skip_gate_init_logit=final_skip_gate_init_logit,
        layers=(
            StackedAnfisLayerConfig(
                name="stacked_layer_1",
                variables=tuple(var3(f"x{i}") for i in range(input_dim)),
                output_dim=hidden_1,
                output_names=tuple(f"h1_{i}" for i in range(hidden_1)),
                output_activation=hidden_output_activation,
                max_rule_arity=2,
                max_rules=_scaled_int(24, rule_scale, minimum=4),
                rule_generation_mode="prototype",
                prototype_scoring_mode=prototype_scoring_mode,
            ),
            StackedAnfisLayerConfig(
                name="stacked_layer_2",
                variables=tuple(var3(f"h1_{i}") for i in range(hidden_1)),
                output_dim=hidden_2,
                output_names=tuple(f"h2_{i}" for i in range(hidden_2)),
                output_activation=hidden_output_activation,
                max_rule_arity=2,
                max_rules=_scaled_int(16, rule_scale, minimum=4),
                rule_generation_mode="prototype",
                prototype_scoring_mode=prototype_scoring_mode,
            ),
            StackedAnfisLayerConfig(
                name="stacked_layer_3",
                variables=final_variables,
                output_dim=1,
                output_names=("target",),
                max_rule_arity=2,
                max_rules=_scaled_int(10, rule_scale, minimum=4),
                rule_generation_mode="prototype",
                prototype_scoring_mode=prototype_scoring_mode,
            ),
        ),
    )


def _resolve_stacked_final_skip_indices(
    *,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    count: int,
    mode: str,
) -> tuple[int, ...]:
    input_dim = int(train_inputs.shape[1])
    requested = min(max(0, int(count)), input_dim)
    if requested <= 0:
        return ()

    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "first":
        return tuple(range(requested))
    if normalized_mode not in {"target_corr", "target_corr_diverse"}:
        raise ValueError(
            f"Unsupported stacked_final_skip_mode={mode!r}. Expected 'first', 'target_corr', or 'target_corr_diverse'."
        )

    relevance = _feature_target_relevance(train_inputs, train_targets)
    if normalized_mode == "target_corr":
        ranked = np.argsort(-relevance, kind="stable")
        return tuple(int(index) for index in ranked[:requested])

    stability = _feature_relevance_stability_proxy(train_inputs, train_targets)
    pairwise_corr = _feature_feature_correlation(train_inputs)
    base_score = 0.75 * relevance + 0.25 * stability
    remaining = set(range(input_dim))
    selected: list[int] = []
    while remaining and len(selected) < requested:
        best_index = None
        best_score = float("-inf")
        for candidate in sorted(remaining):
            redundancy = 0.0 if not selected else max(float(pairwise_corr[candidate, prev]) for prev in selected)
            score = float(base_score[candidate]) - 0.35 * redundancy
            if score > best_score:
                best_score = score
                best_index = candidate
        if best_index is None:
            break
        selected.append(int(best_index))
        remaining.remove(best_index)
    return tuple(selected)


def _make_kanfis_pair_indices(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    *,
    max_pairs: int = 24,
) -> tuple[tuple[int, int], ...]:
    relevance = _feature_target_relevance(train_inputs, train_targets)
    values = train_inputs.detach().float().cpu()
    near_binary = (((values - 0.0).abs() < 1e-5) | ((values - 1.0).abs() < 1e-5)).float().mean(dim=0) > 0.995
    relevance_order = [int(index) for index in np.argsort(-np.asarray(relevance, dtype=np.float64), kind="stable").tolist()]
    non_binary = [index for index in relevance_order if not bool(near_binary[index])]
    ranked = non_binary if len(non_binary) >= 2 else relevance_order
    top_features = tuple(ranked[: min(10, int(train_inputs.shape[1]))])
    candidates: list[tuple[float, int, int]] = []
    for left_pos, left in enumerate(top_features):
        for right in top_features[left_pos + 1 :]:
            score = float(relevance[left] + relevance[right])
            candidates.append((score, int(left), int(right)))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple((left, right) for _, left, right in candidates[: max(0, int(max_pairs))])


def _make_kanfis_projection_indices(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    *,
    projection_count: int = 8,
    projection_width: int = 3,
    mode: str = "ranked",
) -> tuple[tuple[int, ...], ...]:
    count = max(0, int(projection_count))
    width = max(1, int(projection_width))
    if count <= 0:
        return ()
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "fixed_covtype" and int(train_inputs.shape[1]) == 54:
        base_routes = (
            (9, 8, 42, 17),
            (0, 4, 53, 19),
            (7, 13, 36, 24),
            (2, 25, 26, 14),
            (3, 35, 43, 48),
            (6, 10, 23, 45),
            (5, 51, 11, 31),
            (1, 52, 15, 18),
            (9, 0, 7, 2),
            (3, 6, 5, 1),
            (13, 25, 35, 51),
            (52, 36, 42, 53),
        )
        return tuple(tuple(route[:width]) for route in base_routes[:count])
    relevance = _feature_target_relevance(train_inputs, train_targets)
    values = train_inputs.detach().float().cpu()
    near_binary = (((values - 0.0).abs() < 1e-5) | ((values - 1.0).abs() < 1e-5)).float().mean(dim=0) > 0.995
    order = [int(index) for index in np.argsort(-np.asarray(relevance, dtype=np.float64), kind="stable").tolist()]
    continuous = [index for index in order if not bool(near_binary[index])]
    binary = [index for index in order if bool(near_binary[index])]
    ranked = continuous + binary
    if not ranked:
        return ()
    routes: list[tuple[int, ...]] = []
    for route_index in range(count):
        route = []
        for offset in range(width):
            route.append(ranked[(route_index + offset * max(1, count)) % len(ranked)])
        routes.append(tuple(dict.fromkeys(route)))
    return tuple(routes)


def _make_kanfis_active_feature_indices(
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    *,
    max_features: int = 8,
    feature_importances: tuple[float, ...] | list[float] | None = None,
) -> tuple[int, ...]:
    if feature_importances is not None and len(feature_importances) == int(train_inputs.shape[1]):
        requested = min(max(1, int(max_features)), int(train_inputs.shape[1]))
        ranked = np.argsort(-np.asarray(feature_importances, dtype=np.float64), kind="stable")
        return tuple(int(index) for index in ranked[:requested])
    return _resolve_stacked_final_skip_indices(
        train_inputs=train_inputs,
        train_targets=train_targets,
        count=min(max(1, int(max_features)), int(train_inputs.shape[1])),
        mode="target_corr_diverse",
    )


def build_kanfis_model(
    input_dim: int,
    *,
    pair_indices: tuple[tuple[int, int], ...] = (),
    projection_indices: tuple[tuple[int, ...], ...] = (),
    active_feature_indices: tuple[int, ...] | None = None,
    superposition_terms: int = 16,
    depth: int = 1,
    concept_fan_in: int = 0,
    routing: str = "chunk",
    decision_skip_indices: tuple[int, ...] = (),
    train_rule_gates: bool = False,
    decision_skip_gate_init_logit: float = 1.5,
) -> KANFISModel:
    if depth >= 2:
        first_width = max(2, int(superposition_terms))
        concept_depth = max(2, int(depth))
        concept_widths = [first_width, max(2, min(6, first_width))]
        while len(concept_widths) < concept_depth:
            width = round(concept_widths[-1] * 0.75)
            concept_widths.append(max(2, min(first_width, int(width))))
        return DeepKANFISModel(
            input_dim=input_dim,
            concept_widths=tuple(concept_widths),
            term_centers=(0.2, 0.5, 0.8),
            term_width=0.18,
            active_feature_indices=active_feature_indices,
            pair_indices=pair_indices,
            projection_indices=projection_indices,
            first_layer_fan_in=concept_fan_in,
            first_layer_routing=routing,
            decision_skip_indices=decision_skip_indices,
            output_dim=1,
            train_rule_gates=train_rule_gates,
            decision_skip_gate_init_logit=decision_skip_gate_init_logit,
        )
    # Cap the Kolmogorov-style 2n+1 width so high-dimensional tabular runs stay lightweight.
    return KANFISModel(
        input_dim=input_dim,
        superposition_terms=min(2 * input_dim + 1, max(1, int(superposition_terms))),
        term_centers=(0.2, 0.5, 0.8),
        term_width=0.18,
        pair_indices=pair_indices,
        active_feature_indices=active_feature_indices,
        output_dim=1,
        train_rule_gates=train_rule_gates,
    )


def build_hierarchical_anfis_config(
    input_dim: int,
    *,
    group_size: int = 4,
    width_scale: float = 1.0,
    rule_scale: float = 1.0,
    prototype_scoring_mode: str = "max",
    hidden_output_activation: str | None = None,
) -> HierarchicalAnfisModelConfig:
    groups = _make_groups(input_dim, group_size=max(1, int(group_size)))
    local_width = _scaled_int(2, width_scale, minimum=1)
    stage_1_blocks = []
    for block_index, indices in enumerate(groups):
        stage_1_blocks.append(
            HierarchicalAnfisBlockConfig(
                name=f"anfis_local_{block_index}",
                input_indices=indices,
                variables=tuple(var3(f"x{idx}") for idx in indices),
                output_dim=local_width,
                output_names=tuple(f"s1_{block_index}_{i}" for i in range(local_width)),
                output_activation=hidden_output_activation,
                max_rule_arity=min(2, len(indices)),
                max_rules=_scaled_int(8, rule_scale, minimum=2),
                rule_generation_mode="prototype",
                prototype_scoring_mode=prototype_scoring_mode,
            )
        )

    stage_1_width = local_width * len(groups)
    stage_2_width = _scaled_int(min(6, max(3, len(groups) // 2 + 2)), width_scale, minimum=2)

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
                        output_activation=hidden_output_activation,
                        max_rule_arity=2,
                        max_rules=_scaled_int(16, rule_scale, minimum=4),
                        rule_generation_mode="prototype",
                        prototype_scoring_mode=prototype_scoring_mode,
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
            max_rules=_scaled_int(10, rule_scale, minimum=4),
            rule_generation_mode="prototype",
            prototype_scoring_mode=prototype_scoring_mode,
        ),
    )


def build_dffl_config(
    input_dim: int,
    profile: DfflProfile,
    *,
    input_groups: tuple[tuple[int, ...], ...] | None = None,
    binary_feature_indices: tuple[int, ...] | None = None,
    bridge_feature_pairs: tuple[tuple[int, int], ...] | None = None,
    residual_feature_indices: tuple[int, ...] | None = None,
    high_arity_group_indices: tuple[int, ...] | None = None,
    intergroup_interaction_share: float | None = None,
    adaptive_budget_enabled: bool = True,
    adaptive_local_arity_enabled: bool = True,
    component_marginal_gains: dict[str, float] | None = None,
    final_skip_input_indices: tuple[int, ...] | None = None,
) -> HierarchicalModelConfig:
    groups = input_groups or _make_groups(input_dim, group_size=profile.input_group_size)
    binary_feature_set = set(binary_feature_indices or ())
    bridge_pairs = tuple(bridge_feature_pairs or ())
    residual_features = tuple(int(idx) for idx in (residual_feature_indices or ()))
    high_arity_group_set = set(int(idx) for idx in (high_arity_group_indices or ()))
    if not profile.residual_enabled:
        residual_features = ()
    if not profile.bridge_enabled:
        bridge_pairs = ()
    skip_indices = tuple(int(index) for index in (final_skip_input_indices or profile.final_skip_input_indices))
    skip_indices = tuple(index for index in skip_indices if 0 <= index < input_dim)
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
        component_marginal_gains=component_marginal_gains,
    )
    local_max_rules_effective = int(effective_budgets["local_max_rules_effective"])
    bridge_max_rules_effective = int(effective_budgets["bridge_max_rules_effective"])
    decision_max_rules_effective = int(effective_budgets["decision_max_rules_effective"])
    aggregate_max_rules_effective = int(effective_budgets["aggregate_max_rules_effective"])

    stage_1_blocks = []
    bridge_output_indices: list[int] = []
    stage_1_offset = 0
    for block_index, indices in enumerate(groups):
        local_rule_arity = min(profile.local_max_rule_arity, len(indices))
        if adaptive_local_arity_enabled:
            if block_index in high_arity_group_set:
                local_rule_arity = min(max(profile.local_max_rule_arity, 3), len(indices))
            elif input_dim >= 40 and profile.local_max_rule_arity >= 3:
                # Keep high-dimensional blocks mostly compact but allow a wider
                # subset to preserve useful cross-feature interactions.
                relaxed_high_arity = block_index < max(1, len(groups) // 3)
                local_rule_arity = min(3 if relaxed_high_arity else 2, len(indices))
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
                max_rule_arity=local_rule_arity,
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

    if residual_features:
        residual_concepts = max(1, int(profile.residual_concepts))
        residual_max_rules = max(1, int(profile.residual_max_rules))
        for residual_index, feature_idx in enumerate(residual_features):
            if feature_idx < 0 or feature_idx >= input_dim:
                continue
            stage_1_blocks.append(
                TransparentBlockConfig(
                    name=f"dffl_residual_{residual_index}",
                    input_indices=(int(feature_idx),),
                    variables=(
                        var_binary(f"x{feature_idx}") if feature_idx in binary_feature_set else var3(f"x{feature_idx}"),
                    ),
                    n_concepts=residual_concepts,
                    concept_names=tuple(f"s1r_{residual_index}_{i}" for i in range(residual_concepts)),
                    max_rule_arity=1,
                    max_rules=residual_max_rules,
                    rule_generation_mode="prototype",
                    prototype_term_limit=1,
                    prototype_scoring_mode="max",
                    prototype_variable_pool_size=1,
                    prototype_sample_size=profile.local_prototype_sample_size,
                    consequent_mode=profile.local_consequent_mode,
                )
            )
            stage_1_offset += residual_concepts

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
    width_driver = len(groups) + len(bridge_pairs) + len(residual_features)
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
    # Guard against zero-rule aggregate blocks when the effective budget is tight.
    # This can happen under high-dimensional budget caps with many aggregate blocks.
    aggregate_blocks = max(1, min(aggregate_blocks, aggregate_rule_budget_total))
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

    decision_variables = tuple(
        (var3(f"s2_{i}") if profile.decision_three_terms else var2(f"s2_{i}"))
        for i in range(stage_2_width)
    ) + tuple(
        (var_binary(f"x{idx}") if idx in binary_feature_set else var3(f"x{idx}"))
        for idx in skip_indices
    )

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
            variables=decision_variables,
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
        final_skip_input_indices=skip_indices,
        final_skip_gates_enabled=profile.final_skip_gates_enabled,
        final_skip_gate_init_logit=profile.final_skip_gate_init_logit,
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


def _predict_logits_batched(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    *,
    top_k_rules: int | None = None,
    batch_size: int = 8192,
) -> torch.Tensor:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, inputs.size(0), batch_size):
            batch_inputs = inputs[start : start + batch_size].to(device=device, dtype=torch.float32)
            batch_predictions = predict_with_optional_residual_head(
                model,
                batch_inputs,
                top_k_rules=top_k_rules,
            )
            predictions.append(batch_predictions.detach().cpu())
    return torch.cat(predictions, dim=0)


def _choose_best_classification_threshold(
    model: torch.nn.Module,
    *,
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    default_threshold: float,
    top_k_rules: int | None = None,
    strategy: str = "f1",
    prediction_postprocessor: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> float:
    if validation_inputs.numel() == 0 or validation_targets.numel() == 0:
        return default_threshold
    if strategy not in {"f1", "calibration"}:
        raise ValueError("threshold strategy must be 'f1' or 'calibration'.")

    validation_logits = _predict_logits_batched(
        model,
        validation_inputs,
        top_k_rules=top_k_rules,
    )
    if prediction_postprocessor is not None:
        validation_logits = prediction_postprocessor(validation_logits)

    best_threshold = float(default_threshold)
    best_primary = float("-inf")
    best_secondary = float("-inf")
    best_tertiary = float("-inf")
    validation_targets_cpu = validation_targets.detach().cpu()
    target_positive_rate = float((validation_targets_cpu.reshape(-1) >= 0.5).float().mean().item())
    for threshold in np.linspace(0.05, 0.95, 91):
        metrics = compute_metrics(
            "binary_classification",
            validation_logits,
            validation_targets_cpu,
            classification_threshold=float(threshold),
        )
        current_f1 = float(metrics["f1"])
        current_precision = float(metrics["precision"])
        current_recall = float(metrics["recall"])
        if strategy == "f1":
            primary = current_f1
            secondary = current_precision
            tertiary = current_recall
            is_better = (
                primary > best_primary + 1e-12
                or (
                    abs(primary - best_primary) <= 1e-12
                    and (
                        secondary > best_secondary + 1e-12
                        or (
                            abs(secondary - best_secondary) <= 1e-12
                            and tertiary > best_tertiary + 1e-12
                        )
                    )
                )
            )
        else:
            probabilities = torch.sigmoid(validation_logits.reshape(-1))
            predicted_positive_rate = float((probabilities >= float(threshold)).float().mean().item())
            calibration_gap = abs(predicted_positive_rate - target_positive_rate)
            # Maximize negative gap (smaller gap is better), then maximize F1.
            primary = -calibration_gap
            secondary = current_f1
            tertiary = current_precision
            is_better = (
                primary > best_primary + 1e-12
                or (
                    abs(primary - best_primary) <= 1e-12
                    and (
                        secondary > best_secondary + 1e-12
                        or (
                            abs(secondary - best_secondary) <= 1e-12
                            and tertiary > best_tertiary + 1e-12
                        )
                    )
                )
            )
        if is_better:
            best_primary = primary
            best_secondary = secondary
            best_tertiary = tertiary
            best_threshold = float(threshold)
            continue
        if (
            abs(primary - best_primary) <= 1e-12
            and abs(secondary - best_secondary) <= 1e-12
            and abs(tertiary - best_tertiary) <= 1e-12
            and abs(float(threshold) - default_threshold) < abs(best_threshold - default_threshold)
        ):
            best_threshold = float(threshold)

    return best_threshold


def _binary_log_loss_from_probabilities(probabilities: np.ndarray, targets: np.ndarray) -> float:
    probs = np.clip(probabilities.astype(np.float64, copy=False), 1e-7, 1.0 - 1e-7)
    y = targets.astype(np.float64, copy=False)
    return float(-np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))


def _binary_brier_from_probabilities(probabilities: np.ndarray, targets: np.ndarray) -> float:
    probs = probabilities.astype(np.float64, copy=False)
    y = targets.astype(np.float64, copy=False)
    return float(np.mean((probs - y) ** 2))


def _fit_binary_probability_calibrator(
    validation_logits: torch.Tensor,
    validation_targets: torch.Tensor,
) -> dict[str, object] | None:
    logits_np = validation_logits.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
    targets_np = (
        (validation_targets.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False) >= 0.5)
        .astype(np.int64, copy=False)
    )
    if logits_np.size == 0:
        return None
    unique_targets = np.unique(targets_np)
    if unique_targets.size < 2:
        return None

    raw_probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits_np, -30.0, 30.0)))
    n_samples = int(logits_np.shape[0])

    platt = LogisticRegression(solver="lbfgs", max_iter=500, random_state=0)
    platt.fit(logits_np.reshape(-1, 1), targets_np)
    platt_prob = platt.predict_proba(logits_np.reshape(-1, 1))[:, 1]
    platt_log_loss = _binary_log_loss_from_probabilities(platt_prob, targets_np)
    platt_brier = _binary_brier_from_probabilities(platt_prob, targets_np)
    platt_objective = 0.7 * platt_log_loss + 0.3 * platt_brier

    # Isotonic can overfit strongly on small validation splits.
    if n_samples < 200:
        isotonic_objective = float("inf")
        isotonic_log_loss = float("inf")
        isotonic_brier = float("inf")
        isotonic = None
    else:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(raw_probabilities, targets_np)
        isotonic_prob = np.clip(isotonic.predict(raw_probabilities), 1e-7, 1.0 - 1e-7)
        isotonic_log_loss = _binary_log_loss_from_probabilities(isotonic_prob, targets_np)
        isotonic_brier = _binary_brier_from_probabilities(isotonic_prob, targets_np)
        isotonic_objective = 0.7 * isotonic_log_loss + 0.3 * isotonic_brier

    if isotonic is not None and isotonic_objective + 1e-12 < platt_objective:
        return {
            "method": "isotonic",
            "x_thresholds": tuple(float(v) for v in isotonic.X_thresholds_.tolist()),
            "y_thresholds": tuple(float(v) for v in isotonic.y_thresholds_.tolist()),
            "validation_log_loss": float(isotonic_log_loss),
            "validation_brier": float(isotonic_brier),
            "validation_objective": float(isotonic_objective),
        }
    return {
        "method": "platt",
        "coef": float(platt.coef_[0, 0]),
        "intercept": float(platt.intercept_[0]),
        "validation_log_loss": float(platt_log_loss),
        "validation_brier": float(platt_brier),
        "validation_objective": float(platt_objective),
    }


def _apply_binary_probability_calibrator_to_logits(
    logits: torch.Tensor,
    calibrator: dict[str, object] | None,
) -> torch.Tensor:
    if calibrator is None:
        return logits
    flat_logits = logits.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
    raw_probabilities = 1.0 / (1.0 + np.exp(-np.clip(flat_logits, -30.0, 30.0)))
    method = str(calibrator.get("method", "none"))
    if method == "platt":
        coef = float(calibrator["coef"])
        intercept = float(calibrator["intercept"])
        calibrated_prob = 1.0 / (1.0 + np.exp(-(coef * flat_logits + intercept)))
    elif method == "isotonic":
        x_thresholds = np.asarray(calibrator["x_thresholds"], dtype=np.float64)
        y_thresholds = np.asarray(calibrator["y_thresholds"], dtype=np.float64)
        if x_thresholds.size == 0 or y_thresholds.size == 0:
            calibrated_prob = raw_probabilities
        else:
            calibrated_prob = np.interp(raw_probabilities, x_thresholds, y_thresholds)
    else:
        calibrated_prob = raw_probabilities
    calibrated_prob = np.clip(calibrated_prob, 1e-7, 1.0 - 1e-7)
    calibrated_logits = torch.logit(
        torch.from_numpy(calibrated_prob).to(dtype=logits.dtype),
        eps=1e-7,
    ).reshape(logits.shape)
    return calibrated_logits.to(device=logits.device)


def _build_binary_logit_postprocessor(
    calibrator: dict[str, object] | None,
) -> Callable[[torch.Tensor], torch.Tensor] | None:
    if calibrator is None:
        return None

    def _postprocess(logits: torch.Tensor) -> torch.Tensor:
        return _apply_binary_probability_calibrator_to_logits(logits, calibrator)

    return _postprocess


def _resolve_threshold_tuning_strategy(dataset_name: str) -> str:
    if dataset_name == "breast_cancer":
        return "calibration"
    return "f1"


def _resolve_binary_loss_params(dataset_name: str) -> dict[str, float | str | None]:
    if dataset_name == "breast_cancer":
        return {
            "binary_loss_name": "focal",
            "binary_focal_gamma": 1.5,
            "binary_focal_alpha": None,
            "binary_class_balanced_beta": 0.999,
        }
    return {
        "binary_loss_name": "bce",
        "binary_focal_gamma": 2.0,
        "binary_focal_alpha": None,
        "binary_class_balanced_beta": None,
    }


def _resolve_dataset_binary_loss_params(
    *,
    dataset_name: str,
    dataset_override_entry: Mapping[str, Any],
) -> dict[str, float | str | None]:
    params = dict(_resolve_binary_loss_params(dataset_name))
    if "binary_loss_name" in dataset_override_entry:
        params["binary_loss_name"] = str(dataset_override_entry["binary_loss_name"]).strip().lower()
    if "binary_focal_gamma" in dataset_override_entry:
        params["binary_focal_gamma"] = float(dataset_override_entry["binary_focal_gamma"])
    if "binary_focal_alpha" in dataset_override_entry:
        alpha = dataset_override_entry["binary_focal_alpha"]
        params["binary_focal_alpha"] = None if alpha is None else float(alpha)
    if "binary_class_balanced_beta" in dataset_override_entry:
        beta = dataset_override_entry["binary_class_balanced_beta"]
        params["binary_class_balanced_beta"] = None if beta is None else float(beta)
    return params


def _resolve_dataset_threshold_tuning_strategy(
    *,
    dataset_name: str,
    dataset_override_entry: Mapping[str, Any],
) -> str:
    strategy = dataset_override_entry.get("threshold_tuning_strategy", _resolve_threshold_tuning_strategy(dataset_name))
    strategy_norm = str(strategy).strip().lower()
    if strategy_norm not in {"f1", "calibration"}:
        raise ValueError(
            f"Unsupported threshold_tuning_strategy={strategy!r} for dataset {dataset_name!r}. "
            "Expected one of: f1, calibration."
        )
    return strategy_norm


def _resolve_breast_focal_candidates(
    *,
    dataset_name: str,
    task_type: TaskType,
    train_targets: torch.Tensor,
    default_gamma: float,
    default_alpha: float | None,
) -> tuple[tuple[float, float | None], ...]:
    if task_type != "binary_classification" or dataset_name != "breast_cancer":
        return ((float(default_gamma), default_alpha),)
    positive_rate = float((train_targets.detach().reshape(-1) >= 0.5).float().mean().item())
    alpha_auto = float(np.clip(1.0 - positive_rate, 0.25, 0.75))
    candidates = (
        (1.0, None),
        (1.5, None),
        (2.0, alpha_auto),
    )
    unique: list[tuple[float, float | None]] = []
    seen: set[tuple[float, float | None]] = set()
    for gamma, alpha in candidates:
        key = (round(float(gamma), 6), None if alpha is None else round(float(alpha), 6))
        if key in seen:
            continue
        seen.add(key)
        unique.append((float(gamma), None if alpha is None else float(alpha)))
    return tuple(unique)


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


def _build_binary_distillation_targets(
    *,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    seed: int,
    teacher_trees: int,
    blend_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if teacher_trees <= 0:
        raise ValueError("fuzzy_distill_teacher_trees must be positive.")
    if not (0.0 < blend_weight <= 1.0):
        raise ValueError("fuzzy_distill_weight must be in (0, 1].")

    x_train = train_inputs.detach().cpu().numpy().astype(np.float32, copy=False)
    y_train = train_targets.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
    y_train_bin = (y_train >= 0.5).astype(np.int64, copy=False)
    if np.unique(y_train_bin).size < 2:
        return train_targets.clone(), {
            "teacher_trees": float(teacher_trees),
            "teacher_val_f1": float("nan"),
            "teacher_val_auc": float("nan"),
            "teacher_feature_importances": [],
        }

    teacher = ExtraTreesClassifier(
        n_estimators=int(teacher_trees),
        random_state=int(seed),
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    teacher.fit(x_train, y_train_bin)

    p_train = teacher.predict_proba(x_train)[:, 1].astype(np.float32, copy=False)
    p_train_t = torch.from_numpy(p_train).reshape_as(train_targets)
    blended_targets = ((1.0 - blend_weight) * train_targets + blend_weight * p_train_t).clamp(0.0, 1.0)

    x_val = validation_inputs.detach().cpu().numpy().astype(np.float32, copy=False)
    y_val = validation_targets.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
    y_val_bin = (y_val >= 0.5).astype(np.int64, copy=False)
    p_val = teacher.predict_proba(x_val)[:, 1].astype(np.float64, copy=False)
    val_logits = torch.from_numpy(np.log(np.clip(p_val, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - p_val, 1e-6, 1.0)))
    val_targets = torch.from_numpy(y_val_bin.astype(np.float32, copy=False)).reshape_as(validation_targets)
    val_metrics = compute_metrics(
        "binary_classification",
        val_logits.reshape_as(validation_targets),
        val_targets,
        classification_threshold=0.5,
    )
    metadata = {
        "teacher_trees": float(teacher_trees),
        "teacher_val_f1": float(val_metrics.get("f1", float("nan"))),
        "teacher_val_auc": float(val_metrics.get("roc_auc", float("nan"))),
        "blend_weight": float(blend_weight),
        "train_target_mean_before": float(train_targets.mean().item()),
        "train_target_mean_after": float(blended_targets.mean().item()),
        "teacher_feature_importances": teacher.feature_importances_.astype(float).tolist(),
    }
    return blended_targets.to(dtype=train_targets.dtype), metadata


def _compute_sample_hardness_scores(
    *,
    task_type: TaskType,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    aligned_targets = targets.to(device=predictions.device, dtype=predictions.dtype)
    if predictions.ndim == 2 and predictions.size(-1) == 1 and aligned_targets.ndim == 1:
        aligned_targets = aligned_targets.unsqueeze(-1)
    if task_type == "binary_classification":
        if aligned_targets.ndim == 1:
            aligned_targets = aligned_targets.unsqueeze(-1)
        per_output = F.binary_cross_entropy_with_logits(
            predictions,
            aligned_targets,
            reduction="none",
        )
        return per_output.reshape(predictions.size(0), -1).mean(dim=1)
    if task_type == "regression":
        if aligned_targets.ndim == 1:
            aligned_targets = aligned_targets.unsqueeze(-1)
        return torch.abs(predictions - aligned_targets).reshape(predictions.size(0), -1).mean(dim=1)
    raise ValueError(f"Unsupported task_type={task_type!r}.")


def _select_hard_sample_indices(
    *,
    hardness_scores: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    if hardness_scores.ndim != 1:
        raise ValueError("hardness_scores must be a 1D tensor.")
    if hardness_scores.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=hardness_scores.device)
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fuzzy_hard_sample_fraction must be in (0, 1].")
    hard_count = max(1, int(round(float(hardness_scores.numel()) * fraction)))
    hard_count = min(hard_count, int(hardness_scores.numel()))
    return torch.topk(hardness_scores, k=hard_count, largest=True, sorted=False).indices


def _augment_with_hard_samples(
    *,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    hard_indices: torch.Tensor,
    multiplier: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    repeat_factor = int(multiplier) - 1
    if repeat_factor <= 0 or hard_indices.numel() == 0:
        return inputs, targets

    hard_inputs = inputs.index_select(0, hard_indices)
    hard_targets = targets.index_select(0, hard_indices)
    if hard_inputs.ndim == 2:
        hard_inputs = hard_inputs.repeat((repeat_factor, 1))
    else:
        hard_inputs = hard_inputs.repeat(repeat_factor)
    if hard_targets.ndim == 2:
        hard_targets = hard_targets.repeat((repeat_factor, 1))
    else:
        hard_targets = hard_targets.repeat(repeat_factor)
    return torch.cat((inputs, hard_inputs), dim=0), torch.cat((targets, hard_targets), dim=0)


def _compute_baseline_hard_indices(
    *,
    task_type: TaskType,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    fraction: float,
    seed: int,
    teacher_trees: int,
) -> torch.Tensor | None:
    if teacher_trees <= 0:
        raise ValueError("fuzzy_hard_sample_teacher_trees must be positive.")
    if task_type != "binary_classification":
        return None

    x_train = train_inputs.detach().cpu().numpy().astype(np.float32, copy=False)
    y_train = train_targets.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
    y_train_bin = (y_train >= 0.5).astype(np.int64, copy=False)
    if np.unique(y_train_bin).size < 2:
        return None

    teacher = ExtraTreesClassifier(
        n_estimators=int(teacher_trees),
        random_state=int(seed),
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    teacher.fit(x_train, y_train_bin)
    p_train = teacher.predict_proba(x_train)[:, 1].astype(np.float32, copy=False)
    p_train = np.clip(p_train, 1e-6, 1.0 - 1e-6)
    logits = torch.from_numpy(np.log(p_train / (1.0 - p_train))).reshape_as(train_targets)
    scores = _compute_sample_hardness_scores(
        task_type=task_type,
        predictions=logits,
        targets=train_targets,
    )
    return _select_hard_sample_indices(hardness_scores=scores, fraction=fraction)


def _maybe_run_hard_sample_finetune(
    *,
    enabled: bool,
    model_name: str,
    model: torch.nn.Module,
    task_type: TaskType,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    base_training_config: TrainingConfig,
    source: str,
    hard_fraction: float,
    hard_multiplier: int,
    finetune_epochs: int,
    finetune_patience: int,
    baseline_hard_indices: torch.Tensor | None,
    seed: int,
) -> None:
    if not enabled:
        return
    if hard_multiplier <= 1:
        return
    if finetune_epochs <= 0:
        return

    normalized_source = str(source).strip().lower()
    if normalized_source not in {"self", "baseline"}:
        raise ValueError("fuzzy_hard_sample_source must be either 'self' or 'baseline'.")

    hard_indices: torch.Tensor | None = None
    if normalized_source == "baseline" and baseline_hard_indices is not None:
        hard_indices = baseline_hard_indices.to(device=train_inputs.device, dtype=torch.long)
    else:
        model_device = next(model.parameters()).device
        score_inputs = train_inputs.to(device=model_device, dtype=torch.float32)
        score_targets = train_targets.to(device=model_device, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            predictions = predict_with_optional_residual_head(model, score_inputs, top_k_rules=None)
        scores = _compute_sample_hardness_scores(
            task_type=task_type,
            predictions=predictions,
            targets=score_targets,
        )
        hard_indices = _select_hard_sample_indices(
            hardness_scores=scores,
            fraction=hard_fraction,
        ).to(device=train_inputs.device, dtype=torch.long)

    if hard_indices is None or hard_indices.numel() == 0:
        return

    augmented_inputs, augmented_targets = _augment_with_hard_samples(
        inputs=train_inputs,
        targets=train_targets,
        hard_indices=hard_indices,
        multiplier=hard_multiplier,
    )
    finetune_config = replace(
        base_training_config,
        max_epochs=max(1, int(finetune_epochs)),
        patience=min(max(1, int(finetune_patience)), max(1, int(finetune_epochs))),
    )
    # Safe rollback: keep hard-finetune only if validation monitor improves.
    pre_finetune_state = copy.deepcopy(model.state_dict())
    progress_log(
        (
            "seed={seed} model={model}: hard_finetune source={source} hard={hard} "
            "train_before={before} train_after={after}"
        ).format(
            seed=seed,
            model=model_name,
            source=normalized_source,
            hard=int(hard_indices.numel()),
            before=int(train_inputs.size(0)),
            after=int(augmented_inputs.size(0)),
        )
    )
    trainer = FuzzyTrainer(model, finetune_config)
    before_eval = trainer.evaluate(validation_inputs, validation_targets)
    monitor_name = (
        str(finetune_config.monitor_metric)
        if finetune_config.monitor_metric is not None
        else str(PRIMARY_METRIC.get(task_type, "loss"))
    )
    monitor_mode = (
        str(finetune_config.monitor_mode).strip().lower()
        if finetune_config.monitor_mode is not None
        else ("min" if task_type == "regression" else "max")
    )

    def _extract_monitor(eval_result) -> float:
        if monitor_name == "loss":
            return float(eval_result.loss)
        if monitor_name in eval_result.metrics:
            return float(eval_result.metrics[monitor_name])
        return float(eval_result.loss)

    before_value = _extract_monitor(before_eval)
    trainer.fit(augmented_inputs, augmented_targets, validation_inputs, validation_targets)
    after_eval = trainer.evaluate(validation_inputs, validation_targets)
    after_value = _extract_monitor(after_eval)
    min_delta = float(finetune_config.min_delta)
    improved = (
        after_value < (before_value - min_delta)
        if monitor_mode == "min"
        else after_value > (before_value + min_delta)
    )
    if not improved:
        model.load_state_dict(pre_finetune_state)
        progress_log(
            (
                "seed={seed} model={model}: hard_finetune rollback "
                "(monitor={name}, mode={mode}, before={before:.6f}, after={after:.6f})"
            ).format(
                seed=seed,
                model=model_name,
                name=monitor_name,
                mode=monitor_mode,
                before=before_value,
                after=after_value,
            )
        )
    else:
        progress_log(
            (
                "seed={seed} model={model}: hard_finetune keep "
                "(monitor={name}, mode={mode}, before={before:.6f}, after={after:.6f})"
            ).format(
                seed=seed,
                model=model_name,
                name=monitor_name,
                mode=monitor_mode,
                before=before_value,
                after=after_value,
            )
        )


def _select_bootstrap_subset(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_samples: int = 100_000,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_samples = int(inputs.size(0))
    if n_samples <= max_samples:
        return inputs, targets
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 7919)
    indices = torch.randperm(n_samples, generator=generator)[:max_samples]
    indices = indices.to(device=inputs.device)
    return inputs.index_select(0, indices), targets.index_select(0, indices)


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in str(raw).split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return tuple(values)


def _append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _collect_kanfis_layer_inputs(
    model: DeepKANFISModel, inputs: torch.Tensor
) -> tuple[list[torch.Tensor], torch.Tensor | None, torch.Tensor | None]:
    with torch.no_grad():
        full_inputs = inputs.to(device=model.decision_weights.device, dtype=model.decision_weights.dtype)
        features = model._select_inputs(full_inputs)
        layer_inputs: list[torch.Tensor] = []
        pair_input: torch.Tensor | None = None
        projection_input: torch.Tensor | None = None
        for layer_index, layer in enumerate(model.layers):
            layer_inputs.append(features)
            features = layer(features)
            if layer_index == 0 and (model.pair_layer is not None or model.projection_layer is not None):
                extras = []
                if model.pair_layer is not None:
                    pair_input = full_inputs
                    extras.append(model.pair_layer(full_inputs))
                if model.projection_layer is not None:
                    projection_input = full_inputs
                    extras.append(model.projection_layer(full_inputs))
                features = torch.cat((features, *extras), dim=1)
    return layer_inputs, pair_input, projection_input


def _collect_kanfis_importance_entries(
    model: DeepKANFISModel,
    train_inputs: torch.Tensor,
) -> list[dict[str, object]]:
    layer_scales, pair_scale, projection_scale = model._layer_decision_scales()
    layer_inputs, pair_input, projection_input = _collect_kanfis_layer_inputs(model, train_inputs)
    entries: list[dict[str, object]] = []
    with torch.no_grad():
        for layer_idx, (layer, layer_input, scale) in enumerate(zip(model.layers, layer_inputs, layer_scales, strict=True)):
            importances = layer.data_aware_rule_importances(layer_input.detach().cpu(), scale.detach().cpu(), batch_size=8192)
            active_mask = layer.rule_active_mask.detach().cpu().reshape(-1) > 0
            active_indices = torch.nonzero(active_mask, as_tuple=False).reshape(-1).tolist()
            for local_index in active_indices:
                entries.append(
                    {
                        "source": f"layer_{layer_idx}",
                        "rule_local_index": int(local_index),
                        "rule_name": layer.describe_rule(int(local_index)),
                        "importance": float(importances[int(local_index)].item()),
                    }
                )
        if model.pair_layer is not None and pair_input is not None and pair_scale is not None:
            importances = model.pair_layer.data_aware_rule_importances(
                pair_input.detach().cpu(),
                pair_scale.detach().cpu(),
                batch_size=8192,
            )
            active_mask = model.pair_layer.rule_active_mask.detach().cpu().reshape(-1) > 0
            active_indices = torch.nonzero(active_mask, as_tuple=False).reshape(-1).tolist()
            for local_index in active_indices:
                entries.append(
                    {
                        "source": "pair_layer",
                        "rule_local_index": int(local_index),
                        "rule_name": model.pair_layer.describe_rule(int(local_index)),
                        "importance": float(importances[int(local_index)].item()),
                    }
                )
        if model.projection_layer is not None and projection_input is not None and projection_scale is not None:
            importances = model.projection_layer.data_aware_rule_importances(
                projection_input.detach().cpu(),
                projection_scale.detach().cpu(),
                batch_size=8192,
            )
            active_mask = model.projection_layer.rule_active_mask.detach().cpu().reshape(-1) > 0
            active_indices = torch.nonzero(active_mask, as_tuple=False).reshape(-1).tolist()
            for local_index in active_indices:
                entries.append(
                    {
                        "source": "projection_layer",
                        "rule_local_index": int(local_index),
                        "rule_name": model.projection_layer.describe_rule(int(local_index)),
                        "importance": float(importances[int(local_index)].item()),
                    }
                )
    entries.sort(key=lambda item: -float(item["importance"]))
    return entries


def _extract_rule_feature_matrix(
    model: DeepKANFISModel,
    inputs: torch.Tensor,
    selected_rule_keys: list[tuple[str, int]],
) -> np.ndarray:
    if not selected_rule_keys:
        return np.zeros((int(inputs.size(0)), 0), dtype=np.float32)
    layer_inputs, pair_input, projection_input = _collect_kanfis_layer_inputs(model, inputs)
    source_to_scores: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for layer_idx, (layer, layer_input) in enumerate(zip(model.layers, layer_inputs, strict=True)):
            source_to_scores[f"layer_{layer_idx}"] = layer.rule_scores(layer_input).detach().cpu()
        if model.pair_layer is not None and pair_input is not None:
            source_to_scores["pair_layer"] = model.pair_layer.rule_scores(pair_input).detach().cpu()
        if model.projection_layer is not None and projection_input is not None:
            source_to_scores["projection_layer"] = model.projection_layer.rule_scores(projection_input).detach().cpu()
    cols: list[np.ndarray] = []
    for source, local_index in selected_rule_keys:
        values = source_to_scores[source][:, int(local_index)].numpy().astype(np.float32, copy=False)
        cols.append(values)
    return np.stack(cols, axis=1)


def _predict_logits_numpy(model: torch.nn.Module, inputs: torch.Tensor, *, batch_size: int = 8192) -> np.ndarray:
    chunks: list[np.ndarray] = []
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    with torch.no_grad():
        for start in range(0, int(inputs.size(0)), int(batch_size)):
            batch = inputs[start : start + int(batch_size)].to(device=device, dtype=dtype)
            logits = predict_with_optional_residual_head(model, batch, top_k_rules=None)
            chunks.append(logits.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)


def _export_v18_h_artifacts_for_kanfis(
    *,
    dataset_name: str,
    seed: int,
    model: DeepKANFISModel,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    output_dir: Path,
    classification_threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _collect_kanfis_importance_entries(model, train_inputs)
    if not entries:
        return

    selected_keys = [(str(item["source"]), int(item["rule_local_index"])) for item in entries]
    rule_rows = [
        {
            "dataset": dataset_name,
            "seed": int(seed),
            "column": int(column),
            "rank": int(column + 1),
            "rule_key": f'{item["source"]}::{item["rule_name"]}',
            "source": str(item["source"]),
            "rule_local_index": int(item["rule_local_index"]),
            "rule_name": str(item["rule_name"]),
            "importance": float(item["importance"]),
        }
        for column, item in enumerate(entries)
    ]
    _append_csv_rows(
        output_dir / "v18_h_rule_index.csv",
        [
            "dataset",
            "seed",
            "column",
            "rank",
            "rule_key",
            "source",
            "rule_local_index",
            "rule_name",
            "importance",
        ],
        rule_rows,
    )

    h_train = _extract_rule_feature_matrix(model, train_inputs, selected_keys)
    h_test = _extract_rule_feature_matrix(model, test_inputs, selected_keys)
    train_logits = _predict_logits_numpy(model, train_inputs)
    test_logits = _predict_logits_numpy(model, test_inputs)
    artifact_path = output_dir / f"v18_h_artifacts_{dataset_name}_seed{int(seed)}.npz"
    np.savez_compressed(
        artifact_path,
        h_train=h_train.astype(np.float32, copy=False),
        h_test=h_test.astype(np.float32, copy=False),
        y_train=train_targets.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=False),
        y_test=test_targets.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=False),
        full_train_logits=train_logits,
        full_test_logits=test_logits,
        full_train_prob=(1.0 / (1.0 + np.exp(-train_logits))).astype(np.float32, copy=False),
        full_test_prob=(1.0 / (1.0 + np.exp(-test_logits))).astype(np.float32, copy=False),
        importance=np.asarray([float(item["importance"]) for item in entries], dtype=np.float32),
        rule_key=np.asarray([f'{item["source"]}::{item["rule_name"]}' for item in entries]),
        classification_threshold=np.asarray([float(classification_threshold)], dtype=np.float32),
    )


def _export_v18_controls_for_kanfis(
    *,
    dataset_name: str,
    seed: int,
    model: DeepKANFISModel,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    test_inputs: torch.Tensor,
    test_targets: torch.Tensor,
    output_dir: Path,
    lr_budgets: tuple[int, ...],
    random_state: int,
) -> None:
    if train_inputs.ndim != 2 or test_inputs.ndim != 2:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _collect_kanfis_importance_entries(model, train_inputs)
    if not entries:
        return

    importance_rows = [
        {
            "dataset": dataset_name,
            "seed": int(seed),
            "rank": int(rank),
            "source": str(item["source"]),
            "rule_local_index": int(item["rule_local_index"]),
            "rule_name": str(item["rule_name"]),
            "importance": float(item["importance"]),
        }
        for rank, item in enumerate(entries, start=1)
    ]
    _append_csv_rows(
        output_dir / "tables_importance_profile.csv",
        ["dataset", "seed", "rank", "source", "rule_local_index", "rule_name", "importance"],
        importance_rows,
    )

    available_budgets = sorted({int(b) for b in lr_budgets if int(b) > 0 and int(b) <= len(entries)})
    lr_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    for budget in available_budgets:
        selected = entries[:budget]
        selected_keys = [(str(item["source"]), int(item["rule_local_index"])) for item in selected]
        x_train = _extract_rule_feature_matrix(model, train_inputs, selected_keys)
        x_test = _extract_rule_feature_matrix(model, test_inputs, selected_keys)
        y_train = train_targets.detach().cpu().reshape(-1).numpy()
        y_test = test_targets.detach().cpu().reshape(-1).numpy()
        y_train_bin = (y_train >= 0.5).astype(np.int64)
        y_test_bin = (y_test >= 0.5).astype(np.int64)

        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=int(random_state),
        )
        lr.fit(x_train, y_train_bin)
        p_test = lr.predict_proba(x_test)[:, 1]
        y_pred = (p_test >= 0.5).astype(np.int64)
        lr_rows.append(
            {
                "dataset": dataset_name,
                "seed": int(seed),
                "budget": int(budget),
                "method": "lr_topk_rules",
                "f1": float(f1_score(y_test_bin, y_pred)),
                "roc_auc": float(roc_auc_score(y_test_bin, p_test)),
                "pr_auc": float(average_precision_score(y_test_bin, p_test)),
                "n_features": int(x_train.shape[1]),
            }
        )

        if x_train.shape[1] >= 2:
            corr = np.corrcoef(x_train, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            tri = np.abs(corr[np.triu_indices(corr.shape[0], k=1)])
            if tri.size > 0:
                corr_rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": int(seed),
                        "subset": f"top_{budget}",
                        "rule_count": int(budget),
                        "mean_abs_corr": float(np.mean(tri)),
                        "median_abs_corr": float(np.median(tri)),
                        "p75_abs_corr": float(np.quantile(tri, 0.75)),
                        "p90_abs_corr": float(np.quantile(tri, 0.90)),
                    }
                )

        if budget == 400 and len(entries) >= 400:
            rng = np.random.default_rng(int(random_state) + int(seed))
            random_indices = rng.choice(len(entries), size=400, replace=False).tolist()
            random_keys = [
                (str(entries[idx]["source"]), int(entries[idx]["rule_local_index"]))
                for idx in random_indices
            ]
            x_rand = _extract_rule_feature_matrix(model, train_inputs, random_keys)
            if x_rand.shape[1] >= 2:
                corr = np.corrcoef(x_rand, rowvar=False)
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                tri = np.abs(corr[np.triu_indices(corr.shape[0], k=1)])
                if tri.size > 0:
                    corr_rows.append(
                        {
                            "dataset": dataset_name,
                            "seed": int(seed),
                            "subset": "random_400",
                            "rule_count": 400,
                            "mean_abs_corr": float(np.mean(tri)),
                            "median_abs_corr": float(np.median(tri)),
                            "p75_abs_corr": float(np.quantile(tri, 0.75)),
                            "p90_abs_corr": float(np.quantile(tri, 0.90)),
                        }
                    )

    _append_csv_rows(
        output_dir / "tables_lr_topk_baseline.csv",
        ["dataset", "seed", "budget", "method", "f1", "roc_auc", "pr_auc", "n_features"],
        lr_rows,
    )
    _append_csv_rows(
        output_dir / "tables_rule_activation_correlation.csv",
        ["dataset", "seed", "subset", "rule_count", "mean_abs_corr", "median_abs_corr", "p75_abs_corr", "p90_abs_corr"],
        corr_rows,
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
    dffl_concept_expert_labels: Mapping[str, float] | None,
    dffl_dataset_overrides: Mapping[str, Mapping[str, Any]] | None,
    feature_geometry: str,
    binary_heavy_grouping_mode: str,
    dffl_rule_swap_ratio: float | None,
    dffl_rule_swap_min_keep: int | None,
    dffl_adaptive_rule_swap: bool,
    dffl_total_rule_budget: int | None,
    dffl_bridge_score_interaction_weight: float | None,
    dffl_bridge_score_stability_weight: float | None,
    dffl_adaptive_budget: bool,
    dffl_adaptive_local_arity: bool,
    dffl_fast_gpu: bool,
    dffl_tiny_restarts: int,
    stacked_width_scale: float,
    stacked_rule_scale: float,
    stacked_stagewise_init: bool,
    stacked_stagewise_init_epochs: int,
    stacked_final_skip_input_count: int,
    stacked_final_skip_mode: str,
    stacked_final_skip_gates_enabled: bool,
    stacked_final_skip_gate_init_logit: float,
    stacked_final_skip_gate_l1_weight: float,
    hierarchical_width_scale: float,
    hierarchical_rule_scale: float,
    hierarchical_group_size: int,
    stacked_prototype_scoring_mode: str,
    hierarchical_prototype_scoring_mode: str,
    hidden_output_activation: str | None,
    fuzzy_binary_loss_name: str,
    fuzzy_binary_focal_gamma: float,
    fuzzy_binary_auto_pos_weight: bool,
    fuzzy_binary_soft_f1_weight: float,
    fuzzy_regression_loss: str,
    fuzzy_huber_delta: float,
    fuzzy_distill_weight: float,
    fuzzy_distill_models: tuple[str, ...],
    fuzzy_distill_teacher_trees: int,
    fuzzy_hard_sample_training: bool,
    fuzzy_hard_sample_models: tuple[str, ...],
    fuzzy_hard_sample_source: str,
    fuzzy_hard_sample_fraction: float,
    fuzzy_hard_sample_multiplier: int,
    fuzzy_hard_sample_finetune_epochs: int,
    fuzzy_hard_sample_finetune_patience: int,
    fuzzy_hard_sample_teacher_trees: int,
    kanfis_active_features: int,
    kanfis_depth: int,
    kanfis_superposition_terms: int,
    kanfis_concept_fan_in: int,
    kanfis_routing: str,
    kanfis_feature_order: str,
    kanfis_decision_skip_features: int,
    kanfis_pair_count: int,
    kanfis_projection_count: int,
    kanfis_projection_width: int,
    kanfis_projection_mode: str,
    kanfis_prune_rules: int,
    kanfis_importance_split: str,
    kanfis_recovery_epochs: int,
    kanfis_polish_epochs: int,
    kanfis_train_rule_gates: bool,
    kanfis_rule_sparsity_weight: float,
    kanfis_rule_entropy_weight: float,
    kanfis_rule_anchor_weight: float,
    kanfis_skip_gate_l1_weight: float,
    batch_size: int,
    patience: int,
    classification_threshold: float,
    rule_probability_threshold: float,
    tune_fuzzy_threshold: bool,
    tune_fuzzy_threshold_calibrated: bool,
    log_epochs: bool,
    log_epochs_every: int,
    dffl_one_phase: bool,
    device: str | None,
    fuzzy_models: tuple[str, ...] = FUZZY_MODEL_NAMES,
    include_sklearn: bool = True,
    v18_controls_dir: Path | None = None,
    v18_lr_budgets: tuple[int, ...] = (100, 200, 400),
    v18_random_state: int = 42,
    v18_export_h_artifacts: bool = False,
):
    run_rss_start_mb = _read_current_rss_mb()
    feature_geometry_requested = _validate_feature_geometry(feature_geometry)
    binary_heavy_grouping_mode = _validate_binary_heavy_grouping_mode(binary_heavy_grouping_mode)
    stage_runtime: dict[str, float | None] = {
        "full_train_sec": None,
        "h_build_sec": None,
        "selection_sec": None,
        "refit_head_sec": None,
        "inference_ms_per_sample": None,
        "memory_peak_mb": None,
    }
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

    bootstrap_inputs, bootstrap_targets = _select_bootstrap_subset(
        train_inputs,
        train_targets,
        seed=seed,
    )
    if int(bootstrap_inputs.size(0)) < int(train_inputs.size(0)):
        progress_log(
            "seed={seed} bootstrap_subset: {used}/{total} samples for prototype initialization".format(
                seed=seed,
                used=int(bootstrap_inputs.size(0)),
                total=int(train_inputs.size(0)),
            )
        )

    model_train_targets: dict[str, torch.Tensor] = {model_name: train_targets for model_name in FUZZY_MODEL_NAMES}
    distillation_metadata: dict[str, Any] = {}
    if (
        spec.task_type == "binary_classification"
        and fuzzy_distill_weight > 0.0
        and len(fuzzy_distill_models) > 0
    ):
        blended_targets, dist_meta = _build_binary_distillation_targets(
            train_inputs=train_inputs,
            train_targets=train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            seed=seed,
            teacher_trees=fuzzy_distill_teacher_trees,
            blend_weight=fuzzy_distill_weight,
        )
        distillation_metadata = dict(dist_meta)
        applied_models: list[str] = []
        for model_name in fuzzy_distill_models:
            if model_name in model_train_targets:
                model_train_targets[model_name] = blended_targets
                applied_models.append(model_name)
        if applied_models:
            progress_log(
                (
                    "seed={seed} distill: teacher=extra_trees trees={trees} "
                    "blend={blend:.2f} models={models} teacher_val_f1={f1:.4f}"
                ).format(
                    seed=seed,
                    trees=fuzzy_distill_teacher_trees,
                    blend=fuzzy_distill_weight,
                    models=",".join(applied_models),
                    f1=float(distillation_metadata.get("teacher_val_f1", float("nan"))),
                )
            )

    hard_sample_enabled_models = set(fuzzy_hard_sample_models)
    baseline_hard_indices: torch.Tensor | None = None
    if fuzzy_hard_sample_training and fuzzy_hard_sample_source.strip().lower() == "baseline":
        baseline_hard_indices = _compute_baseline_hard_indices(
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=train_targets,
            fraction=fuzzy_hard_sample_fraction,
            seed=seed,
            teacher_trees=fuzzy_hard_sample_teacher_trees,
        )
        if baseline_hard_indices is not None:
            progress_log(
                "seed={seed} hard_source=baseline hard_count={hard}".format(
                    seed=seed,
                    hard=int(baseline_hard_indices.numel()),
                )
            )

    dataset_override_entry = _get_dataset_override_entry(
        dataset_overrides=dffl_dataset_overrides,
        dataset_name=spec.name,
    )
    threshold_tuning_strategy = _resolve_dataset_threshold_tuning_strategy(
        dataset_name=spec.name,
        dataset_override_entry=dataset_override_entry,
    )
    binary_loss_params = (
        _resolve_dataset_binary_loss_params(
            dataset_name=spec.name,
            dataset_override_entry=dataset_override_entry,
        )
        if spec.task_type == "binary_classification"
        else {}
    )
    regression_residual_head_enabled = bool(spec.task_type == "regression" and split.n_samples > 100)
    amp_enabled = bool(
        torch.cuda.is_available()
        and (
            (device is not None and str(device).strip().lower().startswith("cuda"))
            or (device is None)
        )
    )

    trained_fuzzy_models: dict[str, torch.nn.Module] = {}
    model_top_k_rules: dict[str, int | None] = {}
    model_prediction_postprocessors: dict[str, Callable[[torch.Tensor], torch.Tensor] | None] = {}

    if "ruanfis_refined_deep" in fuzzy_models:
        analysis_train_max_samples = 80_000
        analysis_validation_max_samples = 50_000
        if int(split.input_dim) >= 96:
            # High-dimensional interaction probes scale as O(n * d^2) and can trigger host OOM.
            # Cap the structure-analysis subset while keeping the actual model training unchanged.
            analysis_train_max_samples = 20_000
            analysis_validation_max_samples = 12_000
        elif int(split.input_dim) >= 64:
            analysis_train_max_samples = 40_000
            analysis_validation_max_samples = 25_000

        analysis_source_inputs, analysis_source_targets = _select_bootstrap_subset(
            train_inputs,
            train_targets,
            max_samples=analysis_train_max_samples,
            seed=seed + 17,
        )
        analysis_validation_source_inputs, analysis_validation_source_targets = _select_bootstrap_subset(
            validation_inputs,
            validation_targets,
            max_samples=analysis_validation_max_samples,
            seed=seed + 31,
        )
        if int(analysis_source_inputs.size(0)) < int(train_inputs.size(0)):
            progress_log(
                "seed={seed} dffl_structure_subset: train={train_used}/{train_total}, val={val_used}/{val_total}".format(
                    seed=seed,
                    train_used=int(analysis_source_inputs.size(0)),
                    train_total=int(train_inputs.size(0)),
                    val_used=int(analysis_validation_source_inputs.size(0)),
                    val_total=int(validation_inputs.size(0)),
                )
            )
        analysis_device_request = device
        if (
            analysis_device_request is not None
            and str(analysis_device_request).strip().lower().startswith("cuda")
            and int(split.input_dim) >= 96
        ):
            # High-dimensional pairwise interaction probes can exhaust VRAM on mid-range GPUs.
            # Keep model training on CUDA, but run structure-analysis tensors on CPU.
            analysis_device_request = "cpu"
            progress_log(
                "seed={seed} dffl_structure_analysis_device_override: cpu (input_dim={dim})".format(
                    seed=seed,
                    dim=int(split.input_dim),
                )
            )

        analysis_train_inputs, analysis_train_targets, analysis_device = _prepare_structure_analysis_tensors(
            inputs=analysis_source_inputs,
            targets=analysis_source_targets,
            device=analysis_device_request,
        )
        analysis_validation_inputs, analysis_validation_targets, _ = _prepare_structure_analysis_tensors(
            inputs=analysis_validation_source_inputs,
            targets=analysis_validation_source_targets,
            device=analysis_device_request,
        )
        dffl_profile = resolve_dffl_profile(
            profile_name=dffl_profile_name,
            task_type=spec.task_type,
            n_samples=split.n_samples,
            input_dim=split.input_dim,
        )
        dffl_profile, dataset_profile_policy = _resolve_dataset_profile_override(
            dffl_profile,
            dataset_name=spec.name,
            dataset_overrides=dffl_dataset_overrides,
        )
        dffl_profile = apply_dffl_profile_overrides(
            dffl_profile,
            rule_swap_ratio=dffl_rule_swap_ratio,
            rule_swap_min_keep=dffl_rule_swap_min_keep,
            total_rule_budget=dffl_total_rule_budget,
            bridge_score_interaction_weight=dffl_bridge_score_interaction_weight,
            bridge_score_stability_weight=dffl_bridge_score_stability_weight,
        )
        if _dataset_override_skip_builtin_budget_policy(
            dataset_name=spec.name,
            dataset_overrides=dffl_dataset_overrides,
        ):
            dffl_dataset_budget_policy = "skipped_by_dataset_override"
        else:
            dffl_profile, dffl_dataset_budget_policy = apply_dataset_budget_reallocation(
                dffl_profile,
                dataset_name=spec.name,
                task_type=spec.task_type,
                input_dim=split.input_dim,
                total_budget_locked=(dffl_total_rule_budget is not None),
            )
        dffl_profile, dataset_field_policy = _apply_dataset_profile_field_overrides(
            dffl_profile,
            dataset_name=spec.name,
            dataset_overrides=dffl_dataset_overrides,
            total_budget_locked=(dffl_total_rule_budget is not None),
        )
        dffl_fast_gpu_policy = "none"
        if dffl_fast_gpu:
            dffl_profile, dffl_fast_gpu_policy = apply_dffl_fast_gpu_profile(
                dffl_profile,
                task_type=spec.task_type,
                input_dim=split.input_dim,
                n_samples=split.n_samples,
                total_budget_locked=(dffl_total_rule_budget is not None),
            )
        policy_parts = [
            part
            for part in (
                dffl_dataset_budget_policy,
                dataset_profile_policy,
                dataset_field_policy,
                dffl_fast_gpu_policy,
            )
            if part != "none"
        ]
        dffl_dataset_budget_policy = "+".join(policy_parts) if policy_parts else "none"
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
            validation_inputs=analysis_validation_inputs,
            validation_targets=analysis_validation_targets,
            input_groups=dffl_input_groups,
            feature_geometry=feature_geometry_effective,
            adaptive_budget_enabled=dffl_adaptive_budget,
        )
        dffl_residual_features = make_dffl_residual_features(
            dffl_profile,
            train_inputs=analysis_train_inputs,
            train_targets=analysis_train_targets,
            input_groups=dffl_input_groups,
            validation_inputs=analysis_validation_inputs,
            validation_targets=analysis_validation_targets,
        )
        dffl_intergroup_share = estimate_intergroup_interaction_share(
            train_inputs=analysis_train_inputs,
            train_targets=analysis_train_targets,
            input_groups=dffl_input_groups,
        )
        dffl_component_marginal_gains = _estimate_dffl_component_marginal_gains(
            task_type=spec.task_type,
            train_inputs=analysis_train_inputs,
            train_targets=analysis_train_targets,
            validation_inputs=analysis_validation_inputs,
            validation_targets=analysis_validation_targets,
            input_groups=dffl_input_groups,
            bridge_pairs=dffl_bridge_pairs,
        )
        dffl_adaptive_local_arity_effective, dffl_adaptive_local_arity_policy = (
            _resolve_adaptive_local_arity_policy(
                enabled=dffl_adaptive_local_arity,
                task_type=spec.task_type,
                n_samples=split.n_samples,
                input_dim=split.input_dim,
            )
        )
        dffl_high_arity_groups = (
            _select_adaptive_high_arity_groups(
                train_inputs=analysis_train_inputs,
                train_targets=analysis_train_targets,
                input_groups=dffl_input_groups,
                input_dim=split.input_dim,
            )
            if dffl_adaptive_local_arity_effective
            else ()
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
            component_marginal_gains=dffl_component_marginal_gains,
        )
        dffl_adaptive_rule_swap_effective = bool(dffl_adaptive_rule_swap)
        dffl_adaptive_rule_swap_policy = "enabled"
        if dffl_rule_swap_ratio is None and spec.name == "breast_cancer":
            dffl_adaptive_rule_swap_effective = False
            dffl_adaptive_rule_swap_policy = "disabled_for_breast_policy"
        dffl_rule_swap_ratio_effective = float(dffl_profile.stagewise_rule_swap_ratio)
        dffl_rule_swap_reason = "profile_default"
        if dffl_rule_swap_ratio is not None:
            dffl_rule_swap_reason = "cli_override"
        elif dffl_adaptive_rule_swap_effective:
            dffl_rule_swap_ratio_effective, dffl_rule_swap_reason = resolve_adaptive_rule_swap_ratio(
                base_ratio=dffl_rule_swap_ratio_effective,
                input_dim=split.input_dim,
                intergroup_interaction_share=dffl_intergroup_share,
            )
        dffl_rule_swap_min_keep_effective = (
            int(dffl_profile.stagewise_rule_swap_min_keep) if dffl_rule_swap_ratio_effective > 0.0 else 0
        )
        dffl_shuffle_effective, dffl_shuffle_policy = resolve_dffl_shuffle_policy(
            task_type=spec.task_type,
            input_dim=split.input_dim,
            n_samples=split.n_samples,
            intergroup_interaction_share=dffl_intergroup_share,
        )
        is_covtype_binary = _is_covtype_binary_dataset(spec.name)
        dffl_binary_feature_indices = _detect_binary_feature_indices(analysis_train_inputs)
        if dffl_profile.final_skip_input_indices:
            dffl_final_skip_indices = tuple(int(index) for index in dffl_profile.final_skip_input_indices)
        elif int(dffl_profile.final_skip_input_count) > 0:
            dffl_final_skip_indices = _resolve_stacked_final_skip_indices(
                train_inputs=analysis_train_inputs,
                train_targets=analysis_train_targets,
                count=int(dffl_profile.final_skip_input_count),
                mode=str(dffl_profile.final_skip_mode),
            )
        else:
            dffl_final_skip_indices = ()
        if dffl_final_skip_indices:
            progress_log(
                "seed={seed} model=dffl: final_skip mode={mode} indices={indices}".format(
                    seed=seed,
                    mode=dffl_profile.final_skip_mode,
                    indices=",".join(str(index) for index in dffl_final_skip_indices),
                )
            )
        progress_log(
            (
                "seed={seed} profile: dffl={name}, task={task}, device={device}, grouping={grouping}, "
                "analysis_device={analysis_device}, "
                "feature_geometry={feature_geometry}, feature_geometry_policy={feature_geometry_policy}, binary_heavy_grouping={binary_mode}, "
                "groups={groups}, bridges={bridges}, intergroup_share={share:.3f}, "
                "residual_features={residuals}, "
                "dataset_budget_policy={dataset_budget_policy}, bridge_val_gain_w={bridge_val_gain_w:.3f}, "
                "adaptive_local_arity={adaptive_local_arity}, adaptive_local_arity_policy={adaptive_local_arity_policy}, high_arity_groups={high_arity_groups}, "
                "marginal_gain(local={mg_local:.2f}, bridge={mg_bridge:.2f}, aggregate={mg_aggregate:.2f}, decision={mg_decision:.2f}), "
                "bridge_concepts_effective={bridge_concepts}, "
                "rule_swap={swap:.3f}, swap_keep={keep}, swap_policy={policy}, "
                "adaptive_rule_swap={adaptive_rule_swap}, adaptive_rule_swap_policy={adaptive_rule_swap_policy}, "
                "budget_total={budget_total}, budget_policy={budget_policy}, adaptive_budget={adaptive_budget}, "
                "shuffle={shuffle}, shuffle_policy={shuffle_policy}"
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
                residuals=len(dffl_residual_features),
                dataset_budget_policy=dffl_dataset_budget_policy,
                bridge_val_gain_w=float(dffl_profile.bridge_validation_gain_weight),
                adaptive_local_arity=dffl_adaptive_local_arity_effective,
                adaptive_local_arity_policy=dffl_adaptive_local_arity_policy,
                high_arity_groups=len(dffl_high_arity_groups),
                mg_local=float(dffl_component_marginal_gains.get("local", 1.0)),
                mg_bridge=float(dffl_component_marginal_gains.get("bridge", 1.0)),
                mg_aggregate=float(dffl_component_marginal_gains.get("aggregate", 1.0)),
                mg_decision=float(dffl_component_marginal_gains.get("decision", 1.0)),
                bridge_concepts=int(dffl_bridge_concepts_effective),
                swap=dffl_rule_swap_ratio_effective,
                keep=dffl_rule_swap_min_keep_effective,
                policy=dffl_rule_swap_reason,
                adaptive_rule_swap=dffl_adaptive_rule_swap_effective,
                adaptive_rule_swap_policy=dffl_adaptive_rule_swap_policy,
                budget_total=int(dffl_budget_snapshot["total_rule_budget_effective"]),
                budget_policy=str(dffl_budget_snapshot["allocation_policy"]),
                adaptive_budget=dffl_adaptive_budget,
                shuffle=dffl_shuffle_effective,
                shuffle_policy=dffl_shuffle_policy,
            )
        )
        dffl_learning_rate_effective = (
            dffl_learning_rate * dffl_profile.learning_rate_scale_classification
            if spec.task_type == "binary_classification"
            else dffl_learning_rate * dffl_profile.learning_rate_scale_regression
        )
        dffl_refinement_cycles = max(refinement_cycles, dffl_profile.refinement_cycle_floor)
        dffl_train_targets = model_train_targets["ruanfis_refined_deep"]
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
            residual_feature_indices=dffl_residual_features,
            high_arity_group_indices=dffl_high_arity_groups,
            intergroup_interaction_share=dffl_intergroup_share,
            adaptive_budget_enabled=dffl_adaptive_budget,
            adaptive_local_arity_enabled=dffl_adaptive_local_arity_effective,
            component_marginal_gains=dffl_component_marginal_gains,
            final_skip_input_indices=dffl_final_skip_indices,
        )
        dffl_block_agreement_weight, dffl_block_agreement_target_corr = _resolve_dffl_block_agreement_params(
            profile=dffl_profile,
            intergroup_interaction_share=dffl_intergroup_share,
            n_bridge_pairs=len(dffl_bridge_pairs),
        )
        dffl_regularizer_cadence = 1
        if int(split.input_dim) >= 40 or int(split.n_samples) >= 100_000:
            # Reduce overhead of expensive correlation/usage regularizers on large-scale runs.
            dffl_regularizer_cadence = 4
        dffl_patience_effective = min(patience, max_epochs)
        dffl_min_delta = 0.0
        if spec.task_type == "binary_classification" and is_covtype_binary:
            # Covtype is noisy under long tails of small F1 oscillations.
            # Slightly stricter early stopping improves consistency.
            dffl_patience_effective = min(dffl_patience_effective, max(8, max_epochs // 2))
            dffl_min_delta = 5e-4
        dffl_training_config_base = TrainingConfig(
            task_type=spec.task_type,
            max_epochs=max_epochs,
            learning_rate=dffl_learning_rate_effective,
            patience=dffl_patience_effective,
            min_delta=dffl_min_delta,
            batch_size=batch_size,
            shuffle=dffl_shuffle_effective,
            amp_enabled=amp_enabled,
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
            binary_loss_name=str(binary_loss_params.get("binary_loss_name", "bce")),
            binary_focal_gamma=float(binary_loss_params.get("binary_focal_gamma", 2.0)),
            binary_focal_alpha=(
                float(binary_loss_params["binary_focal_alpha"])
                if binary_loss_params.get("binary_focal_alpha") is not None
                else None
            ),
            binary_class_balanced_beta=(
                float(binary_loss_params["binary_class_balanced_beta"])
                if binary_loss_params.get("binary_class_balanced_beta") is not None
                else None
            ),
            regression_linear_residual_head=regression_residual_head_enabled,
            weight_decay=dffl_profile.weight_decay,
            gradient_clip_norm=dffl_profile.gradient_clip_norm,
            rule_sparsity_weight=dffl_profile.rule_sparsity_weight,
            rule_length_weight=dffl_profile.rule_length_weight,
            decision_usage_balance_weight=dffl_profile.decision_usage_balance_weight,
            decision_usage_balance_every_n_steps=dffl_regularizer_cadence,
            block_gate_l1_weight=dffl_profile.block_gate_l1_weight,
            rule_activation_entropy_weight=dffl_profile.rule_activation_entropy_weight,
            rule_anchor_stability_weight=dffl_profile.rule_anchor_stability_weight,
            block_agreement_weight=dffl_block_agreement_weight,
            block_agreement_every_n_steps=dffl_regularizer_cadence,
            block_agreement_target_corr=dffl_block_agreement_target_corr,
            block_agreement_stage_limit=dffl_profile.block_agreement_stage_limit,
            regularization_warmup_epochs=max(1, max_epochs // 3),
            top_k_warmup_epochs=(max(1, max_epochs // 4) if dffl_profile.top_k_rules is not None else 0),
            prune_after_fit=False,
            verbose=bool(log_epochs),
            log_every_n_epochs=max(1, int(log_epochs_every)),
            log_prefix=f"seed={seed} model=dffl",
            device=device,
        )
        if spec.task_type == "binary_classification" and spec.name == "breast_cancer":
            focal_candidates = _resolve_breast_focal_candidates(
                dataset_name=spec.name,
                task_type=spec.task_type,
                train_targets=train_targets,
                default_gamma=float(binary_loss_params.get("binary_focal_gamma", 2.0)),
                default_alpha=(
                    float(binary_loss_params["binary_focal_alpha"])
                    if binary_loss_params.get("binary_focal_alpha") is not None
                    else None
                ),
            )
            dffl_training_candidates = tuple(
                replace(
                    dffl_training_config_base,
                    binary_loss_name="focal",
                    binary_focal_gamma=float(gamma),
                    binary_focal_alpha=(None if alpha is None else float(alpha)),
                )
                for gamma, alpha in focal_candidates
            )
        else:
            dffl_training_candidates = (dffl_training_config_base,)
        tiny_cls = (
            spec.task_type == "binary_classification"
            and split.n_samples <= 300
            and split.input_dim <= 20
        )
        restarts_effective = max(1, int(dffl_tiny_restarts)) if tiny_cls else 1
        if "restarts" in dataset_override_entry:
            restarts_effective = max(restarts_effective, int(dataset_override_entry["restarts"]))
        if tiny_cls and "tiny_restarts" in dataset_override_entry:
            restarts_effective = max(1, int(dataset_override_entry["tiny_restarts"]))
        best_model: torch.nn.Module | None = None
        best_val_score = float("-inf")
        best_restart = 0
        best_prediction_postprocessor: Callable[[torch.Tensor], torch.Tensor] | None = None
        best_training_config: TrainingConfig | None = None
        best_focal_spec = "n/a"
        best_score_label = "validation_score"
        for focal_idx, candidate_training_config in enumerate(dffl_training_candidates):
            focal_spec = (
                f"gamma={candidate_training_config.binary_focal_gamma:.2f},"
                f"alpha={candidate_training_config.binary_focal_alpha if candidate_training_config.binary_focal_alpha is not None else 'auto'}"
            )
            for restart_idx in range(restarts_effective):
                restart_seed = seed + 1009 * restart_idx + 10007 * focal_idx
                set_seed(restart_seed)
                if dffl_one_phase:
                    candidate_model = build_bootstrapped_hierarchical_model(
                        dffl_config,
                        sample_inputs=train_inputs,
                        sample_targets=dffl_train_targets,
                        bootstrap_config=dffl_bootstrap_config,
                        device=device,
                    )
                    candidate_model.concept_dropout_rate = float(dffl_profile.concept_dropout_rate)
                    dffl_trainer = FuzzyTrainer(candidate_model, candidate_training_config)
                    dffl_trainer.fit(train_inputs, dffl_train_targets, validation_inputs, validation_targets)
                else:
                    dffl_result = build_refined_hierarchical_model(
                        dffl_config,
                        train_inputs=train_inputs,
                        train_targets=dffl_train_targets,
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
                            shuffle=dffl_shuffle_effective,
                            rule_sparsity_weight=0.0,
                            stage_selection_metric="auto",
                            stage_selection_threshold=classification_threshold,
                            rule_swap_ratio=dffl_rule_swap_ratio_effective,
                            rule_swap_min_keep=dffl_rule_swap_min_keep_effective,
                        ),
                        training_config=candidate_training_config,
                        refinement_loop_config=RefinementLoopConfig(
                            max_cycles=dffl_refinement_cycles,
                            patience=1,
                            min_delta=1e-4,
                        ),
                    )
                    candidate_model = dffl_result.model
                    candidate_model.concept_dropout_rate = float(dffl_profile.concept_dropout_rate)

                candidate_postprocessor: Callable[[torch.Tensor], torch.Tensor] | None = None
                calibration_method = "none"
                if spec.task_type == "binary_classification":
                    candidate_model.eval()
                    val_logits_raw = _predict_logits_batched(
                        candidate_model,
                        validation_inputs,
                        top_k_rules=dffl_profile.top_k_rules,
                    )

                    candidate_calibrator = _fit_binary_probability_calibrator(
                        val_logits_raw,
                        validation_targets.detach().cpu(),
                    )
                    candidate_postprocessor = _build_binary_logit_postprocessor(candidate_calibrator)
                    if candidate_calibrator is not None:
                        calibration_method = str(candidate_calibrator.get("method", "unknown"))
                    val_logits = (
                        candidate_postprocessor(val_logits_raw)
                        if candidate_postprocessor is not None
                        else val_logits_raw
                    )

                    if spec.name == "breast_cancer":
                        val_metrics = compute_metrics(
                            "binary_classification",
                            val_logits,
                            validation_targets.detach().cpu(),
                            classification_threshold=classification_threshold,
                        )
                        val_score = float(val_metrics["roc_auc"] - val_metrics["log_loss"])
                        score_label = "roc_auc-log_loss"
                    else:
                        val_threshold = _choose_best_classification_threshold(
                            candidate_model,
                            validation_inputs=validation_inputs,
                            validation_targets=validation_targets,
                            default_threshold=classification_threshold,
                            top_k_rules=dffl_profile.top_k_rules,
                            strategy=threshold_tuning_strategy,
                            prediction_postprocessor=candidate_postprocessor,
                        )
                        val_metrics = compute_metrics(
                            "binary_classification",
                            val_logits,
                            validation_targets.detach().cpu(),
                            classification_threshold=val_threshold,
                        )
                        val_score = float(val_metrics["f1"])
                        score_label = "f1"
                else:
                    candidate_model.eval()
                    with torch.no_grad():
                        val_predictions = predict_with_optional_residual_head(
                            candidate_model,
                            validation_inputs.to(
                                device=next(candidate_model.parameters()).device,
                                dtype=torch.float32,
                            ),
                            top_k_rules=dffl_profile.top_k_rules,
                        ).detach().cpu()
                    val_metrics = compute_metrics(
                        "regression",
                        val_predictions,
                        validation_targets.detach().cpu(),
                    )
                    val_score = -float(val_metrics["rmse"])
                    score_label = "-rmse"

                progress_log(
                    (
                        "seed={seed} model=dffl focal={focal} restart={restart}/{total}: "
                        "{score_label}={score:.4f}, calibration={calibration}"
                    ).format(
                        seed=seed,
                        focal=focal_spec,
                        restart=restart_idx + 1,
                        total=restarts_effective,
                        score_label=score_label,
                        score=val_score,
                        calibration=calibration_method,
                    )
                )
                if val_score > best_val_score + 1e-12:
                    best_val_score = val_score
                    best_model = candidate_model
                    best_restart = restart_idx + 1
                    best_prediction_postprocessor = candidate_postprocessor
                    best_training_config = candidate_training_config
                    best_focal_spec = focal_spec
                    best_score_label = score_label
        if best_model is None or best_training_config is None:
            raise RuntimeError("DFFL training did not produce a valid model.")
        if restarts_effective > 1 or len(dffl_training_candidates) > 1:
            progress_log(
                (
                    "seed={seed} model=dffl selected focal={focal} restart={restart}/{restarts} "
                    "with {score_label}={score:.4f}"
                ).format(
                    seed=seed,
                    focal=best_focal_spec,
                    restart=best_restart,
                    restarts=restarts_effective,
                    score_label=best_score_label,
                    score=best_val_score,
                )
            )
        if (
            fuzzy_hard_sample_training
            and "ruanfis_refined_deep" in hard_sample_enabled_models
            and _is_covtype_binary_dataset(spec.name)
        ):
            progress_log(
                "seed={seed} model=dffl: hard_finetune disabled for covtype policy".format(seed=seed)
            )
        _maybe_run_hard_sample_finetune(
            enabled=(
                fuzzy_hard_sample_training
                and "ruanfis_refined_deep" in hard_sample_enabled_models
                and not _is_covtype_binary_dataset(spec.name)
            ),
            model_name="ruanfis_refined_deep",
            model=best_model,
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=dffl_train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            base_training_config=best_training_config,
            source=fuzzy_hard_sample_source,
            hard_fraction=fuzzy_hard_sample_fraction,
            hard_multiplier=fuzzy_hard_sample_multiplier,
            finetune_epochs=fuzzy_hard_sample_finetune_epochs,
            finetune_patience=fuzzy_hard_sample_finetune_patience,
            baseline_hard_indices=baseline_hard_indices,
            seed=seed,
        )
        trained_fuzzy_models["ruanfis_refined_deep"] = best_model
        model_top_k_rules["ruanfis_refined_deep"] = dffl_profile.top_k_rules
        model_prediction_postprocessors["ruanfis_refined_deep"] = best_prediction_postprocessor
        # Keep non-DFFL models deterministic with the original seed.
        set_seed(seed)
        progress_log(f"seed={seed} model=dffl: done in {time.perf_counter() - phase_started_at:.2f}s")

    if "ruanfis_shallow" in fuzzy_models:
        phase_started_at = time.perf_counter()
        progress_log(f"seed={seed} model=shallow: bootstrap+train start")
        shallow_model = build_bootstrapped_shallow_model(
            build_shallow_config(split.input_dim),
            sample_inputs=bootstrap_inputs,
            sample_targets=bootstrap_targets,
            bootstrap_config=BootstrapConfig(decision_task_type=spec.task_type),
            device=device,
        )
        shallow_training_config = TrainingConfig(
            task_type=spec.task_type,
            max_epochs=max_epochs,
            learning_rate=fuzzy_learning_rate,
            patience=min(patience, max_epochs),
            batch_size=batch_size,
            shuffle=True,
            amp_enabled=amp_enabled,
            classification_threshold=classification_threshold,
            binary_auto_pos_weight=spec.task_type == "binary_classification" and fuzzy_binary_auto_pos_weight,
            binary_loss_name=fuzzy_binary_loss_name,
            binary_focal_gamma=fuzzy_binary_focal_gamma,
            binary_soft_f1_weight=(
                fuzzy_binary_soft_f1_weight if spec.task_type == "binary_classification" else 0.0
            ),
            regression_loss=fuzzy_regression_loss,
            huber_delta=fuzzy_huber_delta,
            regression_linear_residual_head=regression_residual_head_enabled,
            monitor_metric="f1" if spec.task_type == "binary_classification" else None,
            monitor_mode="max" if spec.task_type == "binary_classification" else None,
            verbose=bool(log_epochs),
            log_every_n_epochs=max(1, int(log_epochs_every)),
            log_prefix=f"seed={seed} model=shallow",
            device=device,
        )
        shallow_trainer = FuzzyTrainer(shallow_model, shallow_training_config)
        shallow_trainer.fit(
            train_inputs,
            model_train_targets["ruanfis_shallow"],
            validation_inputs,
            validation_targets,
        )
        _maybe_run_hard_sample_finetune(
            enabled=fuzzy_hard_sample_training and "ruanfis_shallow" in hard_sample_enabled_models,
            model_name="ruanfis_shallow",
            model=shallow_model,
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=model_train_targets["ruanfis_shallow"],
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            base_training_config=shallow_training_config,
            source=fuzzy_hard_sample_source,
            hard_fraction=fuzzy_hard_sample_fraction,
            hard_multiplier=fuzzy_hard_sample_multiplier,
            finetune_epochs=fuzzy_hard_sample_finetune_epochs,
            finetune_patience=fuzzy_hard_sample_finetune_patience,
            baseline_hard_indices=baseline_hard_indices,
            seed=seed,
        )
        trained_fuzzy_models["ruanfis_shallow"] = shallow_model
        model_top_k_rules["ruanfis_shallow"] = None
        model_prediction_postprocessors["ruanfis_shallow"] = None
        progress_log(f"seed={seed} model=shallow: done in {time.perf_counter() - phase_started_at:.2f}s")

    if "ruanfis_stacked_anfis" in fuzzy_models:
        phase_started_at = time.perf_counter()
        progress_log(f"seed={seed} model=stacked: build+train start")
        stacked_final_skip_indices = _resolve_stacked_final_skip_indices(
            train_inputs=train_inputs,
            train_targets=train_targets,
            count=stacked_final_skip_input_count,
            mode=stacked_final_skip_mode,
        )
        if stacked_final_skip_indices:
            progress_log(
                "seed={seed} model=stacked: final_skip mode={mode} indices={indices}".format(
                    seed=seed,
                    mode=stacked_final_skip_mode,
                    indices=",".join(str(index) for index in stacked_final_skip_indices),
                )
            )
        stacked_config = build_stacked_config(
            split.input_dim,
            width_scale=stacked_width_scale,
            rule_scale=stacked_rule_scale,
            prototype_scoring_mode=stacked_prototype_scoring_mode,
            hidden_output_activation=hidden_output_activation,
            final_skip_input_count=stacked_final_skip_input_count,
            final_skip_input_indices=stacked_final_skip_indices,
            final_skip_gates_enabled=stacked_final_skip_gates_enabled,
            final_skip_gate_init_logit=stacked_final_skip_gate_init_logit,
        )
        if stacked_stagewise_init:
            stacked_model = build_stagewise_initialized_stacked_anfis_model(
                stacked_config,
                sample_inputs=bootstrap_inputs,
                sample_targets=bootstrap_targets,
                task_type=spec.task_type,
                epochs_per_hidden_layer=stacked_stagewise_init_epochs,
                learning_rate=fuzzy_learning_rate,
                batch_size=batch_size,
                shuffle=True,
                device=device,
            )
        else:
            stacked_model = build_stacked_anfis_model(
                stacked_config,
                sample_inputs=bootstrap_inputs,
            )
        stacked_training_config = TrainingConfig(
            task_type=spec.task_type,
            max_epochs=max_epochs,
            learning_rate=fuzzy_learning_rate,
            patience=min(patience, max_epochs),
            batch_size=batch_size,
            shuffle=True,
            amp_enabled=amp_enabled,
            classification_threshold=classification_threshold,
            binary_auto_pos_weight=spec.task_type == "binary_classification" and fuzzy_binary_auto_pos_weight,
            binary_loss_name=fuzzy_binary_loss_name,
            binary_focal_gamma=fuzzy_binary_focal_gamma,
            binary_soft_f1_weight=(
                fuzzy_binary_soft_f1_weight if spec.task_type == "binary_classification" else 0.0
            ),
            regression_loss=fuzzy_regression_loss,
            huber_delta=fuzzy_huber_delta,
            regression_linear_residual_head=regression_residual_head_enabled,
            monitor_metric="f1" if spec.task_type == "binary_classification" else None,
            monitor_mode="max" if spec.task_type == "binary_classification" else None,
            block_gate_l1_weight=max(0.0, float(stacked_final_skip_gate_l1_weight)),
            verbose=bool(log_epochs),
            log_every_n_epochs=max(1, int(log_epochs_every)),
            log_prefix=f"seed={seed} model=stacked",
            device=device,
        )
        stacked_trainer = FuzzyTrainer(stacked_model, stacked_training_config)
        stacked_trainer.fit(
            train_inputs,
            model_train_targets["ruanfis_stacked_anfis"],
            validation_inputs,
            validation_targets,
        )
        _maybe_run_hard_sample_finetune(
            enabled=fuzzy_hard_sample_training and "ruanfis_stacked_anfis" in hard_sample_enabled_models,
            model_name="ruanfis_stacked_anfis",
            model=stacked_model,
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=model_train_targets["ruanfis_stacked_anfis"],
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            base_training_config=stacked_training_config,
            source=fuzzy_hard_sample_source,
            hard_fraction=fuzzy_hard_sample_fraction,
            hard_multiplier=fuzzy_hard_sample_multiplier,
            finetune_epochs=fuzzy_hard_sample_finetune_epochs,
            finetune_patience=fuzzy_hard_sample_finetune_patience,
            baseline_hard_indices=baseline_hard_indices,
            seed=seed,
        )
        trained_fuzzy_models["ruanfis_stacked_anfis"] = stacked_model
        model_top_k_rules["ruanfis_stacked_anfis"] = None
        model_prediction_postprocessors["ruanfis_stacked_anfis"] = None
        progress_log(f"seed={seed} model=stacked: done in {time.perf_counter() - phase_started_at:.2f}s")

    if "ruanfis_kanfis" in fuzzy_models:
        phase_started_at = time.perf_counter()
        progress_log(f"seed={seed} model=kanfis: build+train start")
        kanfis_active_feature_indices = _make_kanfis_active_feature_indices(
            train_inputs,
            train_targets,
            max_features=split.input_dim if kanfis_active_features <= 0 else min(kanfis_active_features, split.input_dim),
            feature_importances=(
                distillation_metadata.get("teacher_feature_importances")
                if str(kanfis_feature_order).strip().lower() == "teacher_importance"
                else None
            ),
        )
        kanfis_pair_indices = _make_kanfis_pair_indices(
            train_inputs,
            train_targets,
            max_pairs=max(0, int(kanfis_pair_count)),
        )
        kanfis_projection_indices = _make_kanfis_projection_indices(
            train_inputs,
            train_targets,
            projection_count=max(0, int(kanfis_projection_count)),
            projection_width=max(1, int(kanfis_projection_width)),
            mode=str(kanfis_projection_mode),
        )
        progress_log(
            "seed={seed} model=kanfis: active_features={features}".format(
                seed=seed,
                features=",".join(str(index) for index in kanfis_active_feature_indices),
            )
        )
        if kanfis_pair_indices:
            progress_log(
                "seed={seed} model=kanfis: pair_channels={pairs}".format(
                    seed=seed,
                    pairs=",".join(f"{left}-{right}" for left, right in kanfis_pair_indices),
                )
            )
        if kanfis_projection_indices:
            progress_log(
                "seed={seed} model=kanfis: projection_channels={routes}".format(
                    seed=seed,
                    routes=";".join(",".join(str(index) for index in route) for route in kanfis_projection_indices),
                )
            )
        kanfis_decision_skip_indices = tuple(
            kanfis_active_feature_indices[: max(0, min(int(kanfis_decision_skip_features), split.input_dim))]
        )
        kanfis_model = build_kanfis_model(
            split.input_dim,
            pair_indices=kanfis_pair_indices,
            projection_indices=kanfis_projection_indices,
            active_feature_indices=kanfis_active_feature_indices,
            superposition_terms=kanfis_superposition_terms,
            depth=kanfis_depth,
            concept_fan_in=kanfis_concept_fan_in,
            routing=kanfis_routing,
            decision_skip_indices=kanfis_decision_skip_indices,
            train_rule_gates=bool(kanfis_train_rule_gates),
            decision_skip_gate_init_logit=2.0,
        ).to(device=device)
        if hasattr(kanfis_model, "set_feature_names"):
            kanfis_model.set_feature_names(_dataset_feature_names(spec.name, split.input_dim))
        if hasattr(kanfis_model, "fit_membership_terms"):
            kanfis_model.fit_membership_terms(train_inputs)
        kanfis_training_config = TrainingConfig(
            task_type=spec.task_type,
            max_epochs=max_epochs,
            learning_rate=fuzzy_learning_rate,
            patience=min(patience, max_epochs),
            batch_size=batch_size,
            shuffle=True,
            amp_enabled=amp_enabled,
            classification_threshold=classification_threshold,
            binary_auto_pos_weight=spec.task_type == "binary_classification" and fuzzy_binary_auto_pos_weight,
            binary_loss_name=fuzzy_binary_loss_name,
            binary_focal_gamma=fuzzy_binary_focal_gamma,
            binary_soft_f1_weight=(
                fuzzy_binary_soft_f1_weight if spec.task_type == "binary_classification" else 0.0
            ),
            regression_loss=fuzzy_regression_loss,
            huber_delta=fuzzy_huber_delta,
            regression_linear_residual_head=False,
            monitor_metric="f1" if spec.task_type == "binary_classification" else None,
            monitor_mode="max" if spec.task_type == "binary_classification" else None,
            weight_decay=1e-5,
            gradient_clip_norm=5.0,
            rule_sparsity_weight=max(0.0, float(kanfis_rule_sparsity_weight)),
            rule_activation_entropy_weight=max(0.0, float(kanfis_rule_entropy_weight)),
            rule_anchor_stability_weight=max(0.0, float(kanfis_rule_anchor_weight)),
            block_gate_l1_weight=max(0.0, float(kanfis_skip_gate_l1_weight)),
            regularization_warmup_epochs=max(1, max_epochs // 4),
            verbose=bool(log_epochs),
            log_every_n_epochs=max(1, int(log_epochs_every)),
            log_prefix=f"seed={seed} model=kanfis",
            device=device,
        )
        kanfis_trainer = FuzzyTrainer(kanfis_model, kanfis_training_config)
        full_train_started_at = time.perf_counter()
        kanfis_trainer.fit(
            train_inputs,
            model_train_targets["ruanfis_kanfis"],
            validation_inputs,
            validation_targets,
        )
        stage_runtime["full_train_sec"] = float(time.perf_counter() - full_train_started_at)
        selection_sec = 0.0
        refit_head_sec = 0.0
        if kanfis_prune_rules > 0:
            importance_inputs = train_inputs
            if str(kanfis_importance_split).strip().lower() == "val":
                importance_inputs = validation_inputs
            selection_started_at = time.perf_counter()
            pruned_rule_count = kanfis_model.prune_to_top_k_rules(
                int(kanfis_prune_rules),
                inputs=importance_inputs,
                batch_size=8192,
            )
            selection_sec += float(time.perf_counter() - selection_started_at)
            progress_log(
                f"seed={seed} model=kanfis: pruned_active_rules={pruned_rule_count} "
                f"(importance_split={kanfis_importance_split})"
            )
            if kanfis_recovery_epochs > 0:
                kanfis_recovery_config = TrainingConfig(
                    task_type=spec.task_type,
                    max_epochs=int(kanfis_recovery_epochs),
                    learning_rate=fuzzy_learning_rate * 0.35,
                    patience=min(4, max(2, patience)),
                    batch_size=batch_size,
                    shuffle=True,
                    amp_enabled=amp_enabled,
                    classification_threshold=classification_threshold,
                    binary_auto_pos_weight=spec.task_type == "binary_classification" and fuzzy_binary_auto_pos_weight,
                    binary_loss_name=fuzzy_binary_loss_name,
                    binary_focal_gamma=fuzzy_binary_focal_gamma,
                    binary_soft_f1_weight=(
                        fuzzy_binary_soft_f1_weight if spec.task_type == "binary_classification" else 0.0
                    ),
                    regression_loss=fuzzy_regression_loss,
                    huber_delta=fuzzy_huber_delta,
                    monitor_metric="f1" if spec.task_type == "binary_classification" else None,
                    monitor_mode="max" if spec.task_type == "binary_classification" else None,
                    weight_decay=1e-5,
                    gradient_clip_norm=5.0,
                    rule_sparsity_weight=max(0.0, float(kanfis_rule_sparsity_weight)),
                    rule_activation_entropy_weight=max(0.0, float(kanfis_rule_entropy_weight)),
                    rule_anchor_stability_weight=max(0.0, float(kanfis_rule_anchor_weight)),
                    block_gate_l1_weight=max(0.0, float(kanfis_skip_gate_l1_weight)),
                    regularization_warmup_epochs=max(1, int(kanfis_recovery_epochs) // 2),
                    verbose=False,
                    log_prefix=f"seed={seed} model=kanfis-recovery",
                    device=device,
                )
                refit_started_at = time.perf_counter()
                FuzzyTrainer(kanfis_model, kanfis_recovery_config).fit(
                    train_inputs,
                    model_train_targets["ruanfis_kanfis"],
                    validation_inputs,
                    validation_targets,
                )
                refit_head_sec += float(time.perf_counter() - refit_started_at)
        stage_runtime["selection_sec"] = float(selection_sec)
        if kanfis_polish_epochs > 0:
            kanfis_polish_config = TrainingConfig(
                task_type=spec.task_type,
                max_epochs=int(kanfis_polish_epochs),
                learning_rate=fuzzy_learning_rate * 0.2,
                patience=min(4, max(2, patience)),
                batch_size=batch_size,
                shuffle=True,
                amp_enabled=amp_enabled,
                classification_threshold=classification_threshold,
                binary_auto_pos_weight=spec.task_type == "binary_classification" and fuzzy_binary_auto_pos_weight,
                binary_loss_name=fuzzy_binary_loss_name,
                binary_focal_gamma=fuzzy_binary_focal_gamma,
                binary_soft_f1_weight=(
                    fuzzy_binary_soft_f1_weight if spec.task_type == "binary_classification" else 0.0
                ),
                regression_loss=fuzzy_regression_loss,
                huber_delta=fuzzy_huber_delta,
                monitor_metric="f1" if spec.task_type == "binary_classification" else None,
                monitor_mode="max" if spec.task_type == "binary_classification" else None,
                weight_decay=1e-5,
                gradient_clip_norm=5.0,
                rule_sparsity_weight=max(0.0, float(kanfis_rule_sparsity_weight)),
                rule_activation_entropy_weight=max(0.0, float(kanfis_rule_entropy_weight)),
                rule_anchor_stability_weight=max(0.0, float(kanfis_rule_anchor_weight)),
                block_gate_l1_weight=max(0.0, float(kanfis_skip_gate_l1_weight)),
                regularization_warmup_epochs=max(1, int(kanfis_polish_epochs) // 2),
                verbose=False,
                log_prefix=f"seed={seed} model=kanfis-polish",
                device=device,
            )
            polish_started_at = time.perf_counter()
            FuzzyTrainer(kanfis_model, kanfis_polish_config).fit(
                train_inputs,
                train_targets,
                validation_inputs,
                validation_targets,
            )
            refit_head_sec += float(time.perf_counter() - polish_started_at)
        stage_runtime["refit_head_sec"] = float(refit_head_sec)
        _maybe_run_hard_sample_finetune(
            enabled=fuzzy_hard_sample_training and "ruanfis_kanfis" in hard_sample_enabled_models,
            model_name="ruanfis_kanfis",
            model=kanfis_model,
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=model_train_targets["ruanfis_kanfis"],
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            base_training_config=kanfis_training_config,
            source=fuzzy_hard_sample_source,
            hard_fraction=fuzzy_hard_sample_fraction,
            hard_multiplier=fuzzy_hard_sample_multiplier,
            finetune_epochs=fuzzy_hard_sample_finetune_epochs,
            finetune_patience=fuzzy_hard_sample_finetune_patience,
            baseline_hard_indices=baseline_hard_indices,
            seed=seed,
        )
        trained_fuzzy_models["ruanfis_kanfis"] = kanfis_model
        model_top_k_rules["ruanfis_kanfis"] = None
        model_prediction_postprocessors["ruanfis_kanfis"] = None
        progress_log(f"seed={seed} model=kanfis: done in {time.perf_counter() - phase_started_at:.2f}s")

    if "ruanfis_hierarchical_anfis" in fuzzy_models:
        phase_started_at = time.perf_counter()
        progress_log(f"seed={seed} model=hierarchical_anfis: build+train start")
        hierarchical_anfis_model = build_hierarchical_anfis_model(
            build_hierarchical_anfis_config(
                split.input_dim,
                group_size=hierarchical_group_size,
                width_scale=hierarchical_width_scale,
                rule_scale=hierarchical_rule_scale,
                prototype_scoring_mode=hierarchical_prototype_scoring_mode,
                hidden_output_activation=hidden_output_activation,
            ),
            sample_inputs=bootstrap_inputs,
        )
        hierarchical_anfis_training_config = TrainingConfig(
            task_type=spec.task_type,
            max_epochs=max_epochs,
            learning_rate=fuzzy_learning_rate,
            patience=min(patience, max_epochs),
            batch_size=batch_size,
            shuffle=True,
            amp_enabled=amp_enabled,
            classification_threshold=classification_threshold,
            binary_auto_pos_weight=spec.task_type == "binary_classification" and fuzzy_binary_auto_pos_weight,
            binary_loss_name=fuzzy_binary_loss_name,
            binary_focal_gamma=fuzzy_binary_focal_gamma,
            binary_soft_f1_weight=(
                fuzzy_binary_soft_f1_weight if spec.task_type == "binary_classification" else 0.0
            ),
            regression_loss=fuzzy_regression_loss,
            huber_delta=fuzzy_huber_delta,
            regression_linear_residual_head=regression_residual_head_enabled,
            monitor_metric="f1" if spec.task_type == "binary_classification" else None,
            monitor_mode="max" if spec.task_type == "binary_classification" else None,
            verbose=bool(log_epochs),
            log_every_n_epochs=max(1, int(log_epochs_every)),
            log_prefix=f"seed={seed} model=hierarchical",
            device=device,
        )
        hierarchical_anfis_trainer = FuzzyTrainer(hierarchical_anfis_model, hierarchical_anfis_training_config)
        hierarchical_anfis_trainer.fit(
            train_inputs,
            model_train_targets["ruanfis_hierarchical_anfis"],
            validation_inputs,
            validation_targets,
        )
        _maybe_run_hard_sample_finetune(
            enabled=(
                fuzzy_hard_sample_training and "ruanfis_hierarchical_anfis" in hard_sample_enabled_models
            ),
            model_name="ruanfis_hierarchical_anfis",
            model=hierarchical_anfis_model,
            task_type=spec.task_type,
            train_inputs=train_inputs,
            train_targets=model_train_targets["ruanfis_hierarchical_anfis"],
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            base_training_config=hierarchical_anfis_training_config,
            source=fuzzy_hard_sample_source,
            hard_fraction=fuzzy_hard_sample_fraction,
            hard_multiplier=fuzzy_hard_sample_multiplier,
            finetune_epochs=fuzzy_hard_sample_finetune_epochs,
            finetune_patience=fuzzy_hard_sample_finetune_patience,
            baseline_hard_indices=baseline_hard_indices,
            seed=seed,
        )
        trained_fuzzy_models["ruanfis_hierarchical_anfis"] = hierarchical_anfis_model
        model_top_k_rules["ruanfis_hierarchical_anfis"] = None
        model_prediction_postprocessors["ruanfis_hierarchical_anfis"] = None
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
            rule_probability_threshold=rule_probability_threshold,
        )
        progress_log(f"seed={seed} sklearn: done in {time.perf_counter() - phase_started_at:.2f}s")

    phase_started_at = time.perf_counter()
    progress_log(f"seed={seed} fuzzy_eval: start")
    model_thresholds = {model_name: classification_threshold for model_name in trained_fuzzy_models.keys()}
    if tune_fuzzy_threshold and spec.task_type == "binary_classification":
        for model_name, model in trained_fuzzy_models.items():
            if tune_fuzzy_threshold_calibrated and model_prediction_postprocessors.get(model_name) is None:
                val_logits_raw = _predict_logits_batched(
                    model,
                    validation_inputs,
                    top_k_rules=model_top_k_rules.get(model_name),
                )
                calibrator = _fit_binary_probability_calibrator(
                    val_logits_raw,
                    validation_targets.detach().cpu(),
                )
                model_prediction_postprocessors[model_name] = _build_binary_logit_postprocessor(calibrator)

            model_thresholds[model_name] = _choose_best_classification_threshold(
                model,
                validation_inputs=validation_inputs,
                validation_targets=validation_targets,
                default_threshold=classification_threshold,
                top_k_rules=model_top_k_rules.get(model_name),
                strategy=threshold_tuning_strategy,
                prediction_postprocessor=model_prediction_postprocessors.get(model_name),
            )
        progress_log(
            "seed={seed} threshold_tuning: {pairs}".format(
                seed=seed,
                pairs=", ".join(f"{name}={value:.2f}" for name, value in model_thresholds.items()),
            )
        )

    fuzzy_results_list: list[Any] = []
    for model_name in FUZZY_MODEL_NAMES:
        if model_name not in trained_fuzzy_models:
            continue
        fuzzy_results_list.append(
            evaluate_trained_model(
                model_name,
                trained_fuzzy_models[model_name],
                task_type=spec.task_type,
                train_inputs=train_inputs,
                train_targets=train_targets,
                test_inputs=test_inputs,
                test_targets=test_targets,
                classification_threshold=model_thresholds[model_name],
                rule_probability_threshold=rule_probability_threshold,
                top_k_rules=model_top_k_rules.get(model_name),
                prediction_postprocessor=model_prediction_postprocessors.get(model_name),
                dffl_concept_expert_labels=(
                    dffl_concept_expert_labels if model_name == "ruanfis_refined_deep" else None
                ),
            )
        )
        if model_name == "ruanfis_kanfis":
            infer_started_at = time.perf_counter()
            logits = _predict_logits_batched(
                trained_fuzzy_models[model_name],
                test_inputs,
                top_k_rules=model_top_k_rules.get(model_name),
            )
            prediction_postprocessor = model_prediction_postprocessors.get(model_name)
            if prediction_postprocessor is not None:
                _ = prediction_postprocessor(logits)
            infer_elapsed = float(time.perf_counter() - infer_started_at)
            sample_count = max(1, int(test_inputs.size(0)))
            stage_runtime["inference_ms_per_sample"] = infer_elapsed * 1000.0 / float(sample_count)
    fuzzy_results = tuple(fuzzy_results_list)
    if (
        v18_controls_dir is not None
        and spec.task_type == "binary_classification"
        and "ruanfis_kanfis" in trained_fuzzy_models
        and isinstance(trained_fuzzy_models["ruanfis_kanfis"], DeepKANFISModel)
    ):
        _export_v18_controls_for_kanfis(
            dataset_name=str(spec.name),
            seed=int(seed),
            model=trained_fuzzy_models["ruanfis_kanfis"],
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
            output_dir=Path(v18_controls_dir),
            lr_budgets=tuple(int(x) for x in v18_lr_budgets if int(x) > 0),
            random_state=int(v18_random_state),
        )
        if bool(v18_export_h_artifacts):
            h_build_started_at = time.perf_counter()
            _export_v18_h_artifacts_for_kanfis(
                dataset_name=str(spec.name),
                seed=int(seed),
                model=trained_fuzzy_models["ruanfis_kanfis"],
                train_inputs=train_inputs,
                train_targets=train_targets,
                test_inputs=test_inputs,
                test_targets=test_targets,
                output_dir=Path(v18_controls_dir),
                classification_threshold=float(model_thresholds["ruanfis_kanfis"]),
            )
            stage_runtime["h_build_sec"] = float(time.perf_counter() - h_build_started_at)
    elif bool(v18_export_h_artifacts):
        # H-export stage requested but not applicable for this KAFN variant.
        stage_runtime["h_build_sec"] = 0.0
    progress_log(f"seed={seed} fuzzy_eval: done in {time.perf_counter() - phase_started_at:.2f}s")
    progress_log(f"seed={seed} all models: complete")

    peak_rss_mb = _read_peak_rss_mb()
    if peak_rss_mb is None and run_rss_start_mb is not None:
        peak_rss_mb = run_rss_start_mb
    if peak_rss_mb is not None and run_rss_start_mb is not None:
        peak_rss_mb = max(float(peak_rss_mb), float(run_rss_start_mb))
    stage_runtime["memory_peak_mb"] = peak_rss_mb

    return (*sklearn_results, *fuzzy_results), stage_runtime


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


def _summarize_stage_runtime(
    per_seed_stage_runtime: list[dict[str, float | int | None]],
) -> dict[str, dict[str, float | int | None]]:
    stage_keys = (
        "full_train_sec",
        "h_build_sec",
        "selection_sec",
        "refit_head_sec",
        "inference_ms_per_sample",
        "memory_peak_mb",
    )
    summary: dict[str, dict[str, float | int | None]] = {}
    for key in stage_keys:
        values = [
            float(item[key])
            for item in per_seed_stage_runtime
            if item.get(key) is not None
        ]
        if not values:
            summary[key] = {"mean": None, "std": None, "n": 0}
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)) if len(values) > 1 else 0.0,
            "n": int(len(values)),
        }
    return summary


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
    dataset_suite: str,
    dataset_names: tuple[str, ...],
    seeds: tuple[int, ...],
    fuzzy_models: tuple[str, ...],
    dffl_dataset_overrides: Mapping[str, Mapping[str, Any]] | None,
    dataset_protocols: dict[str, dict[str, object]],
) -> dict[str, object]:
    seed_selection_policy = "deterministic_pool" if int(args.seed_count) > 0 else "explicit_argument"
    return {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "script": "examples/run_real_datasets_benchmark.py",
        "runtime_environment": collect_runtime_environment(requested_device=args.device),
        "protocol": {
            "dataset_selection": {
                "dataset_suite": str(dataset_suite),
                "datasets_argument": str(args.datasets),
                "dataset_count_effective": int(len(dataset_names)),
            },
            "seed_selection": {
                "policy": seed_selection_policy,
                "seeds_argument": str(args.seeds),
                "seed_count_requested": int(args.seed_count),
                "seed_count_effective": int(len(seeds)),
            },
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
                "fuzzy_distill_weight": float(args.fuzzy_distill_weight),
                "fuzzy_distill_models": list(parse_optional_fuzzy_model_names(args.fuzzy_distill_models)),
                "fuzzy_distill_teacher_trees": int(args.fuzzy_distill_teacher_trees),
                "fuzzy_hard_sample_training": bool(args.fuzzy_hard_sample_training),
                "fuzzy_hard_sample_models": list(parse_optional_fuzzy_model_names(args.fuzzy_hard_sample_models)),
                "fuzzy_hard_sample_source": str(args.fuzzy_hard_sample_source),
                "fuzzy_hard_sample_fraction": float(args.fuzzy_hard_sample_fraction),
                "fuzzy_hard_sample_multiplier": int(args.fuzzy_hard_sample_multiplier),
                "fuzzy_hard_sample_finetune_epochs": int(args.fuzzy_hard_sample_finetune_epochs),
                "fuzzy_hard_sample_finetune_patience": int(args.fuzzy_hard_sample_finetune_patience),
                "fuzzy_hard_sample_teacher_trees": int(args.fuzzy_hard_sample_teacher_trees),
                "kanfis_active_features": int(args.kanfis_active_features),
                "kanfis_depth": int(args.kanfis_depth),
                "kanfis_superposition_terms": int(args.kanfis_superposition_terms),
                "kanfis_concept_fan_in": int(args.kanfis_concept_fan_in),
                "kanfis_routing": str(args.kanfis_routing),
                "kanfis_feature_order": str(args.kanfis_feature_order),
                "kanfis_decision_skip_features": int(args.kanfis_decision_skip_features),
                "kanfis_pair_count": int(args.kanfis_pair_count),
                "kanfis_projection_count": int(args.kanfis_projection_count),
                "kanfis_projection_width": int(args.kanfis_projection_width),
                "kanfis_projection_mode": str(args.kanfis_projection_mode),
                "kanfis_prune_rules": int(args.kanfis_prune_rules),
                "kanfis_importance_split": str(args.kanfis_importance_split),
                "kanfis_recovery_epochs": int(args.kanfis_recovery_epochs),
                "kanfis_polish_epochs": int(args.kanfis_polish_epochs),
                "kanfis_train_rule_gates": bool(args.kanfis_train_rule_gates),
                "kanfis_rule_sparsity_weight": float(args.kanfis_rule_sparsity_weight),
                "kanfis_rule_entropy_weight": float(args.kanfis_rule_entropy_weight),
                "kanfis_rule_anchor_weight": float(args.kanfis_rule_anchor_weight),
                "kanfis_skip_gate_l1_weight": float(args.kanfis_skip_gate_l1_weight),
                "hidden_output_activation": str(args.hidden_output_activation),
                "stacked_stagewise_init": bool(args.stacked_stagewise_init),
                "stacked_stagewise_init_epochs": int(args.stacked_stagewise_init_epochs),
                "stacked_final_skip_inputs": int(args.stacked_final_skip_inputs),
                "stacked_final_skip_mode": str(args.stacked_final_skip_mode),
                "stacked_final_skip_gates": bool(args.stacked_final_skip_gates),
                "stacked_final_skip_gate_init_logit": float(args.stacked_final_skip_gate_init_logit),
                "stacked_final_skip_gate_l1_weight": float(args.stacked_final_skip_gate_l1_weight),
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
                "dffl_fast_gpu": bool(args.dffl_fast_gpu),
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
                "dffl_dataset_overrides": (
                    {key: dict(value) for key, value in dffl_dataset_overrides.items()}
                    if dffl_dataset_overrides
                    else {}
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
                "rule_probability_threshold": float(args.rule_probability_threshold),
                "tune_fuzzy_threshold": bool(args.tune_fuzzy_threshold),
                "tune_fuzzy_threshold_calibrated": bool(args.tune_fuzzy_threshold_calibrated),
                "threshold_tuning_grid_if_enabled": "0.05..0.95 step=0.01 on validation split",
                "active_rule_criterion": f"rule_probability >= {float(args.rule_probability_threshold):.3f}",
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
                "dffl_fast_gpu": bool(args.dffl_fast_gpu),
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
    dataset_selection = protocol["dataset_selection"]
    seed_selection = protocol["seed_selection"]
    split = protocol["split"]
    preprocessing = protocol["preprocessing"]
    execution = protocol["execution"]
    training = protocol["training_budgets"]
    evaluation = protocol["evaluation"]
    runtime = payload.get("runtime_environment", {})

    lines: list[str] = []
    lines.append("# Reproducibility Manifest")
    lines.append("")
    lines.append(f"- generated_utc: `{payload['generated_utc']}`")
    lines.append(f"- script: `{payload['script']}`")
    lines.append(f"- datasets: `{', '.join(protocol['datasets'])}`")
    lines.append(f"- seeds: `{', '.join(str(seed) for seed in protocol['seeds'])}`")
    lines.append("")
    lines.append("## Dataset/Seed Selection")
    lines.append("")
    lines.append(f"- dataset_suite: `{dataset_selection['dataset_suite']}`")
    lines.append(f"- datasets_argument: `{dataset_selection['datasets_argument']}`")
    lines.append(f"- dataset_count_effective: `{dataset_selection['dataset_count_effective']}`")
    lines.append(f"- seed_policy: `{seed_selection['policy']}`")
    lines.append(f"- seeds_argument: `{seed_selection['seeds_argument']}`")
    lines.append(f"- seed_count_requested: `{seed_selection['seed_count_requested']}`")
    lines.append(f"- seed_count_effective: `{seed_selection['seed_count_effective']}`")
    lines.append("")
    lines.append("## Runtime Environment")
    lines.append("")
    lines.append(f"- python_version: `{runtime.get('python_version', 'unknown')}`")
    lines.append(f"- platform: `{runtime.get('platform', 'unknown')}`")
    lines.append(f"- torch_version: `{runtime.get('torch_version', 'unknown')}`")
    lines.append(f"- cuda_compiled_version: `{runtime.get('cuda_compiled_version', 'none')}`")
    lines.append(f"- cuda_available: `{runtime.get('cuda_available', False)}`")
    lines.append(f"- cuda_device_count: `{runtime.get('cuda_device_count', 0)}`")
    lines.append(
        f"- cuda_devices: `{', '.join(runtime.get('cuda_devices', [])) if runtime.get('cuda_devices') else 'none'}`"
    )
    lines.append(f"- requested_device: `{runtime.get('requested_device', 'default')}`")
    lines.append(f"- git_revision: `{runtime.get('git_revision', 'unknown')}`")
    lines.append(f"- command: `{runtime.get('command', '')}`")
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
    lines.append(f"- fuzzy_distill_weight: `{execution['fuzzy_distill_weight']}`")
    lines.append(f"- fuzzy_distill_models: `{', '.join(execution['fuzzy_distill_models']) if execution['fuzzy_distill_models'] else 'none'}`")
    lines.append(f"- fuzzy_distill_teacher_trees: `{execution['fuzzy_distill_teacher_trees']}`")
    lines.append(f"- fuzzy_hard_sample_training: `{execution['fuzzy_hard_sample_training']}`")
    lines.append(
        f"- fuzzy_hard_sample_models: "
        f"`{', '.join(execution['fuzzy_hard_sample_models']) if execution['fuzzy_hard_sample_models'] else 'none'}`"
    )
    lines.append(f"- fuzzy_hard_sample_source: `{execution['fuzzy_hard_sample_source']}`")
    lines.append(f"- fuzzy_hard_sample_fraction: `{execution['fuzzy_hard_sample_fraction']}`")
    lines.append(f"- fuzzy_hard_sample_multiplier: `{execution['fuzzy_hard_sample_multiplier']}`")
    lines.append(f"- fuzzy_hard_sample_finetune_epochs: `{execution['fuzzy_hard_sample_finetune_epochs']}`")
    lines.append(f"- fuzzy_hard_sample_finetune_patience: `{execution['fuzzy_hard_sample_finetune_patience']}`")
    lines.append(f"- fuzzy_hard_sample_teacher_trees: `{execution['fuzzy_hard_sample_teacher_trees']}`")
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
    lines.append(f"- rule_probability_threshold: `{evaluation['rule_probability_threshold']}`")
    lines.append(f"- tune_fuzzy_threshold: `{evaluation['tune_fuzzy_threshold']}`")
    lines.append(f"- tune_fuzzy_threshold_calibrated: `{evaluation['tune_fuzzy_threshold_calibrated']}`")
    lines.append(f"- threshold_tuning_grid_if_enabled: {evaluation['threshold_tuning_grid_if_enabled']}")
    lines.append(f"- active_rule_criterion: {evaluation['active_rule_criterion']}")
    lines.append(f"- stability_metrics: `{', '.join(evaluation['stability_metrics'])}`")
    lines.append("")
    lines.append("## Dataset-Specific DFFL Resolution")
    lines.append("")
    lines.append(
        "| dataset | task | n_samples | input_dim | runtime_s | profile_resolved | geom_req | geom_eff | one_phase | "
        "lr_effective | groups | stage1_width | stage2_width_total | decision_max_rules |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for dataset_name, entry in payload["dataset_specific_protocols"].items():
        dffl = entry["dffl"]
        runtime_seconds = float(entry.get("runtime_seconds", 0.0))
        if not bool(dffl.get("enabled", True)):
            lines.append(
                "| {dataset} | {task} | {n_samples} | {input_dim} | {runtime:.2f} | {profile} | n/a | n/a | disabled | "
                "0.000000 | 0 | 0 | 0 | 0 |".format(
                    dataset=dataset_name,
                    task=entry["task_type"],
                    n_samples=entry["n_samples"],
                    input_dim=entry["input_dim"],
                    runtime=runtime_seconds,
                    profile=dffl["resolved_profile"],
                )
            )
            continue
        arch = dffl["architecture"]
        lines.append(
            "| {dataset} | {task} | {n_samples} | {input_dim} | {runtime:.2f} | {profile} | {geom_req} | {geom_eff} | "
            "{one_phase} | {lr:.6f} | {groups} | {s1} | {s2} | {dec_rules} |".format(
                dataset=dataset_name,
                task=entry["task_type"],
                n_samples=entry["n_samples"],
                input_dim=entry["input_dim"],
                runtime=runtime_seconds,
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
        "--dataset-suite",
        type=str,
        default="custom",
        choices=("custom",) + tuple(DATASET_SUITES.keys()),
        help=(
            "Named dataset suite. If not 'custom', overrides --datasets. "
            "Use paper_main/paper_extended/paper_all/q1_large/q1_full."
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="diabetes,linnerud_weight,breast_cancer,wine_binary,digits_binary",
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--seeds", type=str, default="19,23,29")
    parser.add_argument(
        "--seed-count",
        type=int,
        default=0,
        help=(
            "If >0, ignore --seeds and use first N deterministic seeds from the internal seed pool "
            "(recommended: 10-30 for stability analysis)."
        ),
    )
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
        "--dffl-dataset-overrides",
        type=str,
        default=None,
        help=(
            "Path to JSON (or inline JSON) with per-dataset DFFL overrides. "
            "Example: "
            '\'{"breast_cancer":{"profile":"quality","learning_rate_scale_classification":1.2},'
            '"diabetes":{"profile":"quality_tiny_reg","learning_rate_scale_regression":1.1}}\''
        ),
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
        "--disable-dffl-adaptive-local-arity",
        action="store_true",
        help="Disable adaptive local rule-arity policy on DFFL stage-1 blocks.",
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
    parser.add_argument(
        "--dffl-tiny-restarts",
        type=int,
        default=1,
        help="Number of DFFL restarts for tiny classification datasets (best model by validation F1).",
    )
    parser.add_argument("--stacked-width-scale", type=float, default=1.0)
    parser.add_argument("--stacked-rule-scale", type=float, default=1.0)
    parser.add_argument(
        "--stacked-stagewise-init",
        action="store_true",
        help="Warm hidden stacked layers before generating upper-layer prototype rules.",
    )
    parser.add_argument(
        "--stacked-stagewise-init-epochs",
        type=int,
        default=12,
        help="Epochs per hidden stacked layer for --stacked-stagewise-init.",
    )
    parser.add_argument(
        "--stacked-final-skip-inputs",
        type=int,
        default=0,
        help="Number of original input features concatenated into the final stacked fuzzy layer.",
    )
    parser.add_argument(
        "--stacked-final-skip-mode",
        type=str,
        default="first",
        choices=("first", "target_corr", "target_corr_diverse"),
        help="Selection rule for raw features concatenated into the final stacked fuzzy layer.",
    )
    parser.add_argument(
        "--stacked-final-skip-gates",
        action="store_true",
        help="Enable trainable gates on stacked final raw-skip features.",
    )
    parser.add_argument(
        "--stacked-final-skip-gate-init-logit",
        type=float,
        default=2.0,
        help="Initial gate logit for stacked final raw-skip features.",
    )
    parser.add_argument(
        "--stacked-final-skip-gate-l1-weight",
        type=float,
        default=0.0,
        help="L1 weight on stacked final raw-skip gates (uses block-gate regularizer slot).",
    )
    parser.add_argument("--hierarchical-width-scale", type=float, default=1.0)
    parser.add_argument("--hierarchical-rule-scale", type=float, default=1.0)
    parser.add_argument("--hierarchical-group-size", type=int, default=4)
    parser.add_argument("--stacked-prototype-scoring-mode", type=str, default="max", choices=("max", "hybrid"))
    parser.add_argument(
        "--hierarchical-prototype-scoring-mode", type=str, default="max", choices=("max", "hybrid")
    )
    parser.add_argument(
        "--hidden-output-activation",
        type=str,
        default="identity",
        choices=("identity", "sigmoid"),
        help="Optional activation for hidden Sugeno outputs in stacked/hierarchical baselines.",
    )
    parser.add_argument("--fuzzy-binary-loss-name", type=str, default="bce", choices=("bce", "focal"))
    parser.add_argument("--fuzzy-binary-focal-gamma", type=float, default=2.0)
    parser.add_argument("--fuzzy-binary-auto-pos-weight", action="store_true")
    parser.add_argument("--fuzzy-binary-soft-f1-weight", type=float, default=0.0)
    parser.add_argument("--fuzzy-regression-loss", type=str, default="mse", choices=("mse", "huber"))
    parser.add_argument("--fuzzy-huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--fuzzy-distill-weight",
        type=float,
        default=0.0,
        help=(
            "Blend weight for binary-label distillation from ExtraTrees teacher: "
            "y'=(1-w)*y + w*p_teacher. 0 disables."
        ),
    )
    parser.add_argument(
        "--fuzzy-distill-models",
        type=str,
        default="none",
        help=(
            "Comma-separated fuzzy models to receive distilled targets "
            "(e.g. stacked,hierarchical). Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--fuzzy-distill-teacher-trees",
        type=int,
        default=400,
        help="Number of trees in ExtraTrees teacher used for distillation.",
    )
    parser.add_argument(
        "--kanfis-active-features",
        type=int,
        default=0,
        help="KANFIS predicate dictionary feature count; 0 uses all features.",
    )
    parser.add_argument(
        "--kanfis-depth",
        type=int,
        default=1,
        help="KANFIS depth: 1 is shallow additive KA-FIS, >=2 is deep interpretable KA-FIS.",
    )
    parser.add_argument(
        "--kanfis-superposition-terms",
        type=int,
        default=16,
        help="KANFIS additive superposition width.",
    )
    parser.add_argument(
        "--kanfis-concept-fan-in",
        type=int,
        default=0,
        help="Deep KANFIS routed first-layer fan-in per concept; 0 connects every concept to every selected feature.",
    )
    parser.add_argument(
        "--kanfis-routing",
        type=str,
        default="chunk",
        choices=("chunk", "banded", "grouped"),
        help="Deep KANFIS first-layer routing strategy.",
    )
    parser.add_argument(
        "--kanfis-feature-order",
        type=str,
        default="target_corr_diverse",
        choices=("target_corr_diverse", "teacher_importance"),
        help="Feature ordering used before KANFIS routing.",
    )
    parser.add_argument(
        "--kanfis-decision-skip-features",
        type=int,
        default=0,
        help="Transparent raw-feature evidence channels appended to the final KANFIS decision layer.",
    )
    parser.add_argument(
        "--kanfis-pair-count",
        type=int,
        default=0,
        help="Number of target-relevant pair predicate channels for KANFIS.",
    )
    parser.add_argument(
        "--kanfis-projection-count",
        type=int,
        default=0,
        help="Number of sparse projection-pursuit fuzzy channels for deep KANFIS.",
    )
    parser.add_argument(
        "--kanfis-projection-width",
        type=int,
        default=3,
        help="Number of source features per KANFIS projection-pursuit channel.",
    )
    parser.add_argument(
        "--kanfis-projection-mode",
        type=str,
        default="ranked",
        choices=("ranked", "fixed_covtype"),
        help="How KANFIS projection-pursuit routes are generated.",
    )
    parser.add_argument(
        "--kanfis-prune-rules",
        type=int,
        default=1024,
        help="Post-train active KANFIS rule budget; <=0 disables pruning.",
    )
    parser.add_argument(
        "--kanfis-importance-split",
        type=str,
        default="train",
        choices=("train", "val"),
        help="Split used to compute data-aware rule importances for KANFIS budget pruning.",
    )
    parser.add_argument(
        "--kanfis-recovery-epochs",
        type=int,
        default=12,
        help="Short recovery fine-tune epochs after KANFIS pruning; <=0 disables recovery.",
    )
    parser.add_argument(
        "--kanfis-polish-epochs",
        type=int,
        default=0,
        help="Final KANFIS fine-tune epochs on true labels after distillation/recovery; <=0 disables.",
    )
    parser.add_argument(
        "--kanfis-train-rule-gates",
        action="store_true",
        help="Allow KANFIS rule gates to learn; pair with sparsity/entropy/anchor regularization.",
    )
    parser.add_argument(
        "--kanfis-rule-sparsity-weight",
        type=float,
        default=0.0,
        help="KANFIS L1-style rule-gate sparsity weight.",
    )
    parser.add_argument(
        "--kanfis-rule-entropy-weight",
        type=float,
        default=0.0,
        help="KANFIS rule-gate entropy penalty weight for crisper active/inactive structure.",
    )
    parser.add_argument(
        "--kanfis-rule-anchor-weight",
        type=float,
        default=0.0,
        help="KANFIS rule-gate stability penalty weight around the current anchor.",
    )
    parser.add_argument(
        "--kanfis-skip-gate-l1-weight",
        type=float,
        default=0.0,
        help="L1 penalty for transparent KANFIS raw-feature skip gates.",
    )
    parser.add_argument(
        "--fuzzy-hard-sample-training",
        action="store_true",
        help="Enable second-stage hard-sample finetune for selected fuzzy models.",
    )
    parser.add_argument(
        "--fuzzy-hard-sample-models",
        type=str,
        default="stacked,hierarchical,dffl",
        help=(
            "Comma-separated fuzzy models for hard-sample finetune "
            "(aliases supported). Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--fuzzy-hard-sample-source",
        type=str,
        default="baseline",
        choices=("self", "baseline"),
        help="Hard-sample source: model self-errors or ExtraTrees baseline errors.",
    )
    parser.add_argument(
        "--fuzzy-hard-sample-fraction",
        type=float,
        default=0.25,
        help="Top fraction of hardest train samples used for oversampling.",
    )
    parser.add_argument(
        "--fuzzy-hard-sample-multiplier",
        type=int,
        default=3,
        help="Train duplication multiplier for selected hard samples.",
    )
    parser.add_argument(
        "--fuzzy-hard-sample-finetune-epochs",
        type=int,
        default=16,
        help="Max epochs for hard-sample finetune stage.",
    )
    parser.add_argument(
        "--fuzzy-hard-sample-finetune-patience",
        type=int,
        default=5,
        help="Early-stopping patience for hard-sample finetune stage.",
    )
    parser.add_argument(
        "--fuzzy-hard-sample-teacher-trees",
        type=int,
        default=400,
        help="ExtraTrees size for baseline-driven hard-sample selection.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--log-epochs",
        action="store_true",
        help="Print per-epoch fuzzy training logs into stdout/run.status.log.",
    )
    parser.add_argument(
        "--log-epochs-every",
        type=int,
        default=5,
        help="Epoch logging frequency when --log-epochs is enabled.",
    )
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument(
        "--rule-probability-threshold",
        type=float,
        default=0.5,
        help="Rule probability threshold used for active-rule structural/stability metrics.",
    )
    parser.add_argument(
        "--tune-fuzzy-threshold",
        action="store_true",
        help=(
            "Tune binary-classification threshold per fuzzy model on validation split "
            "(max F1 by default; calibration-aligned for breast_cancer)."
        ),
    )
    parser.add_argument(
        "--tune-fuzzy-threshold-calibrated",
        action="store_true",
        help="Fit a validation calibrator (Platt/isotonic) before threshold tuning for each fuzzy model.",
    )
    parser.add_argument(
        "--dffl-one-phase",
        action="store_true",
        help="Train DFFL in one-phase mode (bootstrap + single joint fit) without stage-wise refinement.",
    )
    parser.add_argument(
        "--dffl-concept-expert-labels",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping 'stage/block/concept' -> score in [0,1] "
            "for expert semantic validation metrics."
        ),
    )
    parser.add_argument(
        "--dffl-fast-gpu",
        action="store_true",
        help="Apply speed-oriented DFFL caps to reduce CPU-heavy rule/prototype generation in GPU runs.",
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
            "ruanfis_shallow, ruanfis_stacked_anfis, ruanfis_hierarchical_anfis, ruanfis_kanfis, "
            "ruanfis_refined_deep (aliases: shallow, stacked, hierarchical, kanfis, ka_anfis, dffl)."
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
    parser.add_argument(
        "--v18-controls-dir",
        type=Path,
        default=None,
        help="Optional directory for v18 control CSV exports (LR top-K, correlations, importance profile).",
    )
    parser.add_argument(
        "--v18-lr-budgets",
        type=str,
        default="100,200,400",
        help="Comma-separated budgets for LR-on-topK control.",
    )
    parser.add_argument(
        "--v18-random-state",
        type=int,
        default=42,
        help="Random seed for v18 control diagnostics.",
    )
    parser.add_argument(
        "--v18-export-h-artifacts",
        action="store_true",
        help="Export KAFN rule activation matrices and full-model probabilities for H-based stable-selection checks.",
    )
    args = parser.parse_args()
    _apply_gpu_only_mode(args)
    args.feature_geometry = _validate_feature_geometry(args.feature_geometry)
    args.binary_heavy_grouping = _validate_binary_heavy_grouping_mode(args.binary_heavy_grouping)
    dffl_dataset_overrides = parse_dffl_dataset_overrides(args.dffl_dataset_overrides)
    dffl_concept_expert_labels: dict[str, float] | None = None
    if args.dffl_concept_expert_labels is not None:
        payload = json.loads(args.dffl_concept_expert_labels.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--dffl-concept-expert-labels must point to a JSON object.")
        parsed_labels: dict[str, float] = {}
        for raw_key, raw_value in payload.items():
            key = str(raw_key).strip()
            if not key:
                continue
            value = float(raw_value)
            if (not math.isfinite(value)) or value < 0.0 or value > 1.0:
                raise ValueError(
                    "--dffl-concept-expert-labels scores must lie in [0,1]. "
                    f"Invalid value for '{key}': {value}"
                )
            parsed_labels[key] = value
        dffl_concept_expert_labels = parsed_labels or None

    dataset_suite = parse_dataset_suite(args.dataset_suite)
    if dataset_suite != "custom":
        dataset_names = DATASET_SUITES[dataset_suite]
    else:
        dataset_names = parse_dataset_names(args.datasets)
    unknown = [name for name in dataset_names if name not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset names: {', '.join(unknown)}. Available: {', '.join(DATASETS.keys())}")

    seeds = resolve_seeds(args.seeds, args.seed_count)
    if not 0.0 < float(args.rule_probability_threshold) < 1.0:
        raise ValueError("--rule-probability-threshold must lie in (0, 1).")
    fuzzy_models = parse_fuzzy_model_names(args.fuzzy_models)
    fuzzy_distill_models = parse_optional_fuzzy_model_names(args.fuzzy_distill_models)
    fuzzy_hard_sample_models = parse_optional_fuzzy_model_names(args.fuzzy_hard_sample_models)
    v18_lr_budgets = _parse_int_csv(args.v18_lr_budgets)
    hidden_output_activation = None if args.hidden_output_activation == "identity" else args.hidden_output_activation

    dataset_results: dict[str, MultiSeedBenchmarkResult] = {}
    dataset_reports: dict[str, str] = {}
    dataset_protocols: dict[str, dict[str, object]] = {}

    for dataset_name in dataset_names:
        dataset_started_at = time.perf_counter()
        spec = DATASETS[dataset_name]
        dataset_features, dataset_targets = spec.loader()
        resolved_profile = resolve_dffl_profile(
            profile_name=args.dffl_profile,
            task_type=spec.task_type,
            n_samples=int(dataset_features.shape[0]),
            input_dim=int(dataset_features.shape[1]),
        )
        resolved_profile, resolved_profile_policy = _resolve_dataset_profile_override(
            resolved_profile,
            dataset_name=dataset_name,
            dataset_overrides=dffl_dataset_overrides,
        )
        resolved_profile = apply_dffl_profile_overrides(
            resolved_profile,
            rule_swap_ratio=args.dffl_rule_swap_ratio,
            rule_swap_min_keep=args.dffl_rule_swap_min_keep,
            total_rule_budget=args.dffl_total_rule_budget,
            bridge_score_interaction_weight=args.dffl_bridge_score_interaction_weight,
            bridge_score_stability_weight=args.dffl_bridge_score_stability_weight,
        )
        if _dataset_override_skip_builtin_budget_policy(
            dataset_name=dataset_name,
            dataset_overrides=dffl_dataset_overrides,
        ):
            resolved_dataset_budget_policy = "skipped_by_dataset_override"
        else:
            resolved_profile, resolved_dataset_budget_policy = apply_dataset_budget_reallocation(
                resolved_profile,
                dataset_name=dataset_name,
                task_type=spec.task_type,
                input_dim=int(dataset_features.shape[1]),
                total_budget_locked=(args.dffl_total_rule_budget is not None),
            )
        resolved_profile, resolved_field_policy = _apply_dataset_profile_field_overrides(
            resolved_profile,
            dataset_name=dataset_name,
            dataset_overrides=dffl_dataset_overrides,
            total_budget_locked=(args.dffl_total_rule_budget is not None),
        )
        resolved_fast_gpu_policy = "none"
        if args.dffl_fast_gpu:
            resolved_profile, resolved_fast_gpu_policy = apply_dffl_fast_gpu_profile(
                resolved_profile,
                task_type=spec.task_type,
                input_dim=int(dataset_features.shape[1]),
                n_samples=int(dataset_features.shape[0]),
                total_budget_locked=(args.dffl_total_rule_budget is not None),
            )
        resolved_policy_parts = [
            part
            for part in (
                resolved_dataset_budget_policy,
                resolved_profile_policy,
                resolved_field_policy,
                resolved_fast_gpu_policy,
            )
            if part != "none"
        ]
        resolved_dataset_budget_policy = "+".join(resolved_policy_parts) if resolved_policy_parts else "none"
        per_seed_results = []
        per_seed_stage_runtime: list[dict[str, float | int | None]] = []
        total_seeds = len(seeds)
        for seed_index, seed in enumerate(seeds, start=1):
            print(
                f"[progress] dataset={dataset_name} seed={seed} start ({seed_index}/{total_seeds})",
                flush=True,
            )
            seed_results, seed_stage_runtime = run_single_seed_dataset_benchmark(
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
                dffl_concept_expert_labels=dffl_concept_expert_labels,
                dffl_dataset_overrides=dffl_dataset_overrides,
                feature_geometry=args.feature_geometry,
                binary_heavy_grouping_mode=args.binary_heavy_grouping,
                dffl_rule_swap_ratio=args.dffl_rule_swap_ratio,
                dffl_rule_swap_min_keep=args.dffl_rule_swap_min_keep,
                dffl_adaptive_rule_swap=not args.disable_dffl_adaptive_rule_swap,
                dffl_total_rule_budget=args.dffl_total_rule_budget,
                dffl_bridge_score_interaction_weight=args.dffl_bridge_score_interaction_weight,
                dffl_bridge_score_stability_weight=args.dffl_bridge_score_stability_weight,
                dffl_adaptive_budget=not args.disable_dffl_adaptive_budget,
                dffl_adaptive_local_arity=not args.disable_dffl_adaptive_local_arity,
                dffl_fast_gpu=bool(args.dffl_fast_gpu),
                dffl_tiny_restarts=max(1, int(args.dffl_tiny_restarts)),
                stacked_width_scale=args.stacked_width_scale,
                stacked_rule_scale=args.stacked_rule_scale,
                stacked_stagewise_init=bool(args.stacked_stagewise_init),
                stacked_stagewise_init_epochs=int(args.stacked_stagewise_init_epochs),
                stacked_final_skip_input_count=int(args.stacked_final_skip_inputs),
                stacked_final_skip_mode=str(args.stacked_final_skip_mode),
                stacked_final_skip_gates_enabled=bool(args.stacked_final_skip_gates),
                stacked_final_skip_gate_init_logit=float(args.stacked_final_skip_gate_init_logit),
                stacked_final_skip_gate_l1_weight=float(args.stacked_final_skip_gate_l1_weight),
                hierarchical_width_scale=args.hierarchical_width_scale,
                hierarchical_rule_scale=args.hierarchical_rule_scale,
                hierarchical_group_size=args.hierarchical_group_size,
                stacked_prototype_scoring_mode=args.stacked_prototype_scoring_mode,
                hierarchical_prototype_scoring_mode=args.hierarchical_prototype_scoring_mode,
                hidden_output_activation=hidden_output_activation,
                fuzzy_binary_loss_name=args.fuzzy_binary_loss_name,
                fuzzy_binary_focal_gamma=args.fuzzy_binary_focal_gamma,
                fuzzy_binary_auto_pos_weight=args.fuzzy_binary_auto_pos_weight,
                fuzzy_binary_soft_f1_weight=args.fuzzy_binary_soft_f1_weight,
                fuzzy_regression_loss=args.fuzzy_regression_loss,
                fuzzy_huber_delta=args.fuzzy_huber_delta,
                fuzzy_distill_weight=args.fuzzy_distill_weight,
                fuzzy_distill_models=fuzzy_distill_models,
                fuzzy_distill_teacher_trees=args.fuzzy_distill_teacher_trees,
                fuzzy_hard_sample_training=bool(args.fuzzy_hard_sample_training),
                fuzzy_hard_sample_models=fuzzy_hard_sample_models,
                fuzzy_hard_sample_source=str(args.fuzzy_hard_sample_source),
                fuzzy_hard_sample_fraction=float(args.fuzzy_hard_sample_fraction),
                fuzzy_hard_sample_multiplier=int(args.fuzzy_hard_sample_multiplier),
                fuzzy_hard_sample_finetune_epochs=int(args.fuzzy_hard_sample_finetune_epochs),
                fuzzy_hard_sample_finetune_patience=int(args.fuzzy_hard_sample_finetune_patience),
                fuzzy_hard_sample_teacher_trees=int(args.fuzzy_hard_sample_teacher_trees),
                kanfis_active_features=int(args.kanfis_active_features),
                kanfis_depth=int(args.kanfis_depth),
                kanfis_superposition_terms=int(args.kanfis_superposition_terms),
                kanfis_concept_fan_in=int(args.kanfis_concept_fan_in),
                kanfis_routing=str(args.kanfis_routing),
                kanfis_feature_order=str(args.kanfis_feature_order),
                kanfis_decision_skip_features=int(args.kanfis_decision_skip_features),
                kanfis_pair_count=int(args.kanfis_pair_count),
                kanfis_projection_count=int(args.kanfis_projection_count),
                kanfis_projection_width=int(args.kanfis_projection_width),
                kanfis_projection_mode=str(args.kanfis_projection_mode),
                kanfis_prune_rules=int(args.kanfis_prune_rules),
                kanfis_importance_split=str(args.kanfis_importance_split),
                kanfis_recovery_epochs=int(args.kanfis_recovery_epochs),
                kanfis_polish_epochs=int(args.kanfis_polish_epochs),
                kanfis_train_rule_gates=bool(args.kanfis_train_rule_gates),
                kanfis_rule_sparsity_weight=float(args.kanfis_rule_sparsity_weight),
                kanfis_rule_entropy_weight=float(args.kanfis_rule_entropy_weight),
                kanfis_rule_anchor_weight=float(args.kanfis_rule_anchor_weight),
                kanfis_skip_gate_l1_weight=float(args.kanfis_skip_gate_l1_weight),
                batch_size=args.batch_size,
                patience=args.patience,
                classification_threshold=args.classification_threshold,
                tune_fuzzy_threshold=args.tune_fuzzy_threshold,
                tune_fuzzy_threshold_calibrated=args.tune_fuzzy_threshold_calibrated,
                log_epochs=bool(args.log_epochs),
                log_epochs_every=max(1, int(args.log_epochs_every)),
                rule_probability_threshold=float(args.rule_probability_threshold),
                dffl_one_phase=args.dffl_one_phase,
                device=args.device,
                fuzzy_models=fuzzy_models,
                include_sklearn=not args.skip_sklearn,
                v18_controls_dir=args.v18_controls_dir,
                v18_lr_budgets=v18_lr_budgets,
                v18_random_state=int(args.v18_random_state),
                v18_export_h_artifacts=bool(args.v18_export_h_artifacts),
            )
            per_seed_results.append(tuple(seed_results))
            per_seed_stage_runtime.append(
                {
                    "seed": int(seed),
                    "full_train_sec": seed_stage_runtime.get("full_train_sec"),
                    "h_build_sec": seed_stage_runtime.get("h_build_sec"),
                    "selection_sec": seed_stage_runtime.get("selection_sec"),
                    "refit_head_sec": seed_stage_runtime.get("refit_head_sec"),
                    "inference_ms_per_sample": seed_stage_runtime.get("inference_ms_per_sample"),
                    "memory_peak_mb": seed_stage_runtime.get("memory_peak_mb"),
                }
            )
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
        dffl_protocol: dict[str, object]
        if "ruanfis_refined_deep" in fuzzy_models:
            targets_array = np.asarray(dataset_targets).astype(np.float32)
            if targets_array.ndim == 1:
                targets_tensor = torch.from_numpy(targets_array.reshape(-1, 1))
            else:
                targets_tensor = torch.from_numpy(targets_array)
            full_inputs_tensor = torch.from_numpy(np.asarray(dataset_features).astype(np.float32))
            full_analysis_max_samples = 80_000
            if input_dim >= 96:
                full_analysis_max_samples = 20_000
            elif input_dim >= 64:
                full_analysis_max_samples = 40_000
            full_inputs_tensor, targets_tensor = _select_bootstrap_subset(
                full_inputs_tensor,
                targets_tensor,
                max_samples=full_analysis_max_samples,
                seed=seeds[0] + 43,
            )
            full_analysis_device_request = args.device
            if (
                full_analysis_device_request is not None
                and str(full_analysis_device_request).strip().lower().startswith("cuda")
                and input_dim >= 96
            ):
                full_analysis_device_request = "cpu"
                print(
                    (
                        "[progress] dataset={dataset} dffl_post_analysis_device_override: "
                        "cpu (input_dim={dim})"
                    ).format(dataset=dataset_name, dim=input_dim),
                    flush=True,
                )
            analysis_full_inputs, analysis_full_targets, analysis_full_device = _prepare_structure_analysis_tensors(
                inputs=full_inputs_tensor,
                targets=targets_tensor,
                device=full_analysis_device_request,
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
            dffl_protocol = {
                "enabled": True,
                "resolved_profile": resolved_profile.name,
                "feature_geometry_requested": str(args.feature_geometry),
                "feature_geometry_effective": dataset_feature_geometry_effective,
                "feature_geometry_policy": dataset_feature_geometry_policy,
                "analysis_device_effective": analysis_full_device,
                "dataset_budget_policy": resolved_dataset_budget_policy,
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
            }
        else:
            dffl_protocol = {
                "enabled": False,
                "resolved_profile": resolved_profile.name,
                "skip_reason": "DFFL model was not requested for this run.",
            }
        dataset_protocols[dataset_name] = {
            "task_type": spec.task_type,
            "n_samples": n_samples,
            "input_dim": input_dim,
            "primary_metric": PRIMARY_METRIC[spec.task_type],
            "runtime_seconds": float(time.perf_counter() - dataset_started_at),
            "runtime_stages_per_seed": per_seed_stage_runtime,
            "runtime_stages_summary": _summarize_stage_runtime(per_seed_stage_runtime),
            "architectures": {
                "ruanfis_shallow": _summarize_shallow_architecture(input_dim),
                "ruanfis_stacked_anfis": _summarize_stacked_architecture(input_dim),
                "ruanfis_hierarchical_anfis": _summarize_hierarchical_anfis_architecture(input_dim),
            },
            "dffl": dffl_protocol,
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
        dataset_suite=dataset_suite,
        dataset_names=dataset_names,
        seeds=seeds,
        fuzzy_models=fuzzy_models,
        dffl_dataset_overrides=dffl_dataset_overrides,
        dataset_protocols=dataset_protocols,
    )
    reproducibility_manifest_markdown = build_reproducibility_manifest_markdown(reproducibility_manifest_payload)

    full_report_sections = [
        "UNIFIED REAL-DATASET BENCHMARK",
        f"dataset_suite: {dataset_suite}",
        f"datasets: {', '.join(dataset_names)}",
        f"seed_policy: {'deterministic_pool' if int(args.seed_count) > 0 else 'explicit_argument'}",
        f"seed_count_requested: {args.seed_count}",
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
        f"dffl_dataset_overrides: {json.dumps(dffl_dataset_overrides, ensure_ascii=False)}",
        f"dffl_one_phase: {args.dffl_one_phase}",
        f"dffl_fast_gpu: {args.dffl_fast_gpu}",
        f"gpu_only: {args.gpu_only}",
        f"fuzzy_models: {', '.join(fuzzy_models)}",
        f"fuzzy_distill_weight: {args.fuzzy_distill_weight}",
        f"fuzzy_distill_models: {', '.join(fuzzy_distill_models) if fuzzy_distill_models else 'none'}",
        f"fuzzy_distill_teacher_trees: {args.fuzzy_distill_teacher_trees}",
        f"fuzzy_hard_sample_training: {args.fuzzy_hard_sample_training}",
        (
            "fuzzy_hard_sample_models: "
            f"{', '.join(fuzzy_hard_sample_models) if fuzzy_hard_sample_models else 'none'}"
        ),
        f"fuzzy_hard_sample_source: {args.fuzzy_hard_sample_source}",
        f"fuzzy_hard_sample_fraction: {args.fuzzy_hard_sample_fraction}",
        f"fuzzy_hard_sample_multiplier: {args.fuzzy_hard_sample_multiplier}",
        f"fuzzy_hard_sample_finetune_epochs: {args.fuzzy_hard_sample_finetune_epochs}",
        f"fuzzy_hard_sample_finetune_patience: {args.fuzzy_hard_sample_finetune_patience}",
        f"fuzzy_hard_sample_teacher_trees: {args.fuzzy_hard_sample_teacher_trees}",
        f"kanfis_active_features: {args.kanfis_active_features}",
        f"kanfis_depth: {args.kanfis_depth}",
        f"kanfis_superposition_terms: {args.kanfis_superposition_terms}",
        f"kanfis_concept_fan_in: {args.kanfis_concept_fan_in}",
        f"kanfis_routing: {args.kanfis_routing}",
        f"kanfis_feature_order: {args.kanfis_feature_order}",
        f"kanfis_decision_skip_features: {args.kanfis_decision_skip_features}",
        f"kanfis_pair_count: {args.kanfis_pair_count}",
        f"kanfis_projection_count: {args.kanfis_projection_count}",
        f"kanfis_projection_width: {args.kanfis_projection_width}",
        f"kanfis_projection_mode: {args.kanfis_projection_mode}",
        f"kanfis_prune_rules: {args.kanfis_prune_rules}",
        f"kanfis_importance_split: {args.kanfis_importance_split}",
        f"kanfis_recovery_epochs: {args.kanfis_recovery_epochs}",
        f"kanfis_polish_epochs: {args.kanfis_polish_epochs}",
        f"kanfis_train_rule_gates: {args.kanfis_train_rule_gates}",
        f"kanfis_rule_sparsity_weight: {args.kanfis_rule_sparsity_weight}",
        f"kanfis_rule_entropy_weight: {args.kanfis_rule_entropy_weight}",
        f"kanfis_rule_anchor_weight: {args.kanfis_rule_anchor_weight}",
        f"kanfis_skip_gate_l1_weight: {args.kanfis_skip_gate_l1_weight}",
        f"hidden_output_activation: {args.hidden_output_activation}",
        f"stacked_stagewise_init: {args.stacked_stagewise_init}",
        f"stacked_stagewise_init_epochs: {args.stacked_stagewise_init_epochs}",
        f"stacked_final_skip_inputs: {args.stacked_final_skip_inputs}",
        f"stacked_final_skip_mode: {args.stacked_final_skip_mode}",
        f"stacked_final_skip_gates: {args.stacked_final_skip_gates}",
        f"stacked_final_skip_gate_init_logit: {args.stacked_final_skip_gate_init_logit}",
        f"stacked_final_skip_gate_l1_weight: {args.stacked_final_skip_gate_l1_weight}",
        f"tune_fuzzy_threshold: {args.tune_fuzzy_threshold}",
        f"tune_fuzzy_threshold_calibrated: {args.tune_fuzzy_threshold_calibrated}",
        f"rule_probability_threshold: {args.rule_probability_threshold}",
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
            "dataset_suite": str(dataset_suite),
            "datasets": list(dataset_names),
            "seed_policy": "deterministic_pool" if int(args.seed_count) > 0 else "explicit_argument",
            "seed_count_requested": int(args.seed_count),
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
            "dffl_dataset_overrides": {key: dict(value) for key, value in dffl_dataset_overrides.items()},
            "dffl_one_phase": bool(args.dffl_one_phase),
            "dffl_fast_gpu": bool(args.dffl_fast_gpu),
            "gpu_only": bool(args.gpu_only),
            "fuzzy_models": list(fuzzy_models),
            "fuzzy_distill_weight": float(args.fuzzy_distill_weight),
            "fuzzy_distill_models": list(fuzzy_distill_models),
            "fuzzy_distill_teacher_trees": int(args.fuzzy_distill_teacher_trees),
            "fuzzy_hard_sample_training": bool(args.fuzzy_hard_sample_training),
            "fuzzy_hard_sample_models": list(fuzzy_hard_sample_models),
            "fuzzy_hard_sample_source": str(args.fuzzy_hard_sample_source),
            "fuzzy_hard_sample_fraction": float(args.fuzzy_hard_sample_fraction),
            "fuzzy_hard_sample_multiplier": int(args.fuzzy_hard_sample_multiplier),
            "fuzzy_hard_sample_finetune_epochs": int(args.fuzzy_hard_sample_finetune_epochs),
            "fuzzy_hard_sample_finetune_patience": int(args.fuzzy_hard_sample_finetune_patience),
            "fuzzy_hard_sample_teacher_trees": int(args.fuzzy_hard_sample_teacher_trees),
            "kanfis_prune_rules": int(args.kanfis_prune_rules),
            "kanfis_skip_gate_l1_weight": float(args.kanfis_skip_gate_l1_weight),
            "kanfis_importance_split": str(args.kanfis_importance_split),
            "tune_fuzzy_threshold": bool(args.tune_fuzzy_threshold),
            "tune_fuzzy_threshold_calibrated": bool(args.tune_fuzzy_threshold_calibrated),
            "rule_probability_threshold": float(args.rule_probability_threshold),
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
