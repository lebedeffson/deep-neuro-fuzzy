from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

from torch import Tensor

from .bootstrap import (
    BootstrapConfig,
    StagewisePretrainingConfig,
    build_stagewise_pretrained_hierarchical_model,
    build_stagewise_pretrained_shallow_model,
    reestimate_hierarchical_model_rule_base,
    reestimate_shallow_model_rule_base,
)
from .builders import HierarchicalModelConfig, ShallowFuzzyModelConfig
from .model import DeepFuzzyFeatureModel
from .trainer import FuzzyTrainer, TrainingConfig, TrainingResult


@dataclass(frozen=True)
class RefinementLoopConfig:
    max_cycles: int = 2
    patience: int | None = None
    min_delta: float = 0.0


@dataclass(frozen=True)
class RefinementCycleRecord:
    cycle_index: int
    source: str
    train_loss: float
    train_metrics: dict[str, float]
    validation_loss: float | None
    validation_metrics: dict[str, float] | None
    monitor_name: str
    monitor_value: float


@dataclass(frozen=True)
class RefinementResult:
    model: DeepFuzzyFeatureModel
    training_result: TrainingResult
    cycle_records: tuple[RefinementCycleRecord, ...]
    best_cycle_index: int
    monitor_name: str
    best_monitor_value: float
    cycles_ran: int
    stopped_early: bool


def _resolve_monitor_mode(training_config: TrainingConfig) -> str:
    if training_config.monitor_mode is not None:
        return training_config.monitor_mode
    return "min"


def _run_fine_tuning_cycle(
    model: DeepFuzzyFeatureModel,
    *,
    training_config: TrainingConfig,
    train_inputs: Tensor,
    train_targets: Tensor,
    validation_inputs: Tensor | None,
    validation_targets: Tensor | None,
) -> TrainingResult:
    trainer = FuzzyTrainer(model, training_config)
    return trainer.fit(
        train_inputs,
        train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
    )


def _resolve_loop_config(
    *,
    refinement_cycles: int,
    refinement_loop_config: RefinementLoopConfig | None,
) -> RefinementLoopConfig:
    if refinement_loop_config is None:
        loop_config = RefinementLoopConfig(max_cycles=refinement_cycles)
    else:
        loop_config = refinement_loop_config

    if loop_config.max_cycles <= 0:
        raise ValueError("max_cycles must be positive.")
    if loop_config.patience is not None and loop_config.patience < 0:
        raise ValueError("patience must be non-negative when provided.")
    if loop_config.min_delta < 0.0:
        raise ValueError("min_delta must be non-negative.")
    return loop_config


def _resolve_pretraining_device(training_config: TrainingConfig) -> str | None:
    return training_config.device


def _run_refinement_loop(
    *,
    initial_builder: Callable[[], DeepFuzzyFeatureModel],
    reestimate_builder: Callable[[DeepFuzzyFeatureModel], DeepFuzzyFeatureModel],
    training_config: TrainingConfig,
    train_inputs: Tensor,
    train_targets: Tensor,
    validation_inputs: Tensor | None,
    validation_targets: Tensor | None,
    loop_config: RefinementLoopConfig,
) -> RefinementResult:
    current_model = initial_builder()
    best_model: DeepFuzzyFeatureModel | None = None
    best_training_result: TrainingResult | None = None
    best_cycle_index = 0
    monitor_mode = _resolve_monitor_mode(training_config)
    if monitor_mode == "min":
        best_monitor_value = float("inf")
    elif monitor_mode == "max":
        best_monitor_value = float("-inf")
    else:
        raise ValueError(f"Unsupported monitor mode: {monitor_mode}. Expected 'min' or 'max'.")
    best_monitor_name = (
        f"validation_{training_config.monitor_metric}"
        if training_config.monitor_metric is not None and validation_inputs is not None
        else (
            f"train_{training_config.monitor_metric}"
            if training_config.monitor_metric is not None
            else ("validation_loss" if validation_inputs is not None else "train_loss")
        )
    )
    cycles_without_improvement = 0
    stopped_early = False
    cycle_records: list[RefinementCycleRecord] = []

    for cycle_index in range(loop_config.max_cycles):
        if cycle_index > 0:
            reference_model = best_model if best_model is not None else current_model
            current_model = reestimate_builder(reference_model)

        source = "stagewise_pretrained" if cycle_index == 0 else "reestimated_from_best"
        training_result = _run_fine_tuning_cycle(
            current_model,
            training_config=training_config,
            train_inputs=train_inputs,
            train_targets=train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
        )
        monitor_name = training_result.monitor_name
        monitor_value = float(training_result.best_monitor_value)
        cycle_records.append(
            RefinementCycleRecord(
                cycle_index=cycle_index,
                source=source,
                train_loss=training_result.train_loss,
                train_metrics=dict(training_result.train_metrics),
                validation_loss=training_result.validation_loss,
                validation_metrics=(
                    dict(training_result.validation_metrics)
                    if training_result.validation_metrics is not None
                    else None
                ),
                monitor_name=monitor_name,
                monitor_value=monitor_value,
            )
        )

        improved = (
            monitor_value < best_monitor_value - loop_config.min_delta
            if monitor_mode == "min"
            else monitor_value > best_monitor_value + loop_config.min_delta
        )
        if improved:
            best_monitor_value = monitor_value
            best_monitor_name = monitor_name
            best_cycle_index = cycle_index
            best_model = copy.deepcopy(current_model)
            best_training_result = training_result
            cycles_without_improvement = 0
        else:
            cycles_without_improvement += 1
            if loop_config.patience is not None and cycles_without_improvement > loop_config.patience:
                stopped_early = True
                break

    if best_model is None or best_training_result is None:
        raise RuntimeError("Refinement loop failed to produce a trained model.")

    return RefinementResult(
        model=best_model,
        training_result=best_training_result,
        cycle_records=tuple(cycle_records),
        best_cycle_index=best_cycle_index,
        monitor_name=best_monitor_name,
        best_monitor_value=best_monitor_value,
        cycles_ran=len(cycle_records),
        stopped_early=stopped_early,
    )


def build_refined_hierarchical_model(
    config: HierarchicalModelConfig,
    *,
    train_inputs: Tensor,
    train_targets: Tensor,
    validation_inputs: Tensor | None = None,
    validation_targets: Tensor | None = None,
    bootstrap_config: BootstrapConfig | None = None,
    pretraining_config: StagewisePretrainingConfig | None = None,
    training_config: TrainingConfig,
    refinement_cycles: int = 2,
    refinement_loop_config: RefinementLoopConfig | None = None,
) -> RefinementResult:
    if (validation_inputs is None) != (validation_targets is None):
        raise ValueError("validation_inputs and validation_targets must either both be provided or both be omitted.")
    loop_config = _resolve_loop_config(
        refinement_cycles=refinement_cycles,
        refinement_loop_config=refinement_loop_config,
    )
    pretraining_device = _resolve_pretraining_device(training_config)

    return _run_refinement_loop(
        initial_builder=lambda: build_stagewise_pretrained_hierarchical_model(
            config,
            sample_inputs=train_inputs,
            sample_targets=train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            bootstrap_config=bootstrap_config,
            pretraining_config=pretraining_config,
            device=pretraining_device,
        ),
        reestimate_builder=lambda reference_model: reestimate_hierarchical_model_rule_base(
            config,
            reference_model=reference_model,
            sample_inputs=train_inputs,
            sample_targets=train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            bootstrap_config=bootstrap_config,
            pretraining_config=pretraining_config,
            device=pretraining_device,
        ),
        training_config=training_config,
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        loop_config=loop_config,
    )


def build_refined_shallow_model(
    config: ShallowFuzzyModelConfig,
    *,
    train_inputs: Tensor,
    train_targets: Tensor,
    validation_inputs: Tensor | None = None,
    validation_targets: Tensor | None = None,
    bootstrap_config: BootstrapConfig | None = None,
    pretraining_config: StagewisePretrainingConfig | None = None,
    training_config: TrainingConfig,
    refinement_cycles: int = 2,
    refinement_loop_config: RefinementLoopConfig | None = None,
) -> RefinementResult:
    if (validation_inputs is None) != (validation_targets is None):
        raise ValueError("validation_inputs and validation_targets must either both be provided or both be omitted.")
    loop_config = _resolve_loop_config(
        refinement_cycles=refinement_cycles,
        refinement_loop_config=refinement_loop_config,
    )
    pretraining_device = _resolve_pretraining_device(training_config)

    return _run_refinement_loop(
        initial_builder=lambda: build_stagewise_pretrained_shallow_model(
            config,
            sample_inputs=train_inputs,
            sample_targets=train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            bootstrap_config=bootstrap_config,
            pretraining_config=pretraining_config,
            device=pretraining_device,
        ),
        reestimate_builder=lambda reference_model: reestimate_shallow_model_rule_base(
            config,
            reference_model=reference_model,
            sample_inputs=train_inputs,
            sample_targets=train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            bootstrap_config=bootstrap_config,
            pretraining_config=pretraining_config,
            device=pretraining_device,
        ),
        training_config=training_config,
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=validation_inputs,
        validation_targets=validation_targets,
        loop_config=loop_config,
    )
