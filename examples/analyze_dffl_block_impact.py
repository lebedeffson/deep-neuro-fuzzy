from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from examples.run_real_datasets_benchmark import (  # noqa: E402
    DATASETS,
    PRIMARY_METRIC,
    _detect_binary_feature_indices,
    build_dffl_config,
    make_dffl_bridge_pairs,
    make_dffl_input_groups,
    prepare_dataset_split,
    resolve_dffl_profile,
    set_seed,
)
from ruanfis import (  # noqa: E402
    BootstrapConfig,
    RefinementLoopConfig,
    StagewisePretrainingConfig,
    TrainingConfig,
    build_refined_hierarchical_model,
)
from ruanfis.metrics import TaskType, compute_metrics  # noqa: E402


@dataclass(frozen=True)
class BlockImpact:
    stage_index: int
    stage_name: str
    block_index: int
    block_name: str
    input_indices: tuple[int, ...]
    n_rules: int
    output_dim: int
    val_metric: float
    test_metric: float
    val_delta: float
    test_delta: float


def _eval_primary_metric(
    model: torch.nn.Module,
    *,
    task_type: TaskType,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    top_k_rules: int | None,
) -> float:
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        logits = model(inputs.to(device=device, dtype=torch.float32), top_k_rules=top_k_rules).detach().cpu()
    metrics = compute_metrics(
        task_type,
        logits,
        targets.detach().cpu(),
        classification_threshold=threshold,
    )
    return float(metrics[PRIMARY_METRIC[task_type]])


def _zero_forward(self, inputs: torch.Tensor, top_k_rules: int | None = None) -> torch.Tensor:  # noqa: ARG001
    return torch.zeros(
        inputs.size(0),
        self.output_dim,
        dtype=inputs.dtype,
        device=inputs.device,
    )


def _is_better(task_type: TaskType, left: float, right: float) -> bool:
    if task_type == "binary_classification":
        return left > right
    return left < right


def _format_delta(task_type: TaskType, value: float) -> str:
    if task_type == "binary_classification":
        return f"{value:+.4f}"
    return f"{-value:+.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DFFL and compute per-block impact by zero-ablation.")
    parser.add_argument("--dataset", type=str, default="covtype_binary_8000", choices=sorted(DATASETS.keys()))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--dffl-profile", type=str, default="quality_auto")
    parser.add_argument("--dffl-learning-rate", type=float, default=0.015)
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--decision-pretrain-epochs", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--refinement-cycles", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    set_seed(args.seed)
    split = prepare_dataset_split(
        spec,
        seed=args.seed,
        test_size=args.test_size,
        validation_size=args.validation_size,
    )
    profile = resolve_dffl_profile(
        profile_name=args.dffl_profile,
        task_type=spec.task_type,
        n_samples=split.n_samples,
        input_dim=split.input_dim,
    )
    groups = make_dffl_input_groups(profile, train_inputs=split.train_inputs, train_targets=split.train_targets)
    bridge_pairs = make_dffl_bridge_pairs(
        profile,
        train_inputs=split.train_inputs,
        train_targets=split.train_targets,
        input_groups=groups,
    )
    binary_feature_indices = _detect_binary_feature_indices(split.train_inputs)
    dffl_lr = (
        args.dffl_learning_rate * profile.learning_rate_scale_classification
        if spec.task_type == "binary_classification"
        else args.dffl_learning_rate * profile.learning_rate_scale_regression
    )
    dffl_ref_cycles = max(args.refinement_cycles, profile.refinement_cycle_floor)

    result = build_refined_hierarchical_model(
        build_dffl_config(
            split.input_dim,
            profile=profile,
            input_groups=groups,
            binary_feature_indices=binary_feature_indices,
            bridge_feature_pairs=bridge_pairs,
        ),
        train_inputs=split.train_inputs,
        train_targets=split.train_targets,
        validation_inputs=split.validation_inputs,
        validation_targets=split.validation_targets,
        bootstrap_config=BootstrapConfig(
            decision_task_type=spec.task_type,
            hidden_high=0.75 if spec.task_type == "binary_classification" else 0.8,
            hidden_low=0.25 if spec.task_type == "binary_classification" else 0.2,
            gate_floor=0.15,
            gate_ceiling=0.9,
        ),
        pretraining_config=StagewisePretrainingConfig(
            task_type=spec.task_type,
            epochs_per_stage=args.pretrain_epochs,
            decision_epochs=args.decision_pretrain_epochs,
            refinement_rounds=profile.pretrain_refinement_rounds,
            learning_rate=dffl_lr,
            batch_size=args.batch_size,
            shuffle=True,
            rule_sparsity_weight=0.0,
            stage_selection_metric="auto",
            stage_selection_threshold=args.classification_threshold,
        ),
        training_config=TrainingConfig(
            task_type=spec.task_type,
            max_epochs=args.max_epochs,
            learning_rate=dffl_lr,
            patience=min(args.patience, args.max_epochs),
            batch_size=args.batch_size,
            shuffle=True,
            classification_threshold=args.classification_threshold,
            monitor_metric="f1" if spec.task_type == "binary_classification" else None,
            monitor_mode="max" if spec.task_type == "binary_classification" else None,
            top_k_rules=profile.top_k_rules,
            binary_auto_pos_weight=spec.task_type == "binary_classification" and profile.binary_auto_pos_weight,
            weight_decay=profile.weight_decay,
            gradient_clip_norm=profile.gradient_clip_norm,
            rule_sparsity_weight=profile.rule_sparsity_weight,
            rule_length_weight=profile.rule_length_weight,
            decision_usage_balance_weight=profile.decision_usage_balance_weight,
            prune_after_fit=False,
            device=args.device,
        ),
        refinement_loop_config=RefinementLoopConfig(
            max_cycles=dffl_ref_cycles,
            patience=1,
            min_delta=1e-4,
        ),
    )
    model = result.model

    base_val = _eval_primary_metric(
        model,
        task_type=spec.task_type,
        inputs=split.validation_inputs,
        targets=split.validation_targets,
        threshold=args.classification_threshold,
        top_k_rules=profile.top_k_rules,
    )
    base_test = _eval_primary_metric(
        model,
        task_type=spec.task_type,
        inputs=split.test_inputs,
        targets=split.test_targets,
        threshold=args.classification_threshold,
        top_k_rules=profile.top_k_rules,
    )

    impacts: list[BlockImpact] = []
    for stage_index, stage in enumerate(model.stages):
        for block_index, connected_block in enumerate(stage.blocks):
            original_forward = connected_block.block.forward
            connected_block.block.forward = types.MethodType(_zero_forward, connected_block.block)
            try:
                val_metric = _eval_primary_metric(
                    model,
                    task_type=spec.task_type,
                    inputs=split.validation_inputs,
                    targets=split.validation_targets,
                    threshold=args.classification_threshold,
                    top_k_rules=profile.top_k_rules,
                )
                test_metric = _eval_primary_metric(
                    model,
                    task_type=spec.task_type,
                    inputs=split.test_inputs,
                    targets=split.test_targets,
                    threshold=args.classification_threshold,
                    top_k_rules=profile.top_k_rules,
                )
            finally:
                connected_block.block.forward = original_forward

            impacts.append(
                BlockImpact(
                    stage_index=stage_index,
                    stage_name=stage.name,
                    block_index=block_index,
                    block_name=connected_block.block.name,
                    input_indices=tuple(int(idx) for idx in connected_block.input_indices.detach().cpu().tolist()),
                    n_rules=int(connected_block.block.n_rules),
                    output_dim=int(connected_block.block.output_dim),
                    val_metric=float(val_metric),
                    test_metric=float(test_metric),
                    val_delta=float(val_metric - base_val),
                    test_delta=float(test_metric - base_test),
                )
            )

    impacts_sorted = sorted(impacts, key=lambda item: item.test_delta, reverse=True)
    direction = "higher-is-better" if spec.task_type == "binary_classification" else "lower-is-better"
    print(
        f"dataset={args.dataset} seed={args.seed} task={spec.task_type} metric={PRIMARY_METRIC[spec.task_type]} ({direction})"
    )
    print(f"baseline: val={base_val:.4f} test={base_test:.4f}")
    print("block impacts (top harmful first, i.e. removal helps):")
    for item in impacts_sorted:
        print(
            f"- stage={item.stage_index}:{item.stage_name} block={item.block_index}:{item.block_name} "
            f"rules={item.n_rules} out={item.output_dim} "
            f"delta_val={_format_delta(spec.task_type, item.val_delta)} "
            f"delta_test={_format_delta(spec.task_type, item.test_delta)}"
        )

    harmful = [x for x in impacts_sorted if _is_better(spec.task_type, x.test_metric, base_test)]
    print(f"harmful_blocks={len(harmful)} / {len(impacts_sorted)}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": args.dataset,
            "seed": args.seed,
            "task_type": spec.task_type,
            "metric": PRIMARY_METRIC[spec.task_type],
            "dffl_profile_requested": args.dffl_profile,
            "dffl_profile_resolved": profile.name,
            "baseline": {"val": base_val, "test": base_test},
            "harmful_blocks": len(harmful),
            "total_blocks": len(impacts_sorted),
            "impacts": [asdict(item) for item in impacts_sorted],
        }
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
