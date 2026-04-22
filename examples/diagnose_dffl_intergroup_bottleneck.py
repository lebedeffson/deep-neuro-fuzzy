from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_linnerud,
    load_wine,
)

from run_real_datasets_benchmark import (
    DATASETS,
    PRIMARY_METRIC,
    BootstrapConfig,
    RefinementLoopConfig,
    StagewisePretrainingConfig,
    TrainingConfig,
    _detect_binary_feature_indices,
    build_dffl_config,
    build_refined_hierarchical_model,
    make_dffl_bridge_pairs,
    make_dffl_input_groups,
    prepare_dataset_split,
    resolve_dffl_profile,
    set_seed,
)
from ruanfis.metrics import TaskType, compute_metrics


@dataclass(frozen=True)
class PairInteraction:
    feature_i: int
    feature_j: int
    feature_i_name: str
    feature_j_name: str
    strength: float
    is_intergroup: bool


@dataclass(frozen=True)
class Stage2BlockStats:
    block_name: str
    n_rules: int
    n_intergroup_rules: int
    intergroup_rule_ratio: float
    weighted_intergroup_share: float
    dominant_intergroup_share: float
    active_rules: int
    active_intergroup_rules: int
    active_intergroup_ratio: float


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
    names.extend([f"Wilderness_Area_{i}" for i in range(1, 5)])
    names.extend([f"Soil_Type_{i}" for i in range(1, 41)])
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
            names = list(getattr(data, "feature_names", [])) or [f"pixel_{i}" for i in range(data.data.shape[1])]
        elif dataset_name == "california_housing":
            names = list(fetch_california_housing().feature_names)
        elif dataset_name in {"covtype_binary_20000", "covtype_binary_8000"}:
            names = _covtype_feature_names()
        elif dataset_name == "linnerud_weight":
            names = list(load_linnerud().feature_names)
        else:
            names = []
    except Exception:
        names = []
    if len(names) < input_dim:
        names.extend([f"feature_{i}" for i in range(len(names), input_dim)])
    return names[:input_dim]


def _build_feature_to_group(input_groups: tuple[tuple[int, ...], ...], input_dim: int) -> np.ndarray:
    mapping = np.full(input_dim, fill_value=-1, dtype=np.int64)
    for group_index, group in enumerate(input_groups):
        for feature_index in group:
            mapping[int(feature_index)] = group_index
    return mapping


def _pairwise_interaction_strengths(
    *,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    feature_group: np.ndarray,
    top_k: int,
) -> tuple[list[PairInteraction], float, float]:
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    y_norm = float(np.sqrt(np.sum(y_centered * y_centered)) + 1e-12)

    pairs: list[PairInteraction] = []
    total_strength = 0.0
    inter_strength = 0.0
    input_dim = x.shape[1]

    for i in range(input_dim):
        xi = x_centered[:, i]
        for j in range(i + 1, input_dim):
            xj = x_centered[:, j]
            z = xi * xj
            z_centered = z - z.mean()
            z_norm = float(np.sqrt(np.sum(z_centered * z_centered)) + 1e-12)
            strength = float(abs(np.dot(z_centered, y_centered) / (z_norm * y_norm)))
            is_inter = bool(feature_group[i] != feature_group[j])
            total_strength += strength
            if is_inter:
                inter_strength += strength
            pairs.append(
                PairInteraction(
                    feature_i=i,
                    feature_j=j,
                    feature_i_name=feature_names[i],
                    feature_j_name=feature_names[j],
                    strength=strength,
                    is_intergroup=is_inter,
                )
            )

    pairs.sort(key=lambda item: item.strength, reverse=True)
    if top_k > 0 and len(pairs) > top_k:
        pairs = pairs[:top_k]

    inter_share = inter_strength / (total_strength + 1e-12)
    return pairs, float(inter_share), float(total_strength)


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
    metrics = compute_metrics(task_type, logits, targets.detach().cpu(), classification_threshold=threshold)
    return float(metrics[PRIMARY_METRIC[task_type]])


def _stage1_flat_to_group_map(model: torch.nn.Module) -> list[int]:
    stage1 = model.stages[0]
    mapping: list[int] = []
    for group_index, connected_block in enumerate(stage1.blocks):
        mapping.extend([group_index] * int(connected_block.block.output_dim))
    return mapping


def _parse_stage2_variable_groups(model: torch.nn.Module) -> list[list[int]]:
    flat_s1_to_group = _stage1_flat_to_group_map(model)
    stage2 = model.stages[1]
    block_variable_groups: list[list[int]] = []
    for connected_block in stage2.blocks:
        groups: list[int] = []
        for variable in connected_block.block.variables:
            name = variable.name
            if name.startswith("s1_"):
                try:
                    flat_index = int(name.split("_")[1])
                except (ValueError, IndexError):
                    flat_index = -1
                if 0 <= flat_index < len(flat_s1_to_group):
                    groups.append(flat_s1_to_group[flat_index])
                else:
                    groups.append(-1)
            else:
                groups.append(-1)
        block_variable_groups.append(groups)
    return block_variable_groups


def _build_stage2_rule_masks(model: torch.nn.Module) -> list[np.ndarray]:
    stage2 = model.stages[1]
    variable_groups_by_block = _parse_stage2_variable_groups(model)
    masks: list[np.ndarray] = []
    for block_index, connected_block in enumerate(stage2.blocks):
        groups = variable_groups_by_block[block_index]
        intergroup_flags: list[bool] = []
        for rule in connected_block.block.rule_base.rules:
            touched = set()
            for antecedent in rule.antecedents:
                variable_index = int(antecedent.variable_index)
                if 0 <= variable_index < len(groups) and groups[variable_index] >= 0:
                    touched.add(groups[variable_index])
            intergroup_flags.append(len(touched) >= 2)
        masks.append(np.asarray(intergroup_flags, dtype=bool))
    return masks


def _compute_stage2_intergroup_stats(
    model: torch.nn.Module,
    *,
    inputs: torch.Tensor,
    top_k_rules: int | None,
    batch_size: int,
    active_rule_threshold: float,
) -> tuple[list[Stage2BlockStats], float]:
    stage2 = model.stages[1]
    inter_masks = _build_stage2_rule_masks(model)

    per_block_weight_sum = [0.0 for _ in stage2.blocks]
    per_block_inter_sum = [0.0 for _ in stage2.blocks]
    per_block_argmax_inter = [0 for _ in stage2.blocks]
    per_block_samples = [0 for _ in stage2.blocks]
    per_block_avg_weights_acc: list[list[np.ndarray]] = [[] for _ in stage2.blocks]

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for start in range(0, inputs.size(0), batch_size):
            batch = inputs[start : start + batch_size].to(device=device, dtype=torch.float32)
            _, trace = model.forward_with_trace(batch, top_k_rules=top_k_rules)
            stage2_trace = trace.stage_traces[1]

            for block_index, block_trace in enumerate(stage2_trace.block_traces):
                w = block_trace.normalized_rule_weights.detach().cpu().numpy()
                inter_mask = inter_masks[block_index]
                total = float(w.sum())
                inter = float(w[:, inter_mask].sum()) if inter_mask.any() else 0.0
                argmax_idx = w.argmax(axis=1)
                dominant_inter = int(inter_mask[argmax_idx].sum()) if inter_mask.any() else 0

                per_block_weight_sum[block_index] += total
                per_block_inter_sum[block_index] += inter
                per_block_argmax_inter[block_index] += dominant_inter
                per_block_samples[block_index] += w.shape[0]
                per_block_avg_weights_acc[block_index].append(w.mean(axis=0))

    block_stats: list[Stage2BlockStats] = []
    total_weight_sum = 0.0
    total_inter_sum = 0.0

    for block_index, connected_block in enumerate(stage2.blocks):
        inter_mask = inter_masks[block_index]
        n_rules = int(connected_block.block.n_rules)
        n_inter = int(inter_mask.sum())
        ratio = float(n_inter / n_rules) if n_rules > 0 else 0.0
        weighted_share = float(per_block_inter_sum[block_index] / (per_block_weight_sum[block_index] + 1e-12))
        dominant_share = float(per_block_argmax_inter[block_index] / max(1, per_block_samples[block_index]))

        avg_w = (
            np.mean(np.stack(per_block_avg_weights_acc[block_index], axis=0), axis=0)
            if per_block_avg_weights_acc[block_index]
            else np.zeros(n_rules, dtype=np.float64)
        )
        active_mask = avg_w >= active_rule_threshold
        active_rules = int(active_mask.sum())
        active_inter = int((active_mask & inter_mask).sum())
        active_ratio = float(active_inter / active_rules) if active_rules > 0 else 0.0

        block_stats.append(
            Stage2BlockStats(
                block_name=connected_block.block.name,
                n_rules=n_rules,
                n_intergroup_rules=n_inter,
                intergroup_rule_ratio=ratio,
                weighted_intergroup_share=weighted_share,
                dominant_intergroup_share=dominant_share,
                active_rules=active_rules,
                active_intergroup_rules=active_inter,
                active_intergroup_ratio=active_ratio,
            )
        )
        total_weight_sum += per_block_weight_sum[block_index]
        total_inter_sum += per_block_inter_sum[block_index]

    global_weighted_share = float(total_inter_sum / (total_weight_sum + 1e-12))
    return block_stats, global_weighted_share


def _format_metric_name(task_type: TaskType) -> str:
    return "f1" if task_type == "binary_classification" else "rmse"


def _sorted_pairs(pairs: list[PairInteraction], *, intergroup: bool, limit: int) -> list[PairInteraction]:
    filtered = [pair for pair in pairs if pair.is_intergroup == intergroup]
    filtered.sort(key=lambda item: item.strength, reverse=True)
    return filtered[:limit]


def _build_markdown(payload: dict) -> str:
    lines: list[str] = []
    lines.append("# DFFL Local-Block Bottleneck Diagnostics")
    lines.append("")
    lines.append(f"- dataset: `{payload['dataset']}`")
    lines.append(f"- seed: `{payload['seed']}`")
    lines.append(f"- task_type: `{payload['task_type']}`")
    lines.append(f"- dffl_profile: `{payload['dffl_profile_resolved']}`")
    lines.append(f"- metric: `{payload['metric_name']}`")
    lines.append("")
    lines.append("## Core indicators")
    lines.append("")
    lines.append(f"- baseline_metric_test: `{payload['baseline_metric_test']:.4f}`")
    lines.append(f"- data_intergroup_interaction_share: `{payload['data_intergroup_interaction_share']:.4f}`")
    lines.append(f"- stage2_intergroup_weighted_share: `{payload['stage2_intergroup_weighted_share']:.4f}`")
    lines.append(f"- interaction_recovery_ratio: `{payload['interaction_recovery_ratio']:.4f}`")
    lines.append(f"- interaction_loss_index: `{payload['interaction_loss_index']:.4f}`")
    lines.append("")
    lines.append("## Stage-2 block stats")
    lines.append("")
    lines.append("| block | n_rules | inter_rules | inter_ratio | weighted_inter_share | dominant_inter_share | active_rules | active_inter_rules | active_inter_ratio |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in payload["stage2_block_stats"]:
        lines.append(
            f"| {row['block_name']} | {row['n_rules']} | {row['n_intergroup_rules']} | "
            f"{row['intergroup_rule_ratio']:.4f} | {row['weighted_intergroup_share']:.4f} | "
            f"{row['dominant_intergroup_share']:.4f} | {row['active_rules']} | "
            f"{row['active_intergroup_rules']} | {row['active_intergroup_ratio']:.4f} |"
        )
    lines.append("")
    lines.append("## Top inter-group pairs in data")
    lines.append("")
    for pair in payload["top_intergroup_pairs"]:
        lines.append(
            f"- `{pair['feature_i_name']} x {pair['feature_j_name']}` "
            f"(x{pair['feature_i']}, x{pair['feature_j']}): {pair['strength']:.4f}"
        )
    lines.append("")
    lines.append("## Top intra-group pairs in data")
    lines.append("")
    for pair in payload["top_intragroup_pairs"]:
        lines.append(
            f"- `{pair['feature_i_name']} x {pair['feature_j_name']}` "
            f"(x{pair['feature_i']}, x{pair['feature_j']}): {pair['strength']:.4f}"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Высокий `data_intergroup_interaction_share` при умеренном/низком "
        "`stage2_intergroup_weighted_share` указывает на bottleneck: ранняя локальная декомпозиция "
        "теряет часть межгрупповых зависимостей и поздняя агрегация восстанавливает их неполностью."
    )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantify local-block bottleneck via inter-group interaction demand vs stage-2 recovery."
    )
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
    parser.add_argument("--top-pairs", type=int, default=10)
    parser.add_argument("--active-rule-threshold", type=float, default=0.05)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--md-output", type=Path, default=None)
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
    input_groups = make_dffl_input_groups(profile, train_inputs=split.train_inputs, train_targets=split.train_targets)
    bridge_pairs = make_dffl_bridge_pairs(
        profile,
        train_inputs=split.train_inputs,
        train_targets=split.train_targets,
        input_groups=input_groups,
    )
    feature_names = _dataset_feature_names(args.dataset, split.input_dim)
    feature_to_group = _build_feature_to_group(input_groups, split.input_dim)
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
            input_groups=input_groups,
            binary_feature_indices=binary_feature_indices,
            bridge_feature_pairs=bridge_pairs,
        ),
        train_inputs=split.train_inputs,
        train_targets=split.train_targets,
        validation_inputs=split.validation_inputs,
        validation_targets=split.validation_targets,
        bootstrap_config=BootstrapConfig(decision_task_type=spec.task_type),
        pretraining_config=StagewisePretrainingConfig(
            task_type=spec.task_type,
            epochs_per_stage=args.pretrain_epochs,
            decision_epochs=args.decision_pretrain_epochs,
            refinement_rounds=profile.pretrain_refinement_rounds,
            learning_rate=dffl_lr,
            batch_size=args.batch_size,
            shuffle=True,
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
            binary_soft_f1_weight=profile.binary_soft_f1_weight,
            weight_decay=profile.weight_decay,
            gradient_clip_norm=profile.gradient_clip_norm,
            rule_sparsity_weight=profile.rule_sparsity_weight,
            rule_length_weight=profile.rule_length_weight,
            decision_usage_balance_weight=profile.decision_usage_balance_weight,
            block_gate_l1_weight=profile.block_gate_l1_weight,
            device=args.device,
        ),
        refinement_loop_config=RefinementLoopConfig(max_cycles=dffl_ref_cycles, patience=1, min_delta=1e-4),
    )
    model = result.model

    metric_name = _format_metric_name(spec.task_type)
    baseline_test = _eval_primary_metric(
        model,
        task_type=spec.task_type,
        inputs=split.test_inputs,
        targets=split.test_targets,
        threshold=args.classification_threshold,
        top_k_rules=profile.top_k_rules,
    )

    x_train = split.train_inputs.detach().cpu().numpy().astype(np.float64, copy=False)
    y_train = split.train_targets.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False)
    pairs, inter_share_data, total_strength = _pairwise_interaction_strengths(
        x=x_train,
        y=y_train,
        feature_names=feature_names,
        feature_group=feature_to_group,
        top_k=max(1, args.top_pairs * 3),
    )
    top_inter = _sorted_pairs(pairs, intergroup=True, limit=args.top_pairs)
    top_intra = _sorted_pairs(pairs, intergroup=False, limit=args.top_pairs)

    stage2_stats, inter_share_stage2 = _compute_stage2_intergroup_stats(
        model,
        inputs=split.test_inputs,
        top_k_rules=profile.top_k_rules,
        batch_size=args.batch_size,
        active_rule_threshold=args.active_rule_threshold,
    )

    recovery_ratio = float(inter_share_stage2 / (inter_share_data + 1e-12))
    loss_index = float(max(0.0, 1.0 - min(1.0, recovery_ratio)))

    payload = {
        "dataset": args.dataset,
        "seed": args.seed,
        "task_type": spec.task_type,
        "metric_name": metric_name,
        "baseline_metric_test": baseline_test,
        "dffl_profile_requested": args.dffl_profile,
        "dffl_profile_resolved": profile.name,
        "input_dim": int(split.input_dim),
        "n_samples_total": int(split.n_samples),
        "input_groups": [list(group) for group in input_groups],
        "data_total_pair_interaction_strength": total_strength,
        "data_intergroup_interaction_share": inter_share_data,
        "stage2_intergroup_weighted_share": inter_share_stage2,
        "interaction_recovery_ratio": recovery_ratio,
        "interaction_loss_index": loss_index,
        "stage2_block_stats": [asdict(item) for item in stage2_stats],
        "top_intergroup_pairs": [asdict(item) for item in top_inter],
        "top_intragroup_pairs": [asdict(item) for item in top_intra],
    }

    print(f"dataset={args.dataset} seed={args.seed} profile={profile.name} metric={metric_name}")
    print(f"baseline_test={baseline_test:.4f}")
    print(f"data_intergroup_interaction_share={inter_share_data:.4f}")
    print(f"stage2_intergroup_weighted_share={inter_share_stage2:.4f}")
    print(f"interaction_recovery_ratio={recovery_ratio:.4f}")
    print(f"interaction_loss_index={loss_index:.4f}")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[ok] json: {args.json_output}")

    if args.md_output is not None:
        args.md_output.parent.mkdir(parents=True, exist_ok=True)
        args.md_output.write_text(_build_markdown(payload), encoding="utf-8")
        print(f"[ok] md: {args.md_output}")


if __name__ == "__main__":
    main()
