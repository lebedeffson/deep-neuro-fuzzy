from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import torch

from run_real_datasets_benchmark import (
    DATASETS,
    BootstrapConfig,
    FuzzyTrainer,
    RefinementLoopConfig,
    StagewisePretrainingConfig,
    TrainingConfig,
    _choose_best_classification_threshold,
    _detect_binary_feature_indices,
    build_bootstrapped_hierarchical_model,
    build_dffl_config,
    build_refined_hierarchical_model,
    make_dffl_input_groups,
    prepare_dataset_split,
    resolve_dffl_profile,
    set_seed,
)


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _choose_case_index(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> int:
    probs = _sigmoid(logits.squeeze(-1))
    preds = (probs >= threshold).float()
    gold = targets.squeeze(-1).float()
    correct = (preds == gold).nonzero(as_tuple=False).flatten()
    if correct.numel() > 0:
        margins = (probs - 0.5).abs()
        best_local = correct[margins[correct].argmax()]
        return int(best_local.item())
    return int((probs - 0.5).abs().argmax().item())


def _format_memberships(variable_trace, sample_index: int, top_terms: int = 2) -> str:
    memberships = variable_trace.memberships[sample_index]
    top_vals, top_idx = memberships.topk(k=min(top_terms, memberships.numel()))
    parts = []
    for value, idx in zip(top_vals.tolist(), top_idx.tolist(), strict=True):
        term = variable_trace.term_names[idx]
        parts.append(f"{term}={value:.4f}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract case-level DFFL explanation with top-k rules by layers.")
    parser.add_argument("--dataset", type=str, default="breast_cancer")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--dffl-profile", type=str, default="quality_auto")
    parser.add_argument("--one-phase", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--decision-pretrain-epochs", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--refinement-cycles", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--top-k-rules", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available: {', '.join(DATASETS.keys())}")
    spec = DATASETS[args.dataset]
    if spec.task_type != "binary_classification":
        raise ValueError("This extractor currently targets binary classification datasets.")

    set_seed(args.seed)
    split = prepare_dataset_split(
        spec,
        seed=args.seed,
        test_size=args.test_size,
        validation_size=args.validation_size,
    )
    train_inputs = split.train_inputs
    train_targets = split.train_targets
    val_inputs = split.validation_inputs
    val_targets = split.validation_targets
    test_inputs = split.test_inputs
    test_targets = split.test_targets

    profile = resolve_dffl_profile(
        profile_name=args.dffl_profile,
        task_type=spec.task_type,
        n_samples=split.n_samples,
        input_dim=split.input_dim,
    )
    groups = make_dffl_input_groups(profile, train_inputs=train_inputs, train_targets=train_targets)
    binary_feature_indices = _detect_binary_feature_indices(train_inputs)
    config = build_dffl_config(
        split.input_dim,
        profile=profile,
        input_groups=groups,
        binary_feature_indices=binary_feature_indices,
    )

    bootstrap = BootstrapConfig(decision_task_type=spec.task_type)
    train_cfg = TrainingConfig(
        task_type=spec.task_type,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        patience=min(args.patience, args.max_epochs),
        batch_size=args.batch_size,
        shuffle=True,
        classification_threshold=args.classification_threshold,
        monitor_metric="f1",
        monitor_mode="max",
        top_k_rules=profile.top_k_rules,
        binary_auto_pos_weight=True,
        binary_soft_f1_weight=profile.binary_soft_f1_weight if args.one_phase else 0.0,
        weight_decay=profile.weight_decay,
        gradient_clip_norm=profile.gradient_clip_norm,
        rule_sparsity_weight=profile.rule_sparsity_weight,
        rule_length_weight=profile.rule_length_weight,
        decision_usage_balance_weight=profile.decision_usage_balance_weight,
        block_gate_l1_weight=profile.block_gate_l1_weight,
        regularization_warmup_epochs=max(1, args.max_epochs // 3),
        top_k_warmup_epochs=(max(1, args.max_epochs // 4) if profile.top_k_rules is not None else 0),
        device=args.device,
    )

    if args.one_phase:
        model = build_bootstrapped_hierarchical_model(
            config,
            sample_inputs=train_inputs,
            sample_targets=train_targets,
            bootstrap_config=bootstrap,
            device=args.device,
        )
        trainer = FuzzyTrainer(model, train_cfg)
        trainer.fit(train_inputs, train_targets, val_inputs, val_targets)
    else:
        result = build_refined_hierarchical_model(
            config,
            train_inputs=train_inputs,
            train_targets=train_targets,
            validation_inputs=val_inputs,
            validation_targets=val_targets,
            bootstrap_config=bootstrap,
            pretraining_config=StagewisePretrainingConfig(
                task_type=spec.task_type,
                epochs_per_stage=args.pretrain_epochs,
                decision_epochs=args.decision_pretrain_epochs,
                refinement_rounds=1,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                shuffle=True,
                stage_selection_metric="auto",
                stage_selection_threshold=args.classification_threshold,
            ),
            training_config=train_cfg,
            refinement_loop_config=RefinementLoopConfig(
                max_cycles=max(1, args.refinement_cycles),
                patience=1,
                min_delta=1e-4,
            ),
        )
        model = result.model

    tuned_threshold = _choose_best_classification_threshold(
        model,
        validation_inputs=val_inputs,
        validation_targets=val_targets,
        default_threshold=args.classification_threshold,
        top_k_rules=profile.top_k_rules,
    )

    model.eval()
    with torch.no_grad():
        logits = model(test_inputs.to(dtype=torch.float32, device=next(model.parameters()).device), top_k_rules=profile.top_k_rules).detach().cpu()
    case_idx = _choose_case_index(logits, test_targets, tuned_threshold)

    case_input = test_inputs[case_idx : case_idx + 1].to(dtype=torch.float32, device=next(model.parameters()).device)
    case_target = float(test_targets[case_idx].item())
    with torch.no_grad():
        case_logits, case_trace = model.forward_with_trace(case_input, top_k_rules=profile.top_k_rules)
    case_logit = float(case_logits[0, 0].detach().cpu().item())
    case_prob = float(torch.sigmoid(case_logits[0, 0]).detach().cpu().item())
    case_pred = 1.0 if case_prob >= tuned_threshold else 0.0

    lines: list[str] = []
    lines.append("# Case-level explainability for DFFL")
    lines.append("")
    lines.append(f"- timestamp: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- dataset: `{args.dataset}`")
    lines.append(f"- seed: `{args.seed}`")
    lines.append(f"- profile_resolved: `{profile.name}`")
    lines.append(f"- mode: `{'one-phase' if args.one_phase else 'refined'}`")
    lines.append(f"- threshold_tuned: `{tuned_threshold:.4f}`")
    lines.append(f"- case_index_in_test: `{case_idx}`")
    lines.append(f"- y_true: `{case_target:.0f}`")
    lines.append(f"- y_pred: `{int(case_pred)}`")
    lines.append(f"- p_hat: `{case_prob:.4f}`")
    lines.append(f"- logit: `{case_logit:.4f}`")
    lines.append("")

    for stage_idx, stage_trace in enumerate(case_trace.stage_traces, start=1):
        lines.append(f"## Stage {stage_idx}: `{stage_trace.stage_name}`")
        lines.append("")
        for block_idx, block_trace in enumerate(stage_trace.block_traces):
            lines.append(f"### Block `{block_trace.block_name}`")
            lines.append("")
            lines.append("Membership peaks (top terms):")
            for variable_trace in block_trace.variable_traces:
                peak = _format_memberships(variable_trace, sample_index=0, top_terms=2)
                lines.append(f"- `{variable_trace.variable_name}`: {peak}")
            lines.append("")
            lines.append(f"Top-{args.top_k_rules} rules by normalized weight:")
            weights = block_trace.normalized_rule_weights[0]
            top_vals, top_idx = weights.topk(k=min(args.top_k_rules, weights.numel()))
            for rank, (value, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist(), strict=True), start=1):
                rule_name = block_trace.rule_names[idx]
                rule_text = model.stages[stage_idx - 1].blocks[block_idx].block.describe_rule(idx)
                lines.append(f"{rank}. `{rule_name}` | `{rule_text}` | w={value:.4f}")
            lines.append("")

    lines.append("## Decision layer")
    lines.append("")
    dtrace = case_trace.decision_trace
    dweights = dtrace.normalized_rule_weights[0]
    dvals, didx = dweights.topk(k=min(args.top_k_rules, dweights.numel()))
    lines.append(f"Top-{args.top_k_rules} decision rules:")
    for rank, (value, idx) in enumerate(zip(dvals.tolist(), didx.tolist(), strict=True), start=1):
        rule_name = dtrace.rule_names[idx]
        rule_text = model.decision_layer.describe_rule(idx)
        contribution = dtrace.rule_outputs[0, idx, 0].item() * value
        lines.append(f"{rank}. `{rule_name}` | `{rule_text}` | w={value:.4f} | contrib={contribution:.4f}")
    lines.append("")
    lines.append("## Short interpretation")
    lines.append(
        "Решение формируется через ограниченный набор активных правил в локальных блоках первого/второго уровня, "
        "после чего решающий слой агрегирует их в итоговую вероятность класса. "
        "Этот пример показывает, что можно проследить путь `вход -> скрытые правила -> решающие правила -> прогноз`."
    )

    out_path = args.out
    if out_path is None:
        out_path = Path("docs") / f"case_level_dffl_{args.dataset}_seed{args.seed}_2026-04-19.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"[ok] saved: {out_path}")


if __name__ == "__main__":
    main()
