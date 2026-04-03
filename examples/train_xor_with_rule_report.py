from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ruanfis import (  # noqa: E402
    Antecedent,
    ConnectedFuzzyBlock,
    DeepFuzzyFeatureModel,
    FuzzyStage,
    FuzzyTrainer,
    FuzzyVariable,
    GaussianMembership,
    RuleBase,
    RuleSpec,
    SugenoDecisionLayer,
    TrainingConfig,
    TransparentFuzzyBlock,
    analyze_concept_flow,
    export_model_report,
    export_pruning_report,
    format_concept_flows,
    save_model_bundle,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_variable(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"]),
    )


def build_xor_model() -> DeepFuzzyFeatureModel:
    feature_block = TransparentFuzzyBlock(
        name="xor_feature_block",
        variables=[make_variable("x0"), make_variable("x1")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="ll"),
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="lh"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="hl"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="hh"),
            ]
        ),
        n_concepts=2,
        concept_names=["exclusive_low", "exclusive_high"],
    )
    stage = FuzzyStage(
        name="xor_stage",
        blocks=[ConnectedFuzzyBlock(feature_block, input_indices=[0, 1])],
    )
    decision = SugenoDecisionLayer(
        name="xor_decision",
        variables=[make_variable("exclusive_low"), make_variable("exclusive_high")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="xor_on"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="xor_off"),
                RuleSpec((Antecedent(0, 0),), name="weak_auxiliary", gate_init=-6.0),
            ]
        ),
        output_dim=1,
        output_names=["logit"],
    )
    return DeepFuzzyFeatureModel(stages=[stage], decision_layer=decision, input_dim=2)


def build_xor_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    return inputs, targets


def format_metrics(metrics: dict[str, float], decimals: int = 4) -> list[str]:
    return [f"{name}: {value:.{decimals}f}" for name, value in metrics.items()]


def build_trace_summary(model: DeepFuzzyFeatureModel, inputs: torch.Tensor, top_k_rules: int) -> str:
    flows = analyze_concept_flow(model, inputs, top_k_rules=top_k_rules)
    return f"TOP-{top_k_rules} CONCEPT FLOW\n{format_concept_flows(flows)}"


def build_report(
    model: DeepFuzzyFeatureModel,
    training_result,
    inputs: torch.Tensor,
    report_before: str,
    top_k_rules: int,
) -> str:
    lines = [
        "XOR DEEP FUZZY FEATURE LEARNING DEMO",
        "",
        "TRAINING SUMMARY",
        f"epochs_ran: {training_result.epochs_ran}",
        f"best_epoch: {training_result.best_epoch}",
        f"monitor: {training_result.monitor_name}",
        f"best_monitor_value: {training_result.best_monitor_value:.6f}",
        "",
        "TRAIN METRICS",
        *format_metrics(training_result.train_metrics),
    ]

    if training_result.validation_metrics is not None:
        lines.extend(["", "VALIDATION METRICS", *format_metrics(training_result.validation_metrics)])

    lines.extend(
        [
            "",
            "MODEL REPORT BEFORE TRAINING",
            report_before,
            "",
            export_pruning_report(training_result.pruning_report),
            "",
            "MODEL REPORT AFTER TRAINING AND PRUNING",
            export_model_report(model),
            "",
            build_trace_summary(model, inputs, top_k_rules=top_k_rules),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the XOR deep neuro-fuzzy model and export rule reports.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--max-epochs", type=int, default=300, help="Maximum number of training epochs.")
    parser.add_argument("--top-k-rules", type=int, default=1, help="Number of rules to keep in the trace view.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the text report.")
    parser.add_argument("--bundle-output", type=Path, default=None, help="Optional path to save the serialized model.")
    args = parser.parse_args()

    set_seed(args.seed)
    inputs, targets = build_xor_dataset()
    model = build_xor_model()
    report_before = export_model_report(model)

    trainer = FuzzyTrainer(
        model,
        TrainingConfig(
            task_type="binary_classification",
            max_epochs=args.max_epochs,
            learning_rate=0.03,
            patience=40,
            batch_size=4,
            prune_after_fit=True,
            prune_threshold=0.1,
            prune_temperature=0.03,
        ),
    )
    training_result = trainer.fit(inputs, targets, inputs, targets)
    report_text = build_report(
        model,
        training_result,
        inputs,
        report_before=report_before,
        top_k_rules=args.top_k_rules,
    )

    print(report_text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text, encoding="utf-8")
    if args.bundle_output is not None:
        args.bundle_output.parent.mkdir(parents=True, exist_ok=True)
        save_model_bundle(model, args.bundle_output, metadata={"example": "xor", "seed": args.seed})


if __name__ == "__main__":
    main()
