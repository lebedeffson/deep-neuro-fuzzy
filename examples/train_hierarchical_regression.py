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
    BootstrapConfig,
    StagewisePretrainingConfig,
    DecisionLayerConfig,
    FuzzyTrainer,
    FuzzyVariable,
    GaussianMembership,
    HierarchicalModelConfig,
    StageConfig,
    TrainingConfig,
    TransparentBlockConfig,
    analyze_concept_flow,
    analyze_path_concept_flow,
    build_stagewise_pretrained_hierarchical_model,
    count_rule_candidates,
    export_model_config_report,
    export_model_report,
    export_pruning_report,
    format_concept_flows,
    format_path_concept_flows,
    save_model_bundle,
)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def var3(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.2, 0.5, 0.8], [0.18, 0.18, 0.18], term_names=["low", "mid", "high"]),
    )


def var2(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"]),
    )


def build_config() -> HierarchicalModelConfig:
    return HierarchicalModelConfig(
        input_dim=6,
        stages=(
            StageConfig(
                name="stage_1_local_blocks",
                blocks=(
                    TransparentBlockConfig(
                        name="block_01",
                        input_indices=(0, 1),
                        variables=(var3("x0"), var3("x1")),
                        n_concepts=2,
                        concept_names=("pair_01_signal", "pair_01_bias"),
                        max_rule_arity=2,
                        max_rules=6,
                        rule_generation_mode="prototype",
                        prototype_term_limit=2,
                        prototype_sample_size=192,
                    ),
                    TransparentBlockConfig(
                        name="block_23",
                        input_indices=(2, 3),
                        variables=(var3("x2"), var3("x3")),
                        n_concepts=2,
                        concept_names=("pair_23_signal", "pair_23_bias"),
                        max_rule_arity=2,
                        max_rules=6,
                        rule_generation_mode="prototype",
                        prototype_term_limit=2,
                        prototype_sample_size=192,
                    ),
                    TransparentBlockConfig(
                        name="block_45",
                        input_indices=(4, 5),
                        variables=(var3("x4"), var3("x5")),
                        n_concepts=2,
                        concept_names=("pair_45_signal", "pair_45_bias"),
                        max_rule_arity=2,
                        max_rules=6,
                        rule_generation_mode="prototype",
                        prototype_term_limit=2,
                        prototype_sample_size=192,
                    ),
                ),
            ),
            StageConfig(
                name="stage_2_aggregate",
                blocks=(
                    TransparentBlockConfig(
                        name="aggregate_block",
                        input_indices=(0, 2, 4),
                        variables=(var3("pair_01_signal"), var3("pair_23_signal"), var3("pair_45_signal")),
                        n_concepts=3,
                        concept_names=("global_state", "interaction_state", "trend_state"),
                        max_rule_arity=2,
                        max_rules=18,
                    ),
                ),
            ),
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=(var2("global_state"), var2("interaction_state"), var2("trend_state")),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=12,
        ),
    )


def build_dataset(n_samples: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.rand(n_samples, 6)
    targets = (
        0.35 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
        + 0.35 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.15 * (inputs[:, 4:5] + inputs[:, 5:6])
        + 0.15 * inputs[:, 0:1]
    )
    return inputs, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hierarchical deep neuro-fuzzy regression model.")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for reproducibility.")
    parser.add_argument("--max-epochs", type=int, default=220, help="Maximum number of training epochs.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the text report.")
    parser.add_argument("--bundle-output", type=Path, default=None, help="Optional path to save the serialized model.")
    args = parser.parse_args()

    set_seed(args.seed)
    config = build_config()
    train_inputs, train_targets = build_dataset(512)
    val_inputs, val_targets = build_dataset(128)
    model = build_stagewise_pretrained_hierarchical_model(
        config,
        sample_inputs=train_inputs,
        sample_targets=train_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=25,
            decision_epochs=20,
            learning_rate=0.02,
            batch_size=64,
        ),
    )

    trainer = FuzzyTrainer(
        model,
        TrainingConfig(
            task_type="regression",
            max_epochs=args.max_epochs,
            learning_rate=0.025,
            patience=35,
            batch_size=64,
            prune_after_fit=True,
            prune_threshold=0.08,
            prune_temperature=0.03,
        ),
    )
    result = trainer.fit(train_inputs, train_targets, val_inputs, val_targets)

    flat_rule_count = count_rule_candidates([3, 3, 3, 3, 3, 3], max_rule_arity=6)
    report_lines = [
        "HIERARCHICAL REGRESSION DEEP FUZZY DEMO",
        "",
        export_model_config_report(config, flat_term_counts=(3, 3, 3, 3, 3, 3)),
        "",
        f"Flat full-rule count: {flat_rule_count}",
        f"Hierarchical generated-rule count: {config.generated_rule_count()}",
        "",
        "TRAIN METRICS",
        *(f"{name}: {value:.6f}" for name, value in result.train_metrics.items()),
        "",
        "VALIDATION METRICS",
        *(f"{name}: {value:.6f}" for name, value in (result.validation_metrics or {}).items()),
        "",
        export_pruning_report(result.pruning_report),
        "",
        "GLOBAL CONCEPT FLOW",
        format_concept_flows(analyze_concept_flow(model, val_inputs[:2], top_k_rules=2)),
        "",
        "PATH-BASED HIDDEN FLOW",
        format_path_concept_flows(analyze_path_concept_flow(model, val_inputs[:2], top_k_rules=2)),
        "",
        export_model_report(model),
    ]
    report_text = "\n".join(report_lines)
    print(report_text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_text, encoding="utf-8")
    if args.bundle_output is not None:
        args.bundle_output.parent.mkdir(parents=True, exist_ok=True)
        save_model_bundle(model, args.bundle_output, metadata={"example": "hierarchical_regression", "seed": args.seed})


if __name__ == "__main__":
    main()
