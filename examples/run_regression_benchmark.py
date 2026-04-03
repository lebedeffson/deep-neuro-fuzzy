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
    DecisionLayerConfig,
    FuzzyVariable,
    GaussianMembership,
    HierarchicalModelConfig,
    MultiSeedBenchmarkResult,
    RefinementLoopConfig,
    ShallowFuzzyModelConfig,
    StageConfig,
    StagewisePretrainingConfig,
    TrainingConfig,
    TransparentBlockConfig,
    build_bootstrapped_shallow_model,
    build_refined_hierarchical_model,
    evaluate_trained_model,
    format_aggregated_benchmark_results,
    format_benchmark_results,
    format_paper_benchmark_markdown_table,
    run_multi_seed_benchmark,
    run_tabular_benchmark,
    save_multi_seed_benchmark_results_json,
    save_paper_benchmark_markdown_table,
)
from ruanfis.trainer import FuzzyTrainer  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(chunk.strip()) for chunk in raw.split(",") if chunk.strip())
    if not seeds:
        raise ValueError("At least one seed must be provided.")
    return seeds


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
        input_dim=4,
        stages=(
            StageConfig(
                name="stage_1",
                blocks=(
                    TransparentBlockConfig(
                        name="left_block",
                        input_indices=(0, 1),
                        variables=(var3("x0"), var3("x1")),
                        n_concepts=2,
                        concept_names=("left_signal", "left_bias"),
                        max_rule_arity=2,
                        max_rules=4,
                        rule_generation_mode="prototype",
                    ),
                    TransparentBlockConfig(
                        name="right_block",
                        input_indices=(2, 3),
                        variables=(var3("x2"), var3("x3")),
                        n_concepts=2,
                        concept_names=("right_signal", "right_bias"),
                        max_rule_arity=2,
                        max_rules=4,
                        rule_generation_mode="prototype",
                    ),
                ),
            ),
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=(var2("left_signal"), var2("left_bias"), var2("right_signal"), var2("right_bias")),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=8,
        ),
    )


def build_shallow_config() -> ShallowFuzzyModelConfig:
    return ShallowFuzzyModelConfig(
        input_dim=4,
        feature_block=TransparentBlockConfig(
            name="shallow_block",
            input_indices=(0, 1, 2, 3),
            variables=(var3("x0"), var3("x1"), var3("x2"), var3("x3")),
            n_concepts=4,
            concept_names=("signal_0", "signal_1", "signal_2", "signal_3"),
            max_rule_arity=2,
            max_rules=12,
            rule_generation_mode="prototype",
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=(var2("signal_0"), var2("signal_1"), var2("signal_2"), var2("signal_3")),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=8,
        ),
    )


def build_dataset(n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.rand(n_samples, 4)
    targets = (
        0.45 * torch.sin(torch.pi * inputs[:, 0:1] * inputs[:, 1:2])
        + 0.30 * (inputs[:, 2:3] * inputs[:, 3:4])
        + 0.15 * inputs[:, 0:1]
    )
    return inputs, targets


def run_single_seed_benchmark(
    seed: int,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    pretrain_epochs: int,
    decision_pretrain_epochs: int,
    max_epochs: int,
    refinement_cycles: int,
) -> tuple:
    set_seed(seed)
    train_inputs, train_targets = build_dataset(train_size)
    val_inputs, val_targets = build_dataset(validation_size)
    test_inputs, test_targets = build_dataset(test_size)

    refined_result = build_refined_hierarchical_model(
        build_config(),
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=val_inputs,
        validation_targets=val_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=pretrain_epochs,
            decision_epochs=decision_pretrain_epochs,
            learning_rate=0.02,
            batch_size=64,
            shuffle=False,
        ),
        training_config=TrainingConfig(
            task_type="regression",
            max_epochs=max_epochs,
            learning_rate=0.02,
            patience=min(20, max_epochs),
            batch_size=64,
            shuffle=False,
            concept_orthogonality_weight=1e-3,
            membership_overlap_weight=1e-3,
            membership_coverage_weight=1e-3,
        ),
        refinement_loop_config=RefinementLoopConfig(
            max_cycles=refinement_cycles,
            patience=1,
            min_delta=1e-4,
        ),
    )

    shallow_model = build_bootstrapped_shallow_model(
        build_shallow_config(),
        sample_inputs=train_inputs,
        sample_targets=train_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
    )
    shallow_training = FuzzyTrainer(
        shallow_model,
        TrainingConfig(
            task_type="regression",
            max_epochs=max_epochs,
            learning_rate=0.02,
            patience=min(20, max_epochs),
            batch_size=64,
            shuffle=False,
        ),
    )
    shallow_training.fit(train_inputs, train_targets, val_inputs, val_targets)

    sklearn_results = run_tabular_benchmark(
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        task_type="regression",
        random_state=seed,
    )
    fuzzy_results = (
        evaluate_trained_model(
            "ruanfis_shallow",
            shallow_model,
            task_type="regression",
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
        ),
        evaluate_trained_model(
            "ruanfis_refined_deep",
            refined_result.model,
            task_type="regression",
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
        ),
    )
    return (*sklearn_results, *fuzzy_results)


def render_multi_seed_report(result: MultiSeedBenchmarkResult) -> str:
    lines = [
        "MULTI-SEED TABULAR BENCHMARK",
        f"seeds: {', '.join(str(seed) for seed in result.seeds)}",
        "",
    ]

    for seed, seed_results in zip(result.seeds, result.per_seed_results, strict=True):
        lines.append(f"SEED {seed}")
        lines.append(format_benchmark_results(seed_results))
        lines.append("")

    lines.append(format_aggregated_benchmark_results(result.aggregated_results))
    lines.append("")
    lines.append("INTERPRETATION STABILITY")
    lines.append("Included in the aggregated section and in the paper-ready summary table.")
    lines.append("")
    lines.append("PAPER-READY SUMMARY TABLE")
    lines.append(format_paper_benchmark_markdown_table(result.aggregated_results))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a multi-seed regression benchmark against sklearn baselines and RuANFIS models."
    )
    parser.add_argument("--seeds", type=str, default="19,23,29")
    parser.add_argument("--train-size", type=int, default=384)
    parser.add_argument("--validation-size", type=int, default=128)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=25)
    parser.add_argument("--decision-pretrain-epochs", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--refinement-cycles", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--table-output", type=Path, default=None)
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    benchmark = run_multi_seed_benchmark(
        seeds=seeds,
        seed_runner=lambda seed: run_single_seed_benchmark(
            seed,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            pretrain_epochs=args.pretrain_epochs,
            decision_pretrain_epochs=args.decision_pretrain_epochs,
            max_epochs=args.max_epochs,
            refinement_cycles=args.refinement_cycles,
        ),
    )

    report = render_multi_seed_report(benchmark)
    paper_table = format_paper_benchmark_markdown_table(benchmark.aggregated_results)
    print(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        save_multi_seed_benchmark_results_json(benchmark, args.json_output)
    if args.table_output is not None:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        save_paper_benchmark_markdown_table(benchmark.aggregated_results, args.table_output)

    if args.output is None and args.table_output is None:
        print()
        print(paper_table)


if __name__ == "__main__":
    main()
