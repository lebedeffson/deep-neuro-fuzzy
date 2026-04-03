import json

import torch

from ruanfis.benchmarks import (
    AggregatedBenchmarkEntryResult,
    BenchmarkEntryResult,
    MetricSummary,
    aggregate_benchmark_results,
    evaluate_trained_model,
    format_aggregated_benchmark_markdown_table,
    format_aggregated_benchmark_results,
    format_benchmark_markdown_table,
    format_benchmark_results,
    format_paper_benchmark_markdown_table,
    run_multi_seed_benchmark,
    run_tabular_benchmark,
    save_benchmark_markdown_table,
    save_benchmark_results_json,
    save_multi_seed_benchmark_results_json,
    save_paper_benchmark_markdown_table,
    serialize_aggregated_benchmark_results,
    serialize_benchmark_results,
    serialize_multi_seed_benchmark_result,
)
from ruanfis.bootstrap import (
    BootstrapConfig,
    StagewisePretrainingConfig,
    build_bootstrapped_shallow_model,
)
from ruanfis.builders import (
    DecisionLayerConfig,
    HierarchicalModelConfig,
    ShallowFuzzyModelConfig,
    StageConfig,
    TransparentBlockConfig,
)
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.refinement import RefinementLoopConfig, build_refined_hierarchical_model
from ruanfis.trainer import FuzzyTrainer, TrainingConfig


def _var3(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.2, 0.5, 0.8], [0.18, 0.18, 0.18], term_names=["low", "mid", "high"]),
    )


def _var2(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"]),
    )


def _benchmark_config() -> HierarchicalModelConfig:
    return HierarchicalModelConfig(
        input_dim=4,
        stages=(
            StageConfig(
                name="stage_1",
                blocks=(
                    TransparentBlockConfig(
                        name="left_block",
                        input_indices=(0, 1),
                        variables=(_var3("x0"), _var3("x1")),
                        n_concepts=2,
                        concept_names=("left_signal", "left_bias"),
                        max_rule_arity=2,
                        max_rules=4,
                        rule_generation_mode="prototype",
                    ),
                    TransparentBlockConfig(
                        name="right_block",
                        input_indices=(2, 3),
                        variables=(_var3("x2"), _var3("x3")),
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
            variables=(
                _var2("left_signal"),
                _var2("left_bias"),
                _var2("right_signal"),
                _var2("right_bias"),
            ),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=8,
        ),
    )


def _shallow_benchmark_config() -> ShallowFuzzyModelConfig:
    return ShallowFuzzyModelConfig(
        input_dim=4,
        feature_block=TransparentBlockConfig(
            name="shallow_block",
            input_indices=(0, 1, 2, 3),
            variables=(_var3("x0"), _var3("x1"), _var3("x2"), _var3("x3")),
            n_concepts=4,
            concept_names=("signal_0", "signal_1", "signal_2", "signal_3"),
            max_rule_arity=2,
            max_rules=12,
            rule_generation_mode="prototype",
        ),
        decision_layer=DecisionLayerConfig(
            name="decision",
            variables=(_var2("signal_0"), _var2("signal_1"), _var2("signal_2"), _var2("signal_3")),
            output_dim=1,
            output_names=("target",),
            max_rule_arity=2,
            max_rules=8,
        ),
    )


def test_regression_benchmark_runs_with_sklearn_and_fuzzy_models() -> None:
    torch.manual_seed(23)
    train_inputs = torch.rand(192, 4)
    test_inputs = torch.rand(64, 4)
    train_targets = (
        0.45 * torch.sin(torch.pi * train_inputs[:, 0:1] * train_inputs[:, 1:2])
        + 0.30 * (train_inputs[:, 2:3] * train_inputs[:, 3:4])
        + 0.15 * train_inputs[:, 0:1]
    )
    test_targets = (
        0.45 * torch.sin(torch.pi * test_inputs[:, 0:1] * test_inputs[:, 1:2])
        + 0.30 * (test_inputs[:, 2:3] * test_inputs[:, 3:4])
        + 0.15 * test_inputs[:, 0:1]
    )

    fuzzy_result = build_refined_hierarchical_model(
        _benchmark_config(),
        train_inputs=train_inputs,
        train_targets=train_targets,
        validation_inputs=test_inputs,
        validation_targets=test_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
        pretraining_config=StagewisePretrainingConfig(
            task_type="regression",
            epochs_per_stage=15,
            decision_epochs=10,
            learning_rate=0.02,
            batch_size=64,
            shuffle=False,
        ),
        training_config=TrainingConfig(
            task_type="regression",
            max_epochs=60,
            learning_rate=0.02,
            patience=10,
            batch_size=64,
            shuffle=False,
        ),
        refinement_loop_config=RefinementLoopConfig(max_cycles=2, patience=1, min_delta=1e-4),
    )
    shallow_model = build_bootstrapped_shallow_model(
        _shallow_benchmark_config(),
        sample_inputs=train_inputs,
        sample_targets=train_targets,
        bootstrap_config=BootstrapConfig(decision_task_type="regression"),
    )
    shallow_trainer = TrainingConfig(
        task_type="regression",
        max_epochs=50,
        learning_rate=0.02,
        patience=10,
        batch_size=64,
        shuffle=False,
    )

    sklearn_results = run_tabular_benchmark(
        train_inputs=train_inputs,
        train_targets=train_targets,
        test_inputs=test_inputs,
        test_targets=test_targets,
        task_type="regression",
        random_state=23,
    )
    FuzzyTrainer(shallow_model, shallow_trainer).fit(train_inputs, train_targets, test_inputs, test_targets)
    results = (
        *sklearn_results,
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
            fuzzy_result.model,
            task_type="regression",
            train_inputs=train_inputs,
            train_targets=train_targets,
            test_inputs=test_inputs,
            test_targets=test_targets,
        ),
    )
    rendered = format_benchmark_results(results)
    markdown = format_benchmark_markdown_table(results, split="test")
    payload = serialize_benchmark_results(results)
    names = {result.model_name for result in results}
    fuzzy_entries = [result for result in results if result.family == "ruanfis"]

    assert {
        "linear_regression",
        "random_forest_regressor",
        "mlp_regressor",
        "ruanfis_shallow",
        "ruanfis_refined_deep",
    } <= names
    assert "TABULAR BENCHMARK" in rendered
    assert "| model | family |" in markdown
    assert "ruanfis_shallow [ruanfis]" in rendered
    assert "ruanfis_refined_deep [ruanfis]" in rendered
    assert payload[-1]["model_name"] == "ruanfis_refined_deep"
    assert all("total_rules" in entry.structural_metrics for entry in fuzzy_entries)
    assert all("decision_top1_mass" in entry.explainability_metrics for entry in fuzzy_entries)
    assert all("active:all" in entry.stability_artifacts for entry in fuzzy_entries)


def test_multi_seed_aggregation_and_paper_export(tmp_path) -> None:
    run_a = (
        BenchmarkEntryResult(
            "baseline",
            "sklearn",
            {"rmse": 0.20},
            {"rmse": 0.30},
        ),
        BenchmarkEntryResult(
            "deep",
            "ruanfis",
            {"rmse": 0.10},
            {"rmse": 0.15},
            structural_metrics={"total_rules": 20.0, "active_rules": 12.0},
            explainability_metrics={"decision_top1_mass": 0.60, "decision_entropy": 0.90},
            stability_artifacts={
                "generated:all": ("rule_a", "rule_b", "rule_c"),
                "generated:hidden": ("rule_a", "rule_b"),
                "generated:decision": ("rule_c",),
                "active:all": ("rule_a", "rule_c"),
                "active:hidden": ("rule_a",),
                "active:decision": ("rule_c",),
                "active:stage_1/left_block": ("rule_a",),
                "generated:stage_1/left_block": ("rule_a", "rule_b"),
            },
        ),
    )
    run_b = (
        BenchmarkEntryResult(
            "baseline",
            "sklearn",
            {"rmse": 0.30},
            {"rmse": 0.50},
        ),
        BenchmarkEntryResult(
            "deep",
            "ruanfis",
            {"rmse": 0.20},
            {"rmse": 0.25},
            structural_metrics={"total_rules": 22.0, "active_rules": 10.0},
            explainability_metrics={"decision_top1_mass": 0.70, "decision_entropy": 0.80},
            stability_artifacts={
                "generated:all": ("rule_a", "rule_d", "rule_c"),
                "generated:hidden": ("rule_a", "rule_d"),
                "generated:decision": ("rule_c",),
                "active:all": ("rule_a", "rule_c", "rule_d"),
                "active:hidden": ("rule_a", "rule_d"),
                "active:decision": ("rule_c",),
                "active:stage_1/left_block": ("rule_a", "rule_d"),
                "generated:stage_1/left_block": ("rule_a", "rule_d"),
            },
        ),
    )

    aggregates = aggregate_benchmark_results((run_a, run_b))
    multi_seed = run_multi_seed_benchmark(seeds=(3, 7), seed_runner=lambda seed: run_a if seed == 3 else run_b)
    rendered = format_aggregated_benchmark_results(aggregates)
    aggregate_table = format_aggregated_benchmark_markdown_table(aggregates, split="test")
    paper_table = format_paper_benchmark_markdown_table(aggregates)

    json_path = tmp_path / "multiseed.json"
    paper_path = tmp_path / "paper.md"
    save_multi_seed_benchmark_results_json(multi_seed, json_path)
    save_paper_benchmark_markdown_table(aggregates, paper_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    saved_paper = paper_path.read_text(encoding="utf-8")
    serialized = serialize_aggregated_benchmark_results(aggregates)
    full_serialized = serialize_multi_seed_benchmark_result(multi_seed)

    assert len(aggregates) == 2
    deep_result = next(result for result in aggregates if result.model_name == "deep")
    assert abs(deep_result.test_metrics["rmse"].mean - 0.20) < 1e-6
    assert abs(deep_result.test_metrics["rmse"].std - 0.05) < 1e-6
    assert abs(deep_result.structural_metrics["total_rules"].mean - 21.0) < 1e-6
    assert abs(deep_result.stability_metrics["decision_active_rule_jaccard"] - 1.0) < 1e-6
    assert 0.0 < deep_result.stability_metrics["active_rule_jaccard"] < 1.0
    assert "AGGREGATED TABULAR BENCHMARK" in rendered
    assert "stability:" in rendered
    assert "| model | family | rmse |" in aggregate_table
    assert "test:rmse" in paper_table
    assert "rules:total_rules" in paper_table
    assert "expl:decision_top1_mass" in paper_table
    assert "stab:active_rule_jaccard" in paper_table
    assert payload["seeds"] == [3, 7]
    assert payload["aggregated_results"][1]["model_name"] == "deep"
    assert saved_paper == paper_table
    assert serialized[1]["structural_metrics"]["total_rules"]["mean"] == 21.0
    assert serialized[1]["stability_metrics"]["decision_active_rule_jaccard"] == 1.0
    assert full_serialized["per_seed_results"][0]["seed"] == 3


def test_benchmark_serialization_and_markdown_export(tmp_path) -> None:
    actual_results = (
        BenchmarkEntryResult("a", "sklearn", {"rmse": 0.1}, {"rmse": 0.2}),
        BenchmarkEntryResult(
            "b",
            "ruanfis",
            {"rmse": 0.05},
            {"rmse": 0.08},
            structural_metrics={"total_rules": 18.0},
            explainability_metrics={"decision_top1_mass": 0.7},
            stability_artifacts={"active:all": ("rule_a",)},
        ),
    )
    json_path = tmp_path / "benchmark.json"
    table_path = tmp_path / "benchmark.md"

    save_benchmark_results_json(actual_results, json_path)
    save_benchmark_markdown_table(actual_results, table_path, split="test")

    saved_payload = json.loads(json_path.read_text(encoding="utf-8"))
    saved_table = table_path.read_text(encoding="utf-8")

    assert len(saved_payload) == 2
    assert saved_payload[1]["model_name"] == "b"
    assert saved_payload[1]["structural_metrics"]["total_rules"] == 18.0
    assert saved_payload[1]["stability_artifacts"]["active:all"] == ["rule_a"]
    assert "| b | ruanfis | 0.0800 |" in saved_table


def test_aggregated_result_dataclasses_are_instantiable() -> None:
    result = AggregatedBenchmarkEntryResult(
        model_name="demo",
        family="ruanfis",
        runs=2,
        train_metrics={"rmse": MetricSummary(mean=0.1, std=0.01)},
        test_metrics={"rmse": MetricSummary(mean=0.2, std=0.02)},
        structural_metrics={"total_rules": MetricSummary(mean=20.0, std=1.0)},
        explainability_metrics={"decision_top1_mass": MetricSummary(mean=0.7, std=0.05)},
        stability_metrics={"active_rule_jaccard": 0.8},
    )

    assert result.model_name == "demo"
    assert result.test_metrics["rmse"].std == 0.02
    assert result.stability_metrics["active_rule_jaccard"] == 0.8
