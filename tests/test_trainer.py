import torch

from ruanfis.blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.model import DeepFuzzyFeatureModel
from ruanfis.rules import Antecedent, RuleBase, RuleSpec
from ruanfis.stages import ConnectedFuzzyBlock, FuzzyStage
from ruanfis.trainer import FuzzyTrainer, TrainingConfig


def _var(name: str) -> FuzzyVariable:
    return FuzzyVariable(name, GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"]))


def _build_regression_model() -> DeepFuzzyFeatureModel:
    feature_block = TransparentFuzzyBlock(
        name="feature_block",
        variables=[_var("x0"), _var("x1")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="ll"),
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="lh"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="hl"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="hh"),
            ]
        ),
        n_concepts=2,
        concept_names=["load", "risk"],
    )
    stage = FuzzyStage(
        name="feature_stage",
        blocks=[ConnectedFuzzyBlock(feature_block, input_indices=[0, 1])],
    )
    decision = SugenoDecisionLayer(
        name="decision",
        variables=[_var("load"), _var("risk")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="low"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="high"),
                RuleSpec((Antecedent(0, 0),), name="weak", gate_init=-6.0),
            ]
        ),
        output_dim=1,
        output_names=["target"],
    )
    return DeepFuzzyFeatureModel(stages=[stage], decision_layer=decision, input_dim=2)


def _build_xor_model() -> DeepFuzzyFeatureModel:
    feature_block = TransparentFuzzyBlock(
        name="xor_feature_block",
        variables=[_var("x0"), _var("x1")],
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
        variables=[_var("exclusive_low"), _var("exclusive_high")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="xor_on"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 0)), name="xor_off"),
                RuleSpec((Antecedent(0, 0),), name="weak", gate_init=-6.0),
            ]
        ),
        output_dim=1,
        output_names=["logit"],
    )
    return DeepFuzzyFeatureModel(stages=[stage], decision_layer=decision, input_dim=2)


def test_trainer_stops_early_when_loss_plateaus() -> None:
    torch.manual_seed(13)
    model = _build_regression_model()
    trainer = FuzzyTrainer(
        model,
        TrainingConfig(
            task_type="regression",
            max_epochs=20,
            learning_rate=0.0,
            patience=3,
            min_delta=1e-6,
        ),
    )

    inputs = torch.rand(32, 2)
    targets = 0.7 * inputs[:, :1] + 0.3 * inputs[:, 1:2]
    result = trainer.fit(inputs, targets)

    assert result.epochs_ran == 4
    assert result.best_epoch == 1
    assert result.monitor_name == "train_loss"


def test_trainer_reports_metrics_and_soft_pruning() -> None:
    torch.manual_seed(5)
    model = _build_xor_model()
    trainer = FuzzyTrainer(
        model,
        TrainingConfig(
            task_type="binary_classification",
            max_epochs=300,
            learning_rate=0.03,
            patience=40,
            prune_after_fit=True,
            prune_threshold=0.1,
            prune_temperature=0.03,
            batch_size=4,
        ),
    )

    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    result = trainer.fit(inputs, targets, inputs, targets)

    assert result.train_metrics["accuracy"] == 1.0
    assert result.validation_metrics is not None
    assert result.validation_metrics["f1"] > 0.99
    assert result.pruning_report is not None
    decision_report = result.pruning_report.layers[-1]
    assert decision_report.layer_name == "xor_decision"
    assert decision_report.probabilities_after[-1] < decision_report.probabilities_before[-1]

    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs))
        predictions = (probabilities >= 0.5).float()
        accuracy = (predictions == targets).float().mean().item()

    assert accuracy == 1.0
