import torch

from ruanfis.blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.model import DeepFuzzyFeatureModel
from ruanfis.regularizers import membership_center_order_penalty, rule_sparsity_penalty
from ruanfis.rules import Antecedent, RuleBase, RuleSpec
from ruanfis.stages import ConnectedFuzzyBlock, FuzzyStage


def _var(name: str) -> FuzzyVariable:
    return FuzzyVariable(name, GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"]))


def _build_trainable_model() -> DeepFuzzyFeatureModel:
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
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="low_state"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="high_state"),
            ]
        ),
        output_dim=1,
        output_names=["target"],
    )
    return DeepFuzzyFeatureModel(stages=[stage], decision_layer=decision, input_dim=2)


def _var3(name: str) -> FuzzyVariable:
    return FuzzyVariable(
        name,
        GaussianMembership([0.2, 0.5, 0.8], [0.18, 0.18, 0.18], term_names=["low", "mid", "high"]),
    )


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
            ]
        ),
        output_dim=1,
        output_names=["logit"],
    )
    return DeepFuzzyFeatureModel(stages=[stage], decision_layer=decision, input_dim=2)


def _build_nonlinear_regression_model() -> DeepFuzzyFeatureModel:
    feature_block = TransparentFuzzyBlock(
        name="nonlinear_feature_block",
        variables=[_var3("x0"), _var3("x1")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 0)), name="low_low"),
                RuleSpec((Antecedent(0, 0), Antecedent(1, 2)), name="low_high"),
                RuleSpec((Antecedent(0, 2), Antecedent(1, 0)), name="high_low"),
                RuleSpec((Antecedent(0, 2), Antecedent(1, 2)), name="high_high"),
                RuleSpec((Antecedent(0, 1), Antecedent(1, 1)), name="mid_mid"),
            ]
        ),
        n_concepts=3,
        concept_names=["curvature", "bias", "interaction"],
    )
    stage = FuzzyStage(
        name="nonlinear_stage",
        blocks=[ConnectedFuzzyBlock(feature_block, input_indices=[0, 1])],
    )
    decision = SugenoDecisionLayer(
        name="nonlinear_decision",
        variables=[_var3("curvature"), _var3("bias"), _var3("interaction")],
        rule_base=RuleBase(
            [
                RuleSpec((Antecedent(0, 0), Antecedent(1, 1)), name="regime_a"),
                RuleSpec((Antecedent(1, 2), Antecedent(2, 0)), name="regime_b"),
                RuleSpec((Antecedent(0, 2), Antecedent(2, 2)), name="regime_c"),
            ]
        ),
        output_dim=1,
        output_names=["target"],
    )
    return DeepFuzzyFeatureModel(stages=[stage], decision_layer=decision, input_dim=2)


def test_end_to_end_training_reduces_regression_loss() -> None:
    torch.manual_seed(7)
    model = _build_trainable_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.MSELoss()

    inputs = torch.rand(96, 2)
    targets = (0.7 * inputs[:, :1] + 0.3 * inputs[:, 1:2])

    initial_loss = loss_fn(model(inputs), targets).item()
    for _ in range(150):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss = loss + 1e-3 * rule_sparsity_penalty(model) + 1e-3 * membership_center_order_penalty(model)
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(model(inputs), targets).item()

    assert final_loss < initial_loss
    assert final_loss < 0.05


def test_model_learns_xor_classification_with_high_accuracy() -> None:
    torch.manual_seed(5)
    model = _build_xor_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

    for _ in range(200):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs))
        predictions = (probabilities > 0.5).float()
        accuracy = (predictions == targets).float().mean().item()

    assert accuracy == 1.0
    assert probabilities[0].item() < 0.05
    assert probabilities[1].item() > 0.95
    assert probabilities[2].item() > 0.95
    assert probabilities[3].item() < 0.05


def test_model_learns_nonlinear_regression_with_low_error() -> None:
    torch.manual_seed(7)
    model = _build_nonlinear_regression_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = torch.nn.MSELoss()

    inputs = torch.rand(256, 2)
    targets = torch.sin(torch.pi * inputs[:, :1]) + inputs[:, 1:2].pow(2)

    for _ in range(350):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        mse = loss_fn(model(inputs), targets).item()

    assert mse < 0.002
