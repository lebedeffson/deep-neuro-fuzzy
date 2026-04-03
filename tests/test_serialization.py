from pathlib import Path

import torch

from ruanfis.blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from ruanfis.memberships import FuzzyVariable, GaussianMembership
from ruanfis.model import DeepFuzzyFeatureModel
from ruanfis.rules import Antecedent, RuleBase, RuleSpec
from ruanfis.serialization import (
    deserialize_model_config,
    load_model_bundle,
    load_model_config_json,
    save_model_bundle,
    save_model_config_json,
    serialize_model_config,
)
from ruanfis.stages import ConnectedFuzzyBlock, FuzzyStage


def _var(name: str) -> FuzzyVariable:
    return FuzzyVariable(name, GaussianMembership([0.25, 0.75], [0.2, 0.2], term_names=["low", "high"]))


def _build_model() -> DeepFuzzyFeatureModel:
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
        concept_names=["exclusive_low", "exclusive_high"],
    )
    stage = FuzzyStage("xor_stage", [ConnectedFuzzyBlock(feature_block, [0, 1])])
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
    return DeepFuzzyFeatureModel([stage], decision, input_dim=2)


def _train_model(model: DeepFuzzyFeatureModel) -> None:
    torch.manual_seed(4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    inputs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
    for _ in range(150):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()


def test_model_bundle_roundtrip_preserves_predictions(tmp_path: Path) -> None:
    model = _build_model()
    _train_model(model)
    inputs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.2, 0.8]], dtype=torch.float32)

    expected = model(inputs).detach()
    bundle_path = tmp_path / "model_bundle.pt"
    save_model_bundle(model, bundle_path, metadata={"task": "xor"})
    loaded = load_model_bundle(bundle_path)

    restored = loaded.model(inputs).detach()

    assert torch.allclose(expected, restored, atol=1e-6)
    assert loaded.metadata == {"task": "xor"}
    assert loaded.config["decision_layer"]["name"] == "xor_decision"


def test_model_config_json_roundtrip_rebuilds_architecture(tmp_path: Path) -> None:
    model = _build_model()
    config = serialize_model_config(model)
    rebuilt = deserialize_model_config(config)
    json_path = tmp_path / "model_config.json"
    save_model_config_json(model, json_path)
    loaded_config = load_model_config_json(json_path)

    assert rebuilt.input_dim == model.input_dim
    assert rebuilt.decision_layer.output_dim == model.decision_layer.output_dim
    assert loaded_config == config
