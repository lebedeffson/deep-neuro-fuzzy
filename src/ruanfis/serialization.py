from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .blocks import SugenoDecisionLayer, TransparentFuzzyBlock
from .memberships import FuzzyVariable, GaussianMembership, GeneralizedBellMembership
from .model import DeepFuzzyFeatureModel
from .rules import Antecedent, RuleBase, RuleSpec
from .stages import ConnectedFuzzyBlock, FuzzyStage


@dataclass(frozen=True)
class LoadedModelBundle:
    model: DeepFuzzyFeatureModel
    config: dict[str, Any]
    metadata: dict[str, Any] | None


def _tensor_to_list(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().tolist()]


def _serialize_membership(membership: GaussianMembership | GeneralizedBellMembership) -> dict[str, Any]:
    if isinstance(membership, GaussianMembership):
        return {
            "kind": "gaussian",
            "term_names": list(membership.term_names),
            "centers": _tensor_to_list(membership.centers),
            "spreads": _tensor_to_list(membership.spreads),
            "min_spread": float(membership.min_spread),
        }
    if isinstance(membership, GeneralizedBellMembership):
        return {
            "kind": "generalized_bell",
            "term_names": list(membership.term_names),
            "centers": _tensor_to_list(membership.centers),
            "widths": _tensor_to_list(membership.widths),
            "slopes": _tensor_to_list(membership.slopes),
            "min_width": float(membership.min_width),
            "min_slope": float(membership.min_slope),
        }
    raise TypeError(f"Unsupported membership type: {type(membership)!r}.")


def _deserialize_membership(config: dict[str, Any]) -> GaussianMembership | GeneralizedBellMembership:
    kind = config["kind"]
    if kind == "gaussian":
        return GaussianMembership(
            centers=config["centers"],
            spreads=config["spreads"],
            term_names=config.get("term_names"),
            min_spread=config.get("min_spread", 1e-3),
        )
    if kind == "generalized_bell":
        return GeneralizedBellMembership(
            centers=config["centers"],
            widths=config["widths"],
            slopes=config["slopes"],
            term_names=config.get("term_names"),
            min_width=config.get("min_width", 1e-3),
            min_slope=config.get("min_slope", 1e-2),
        )
    raise ValueError(f"Unsupported membership kind: {kind}.")


def _serialize_variable(variable: FuzzyVariable) -> dict[str, Any]:
    return {
        "name": variable.name,
        "membership": _serialize_membership(variable.membership),
    }


def _deserialize_variable(config: dict[str, Any]) -> FuzzyVariable:
    return FuzzyVariable(
        name=config["name"],
        membership=_deserialize_membership(config["membership"]),
    )


def _serialize_rule_base(layer: TransparentFuzzyBlock | SugenoDecisionLayer) -> list[dict[str, Any]]:
    rules = []
    gate_logits = layer.rule_logits.detach().cpu().tolist()
    for rule_index, rule in enumerate(layer.rule_base.rules):
        rules.append(
            {
                "name": rule.name,
                "gate_init": float(gate_logits[rule_index]),
                "antecedents": [
                    {
                        "variable_index": antecedent.variable_index,
                        "term_index": antecedent.term_index,
                    }
                    for antecedent in rule.antecedents
                ],
            }
        )
    return rules


def _deserialize_rule_base(config: list[dict[str, Any]]) -> RuleBase:
    return RuleBase(
        [
            RuleSpec(
                antecedents=tuple(
                    Antecedent(
                        variable_index=antecedent["variable_index"],
                        term_index=antecedent["term_index"],
                    )
                    for antecedent in rule["antecedents"]
                ),
                name=rule.get("name"),
                gate_init=rule.get("gate_init", 0.0),
            )
            for rule in config
        ]
    )


def serialize_model_config(model: DeepFuzzyFeatureModel) -> dict[str, Any]:
    stages = []
    for stage in model.stages:
        blocks = []
        for connected_block in stage.blocks:
            block = connected_block.block
            blocks.append(
                {
                    "name": block.name,
                    "input_indices": [int(index) for index in connected_block.input_indices.tolist()],
                    "variables": [_serialize_variable(variable) for variable in block.variables],
                    "rule_base": _serialize_rule_base(block),
                    "n_concepts": block.n_concepts,
                    "concept_names": list(block.output_names),
                }
            )
        stages.append({"name": stage.name, "blocks": blocks})

    decision = model.decision_layer
    return {
        "input_dim": model.input_dim,
        "stages": stages,
        "decision_layer": {
            "name": decision.name,
            "variables": [_serialize_variable(variable) for variable in decision.variables],
            "rule_base": _serialize_rule_base(decision),
            "output_dim": decision.output_dim,
            "output_names": list(decision.output_names),
        },
    }


def deserialize_model_config(config: dict[str, Any]) -> DeepFuzzyFeatureModel:
    stages = []
    for stage_config in config["stages"]:
        blocks = []
        for block_config in stage_config["blocks"]:
            block = TransparentFuzzyBlock(
                name=block_config["name"],
                variables=[_deserialize_variable(variable) for variable in block_config["variables"]],
                rule_base=_deserialize_rule_base(block_config["rule_base"]),
                n_concepts=block_config["n_concepts"],
                concept_names=block_config.get("concept_names"),
            )
            blocks.append(
                ConnectedFuzzyBlock(
                    block=block,
                    input_indices=block_config["input_indices"],
                )
            )
        stages.append(FuzzyStage(name=stage_config["name"], blocks=blocks))

    decision_config = config["decision_layer"]
    decision_layer = SugenoDecisionLayer(
        name=decision_config["name"],
        variables=[_deserialize_variable(variable) for variable in decision_config["variables"]],
        rule_base=_deserialize_rule_base(decision_config["rule_base"]),
        output_dim=decision_config["output_dim"],
        output_names=decision_config.get("output_names"),
    )
    return DeepFuzzyFeatureModel(
        stages=stages,
        decision_layer=decision_layer,
        input_dim=config.get("input_dim"),
    )


def save_model_bundle(
    model: DeepFuzzyFeatureModel,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    bundle = {
        "config": serialize_model_config(model),
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }
    torch.save(bundle, Path(path))


def load_model_bundle(path: str | Path, map_location: str | torch.device | None = None) -> LoadedModelBundle:
    payload = torch.load(Path(path), map_location=map_location)
    config = payload["config"]
    model = deserialize_model_config(config)
    model.load_state_dict(payload["state_dict"])
    return LoadedModelBundle(
        model=model,
        config=config,
        metadata=payload.get("metadata"),
    )


def save_model_config_json(model: DeepFuzzyFeatureModel, path: str | Path) -> None:
    config = serialize_model_config(model)
    Path(path).write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_model_config_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
