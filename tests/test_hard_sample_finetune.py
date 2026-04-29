from __future__ import annotations

import torch

import examples.run_real_datasets_benchmark as bench
from ruanfis import TrainingConfig


class _Eval:
    def __init__(self, loss: float, f1: float) -> None:
        self.loss = float(loss)
        self.metrics = {"f1": float(f1)}


class _BaseFakeTrainer:
    def __init__(self, model: torch.nn.Module, config: TrainingConfig) -> None:
        self.model = model
        self.config = config
        self._fit_called = False

    def fit(self, *_args, **_kwargs) -> None:
        self._fit_called = True
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.add_(1.0)


class _RollbackTrainer(_BaseFakeTrainer):
    def evaluate(self, *_args, **_kwargs):
        if not self._fit_called:
            return _Eval(loss=0.2, f1=0.90)
        return _Eval(loss=0.3, f1=0.80)


class _KeepTrainer(_BaseFakeTrainer):
    def evaluate(self, *_args, **_kwargs):
        if not self._fit_called:
            return _Eval(loss=0.2, f1=0.80)
        return _Eval(loss=0.1, f1=0.90)


def _make_inputs_targets() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(12, 4)
    y = torch.randint(0, 2, (12, 1), dtype=torch.int64)
    return x, y


def _make_cfg() -> TrainingConfig:
    return TrainingConfig(
        task_type="binary_classification",
        max_epochs=2,
        patience=1,
        learning_rate=1e-3,
        monitor_metric="f1",
        monitor_mode="max",
        min_delta=0.0,
        device="cpu",
    )


def test_hard_sample_finetune_rolls_back_when_validation_worsens(monkeypatch) -> None:
    monkeypatch.setattr(bench, "FuzzyTrainer", _RollbackTrainer)
    monkeypatch.setattr(bench, "progress_log", lambda *_args, **_kwargs: None)

    model = torch.nn.Linear(4, 1)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_x, train_y = _make_inputs_targets()
    val_x, val_y = _make_inputs_targets()

    bench._maybe_run_hard_sample_finetune(
        enabled=True,
        model_name="test",
        model=model,
        task_type="binary_classification",
        train_inputs=train_x,
        train_targets=train_y,
        validation_inputs=val_x,
        validation_targets=val_y,
        base_training_config=_make_cfg(),
        source="baseline",
        hard_fraction=0.25,
        hard_multiplier=2,
        finetune_epochs=2,
        finetune_patience=1,
        baseline_hard_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        seed=19,
    )

    after = model.state_dict()
    for key, value in before.items():
        assert torch.equal(value, after[key])


def test_hard_sample_finetune_keeps_weights_when_validation_improves(monkeypatch) -> None:
    monkeypatch.setattr(bench, "FuzzyTrainer", _KeepTrainer)
    monkeypatch.setattr(bench, "progress_log", lambda *_args, **_kwargs: None)

    model = torch.nn.Linear(4, 1)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    train_x, train_y = _make_inputs_targets()
    val_x, val_y = _make_inputs_targets()

    bench._maybe_run_hard_sample_finetune(
        enabled=True,
        model_name="test",
        model=model,
        task_type="binary_classification",
        train_inputs=train_x,
        train_targets=train_y,
        validation_inputs=val_x,
        validation_targets=val_y,
        base_training_config=_make_cfg(),
        source="baseline",
        hard_fraction=0.25,
        hard_multiplier=2,
        finetune_epochs=2,
        finetune_patience=1,
        baseline_hard_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        seed=23,
    )

    after = model.state_dict()
    changed = any(not torch.equal(before[key], after[key]) for key in before)
    assert changed
