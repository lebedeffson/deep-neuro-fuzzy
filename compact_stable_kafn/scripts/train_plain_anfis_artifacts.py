from __future__ import annotations

import argparse
import csv
import gzip
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _load_breast_cancer() -> tuple[np.ndarray, np.ndarray]:
    d = load_breast_cancer()
    return _to_float32(d.data), d.target.astype(np.int64)


def _load_susy_binary_200000() -> tuple[np.ndarray, np.ndarray]:
    p = Path("data/external/susy/SUSY.csv.gz")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 200_000:
                break
            vals = line.strip().split(",")
            if len(vals) < 19:
                continue
            y_rows.append(int(float(vals[0]) >= 0.5))
            x_rows.append([float(v) for v in vals[1:19]])
    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    return x, y


@dataclass
class DatasetSpec:
    name: str
    x: np.ndarray
    y: np.ndarray


def _load_dataset(name: str) -> DatasetSpec:
    if name == "breast_cancer":
        x, y = _load_breast_cancer()
        return DatasetSpec(name=name, x=x, y=y)
    if name == "susy_binary_200000":
        x, y = _load_susy_binary_200000()
        return DatasetSpec(name=name, x=x, y=y)
    raise ValueError(f"Unsupported dataset: {name}")


class PlainANFISZeroOrder(nn.Module):
    """Standalone vanilla ANFIS-like model (zero-order TSK) for binary logits.

    Output:
      logits = bias + sum_r h_r(x) * c_r
      where h_r are normalized Gaussian rule firing strengths.
    """

    def __init__(self, n_features: int, n_rules: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_rules = int(n_rules)
        self.centers = nn.Parameter(torch.empty(self.n_rules, self.n_features))
        self.log_sigmas = nn.Parameter(torch.empty(self.n_rules, self.n_features))
        self.consequents = nn.Parameter(torch.zeros(self.n_rules))
        self.bias = nn.Parameter(torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.centers, mean=0.0, std=0.8)
        nn.init.constant_(self.log_sigmas, -0.2)
        nn.init.normal_(self.consequents, mean=0.0, std=0.2)
        nn.init.constant_(self.bias, 0.0)

    def normalized_firing(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D], centers/sigmas: [R, D]
        sigmas = torch.exp(self.log_sigmas).clamp_min(1e-4)
        diff = (x[:, None, :] - self.centers[None, :, :]) / sigmas[None, :, :]
        sq = torch.sum(diff * diff, dim=2)  # [N, R]
        firing = torch.exp(-0.5 * sq).clamp_min(1e-12)
        h = firing / (torch.sum(firing, dim=1, keepdim=True) + 1e-12)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.normalized_firing(x)
        logits = self.bias + torch.matmul(h, self.consequents)
        return logits, h


def _train_one_seed(
    dataset: DatasetSpec,
    seed: int,
    n_rules: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> dict[str, np.ndarray]:
    _seed_everything(seed)
    x = dataset.x
    y = dataset.y

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=int(seed), stratify=y
    )
    mu = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = PlainANFISZeroOrder(n_features=x_train.shape[1], n_rules=int(n_rules)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(torch.from_numpy(_to_float32(x_train)), torch.from_numpy(y_train.astype(np.float32)))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True)

    model.train()
    for _ in range(int(epochs)):
        for xb, yb in dl:
            xb = xb.to(dev)
            yb = yb.to(dev)
            logits, _ = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(_to_float32(x_train)).to(dev)
        xq = torch.from_numpy(_to_float32(x_test)).to(dev)
        tr_logits_t, tr_h_t = model(xt)
        te_logits_t, te_h_t = model(xq)

    tr_logits = tr_logits_t.detach().cpu().numpy().reshape(-1)
    te_logits = te_logits_t.detach().cpu().numpy().reshape(-1)
    h_train = tr_h_t.detach().cpu().numpy().astype(np.float32)
    h_test = te_h_t.detach().cpu().numpy().astype(np.float32)

    tr_prob = _sigmoid(tr_logits).astype(np.float32)
    te_prob = _sigmoid(te_logits).astype(np.float32)

    # Importance proxy for downstream pruning.
    with torch.no_grad():
        c = model.consequents.detach().cpu().numpy().astype(np.float64)
    importance = np.abs(c) * np.mean(np.abs(h_train), axis=0)
    rule_key = np.asarray([f"anfis_rule_{i}" for i in range(int(n_rules))], dtype=object)

    # quick sanity metrics
    y_test_bin = y_test.astype(np.int64)
    pred = (te_prob >= 0.5).astype(np.int64)
    f1 = float(f1_score(y_test_bin, pred))
    auc = float(roc_auc_score(y_test_bin, te_prob))
    ap = float(average_precision_score(y_test_bin, te_prob))
    print(f"[seed={seed}] f1={f1:.4f} auc={auc:.4f} pr_auc={ap:.4f}")

    return {
        "h_train": h_train,
        "h_test": h_test,
        "y_train": y_train.astype(np.int64),
        "y_test": y_test.astype(np.int64),
        "full_train_logits": tr_logits.astype(np.float32),
        "full_test_logits": te_logits.astype(np.float32),
        "full_train_prob": tr_prob,
        "full_test_prob": te_prob,
        "importance": importance.astype(np.float32),
        "rule_key": rule_key,
        "classification_threshold": np.asarray([0.5], dtype=np.float32),
    }


def _save_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="breast_cancer")
    ap.add_argument("--seeds", type=str, default="19,23,29")
    ap.add_argument("--n-rules", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("compact_stable_kafn/paper_tables/_tmp_plain_anfis_artifacts"),
    )
    ap.add_argument(
        "--out-metrics",
        type=Path,
        default=Path("docs/tables_plain_anfis_train_metrics.csv"),
    )
    args = ap.parse_args()

    ds = _load_dataset(args.dataset)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    metrics_rows: list[dict[str, object]] = []
    for seed in seeds:
        payload = _train_one_seed(
            dataset=ds,
            seed=seed,
            n_rules=int(args.n_rules),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device=args.device,
        )
        out_path = args.out_dir / f"v18_h_artifacts_{args.dataset}_seed{seed}.npz"
        _save_npz(out_path, payload)
        y = payload["y_test"].astype(np.int64)
        p = payload["full_test_prob"].astype(np.float64)
        pred = (p >= 0.5).astype(np.int64)
        metrics_rows.append(
            {
                "dataset": args.dataset,
                "seed": seed,
                "n_rules": int(args.n_rules),
                "f1": float(f1_score(y, pred)),
                "roc_auc": float(roc_auc_score(y, p)),
                "pr_auc": float(average_precision_score(y, p)),
                "artifact_path": str(out_path),
            }
        )
        print(f"Wrote {out_path}")

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.out_metrics.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["dataset", "seed", "n_rules", "f1", "roc_auc", "pr_auc", "artifact_path"]
        )
        w.writeheader()
        w.writerows(metrics_rows)
    print(f"Wrote {args.out_metrics}")


if __name__ == "__main__":
    main()
