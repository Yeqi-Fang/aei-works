# train_nfs_refactored.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import seaborn as sns

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
)
from nflows.transforms.permutations import RandomPermutation

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_dataset(x_path: Path | str, y_path: Path | str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read CSVs, align length mismatches, and return **float32** tensors."""
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    if len(x_df) != len(y_df):
        print(
            f"⚠️  Length mismatch – x: {len(x_df)} rows, y: {len(y_df)} rows. "
            "Truncating to smallest."
        )
    n = min(len(x_df), len(y_df))
    x_tensor = torch.tensor(x_df.iloc[:n].values, dtype=torch.float32)
    y_tensor = torch.tensor(y_df.iloc[:n].values.reshape(-1, 1), dtype=torch.float32)
    return x_tensor, y_tensor


def build_nfs_model(context_features: int, flow_features: int = 1) -> Flow:
    """Factory: a shallow MAF‑style conditional normalising flow."""
    base_dist = StandardNormal([flow_features])
    transforms: List = []
    for _ in range(6):
        transforms += [
            RandomPermutation(features=flow_features),
            MaskedAffineAutoregressiveTransform(
                features=flow_features,
                hidden_features=32,
                context_features=context_features,
            ),
        ]
    return Flow(CompositeTransform(transforms), base_dist)


def train(model: Flow, train_loader: DataLoader, *, epochs: int = 200, lr: float = 1e-3) -> None:
    """Single‑loop optimiser; we checkpoint the final model only."""
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for cx, y in train_loader:
            cx, y = cx.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = -model.log_prob(inputs=y, context=cx).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        ep_loss /= len(train_loader)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | train nll {ep_loss:.4f}")

    torch.save(model.state_dict(), "trained_flow.pt")
    print("✔️  Training complete – model saved to *trained_flow.pt*.")


def evaluate(
    model: Flow,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    samples_per_cond: int = 100,
    eval_subset: int | None = None,
    batch_size: int = 512,
) -> None:
    """Memory‑safe posterior sampling + diagnostic plots on the test set.

    Args
    ----
    samples_per_cond:  how many draws to take **per test row**
    eval_subset      : if set, randomly down‑sample the test set to this many rows
    batch_size       : context batch size to keep GPU memory in check
    """
    model.eval()

    # ——— optionally pare down the test set (useful when rows ≫ 10k) ———
    if eval_subset is not None and eval_subset < len(x_test):
        idx = torch.randperm(len(x_test))[:eval_subset]
        x_test = x_test[idx]
        y_test = y_test[idx]

    empirical: List[torch.Tensor] = []
    generated: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            cx = x_test[start : start + batch_size].to(device)
            y = y_test[start : start + batch_size]

            # Draw [samples_per_cond] for each condition in the mini‑batch
            batch_samples = model.sample(samples_per_cond, context=cx).cpu()

            # Align shapes for later flattening
            generated.append(batch_samples)
            empirical.append(y.repeat(samples_per_cond, 1))

    y_emp = torch.cat(empirical).numpy().flatten()
    y_gen = torch.cat(generated).numpy().flatten()

    # ——— Quick‑and‑dirty density check + Q‑Q plot ———
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.kdeplot(y_emp, label="empirical", fill=True, alpha=0.5)
    sns.kdeplot(y_gen, label="flow", fill=True, alpha=0.5)
    plt.title("Test‑set distribution overlap")
    plt.legend()

    plt.subplot(1, 2, 2)
    percs = np.linspace(1, 99, 99)
    plt.scatter(
        np.percentile(y_emp, percs),
        np.percentile(y_gen, percs),
        s=8,
    )
    lims = [y_emp.min(), y_emp.max()]
    plt.plot(lims, lims, "--")
    plt.title("Q–Q plot (test set)")
    plt.xlabel("empirical quantiles")
    plt.ylabel("flow quantiles")
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_csv", type=str, default="data/x_data.csv", help="Path to x CSV file")
    parser.add_argument("--y_csv", type=str, default="data/y_data.csv", help="Path to y CSV file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of data held out for testing")
    parser.add_argument("--samples_per_cond", type=int, default=100, help="Samples per test condition during evaluation")
    parser.add_argument("--eval_subset", type=int, default=10000, help="Random subset of test rows to evaluate (None = all)")
    args = parser.parse_args()

    # ——— Load + split ———
    x, y = load_dataset(args.x_csv, args.y_csv)
    print(f"Dataset loaded – {len(x)} rows, {x.shape[1]} features ➜ target dim 1")

    data = TensorDataset(x, y)
    test_sz = int(args.test_ratio * len(data))
    train_sz = len(data) - test_sz
    train_ds, test_ds = random_split(data, [train_sz, test_sz], generator=torch.Generator().manual_seed(42))
    print(f"Split → train: {train_sz} | test: {test_sz}")

    # ——— DataLoaders ———
    batch_size = min(256, train_sz)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ——— Model + training ———
    flow = build_nfs_model(context_features=x.shape[1])
    train(flow, train_loader, epochs=args.epochs)

    # ——— Reload best weights & evaluate on the held‑out test set ———
    flow.load_state_dict(torch.load("trained_flow.pt", map_location=device))

    # Drop dataset wrappers to get raw tensors for evaluation
    x_test = torch.stack([sample[0] for sample in test_ds])
    y_test = torch.stack([sample[1] for sample in test_ds])

    evaluate(
        flow,
        x_test,
        y_test,
        samples_per_cond=args.samples_per_cond,
        eval_subset=None if args.eval_subset <= 0 else args.eval_subset,
    )


if __name__ == "__main__":
    main()
