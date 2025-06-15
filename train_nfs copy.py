#!/usr/bin/env python
"""
train_nfs_updated.py
====================
A minimal working example of using a **conditional normalizing flow** to model the
scalar target `mismatch` conditioned on three input features (`mf`, `mf1`, `mf2`).

Key upgrades compared with the original script:

* **Data ingestion** – reads the new CSV files with Pandas instead of relying on a
  pre‑baked `.pth` file. Length mismatches between x and y are handled safely by
  truncating to the smaller of the two.
* **Model dimensions** – `context_features = 3` and `flow_features = 1` now match
  the (x, y) shapes.
* **Training loop** – pared‑down but still includes early model‑checkpointing to
  `best_flow.pt`.
* **Evaluation & visualisation** – simple KDE overlay + Q‑Q plot tailored to the
  1‑D target.

Usage example:

```bash
python train_nfs_updated.py --x_csv x.csv --y_csv y.csv
```

If you omit the flags the script looks for `x.csv` & `y.csv` in the CWD.
"""

import argparse
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
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import RandomPermutation

# ---------------------------
# Config & helpers
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_dataset(x_path: str, y_path: str):
    """Read CSVs, align lengths, return (x, y) tensors."""
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


def build_nfs_model(context_features: int, flow_features: int = 1):
    """Create a simple MAF‑based conditional flow."""
    base_dist = StandardNormal([flow_features])
    transforms = []
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


def train(model, train_loader, val_loader, epochs: int = 200, lr: float = 1e-3):
    model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # ----- Train -----
        model.train()
        tr_loss = 0.0
        for cx, y in train_loader:
            cx, y = cx.to(device), y.to(device)
            optimiser.zero_grad()
            loss = -model.log_prob(inputs=y, context=cx).mean()
            loss.backward()
            optimiser.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # ----- Validate -----
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for cx, y in val_loader:
                cx, y = cx.to(device), y.to(device)
                val_loss += (-model.log_prob(inputs=y, context=cx).mean()).item()
            val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_flow.pt")

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | train {tr_loss:.4f} | val {val_loss:.4f}")


def evaluate(model, x_tensor: torch.Tensor, y_tensor: torch.Tensor, n_samples: int = 500):
    """Draw posterior samples & compare with empirical y."""
    model.eval()
    with torch.no_grad():
        # Repeat each condition n_samples times so we can sample per‑row
        context = x_tensor.to(device).repeat_interleave(n_samples, dim=0)
        samples = model.sample(len(context), context=context).cpu().numpy().flatten()
    y_rep = np.repeat(y_tensor.numpy().flatten(), n_samples)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    sns.kdeplot(y_rep, label="empirical", fill=True, alpha=0.5)
    sns.kdeplot(samples, label="flow", fill=True, alpha=0.5)
    plt.title("Distribution comparison")
    plt.legend()

    plt.subplot(1, 2, 2)
    percs = np.linspace(1, 99, 99)
    plt.scatter(
        np.percentile(y_rep, percs),
        np.percentile(samples, percs),
        s=8,
    )
    lims = [y_rep.min(), y_rep.max()]
    plt.plot(lims, lims, "--")
    plt.title("Q–Q plot")
    plt.xlabel("empirical quantiles")
    plt.ylabel("flow quantiles")
    plt.tight_layout()
    plt.show()


# ---------------------------
# Entry point
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_csv", type=str, default="data/x_data.csv", help="Path to x CSV file")
    parser.add_argument("--y_csv", type=str, default="data/y_data.csv", help="Path to y CSV file")

    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    x, y = load_dataset(args.x_csv, args.y_csv)
    print(f"Dataset loaded – {len(x)} rows, {x.shape[1]} features ➜ target dim 1")

    dataset = TensorDataset(x, y)
    train_sz = int(0.8 * len(dataset))
    val_sz = len(dataset) - train_sz
    train_ds, val_ds = random_split(dataset, [train_sz, val_sz], generator=torch.Generator().manual_seed(42))

    batch_size = min(64, train_sz)  # sensible for the very small dataset
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    flow = build_nfs_model(context_features=x.shape[1])
    train(flow, loader(train_ds, True), loader(val_ds, False), epochs=args.epochs)

    flow.load_state_dict(torch.load("best_flow.pt", map_location=device))
    evaluate(flow, x, y)


if __name__ == "__main__":
    main()
