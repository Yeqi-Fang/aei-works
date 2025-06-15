#!/usr/bin/env python
"""
train_nfs_kfold.py
------------------
5-fold CV version of the normalising-flow script.

Running it will create **cv_results.csv** in the working directory, containing
one line per original sample with columns: mf, mf1, mf2, mismatch.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold       # <- new
from torch.utils.data import DataLoader, TensorDataset

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import RandomPermutation

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_dataset(x_path: str | Path, y_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load two CSV files and return float32 tensors of equal length."""
    x_df, y_df = pd.read_csv(x_path), pd.read_csv(y_path)
    if len(x_df) != len(y_df):
        n = min(len(x_df), len(y_df))
        print(f"⚠️  Length mismatch – truncating both to {n} rows.")
        x_df, y_df = x_df.iloc[:n], y_df.iloc[:n]
    x = torch.tensor(x_df.values, dtype=torch.float32)
    y = torch.tensor(y_df.values.reshape(-1, 1), dtype=torch.float32)
    return x, y

def build_nfs_model(context_features: int, flow_features: int = 1) -> Flow:
    """6-layer MAF-style conditional flow."""
    base = StandardNormal([flow_features])
    tfms: List = []
    for _ in range(6):
        tfms += [
            RandomPermutation(features=flow_features),
            MaskedAffineAutoregressiveTransform(
                features=flow_features,
                hidden_features=32,
                context_features=context_features,
            ),
        ]
    return Flow(CompositeTransform(tfms), base)

def train_one_fold(model: Flow, loader: DataLoader, *, epochs: int = 50, lr: float = 1e-3) -> None:
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for cx, y in loader:
            cx, y = cx.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = -model.log_prob(inputs=y, context=cx).mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        ep_loss /= len(loader)
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"  epoch {ep:3d}/{epochs} | nll {ep_loss:.4f}")

def posterior_summary(model: Flow,
                      context: torch.Tensor,
                      *,
                      n_samp: int = 100) -> tuple[float, float, float]:
    """Return mean, 5-th perc, 95-th perc of y|x for a single row."""
    with torch.no_grad():
        samp = model.sample(n_samp, context=context.unsqueeze(0).to(device)).cpu().numpy().flatten()
    mean = samp.mean()
    p5, p95 = np.percentile(samp, [5, 95])
    return mean, p5, p95

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x_csv", default="data/x_data.csv")
    ap.add_argument("--y_csv", default="data/y_data.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",  type=int, default=256)
    args = ap.parse_args()

    # 1) data -----------------------------------------------------------------
    x, y = load_dataset(args.x_csv, args.y_csv)
    print(f"Dataset → {len(x)} rows | {x.shape[1]} features | target dim 1")

    # 2) prepare result holders ----------------------------------------------
    mf     = np.empty(len(x), dtype=np.float32)
    mf1    = np.empty(len(x), dtype=np.float32)
    mf2    = np.empty(len(x), dtype=np.float32)
    misfit = np.empty(len(x), dtype=np.float32)

    # 3) 5-fold CV ------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(x), 1):
        print(f"\n🟦 Fold {fold}/5 – train {len(train_idx)} | val {len(val_idx)}")

        # ­­— build loader for this fold
        train_ds = TensorDataset(x[train_idx], y[train_idx])
        loader   = DataLoader(train_ds, batch_size=min(args.batch, len(train_ds)), shuffle=True)

        # ­­— fresh model
        flow = build_nfs_model(context_features=x.shape[1])
        train_one_fold(flow, loader, epochs=args.epochs)

        # ­­— posterior summaries for validation rows
        flow.eval()
        for idx in val_idx:
            mean, p5, p95 = posterior_summary(flow, x[idx])
            mf[idx], mf1[idx], mf2[idx] = mean, p5, p95
            misfit[idx] = abs(mean - float(y[idx]))

    # 4) write CSV ------------------------------------------------------------
    out_df = pd.DataFrame({
        "mf": mf,
        "mf1": mf1,
        "mf2": mf2,
        "mismatch": misfit,
    })
    out_path = Path("cv_results.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n📄  Saved per-row CV summaries to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
