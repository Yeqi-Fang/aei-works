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
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform
)
from nflows.transforms.permutations import RandomPermutation

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

# Support for Apple Silicon MPS
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
device = torch.device("cpu")

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
    # base_dist = StandardNormal([flow_features])
    transforms: List = []
    base_dist = StandardNormal([flow_features])

    for _ in range(3):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=1,
                hidden_features=8,  # 减少隐藏单元
                context_features=context_features,
                num_bins=6,  # 减少样条段数
                tail_bound=1.0,  # 设置为[0,1]边界
                min_bin_width=1e-3,
                min_bin_height=1e-3,
                min_derivative=1e-3,
            )
        )
    flow =  Flow(
            CompositeTransform(transforms), 
            base_dist
        )
    return flow.float()


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
    """Enhanced evaluation with more metrics."""
    model.eval()

    if eval_subset is not None and eval_subset < len(x_test):
        idx = torch.randperm(len(x_test))[:eval_subset]
        x_test = x_test[idx]
        y_test = y_test[idx]

    empirical: List[torch.Tensor] = []
    generated: List[torch.Tensor] = []
    log_probs: List[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            cx = x_test[start : start + batch_size].to(device)
            y = y_test[start : start + batch_size]

            try:
                # 生成样本
                batch_samples = model.sample(samples_per_cond, context=cx).cpu()
                
                # 计算对数概率
                batch_log_probs = model.log_prob(inputs=y.to(device), context=cx).cpu()
                
                generated.append(batch_samples)
                empirical.append(y.repeat(samples_per_cond, 1))
                log_probs.append(batch_log_probs)
                
            except Exception as e:
                print(f"⚠️ Error in evaluation batch: {e}")
                continue

    if not generated:
        print("❌ No valid samples generated!")
        return

    y_emp = torch.cat(empirical).numpy().flatten()
    y_gen = torch.cat(generated).numpy().flatten()
    all_log_probs = torch.cat(log_probs).numpy()

    # ✅ 计算评估指标
    print(f"\n📊 Evaluation Metrics:")
    print(f"Average log-likelihood: {all_log_probs.mean():.4f} ± {all_log_probs.std():.4f}")
    print(f"Generated samples range: [{y_gen.min():.4f}, {y_gen.max():.4f}]")
    print(f"Empirical samples range: [{y_emp.min():.4f}, {y_emp.max():.4f}]")
    
    # Wasserstein距离 (简化版)
    from scipy import stats
    try:
        ks_stat, ks_p = stats.ks_2samp(y_emp, y_gen)
        print(f"KS test statistic: {ks_stat:.4f} (p-value: {ks_p:.4f})")
    except:
        print("KS test failed")

    # ✅ 改进的可视化
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.kdeplot(y_emp, label="Empirical", fill=True, alpha=0.5)
    sns.kdeplot(y_gen, label="Generated", fill=True, alpha=0.5)
    plt.title("Distribution Overlap")
    plt.legend()

    plt.subplot(1, 3, 2)
    percs = np.linspace(1, 99, 99)
    plt.scatter(
        np.percentile(y_emp, percs),
        np.percentile(y_gen, percs),
        s=8, alpha=0.7
    )
    lims = [y_emp.min(), y_emp.max()]
    plt.plot(lims, lims, "r--", alpha=0.8)
    plt.title("Q–Q Plot")
    plt.xlabel("Empirical Quantiles")
    plt.ylabel("Generated Quantiles")

    plt.subplot(1, 3, 3)
    plt.hist(all_log_probs, bins=50, alpha=0.7, density=True)
    plt.title("Log-Likelihood Distribution")
    plt.xlabel("Log Probability")
    plt.ylabel("Density")

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
    # set y all positive and set all elements < 0 to 0
    y = torch.clamp(y, min=0.0)
    print(f"Dataset loaded – {len(x)} rows, {x.shape[1]} features ➜ target dim 1")

    # 按setup分组划分（假设前3列是mf, mf1, mf2）
    setup_cols = x[:, :3]  # 提取setup列
    unique_setups, indices = torch.unique(setup_cols, dim=0, return_inverse=True)
    n_setups = len(unique_setups)

    # 按setup划分训练测试集
    torch.manual_seed(42)
    setup_perm = torch.randperm(n_setups)
    n_test_setups = int(args.test_ratio * n_setups)
    test_setup_indices = setup_perm[:n_test_setups]
    train_setup_indices = setup_perm[n_test_setups:]

    # 创建训练测试mask
    test_mask = torch.isin(indices, test_setup_indices)
    train_mask = ~test_mask

    # 分割数据
    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    print(f"Split by setup → train: {len(train_ds)} samples ({len(train_setup_indices)} setups) | test: {len(test_ds)} samples ({len(test_setup_indices)} setups)")
    # ——— DataLoaders ———
    batch_size = min(256, len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ——— Model + training ———
    flow = build_nfs_model(context_features=x.shape[1])
    train(flow, train_loader, epochs=args.epochs)

    # ——— Reload best weights & evaluate on the held‑out test set ———
    flow.load_state_dict(torch.load("trained_flow.pt", map_location=device))

    # Drop dataset wrappers to get raw tensors for evaluation

    evaluate(
        flow,
        x_test,
        y_test,
        samples_per_cond=args.samples_per_cond,
        eval_subset=None if args.eval_subset <= 0 else args.eval_subset,
    )


if __name__ == "__main__":
    main()
