# train_nfs_refactored.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List
import os
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
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f"logs/{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

# Support for Apple Silicon MPS
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")
# else:
# device = torch.device("cpu")

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
                hidden_features=16, 
                context_features=context_features,
                num_bins=15,  
                min_bin_width=1e-3,
                min_bin_height=1e-3,
                min_derivative=1e-3,
                tails="linear",        # ← extend outside the bound
                tail_bound=4.0,        # ← big enough for ~99.7 % of N(0,1)
            )
        )
    flow =  Flow(
            CompositeTransform(transforms), 
            base_dist
        )
    return flow.float()

# def build_nfs_model(context_features: int, flow_features: int = 1) -> Flow:
#     """Factory: a shallow MAF‑style conditional normalising flow."""
#     base_dist = StandardNormal([flow_features])
#     transforms: List = []
#     for _ in range(6):
#         transforms += [
#             RandomPermutation(features=flow_features),
#             MaskedAffineAutoregressiveTransform(
#                 features=flow_features,
#                 hidden_features=32,
#                 context_features=context_features,
#             ),
#         ]
#     return Flow(CompositeTransform(transforms), base_dist)


def train(model: Flow, train_loader: DataLoader, *, epochs: int = 200, lr: float = 5e-3) -> None:
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
    save_path: str = "images/evaluation.pdf",
    error_csv_path: str = "errors/setup_errors.csv"
) -> torch.Tensor:
    """Enhanced evaluation with more metrics."""
    model.eval()

    # 从测试集中选择4个不同的setup进行可视化
    test_setup_cols = x_test[:, :3]  # 前3列是setup特征
    test_unique_setups, test_indices = torch.unique(test_setup_cols, dim=0, return_inverse=True)
    
    # 选择4个setup用于画图
    n_viz_setups = min(4, len(test_unique_setups))
    selected_setups = torch.randperm(len(test_unique_setups))[:n_viz_setups]
    
    # 对选中的4个setup画图
    for i, setup_idx in enumerate(selected_setups):
        setup_mask = test_indices == setup_idx
        x_setup = x_test[setup_mask]
        y_setup = y_test[setup_mask]
        
        if len(x_setup) == 0:
            continue
            
        print(f"\n🎨 Visualizing Setup {i+1}/4 (Setup ID: {setup_idx}) - {len(x_setup)} samples")
        print(f"Setup parameters: {test_unique_setups[setup_idx].tolist()}")
        
        # 为该setup生成可视化数据
        empirical = []
        generated = []
        log_probs = []
        
        with torch.no_grad():
            for start in range(0, len(x_setup), batch_size):
                cx = x_setup[start : start + batch_size].to(device)
                y = y_setup[start : start + batch_size]
                
                try:
                    batch_samples = model.sample(samples_per_cond, context=cx).cpu()
                    batch_log_probs = model.log_prob(inputs=y.to(device), context=cx).cpu()
                    
                    generated.append(batch_samples)
                    empirical.append(y.repeat(samples_per_cond, 1))
                    log_probs.append(batch_log_probs)
                except Exception as e:
                    print(f"⚠️ Error in visualization batch: {e}")
                    continue
        
        if generated:
            y_emp = torch.cat(empirical).numpy().flatten()
            y_gen = torch.cat(generated).numpy().flatten()
            all_log_probs = torch.cat(log_probs).numpy()
            
            # 为每个setup创建单独的图
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            sns.kdeplot(y_emp, label="Empirical", fill=True, alpha=0.5)
            sns.kdeplot(y_gen, label="Generated", fill=True, alpha=0.5)
            plt.title(f"Setup {i+1} Distribution Overlap")
            plt.xlim([-0.1, 1.1])
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
            plt.title(f"Setup {i+1} Q–Q Plot")
            plt.xlabel("Empirical Quantiles")
            plt.ylabel("Generated Quantiles")
            
            plt.subplot(1, 3, 3)
            plt.hist(all_log_probs, bins=50, alpha=0.7, density=True)
            plt.title(f"Setup {i+1} Log-Likelihood")
            plt.xlabel("Log Probability")
            plt.ylabel("Density")
            
            plt.tight_layout()
            plt.savefig(f"{log_dir}/evaluation_setup_{i+1}.pdf", bbox_inches='tight')
            plt.show()

    # ✅ 计算所有测试集setup的平均误差
    print(f"\n📊 Computing errors for all {len(test_unique_setups)} test setups...")
    
    all_setup_errors = []
    setup_error_details = []
    
    for setup_idx in range(len(test_unique_setups)):
        setup_mask = test_indices == setup_idx
        x_setup = x_test[setup_mask]
        y_setup = y_test[setup_mask]
        
        if len(x_setup) == 0:
            continue
            
        # 计算该setup的真实均值
        true_setup_mean = y_setup.mean().item()
        
        # 生成预测样本并计算均值
        generated_samples = []
        with torch.no_grad():
            for start in range(0, len(x_setup), batch_size):
                cx = x_setup[start : start + batch_size].to(device)
                batch_samples = model.sample(samples_per_cond, context=cx).cpu()
                generated_samples.append(batch_samples)
        
        if generated_samples:
            all_generated = torch.cat(generated_samples)
            pred_setup_mean = all_generated.mean().item()
            
            # 计算相对误差
            setup_relative_error = abs(pred_setup_mean - true_setup_mean) / (true_setup_mean + 1e-8)
            
            all_setup_errors.append(setup_relative_error)
            setup_error_details.append({
                'setup_id': setup_idx,
                'setup_params': test_unique_setups[setup_idx].tolist(),
                'true_mean': true_setup_mean,
                'pred_mean': pred_setup_mean,
                'relative_error': setup_relative_error,
                'n_samples': len(x_setup)
            })
    
    # 计算平均误差
    if all_setup_errors:
        mean_error = np.mean(all_setup_errors)
        std_error = np.std(all_setup_errors)
        
        print(f"\n🎯 Overall Results:")
        print(f"Number of test setups: {len(all_setup_errors)}")
        print(f"Average relative error across all setups: {mean_error:.4f} ± {std_error:.4f}")
        
        # 保存详细误差数据
        error_df = pd.DataFrame(setup_error_details)
        error_df.to_csv(error_csv_path, index=False)
        
        print(f"Detailed error data saved to: {error_csv_path}")
        
        return torch.tensor(all_setup_errors)
    else:
        print("❌ No valid setup errors computed!")
        return torch.tensor([])

# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_csv", type=str, default="data/x_data.csv", help="Path to x CSV file")
    parser.add_argument("--y_csv", type=str, default="data/y_data.csv", help="Path to y CSV file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of data held out for testing")
    parser.add_argument("--samples_per_cond", type=int, default=100, help="Samples per test condition during evaluation")
    parser.add_argument("--eval_subset", type=int, default=10000, help="Random subset of test rows to evaluate (None = all)")
    args = parser.parse_args()

    # ——— Load + split ———
    x, y = load_dataset(args.x_csv, args.y_csv)
    # set y all positive and set all elements < 0 to 0
    y = torch.clamp(y, min=0.0)
    print(f"Dataset loaded – {len(x)} rows, {x.shape[1]} features ➜ target dim 1")
    print(f"Y data range: min={y.min():.4f}, max={y.max():.4f}")
    print(f"Y data statistics: mean={y.mean():.4f}, std={y.std():.4f}")
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
    batch_size = min(1024, len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ——— Model + training ———
    flow = build_nfs_model(context_features=x.shape[1])
    train(flow, train_loader, epochs=args.epochs)

    # ——— Reload best weights & evaluate on all test setups ———
    flow.load_state_dict(torch.load("trained_flow.pt", map_location=device))

    # 直接对整个测试集进行评估
    print(f"\n🔍 Evaluating all test setups...")
    
    all_errors = evaluate(
        flow,
        x_test,
        y_test,
        samples_per_cond=args.samples_per_cond,
        eval_subset=None if args.eval_subset <= 0 else args.eval_subset,
        save_path=f"{log_dir}/evaluation_overview.pdf",
        error_csv_path=f"{log_dir}/all_setup_errors.csv"
    )
    
    # 保存最终统计
    if len(all_errors) > 0:
        overall_df = pd.DataFrame({
            'setup_relative_errors': all_errors.numpy()
        })
        overall_df.to_csv(f"{log_dir}/overall_error_summary.csv", index=False)
        print(f"Final summary saved to: {log_dir}/overall_error_summary.csv")


if __name__ == "__main__":
    main()
