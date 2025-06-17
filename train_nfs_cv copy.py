# train_nfs_5fold_cv_final.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import os
from datetime import datetime
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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

def create_log_directory(base_dir: str = "logs") -> Path:
    """Create a timestamped log directory for this experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir) / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (log_dir / "checkpoints").mkdir(exist_ok=True)
    (log_dir / "visualizations").mkdir(exist_ok=True)
    (log_dir / "predictions").mkdir(exist_ok=True)
    
    return log_dir

def load_dataset(x_path: Path | str, y_path: Path | str) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame, pd.DataFrame]:
    """Read CSVs, align length mismatches, and return float32 tensors plus original dataframes."""
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    if len(x_df) != len(y_df):
        print(
            f"⚠️  Length mismatch – x: {len(x_df)} rows, y: {len(y_df)} rows. "
            "Truncating to smallest."
        )
    n = min(len(x_df), len(y_df))
    x_df = x_df.iloc[:n]
    y_df = y_df.iloc[:n]
    
    x_tensor = torch.tensor(x_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(y_df.values.reshape(-1, 1), dtype=torch.float32)
    
    return x_tensor, y_tensor, x_df, y_df

def analyze_setups(x_df: pd.DataFrame, precision: int = 6) -> Dict[Tuple, List[int]]:
    """Analyze which setups have multiple repetitions with floating point precision control."""
    setup_indices = defaultdict(list)
    
    for idx, row in x_df.iterrows():
        # Round to avoid floating point precision issues
        setup = tuple(round(val, precision) for val in row.values)
        setup_indices[setup].append(idx)
    
    return dict(setup_indices)

def select_diverse_setups(test_multi_setups, x_df, n_viz=2, max_total_samples=5000):
    """Select diverse setups for visualization considering both diversity and computational cost."""
    
    if len(test_multi_setups) <= n_viz:
        return test_multi_setups
    
    # Sort by sample count (descending)
    test_multi_setups.sort(key=lambda x: len(x[1]), reverse=True)
    
    selected = []
    used_feature_values = set()
    total_samples = 0
    
    for setup, indices in test_multi_setups:
        n_samples = len(indices)
        
        # Skip if this would exceed our computational budget
        if total_samples + n_samples > max_total_samples:
            continue
            
        # Use diversity criterion (adjust index as needed)
        if x_df.shape[1] >= 3:
            feature_val = round(setup[2], 4)  # Third feature for diversity
        else:
            feature_val = setup
            
        if feature_val not in used_feature_values:
            selected.append((setup, indices))
            used_feature_values.add(feature_val)
            total_samples += n_samples
            
            if len(selected) >= n_viz:
                break
    
    # If we still need more and have budget, add by sample count
    if len(selected) < n_viz:
        for setup, indices in test_multi_setups:
            if (setup, indices) not in selected and total_samples + len(indices) <= max_total_samples:
                selected.append((setup, indices))
                total_samples += len(indices)
                if len(selected) >= n_viz:
                    break
    
    return selected

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

def train(model: Flow, train_loader: DataLoader, *, epochs: int = 200, lr: float = 1e-3) -> List[float]:
    """Single‑loop optimiser with loss tracking."""
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []

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
        losses.append(ep_loss)

        if epoch % 20 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs} | train nll {ep_loss:.4f}")

    return losses

def create_fast_setup_mapping(test_indices: List[int], setup_indices: Dict) -> Dict[int, Tuple]:
    """Create fast O(1) lookup mapping from index to setup"""
    index_to_setup = {}
    for setup, indices in setup_indices.items():
        for idx in indices:
            if idx in test_indices:  # Only map test indices
                index_to_setup[idx] = setup
    return index_to_setup

def generate_predictions_for_setups(
    model: Flow,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    test_indices: List[int],
    setup_indices: Dict
) -> torch.Tensor:
    """Generate predictions matching the number of repetitions for each setup."""
    model.eval()
    predictions = torch.zeros_like(y_test)
    
    # Fast O(1) lookup instead of nested loops
    index_to_setup = create_fast_setup_mapping(test_indices, setup_indices)
    
    # Group by setup more efficiently
    test_setups = defaultdict(list)
    for test_idx in test_indices:
        if test_idx in index_to_setup:
            setup = index_to_setup[test_idx]
            test_setups[setup].append(test_idx)
    
    with torch.no_grad():
        for setup, indices in test_setups.items():
            n_samples = len(indices)
            
            # Use the first instance of this setup as context
            first_idx = indices[0]
            context = x_test[first_idx:first_idx+1].to(device)
            
            # Generate n_samples predictions
            samples = model.sample(n_samples, context=context).cpu().squeeze()
            
            # Assign predictions
            for i, idx in enumerate(indices):
                predictions[idx] = samples[i] if n_samples > 1 else samples
    
    return predictions

def visualize_setup_distribution(
    model: Flow,
    setup: Tuple,
    indices: List[int],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    fold_idx: int,
    log_dir: Path,
    setup_name: str,
    max_viz_samples: int = 1000
) -> None:
    """Create visualization comparing predicted vs ground truth distribution for a setup."""
    model.eval()
    
    # Get ground truth values for this setup
    y_true = y_data[indices].numpy().flatten()
    
    # Generate predictions - limit the number for visualization
    n_samples = len(indices)
    context = x_data[indices[0]:indices[0]+1].to(device)
    
    # Limit visualization samples to avoid excessive computation
    viz_samples = min(max_viz_samples, n_samples * 10)
    
    with torch.no_grad():
        y_pred = model.sample(viz_samples, context=context).cpu().numpy().flatten()
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Distribution comparison
    plt.subplot(1, 2, 1)
    sns.kdeplot(y_true, label='Ground Truth', fill=True, alpha=0.5, color='blue')
    sns.kdeplot(y_pred, label=f'Predicted ({viz_samples} samples)', fill=True, alpha=0.5, color='orange')
    
    # Dynamic title generation (avoid hardcoding feature names)
    feature_str = ', '.join([f'f{i}={val:.6f}' for i, val in enumerate(setup)])
    plt.title(f'Distribution Comparison\nSetup: {feature_str}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    
    # Subplot 2: Q-Q plot
    plt.subplot(1, 2, 2)
    percs = np.linspace(1, 99, 99)
    plt.scatter(
        np.percentile(y_true, percs),
        np.percentile(y_pred, percs),
        s=8,
        alpha=0.6
    )
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, '--', color='red', alpha=0.5)
    plt.title(f'Q-Q Plot (n={n_samples} samples)')
    plt.xlabel('Ground Truth Quantiles')
    plt.ylabel('Predicted Quantiles')
    
    plt.tight_layout()
    
    # Dynamic filename generation
    setup_suffix = '_'.join([f'{val:.6f}' for val in setup])
    filename = log_dir / "visualizations" / f"fold_{fold_idx}_{setup_name}_setup_{setup_suffix}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization for {setup_name} setup to {filename}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_csv", type=str, default="data/x_data.csv", help="Path to x CSV file")
    parser.add_argument("--y_csv", type=str, default="data/y_data.csv", help="Path to y CSV file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--min_samples_for_viz", type=int, default=10, help="Minimum samples for visualization")
    parser.add_argument("--max_viz_samples", type=int, default=1000, help="Max samples for visualization")
    parser.add_argument("--max_total_viz_samples", type=int, default=5000, help="Max total samples across all visualizations")
    
    args = parser.parse_args()

    # Create log directory
    log_dir = create_log_directory()
    print(f"Created log directory: {log_dir}")
    
    # Load data
    x, y, x_df, y_df = load_dataset(args.x_csv, args.y_csv)
    print(f"Dataset loaded – {len(x)} rows, {x.shape[1]} features ➜ target dim 1")
    
    # Analyze setups with precision control
    setup_indices = analyze_setups(x_df, precision=6)
    print(f"Found {len(setup_indices)} unique setups")
    
    # Find setups with many repetitions for visualization
    multi_sample_setups = [(setup, indices) for setup, indices in setup_indices.items() 
                          if len(indices) >= args.min_samples_for_viz]
    multi_sample_setups.sort(key=lambda x: len(x[1]), reverse=True)
    print(f"Found {len(multi_sample_setups)} setups with >= {args.min_samples_for_viz} samples")
    
    # K-Fold Cross Validation
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    all_predictions = np.zeros((len(x), 1))
    
    # Save experiment info
    with open(log_dir / "experiment_info.txt", 'w') as f:
        f.write(f"Experiment started at: {datetime.now()}\n")
        f.write(f"Number of folds: {args.n_folds}\n")
        f.write(f"Epochs per fold: {args.epochs}\n")
        f.write(f"Total samples: {len(x)}\n")
        f.write(f"Unique setups: {len(setup_indices)}\n")
        f.write(f"Setups with >= {args.min_samples_for_viz} samples: {len(multi_sample_setups)}\n")
        f.write(f"Max viz samples: {args.max_viz_samples}\n\n")
    
    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(x)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*60}")
        print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
        
        # Create datasets
        train_dataset = TensorDataset(x[train_indices], y[train_indices])
        test_dataset = TensorDataset(x[test_indices], y[test_indices])
        
        # DataLoader
        batch_size = min(1024, len(train_indices))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Build and train model
        flow = build_nfs_model(context_features=x.shape[1])
        losses = train(flow, train_loader, epochs=args.epochs)
        
        # Save checkpoint
        checkpoint_path = log_dir / "checkpoints" / f"fold_{fold_idx + 1}_model.pt"
        torch.save(flow.state_dict(), checkpoint_path)
        print(f"✔️  Saved checkpoint to {checkpoint_path}")
        
        # Generate predictions
        print("Generating predictions...")
        start_time = time.time()
        
        predictions = generate_predictions_for_setups(
            flow, x, y, test_indices, setup_indices
        )
        
        end_time = time.time()
        print(f"✔️  Prediction generation took {end_time - start_time:.2f}s")
        
        # Save predictions with setup information
        pred_df = pd.DataFrame({
            'index': test_indices,
            'prediction': predictions[test_indices].numpy().flatten()
        })

        # Add setup features for each prediction
        for col in x_df.columns:
            pred_df[col] = x_df.iloc[test_indices][col].values

        pred_path = log_dir / "predictions" / f"fold_{fold_idx + 1}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        
        # Store predictions in the global array
        all_predictions[test_indices] = predictions[test_indices].numpy()
        
        # Find setups in test set that have many samples
        test_multi_setups = []
        for setup, indices in multi_sample_setups:
            test_setup_indices = [idx for idx in indices if idx in test_indices]
            if len(test_setup_indices) >= args.min_samples_for_viz:
                test_multi_setups.append((setup, test_setup_indices))

        if len(test_multi_setups) > 0:
            # Select diverse setups for visualization
            diverse_setups = select_diverse_setups(
                test_multi_setups, x_df, n_viz=4, 
                max_total_samples=args.max_total_viz_samples
            )
            print(f"\nGenerating visualizations for {len(diverse_setups)} diverse setups in test set...")
            
            viz_start_time = time.time()
            for i, (setup, indices) in enumerate(diverse_setups):
                visualize_setup_distribution(
                    flow, setup, indices, x, y, 
                    fold_idx + 1, log_dir, f"diverse{i+1}",
                    max_viz_samples=args.max_viz_samples
                )
            viz_end_time = time.time()
            print(f"✔️  Visualization took {viz_end_time - viz_start_time:.2f}s")

    # Merge all predictions
    print(f"\n{'='*60}")
    print("Merging all predictions...")
    
    # Create final prediction dataframe matching original data structure
    final_predictions_df = x_df.copy()
    final_predictions_df['prediction'] = all_predictions
    
    # Save merged predictions
    merged_path = log_dir / "merged_predictions.csv"
    final_predictions_df.to_csv(merged_path, index=False)
    print(f"✔️  Saved merged predictions to {merged_path}")
    
    # Create final summary
    with open(log_dir / "experiment_summary.txt", 'w') as f:
        f.write(f"Experiment completed at: {datetime.now()}\n")
        f.write(f"Total predictions generated: {len(all_predictions)}\n")
        f.write(f"Final prediction shape: {final_predictions_df.shape}\n")
        f.write(f"Log directory: {log_dir}\n")
    
    print(f"\n✅ 5-fold cross-validation complete!")
    print(f"All results saved to: {log_dir}")

if __name__ == "__main__":
    main()