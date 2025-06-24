# train_gpr_refactored.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f"logs/{timestamp}"
os.makedirs(log_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Configuration helpers
# -----------------------------------------------------------------------------

def load_dataset(x_path: Path | str, y_path: Path | str) -> Tuple[np.ndarray, np.ndarray]:
    """Read CSVs, align length mismatches, and return numpy arrays."""
    x_df = pd.read_csv(x_path)
    y_df = pd.read_csv(y_path)

    if len(x_df) != len(y_df):
        print(
            f"⚠️  Length mismatch – x: {len(x_df)} rows, y: {len(y_df)} rows. "
            "Truncating to smallest."
        )
    n = min(len(x_df), len(y_df))
    x_array = x_df.iloc[:n].values.astype(np.float32)
    y_array = y_df.iloc[:n].values.reshape(-1).astype(np.float32)
    return x_array, y_array


def build_gpr_model(n_features: int, kernel_type: str = "rbf") -> GaussianProcessRegressor:
    """Factory: Build a Gaussian Process Regressor with specified kernel."""
    
    if kernel_type == "rbf":
        # RBF kernel with different length scales for different features
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e2))
    elif kernel_type == "matern":
        # Matérn kernel with nu=1.5 (once differentiable)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e2), nu=1.5)
    elif kernel_type == "rbf_white":
        # RBF kernel with white noise
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=[1.0] * n_features, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-2)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # Add white noise for numerical stability
    kernel = kernel + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,  # Additional regularization
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=5,  # Multiple random starts for optimization
        normalize_y=True,  # Normalize target values
        copy_X_train=True,
        random_state=42
    )
    
    return gpr


def train_gpr(model: GaussianProcessRegressor, x_train: np.ndarray, y_train: np.ndarray, 
              scaler_x: StandardScaler = None) -> Tuple[GaussianProcessRegressor, StandardScaler]:
    """Train the Gaussian Process Regressor."""
    print(f"🚀 Training GP on {len(x_train)} samples with {x_train.shape[1]} features...")
    
    # Standardize features
    if scaler_x is None:
        scaler_x = StandardScaler()
        x_train_scaled = scaler_x.fit_transform(x_train)
    else:
        x_train_scaled = scaler_x.transform(x_train)
    
    # Fit the GP
    model.fit(x_train_scaled, y_train)
    
    print(f"✔️  GP training complete!")
    print(f"📊 Kernel parameters:")
    print(f"   {model.kernel_}")
    print(f"   Log-marginal-likelihood: {model.log_marginal_likelihood():.4f}")
    
    # Save the trained model and scaler
    joblib.dump(model, f"{log_dir}/trained_gpr.pkl")
    joblib.dump(scaler_x, f"{log_dir}/scaler_x.pkl")
    print(f"✔️  Model saved to {log_dir}/trained_gpr.pkl")
    
    return model, scaler_x


def evaluate_gpr(
    model: GaussianProcessRegressor,
    scaler_x: StandardScaler,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_samples: int = 100,
    eval_subset: int | None = None,
    save_path: str = "images/evaluation.pdf",
    error_csv_path: str = "errors/setup_errors.csv"
) -> np.ndarray:
    """Enhanced evaluation with GP predictions and uncertainty quantification."""
    
    print(f"🔍 Evaluating GP model...")
    
    # Scale test features
    x_test_scaled = scaler_x.transform(x_test)
    
    # Get predictions with uncertainty
    y_pred_mean, y_pred_std = model.predict(x_test_scaled, return_std=True)
    
    # Overall metrics
    mse = mean_squared_error(y_test, y_pred_mean)
    mae = mean_absolute_error(y_test, y_pred_mean)
    r2 = r2_score(y_test, y_pred_mean)
    
    print(f"\n📊 Overall Test Results:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Mean prediction std: {y_pred_std.mean():.4f}")
    
    # Analyze by setup (assuming first 3 columns are setup parameters)
    test_setup_cols = x_test[:, :3]
    test_unique_setups, test_indices = np.unique(test_setup_cols, axis=0, return_inverse=True)
    
    # Select 4 setups for visualization
    n_viz_setups = min(4, len(test_unique_setups))
    selected_setups = np.random.choice(len(test_unique_setups), n_viz_setups, replace=False)
    
    # Visualize selected setups
    for i, setup_idx in enumerate(selected_setups):
        setup_mask = test_indices == setup_idx
        x_setup = x_test[setup_mask]
        y_setup = y_test[setup_mask]
        y_pred_setup = y_pred_mean[setup_mask]
        y_std_setup = y_pred_std[setup_mask]
        
        if len(x_setup) == 0:
            continue
            
        print(f"\n🎨 Visualizing Setup {i+1}/4 (Setup ID: {setup_idx}) - {len(x_setup)} samples")
        print(f"Setup parameters: {test_unique_setups[setup_idx].tolist()}")
        
        # Generate samples from GP predictive distribution
        x_setup_scaled = scaler_x.transform(x_setup)
        y_samples = model.sample_y(x_setup_scaled, n_samples=n_samples, random_state=42)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Distribution comparison
        plt.subplot(1, 3, 1)
        sns.kdeplot(y_setup, label="True", fill=True, alpha=0.5)
        sns.kdeplot(y_samples.flatten(), label="GP Samples", fill=True, alpha=0.5)
        plt.axvline(y_setup.mean(), color='blue', linestyle='--', alpha=0.7, label=f'True Mean: {y_setup.mean():.3f}')
        plt.axvline(y_pred_setup.mean(), color='orange', linestyle='--', alpha=0.7, label=f'Pred Mean: {y_pred_setup.mean():.3f}')
        plt.title(f"Setup {i+1} Distribution Comparison")
        plt.legend()
        
        # Prediction vs True scatter plot
        plt.subplot(1, 3, 2)
        plt.scatter(y_setup, y_pred_setup, alpha=0.6, s=20)
        plt.errorbar(y_setup, y_pred_setup, yerr=y_std_setup, fmt='o', alpha=0.3, capsize=3)
        
        # Perfect prediction line
        min_val, max_val = min(y_setup.min(), y_pred_setup.min()), max(y_setup.max(), y_pred_setup.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Setup {i+1} Predictions with Uncertainty")
        
        # Residuals vs Predictions
        plt.subplot(1, 3, 3)
        residuals = y_setup - y_pred_setup
        plt.scatter(y_pred_setup, residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f"Setup {i+1} Residual Plot")
        
        plt.tight_layout()
        plt.savefig(f"{log_dir}/evaluation_setup_{i+1}.pdf", bbox_inches='tight')
        plt.show()
    
    # Compute errors for all test setups
    print(f"\n📊 Computing errors for all {len(test_unique_setups)} test setups...")
    
    all_setup_errors = []
    setup_error_details = []
    
    for setup_idx in range(len(test_unique_setups)):
        setup_mask = test_indices == setup_idx
        x_setup = x_test[setup_mask]
        y_setup = y_test[setup_mask]
        y_pred_setup = y_pred_mean[setup_mask]
        y_std_setup = y_pred_std[setup_mask]
        
        if len(x_setup) == 0:
            continue
            
        # Calculate setup-specific metrics
        true_setup_mean = y_setup.mean()
        pred_setup_mean = y_pred_setup.mean()
        setup_mse = mean_squared_error(y_setup, y_pred_setup)
        setup_mae = mean_absolute_error(y_setup, y_pred_setup)
        setup_r2 = r2_score(y_setup, y_pred_setup) if len(y_setup) > 1 else 0.0
        
        # Relative error in mean prediction
        setup_relative_error = abs(pred_setup_mean - true_setup_mean) / (true_setup_mean + 1e-8)
        
        all_setup_errors.append(setup_relative_error)
        setup_error_details.append({
            'setup_id': setup_idx,
            'setup_params': test_unique_setups[setup_idx].tolist(),
            'true_mean': true_setup_mean,
            'pred_mean': pred_setup_mean,
            'relative_error': setup_relative_error,
            'mse': setup_mse,
            'mae': setup_mae,
            'r2': setup_r2,
            'mean_uncertainty': y_std_setup.mean(),
            'n_samples': len(x_setup)
        })
    
    # Save and report results
    if all_setup_errors:
        mean_error = np.mean(all_setup_errors)
        std_error = np.std(all_setup_errors)
        
        print(f"\n🎯 Setup-wise Results:")
        print(f"Number of test setups: {len(all_setup_errors)}")
        print(f"Average relative error across all setups: {mean_error:.4f} ± {std_error:.4f}")
        
        # Save detailed error data
        error_df = pd.DataFrame(setup_error_details)
        error_df.to_csv(error_csv_path, index=False)
        
        print(f"Detailed error data saved to: {error_csv_path}")
        
        return np.array(all_setup_errors)
    else:
        print("❌ No valid setup errors computed!")
        return np.array([])


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_csv", type=str, default="data/x_data.csv", help="Path to x CSV file")
    parser.add_argument("--y_csv", type=str, default="data/y_data.csv", help="Path to y CSV file")
    parser.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "matern", "rbf_white"], 
                       help="Kernel type for GP")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of data held out for testing")
    parser.add_argument("--n_samples", type=int, default=100, help="Samples per test condition during evaluation")
    parser.add_argument("--eval_subset", type=int, default=10000, help="Random subset of test rows to evaluate (None = all)")
    args = parser.parse_args()

    # Load data
    x, y = load_dataset(args.x_csv, args.y_csv)
    
    # Clamp y to be non-negative
    y = np.clip(y, 0.0, None)
    
    print(f"Dataset loaded – {len(x)} rows, {x.shape[1]} features ➜ target dim 1")
    print(f"Y data range: min={y.min():.4f}, max={y.max():.4f}")
    print(f"Y data statistics: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Split by setup (assuming first 3 columns are setup parameters)
    setup_cols = x[:, :3]
    unique_setups, indices = np.unique(setup_cols, axis=0, return_inverse=True)
    n_setups = len(unique_setups)

    # Split train/test by setup
    np.random.seed(42)
    setup_perm = np.random.permutation(n_setups)
    n_test_setups = int(args.test_ratio * n_setups)
    test_setup_indices = setup_perm[:n_test_setups]
    train_setup_indices = setup_perm[n_test_setups:]

    # Create train/test masks
    test_mask = np.isin(indices, test_setup_indices)
    train_mask = ~test_mask

    # Split data
    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    print(f"Split by setup → train: {len(x_train)} samples ({len(train_setup_indices)} setups) | test: {len(x_test)} samples ({len(test_setup_indices)} setups)")

    # Build and train GP model
    gpr = build_gpr_model(n_features=x.shape[1], kernel_type=args.kernel)
    gpr, scaler_x = train_gpr(gpr, x_train, y_train)

    # Evaluate on test set
    print(f"\n🔍 Evaluating on test setups...")
    
    all_errors = evaluate_gpr(
        gpr,
        scaler_x,
        x_test,
        y_test,
        n_samples=args.n_samples,
        eval_subset=None if args.eval_subset <= 0 else args.eval_subset,
        save_path=f"{log_dir}/evaluation_overview.pdf",
        error_csv_path=f"{log_dir}/all_setup_errors.csv"
    )
    
    # Save final summary
    if len(all_errors) > 0:
        overall_df = pd.DataFrame({
            'setup_relative_errors': all_errors
        })
        overall_df.to_csv(f"{log_dir}/overall_error_summary.csv", index=False)
        print(f"Final summary saved to: {log_dir}/overall_error_summary.csv")


if __name__ == "__main__":
    main()