import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ks_2samp
import os
import glob
from itertools import combinations

def load_all_csv_files(file_pattern='PyFstat_example_data/PyFstatExampleSimpleMCMCvsGridComparisonSemi/new-mismatches-0.05-*.csv', column_name='Empirical Mismatch (μ)'):
    """
    Load all CSV files matching the pattern and extract specified column
    
    Parameters:
    file_pattern: str, pattern to match CSV files
    column_name: str, name of column to extract
    
    Returns:
    dict: Dictionary with filename as key and data array as value
    """
    
    # Find all matching CSV files
    csv_files = glob.glob(file_pattern)
    csv_files.sort()  # Sort to ensure consistent ordering
    
    data_dict = {}
    
    print(f"Found {len(csv_files)} CSV files:")
    
    for file_path in csv_files:
        try:
            # Extract filename without extension for labeling
            filename = os.path.basename(file_path).replace('.csv', '')
            
            # Load CSV file
            df = pd.read_csv(file_path)
            
            # Check if column exists
            if column_name not in df.columns:
                print(f"Warning: Column '{column_name}' not found in {filename}")
                print(f"Available columns: {list(df.columns)}")
                continue
            
            # Extract the data column and remove any NaN values
            data = df[column_name].dropna().values
            
            data_dict[filename] = data
            
            print(f"  {filename}: {len(data)} data points, mean={np.mean(data):.4f}, std={np.std(data):.4f}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data_dict

def create_pairwise_comparison_matrices(data_dict, alpha=0.05):
    """
    Create pairwise comparison matrices for Mann-Whitney U and Kolmogorov-Smirnov tests
    
    Parameters:
    data_dict: dict, dictionary with dataset names and data arrays
    alpha: float, significance level
    
    Returns:
    tuple: (mw_pvalue_matrix, ks_pvalue_matrix, mw_significant_matrix, ks_significant_matrix, labels, display_labels)
    """
    
    # Sort labels numerically by extracting the last number from filename
    def extract_number(filename):
        import re
        numbers = re.findall(r'-(\d+)', filename)
        return int(numbers[-1]) if numbers else 0
    
    # Create mapping from original filename to clear display label
    def create_display_label(filename):
        import re
        numbers = re.findall(r'-(\d+)', filename)
        if numbers:
            grid_size = numbers[-1]
            return f"{grid_size}³ grids"
        return filename
    
    # Sort by numerical value but keep original filenames for data access
    sorted_items = sorted(data_dict.items(), key=lambda x: extract_number(x[0]))
    labels = [item[0] for item in sorted_items]  # Original filenames for data access
    display_labels = [create_display_label(label) for label in labels]  # Clear labels for display
    
    n_datasets = len(labels)
    
    print(f"Found {n_datasets} datasets:")
    for orig, display in zip(labels, display_labels):
        print(f"  {orig} → {display}")
    print(f"Sorted by grid size: {display_labels}")
    
    # Initialize matrices
    mw_pvalue_matrix = np.ones((n_datasets, n_datasets))
    ks_pvalue_matrix = np.ones((n_datasets, n_datasets))
    mw_significant_matrix = np.zeros((n_datasets, n_datasets))
    ks_significant_matrix = np.zeros((n_datasets, n_datasets))
    
    print("\nPerforming pairwise comparisons...")
    print("=" * 60)
    
    # Perform all pairwise comparisons
    for i in range(n_datasets):
        for j in range(n_datasets):
            if i != j:  # Don't compare dataset with itself
                data1 = data_dict[labels[i]]
                data2 = data_dict[labels[j]]
                
                # Mann-Whitney U test
                try:
                    mw_stat, mw_p = mannwhitneyu(data1, data2, alternative='two-sided')
                    mw_pvalue_matrix[i, j] = mw_p
                    mw_significant_matrix[i, j] = 1 if mw_p < alpha else 0
                except Exception as e:
                    print(f"Mann-Whitney U test failed for {labels[i]} vs {labels[j]}: {e}")
                    mw_pvalue_matrix[i, j] = np.nan
                
                # Kolmogorov-Smirnov test
                try:
                    ks_stat, ks_p = ks_2samp(data1, data2)
                    ks_pvalue_matrix[i, j] = ks_p
                    ks_significant_matrix[i, j] = 1 if ks_p < alpha else 0
                except Exception as e:
                    print(f"Kolmogorov-Smirnov test failed for {labels[i]} vs {labels[j]}: {e}")
                    ks_pvalue_matrix[i, j] = np.nan
    
    return mw_pvalue_matrix, ks_pvalue_matrix, mw_significant_matrix, ks_significant_matrix, labels, display_labels

def plot_individual_matrices(mw_pvalue_matrix, ks_pvalue_matrix, 
                           mw_significant_matrix, ks_significant_matrix, 
                           labels, display_labels, alpha=0.05, save_plots=True):
    """
    Create individual visualizations for each comparison matrix and save them separately
    """
    
    # Set up common parameters
    plt.rcParams.update({'font.size': 12})
    
    # 1. Mann-Whitney U p-values heatmap
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    im1 = ax1.imshow(mw_pvalue_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.1)
    ax1.set_title('Mann-Whitney U Test - P-values', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(display_labels)))
    ax1.set_yticks(range(len(display_labels)))
    ax1.set_xticklabels(display_labels, rotation=45, ha='right')
    ax1.set_yticklabels(display_labels)
    
    # Add text annotations for p-values
    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            if i != j:  # Skip diagonal
                text = f'{mw_pvalue_matrix[i, j]:.3f}' if not np.isnan(mw_pvalue_matrix[i, j]) else 'NaN'
                color = 'white' if mw_pvalue_matrix[i, j] < 0.05 else 'black'
                ax1.text(j, i, text, ha='center', va='center', 
                        color=color, fontsize=9, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=ax1, label='P-value', shrink=0.8)
    plt.tight_layout()
    if save_plots:
        plt.savefig('images/mann_whitney_pvalues.pdf', bbox_inches='tight')
        print("Saved: images/mann_whitney_pvalues.pdf")
    plt.show()
    
    # 2. Kolmogorov-Smirnov p-values heatmap
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    im2 = ax2.imshow(ks_pvalue_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.1)
    ax2.set_title('Kolmogorov-Smirnov Test - P-values', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(display_labels)))
    ax2.set_yticks(range(len(display_labels)))
    ax2.set_xticklabels(display_labels, rotation=45, ha='right')
    ax2.set_yticklabels(display_labels)
    
    # Add text annotations for p-values
    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            if i != j:  # Skip diagonal
                text = f'{ks_pvalue_matrix[i, j]:.3f}' if not np.isnan(ks_pvalue_matrix[i, j]) else 'NaN'
                color = 'white' if ks_pvalue_matrix[i, j] < 0.05 else 'black'
                ax2.text(j, i, text, ha='center', va='center', 
                        color=color, fontsize=9, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=ax2, label='P-value', shrink=0.8)
    plt.tight_layout()
    if save_plots:
        plt.savefig('images/kolmogorov_smirnov_pvalues.pdf', bbox_inches='tight')
        print("Saved: images/kolmogorov_smirnov_pvalues.pdf")
    plt.show()
    
    # 3. Mann-Whitney U significance matrix
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    im3 = ax3.imshow(mw_significant_matrix, cmap='RdBu_r', vmin=0, vmax=1)
    ax3.set_title('Mann-Whitney U Test - Significance Matrix', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xticks(range(len(display_labels)))
    ax3.set_yticks(range(len(display_labels)))
    ax3.set_xticklabels(display_labels, rotation=45, ha='right')
    ax3.set_yticklabels(display_labels)
    
    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            if i != j:
                text = 'Sig' if mw_significant_matrix[i, j] == 1 else 'ns'
                color = 'white' if mw_significant_matrix[i, j] == 1 else 'black'
                ax3.text(j, i, text, ha='center', va='center', 
                        color=color, fontsize=10, fontweight='bold')
    
    cbar3 = plt.colorbar(im3, ax=ax3, label='Significant (1=Yes, 0=No)', shrink=0.8)
    plt.tight_layout()
    if save_plots:
        plt.savefig('images/mann_whitney_significance.pdf', bbox_inches='tight')
        print("Saved: images/mann_whitney_significance.pdf")
    plt.show()
    
    # 4. Kolmogorov-Smirnov significance matrix
    fig4, ax4 = plt.subplots(figsize=(12, 10))
    im4 = ax4.imshow(ks_significant_matrix, cmap='RdBu_r', vmin=0, vmax=1)
    ax4.set_title('Kolmogorov-Smirnov Test - Significance Matrix', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xticks(range(len(display_labels)))
    ax4.set_yticks(range(len(display_labels)))
    ax4.set_xticklabels(display_labels, rotation=45, ha='right')
    ax4.set_yticklabels(display_labels)
    
    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            if i != j:
                text = 'Sig' if ks_significant_matrix[i, j] == 1 else 'ns'
                color = 'white' if ks_significant_matrix[i, j] == 1 else 'black'
                ax4.text(j, i, text, ha='center', va='center', 
                        color=color, fontsize=10, fontweight='bold')
    
    cbar4 = plt.colorbar(im4, ax=ax4, label='Significant (1=Yes, 0=No)', shrink=0.8)
    plt.tight_layout()
    if save_plots:
        plt.savefig('images/kolmogorov_smirnov_significance.pdf', bbox_inches='tight')
        print("Saved: images/kolmogorov_smirnov_significance.pdf")
    plt.show()
    
    # 5. Combined significance matrix
    combined_significance = mw_significant_matrix + ks_significant_matrix
    fig5, ax5 = plt.subplots(figsize=(12, 10))
    im5 = ax5.imshow(combined_significance, cmap='Reds', vmin=0, vmax=2)
    ax5.set_title('Combined Significance Matrix\n(Both Tests)', fontsize=16, fontweight='bold', pad=20)
    ax5.set_xticks(range(len(display_labels)))
    ax5.set_yticks(range(len(display_labels)))
    ax5.set_xticklabels(display_labels, rotation=45, ha='right')
    ax5.set_yticklabels(display_labels)
    
    # Add text annotations
    for i in range(len(display_labels)):
        for j in range(len(display_labels)):
            if i != j:
                value = int(combined_significance[i, j])
                if value == 2:
                    text = '***'  # Both tests significant
                elif value == 1:
                    text = '*'    # One test significant
                else:
                    text = 'ns'   # Not significant
                
                color = 'white' if value > 0 else 'black'
                ax5.text(j, i, text, ha='center', va='center', 
                        color=color, fontsize=11, fontweight='bold')
    
    cbar5 = plt.colorbar(im5, ax=ax5, shrink=0.8)
    cbar5.set_ticks([0, 1, 2])
    cbar5.set_ticklabels(['Not Sig.', 'One Test', 'Both Tests'])
    cbar5.set_label('Significance Level')
    plt.tight_layout()
    if save_plots:
        plt.savefig('images/combined_significance.pdf', bbox_inches='tight')
        print("Saved: images/combined_significance.pdf")
    plt.show()
    
    # 6. Summary statistics plot
    fig6, ax6 = plt.subplots(figsize=(12, 10))
    ax6.axis('off')
    
    # Calculate summary statistics
    total_comparisons = len(display_labels) * (len(display_labels) - 1)
    mw_significant_count = np.sum(mw_significant_matrix)
    ks_significant_count = np.sum(ks_significant_matrix)
    both_significant_count = np.sum((mw_significant_matrix == 1) & (ks_significant_matrix == 1))
    
    summary_text = f"""
    PAIRWISE DISTRIBUTION COMPARISON SUMMARY
    ========================================
    
    Total Datasets: {len(display_labels)}
    Total Pairwise Comparisons: {total_comparisons}
    
    Mann-Whitney U Test Results:
    • Significant Differences: {int(mw_significant_count)} ({mw_significant_count/total_comparisons*100:.1f}%)
    • Not Significant: {int(total_comparisons - mw_significant_count)} ({(total_comparisons - mw_significant_count)/total_comparisons*100:.1f}%)
    
    Kolmogorov-Smirnov Test Results:
    • Significant Differences: {int(ks_significant_count)} ({ks_significant_count/total_comparisons*100:.1f}%)
    • Not Significant: {int(total_comparisons - ks_significant_count)} ({(total_comparisons - ks_significant_count)/total_comparisons*100:.1f}%)
    
    Agreement Between Tests:
    • Both Tests Significant: {int(both_significant_count)} ({both_significant_count/total_comparisons*100:.1f}%)
    • Only Mann-Whitney Significant: {int(mw_significant_count - both_significant_count)}
    • Only Kolmogorov-Smirnov Significant: {int(ks_significant_count - both_significant_count)}
    
    Test Parameters:
    • Significance Level: α = {alpha}
    • Alternative Hypothesis: Two-sided
    
    Legend for Combined Matrix:
    *** = Both tests show significant difference
    *   = One test shows significant difference
    ns  = No significant difference found
    
    Grid Configurations (sorted by size):
    {chr(10).join([f"  {i+1}. {label}" for i, label in enumerate(display_labels)])}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('images/summary_statistics.pdf', bbox_inches='tight')
        print("Saved: images/summary_statistics.pdf")
    plt.show()
    
    return [fig1, fig2, fig3, fig4, fig5, fig6]

def plot_distribution_violin(data_dict, labels, display_labels, save_plot=True):
    """
    Create violin plot showing the distribution of mismatch values for each grid configuration
    """
    
    # Prepare data for violin plot
    all_data = []
    all_labels = []
    
    for i, label in enumerate(labels):
        data = data_dict[label]
        all_data.extend(data)
        all_labels.extend([display_labels[i]] * len(data))
    
    # Create DataFrame for easier plotting
    df_violin = pd.DataFrame({
        'Grid Configuration': all_labels,
        'Empirical Mismatch (μ)': all_data
    })
    
    # Create the violin plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create violin plot with seaborn for better aesthetics
    violin_parts = ax.violinplot([data_dict[label] for label in labels], 
                                positions=range(len(labels)), 
                                showmeans=True, showmedians=True, 
                                showextrema=True)
    
    # Customize violin plot colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    for pc, color in zip(violin_parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    # Customize other elements
    violin_parts['cmeans'].set_color('red')
    violin_parts['cmeans'].set_linewidth(2)
    violin_parts['cmedians'].set_color('white')
    violin_parts['cmedians'].set_linewidth(2)
    violin_parts['cbars'].set_color('black')
    violin_parts['cmins'].set_color('black')
    violin_parts['cmaxes'].set_color('black')
    
    # Add box plots on top for additional statistical info
    box_parts = ax.boxplot([data_dict[label] for label in labels], 
                          positions=range(len(labels)), 
                          patch_artist=False, 
                          widths=0.3,
                          showfliers=False,  # Don't show outliers (already in violin)
                          medianprops=dict(color='orange', linewidth=2),
                          boxprops=dict(color='black', linewidth=1),
                          whiskerprops=dict(color='black', linewidth=1),
                          capprops=dict(color='black', linewidth=1))
    
    # Formatting
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha='right')
    ax.set_ylabel('Empirical Mismatch (μ)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Grid Configuration', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Empirical Mismatch Values by Grid Configuration', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistical annotations
    stats_text = ""
    for i, (label, display_label) in enumerate(zip(labels, display_labels)):
        data = data_dict[label]
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        stats_text += f"{display_label}: μ={mean_val:.4f}, Med={median_val:.4f}, σ={std_val:.4f}\n"
    
    # Add statistics box
    ax.text(0.02, 0.98, f"Statistics Summary:\n{stats_text}", 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
           fontfamily='monospace')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', linewidth=2, label='Mean'),
        plt.Line2D([0], [0], color='white', linewidth=2, label='Median (violin)'),
        plt.Line2D([0], [0], color='orange', linewidth=2, label='Median (box)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, label='Distribution density')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('images/mismatch_distribution_violin.pdf', bbox_inches='tight')
        print("Saved: images/mismatch_distribution_violin.pdf")
    
    plt.show()
    
    return fig

def plot_distribution_summary_stats(data_dict, labels, display_labels, save_plot=True):
    """
    Create a comprehensive statistical summary plot
    """
    
    # Calculate statistics for each dataset
    stats_data = []
    for label, display_label in zip(labels, display_labels):
        data = data_dict[label]
        stats_data.append({
            'Grid_Config': display_label,
            'Mean': np.mean(data),
            'Median': np.median(data),
            'Std': np.std(data),
            'Min': np.min(data),
            'Max': np.max(data),
            'Q25': np.percentile(data, 25),
            'Q75': np.percentile(data, 75),
            'IQR': np.percentile(data, 75) - np.percentile(data, 25),
            'Skewness': pd.Series(data).skew(),
            'Kurtosis': pd.Series(data).kurtosis(),
            'Count': len(data)
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create subplots for different statistical measures
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Mean and Standard Deviation
    x_pos = np.arange(len(display_labels))
    axes[0, 0].errorbar(x_pos, stats_df['Mean'], yerr=stats_df['Std'], 
                       fmt='o-', capsize=5, capthick=2, markersize=8)
    axes[0, 0].set_title('Mean ± Standard Deviation', fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(display_labels, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylabel('Empirical Mismatch (μ)')
    
    # 2. Range (Min-Max)
    axes[0, 1].fill_between(x_pos, stats_df['Min'], stats_df['Max'], alpha=0.3, label='Range')
    axes[0, 1].plot(x_pos, stats_df['Median'], 'ro-', label='Median', markersize=6)
    axes[0, 1].set_title('Range and Median', fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(display_labels, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylabel('Empirical Mismatch (μ)')
    
    # 3. Interquartile Range
    axes[0, 2].fill_between(x_pos, stats_df['Q25'], stats_df['Q75'], alpha=0.5, label='IQR')
    axes[0, 2].plot(x_pos, stats_df['Median'], 'ko-', label='Median', markersize=6)
    axes[0, 2].set_title('Interquartile Range (IQR)', fontweight='bold')
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(display_labels, rotation=45, ha='right')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    axes[0, 2].set_ylabel('Empirical Mismatch (μ)')
    
    # 4. Coefficient of Variation
    cv = stats_df['Std'] / stats_df['Mean']
    axes[1, 0].bar(x_pos, cv, alpha=0.7, color='orange')
    axes[1, 0].set_title('Coefficient of Variation (σ/μ)', fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(display_labels, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].set_ylabel('CV')
    
    # 5. Skewness
    colors = ['red' if s > 0 else 'blue' for s in stats_df['Skewness']]
    axes[1, 1].bar(x_pos, stats_df['Skewness'], alpha=0.7, color=colors)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].set_title('Skewness', fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(display_labels, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylabel('Skewness')
    
    # 6. Sample Size
    axes[1, 2].bar(x_pos, stats_df['Count'], alpha=0.7, color='green')
    axes[1, 2].set_title('Sample Size', fontweight='bold')
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(display_labels, rotation=45, ha='right')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].set_ylabel('Number of Samples')
    
    plt.suptitle('Comprehensive Statistical Summary by Grid Configuration', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('images/statistical_summary.pdf', bbox_inches='tight')
        print("Saved: images/statistical_summary.pdf")
    
    plt.show()
    
    # Also save statistics as CSV in data/ directory
    stats_df.to_csv('data/grid_configuration_statistics.csv', index=False, float_format='%.6f')
    print("Saved: data/grid_configuration_statistics.csv")
    
    return fig, stats_df

def save_matrices_to_csv(mw_pvalue_matrix, ks_pvalue_matrix, 
                        mw_significant_matrix, ks_significant_matrix, 
                        labels, display_labels):
    """
    Save all matrices to separate CSV files with proper labeling
    """
    
    # Create DataFrames with clear display labels for row and column labels
    mw_pvalues_df = pd.DataFrame(mw_pvalue_matrix, index=display_labels, columns=display_labels)
    ks_pvalues_df = pd.DataFrame(ks_pvalue_matrix, index=display_labels, columns=display_labels)
    mw_significant_df = pd.DataFrame(mw_significant_matrix, index=display_labels, columns=display_labels)
    ks_significant_df = pd.DataFrame(ks_significant_matrix, index=display_labels, columns=display_labels)
    
    # Save to CSV files in data/ directory
    mw_pvalues_df.to_csv('data/mann_whitney_pvalues_matrix.csv')
    ks_pvalues_df.to_csv('data/kolmogorov_smirnov_pvalues_matrix.csv')
    mw_significant_df.to_csv('data/mann_whitney_significance_matrix.csv')
    ks_significant_df.to_csv('data/kolmogorov_smirnov_significance_matrix.csv')
    
    print("\nMatrices saved to CSV files:")
    print("  • data/mann_whitney_pvalues_matrix.csv")
    print("  • data/kolmogorov_smirnov_pvalues_matrix.csv")
    print("  • data/mann_whitney_significance_matrix.csv")
    print("  • data/kolmogorov_smirnov_significance_matrix.csv")
    
    return {
        'mann_whitney_pvalues': mw_pvalues_df,
        'kolmogorov_smirnov_pvalues': ks_pvalues_df,
        'mann_whitney_significance': mw_significant_df,
        'kolmogorov_smirnov_significance': ks_significant_df
    }
def create_detailed_results_table(mw_pvalue_matrix, ks_pvalue_matrix, labels, display_labels, alpha=0.05):
    """
    Create a detailed results table for all pairwise comparisons
    """
    
    results_list = []
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:  # Skip diagonal comparisons
                mw_p = mw_pvalue_matrix[i, j]
                ks_p = ks_pvalue_matrix[i, j]
                
                results_list.append({
                    'Grid_Configuration_1': display_labels[i],
                    'Grid_Configuration_2': display_labels[j],
                    'Original_File_1': labels[i],
                    'Original_File_2': labels[j],
                    'Mann_Whitney_p': mw_p,
                    'Mann_Whitney_Significant': 'Yes' if mw_p < alpha else 'No',
                    'KS_p': ks_p,
                    'KS_Significant': 'Yes' if ks_p < alpha else 'No',
                    'Both_Significant': 'Yes' if (mw_p < alpha and ks_p < alpha) else 'No'
                })
    
    results_df = pd.DataFrame(results_list)
    
    print("\nDETAILED PAIRWISE COMPARISON RESULTS")
    print("=" * 80)
    # Display only the grid configurations and test results (not the long filenames)
    display_df = results_df[['Grid_Configuration_1', 'Grid_Configuration_2', 
                           'Mann_Whitney_p', 'Mann_Whitney_Significant',
                           'KS_p', 'KS_Significant', 'Both_Significant']]
    print(display_df.to_string(index=False, float_format='%.4f'))
    
    return results_df

# Main execution function
def main():
    """
    Main function to execute the complete analysis
    """
    
    print("PAIRWISE DISTRIBUTION COMPARISON ANALYSIS")
    print("=" * 60)
    
    # 0. Ensure output directories exist
    # ensure_directories()
    
    # 1. Load all CSV files
    data_dict = load_all_csv_files()
    
    if len(data_dict) < 2:
        print("Error: Need at least 2 datasets for comparison")
        return
    
    # 2. Create pairwise comparison matrices
    mw_pvalues, ks_pvalues, mw_significant, ks_significant, labels, display_labels = create_pairwise_comparison_matrices(data_dict)
    
    # 3. Create distribution visualizations
    print("\n" + "="*60)
    print("CREATING DISTRIBUTION VISUALIZATIONS")
    print("="*60)
    
    # Violin plot showing mismatch distributions
    violin_fig = plot_distribution_violin(data_dict, labels, display_labels)
    
    # Comprehensive statistical summary
    summary_fig, stats_df = plot_distribution_summary_stats(data_dict, labels, display_labels)
    
    # 4. Save matrices to CSV files (with clear display labels)
    print("\n" + "="*60)
    print("PERFORMING STATISTICAL COMPARISONS")
    print("="*60)
    
    matrix_dataframes = save_matrices_to_csv(mw_pvalues, ks_pvalues, mw_significant, ks_significant, labels, display_labels)
    
    # 5. Create individual comparison matrix visualizations (saved separately)
    figures = plot_individual_matrices(mw_pvalues, ks_pvalues, mw_significant, ks_significant, labels, display_labels)
    
    # 6. Create detailed results table
    results_df = create_detailed_results_table(mw_pvalues, ks_pvalues, labels, display_labels)
    
    # 7. Save detailed results to CSV in data/ directory
    results_df.to_csv('data/detailed_pairwise_comparison_results.csv', index=False)
    print(f"\nDetailed results saved to 'data/detailed_pairwise_comparison_results.csv'")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("  PDF files (images/): 8 visualizations")
    print("    • mismatch_distribution_violin.pdf")
    print("    • statistical_summary.pdf")
    print("    • mann_whitney_pvalues.pdf")
    print("    • kolmogorov_smirnov_pvalues.pdf")
    print("    • mann_whitney_significance.pdf")
    print("    • kolmogorov_smirnov_significance.pdf")
    print("    • combined_significance.pdf")
    print("    • summary_statistics.pdf")
    print("  CSV files (data/): 6 data files")
    print("    • mann_whitney_pvalues_matrix.csv")
    print("    • kolmogorov_smirnov_pvalues_matrix.csv")
    print("    • mann_whitney_significance_matrix.csv")
    print("    • kolmogorov_smirnov_significance_matrix.csv")
    print("    • detailed_pairwise_comparison_results.csv")
    print("    • grid_configuration_statistics.csv")
    print(f"\nGrid configurations analyzed:")
    for i, (orig, display) in enumerate(zip(labels, display_labels), 1):
        print(f"  {i}. {display} (from {orig})")
    
    return {
        'data': data_dict,
        'mann_whitney_pvalues': mw_pvalues,
        'kolmogorov_smirnov_pvalues': ks_pvalues,
        'mann_whitney_significant': mw_significant,
        'kolmogorov_smirnov_significant': ks_significant,
        'labels': labels,
        'display_labels': display_labels,
        'matrix_dataframes': matrix_dataframes,
        'results_dataframe': results_df,
        'figures': figures,
        'violin_figure': violin_fig,
        'summary_figure': summary_fig,
        'statistics_dataframe': stats_df
    }

# Execute the analysis
if __name__ == "__main__":
    results = main()