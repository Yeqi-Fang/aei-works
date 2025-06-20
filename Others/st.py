import numpy as np
import scipy.stats as stats
from scipy.stats import mannwhitneyu, ks_2samp, ttest_ind, shapiro, levene
import matplotlib.pyplot as plt
import seaborn as sns

def test_normality(data, alpha=0.05):
    """
    Test if data follows a normal distribution using Shapiro-Wilk test
    
    Parameters:
    data: array-like, sample data
    alpha: float, significance level (default 0.05)
    
    Returns:
    bool: True if data appears normal, False otherwise
    """
    if len(data) < 3:
        print("Warning: Sample size too small for normality testing")
        return False
    
    stat, p_value = shapiro(data)
    is_normal = p_value > alpha
    
    print(f"Shapiro-Wilk test:")
    print(f"  Statistic: {stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Normal distribution: {'Yes' if is_normal else 'No'} (α={alpha})")
    
    return is_normal

def compare_distributions(data1, data2, alpha=0.05, verbose=True):
    """
    Comprehensive comparison of two distributions
    
    Parameters:
    data1, data2: array-like, the two samples to compare
    alpha: float, significance level
    verbose: bool, whether to print detailed results
    
    Returns:
    dict: Dictionary containing test results
    """
    
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    results = {}
    
    if verbose:
        print("=" * 60)
        print("STATISTICAL COMPARISON OF TWO DISTRIBUTIONS")
        print("=" * 60)
        print(f"Sample 1: n={len(data1)}, mean={np.mean(data1):.4f}, std={np.std(data1):.4f}")
        print(f"Sample 2: n={len(data2)}, mean={np.mean(data2):.4f}, std={np.std(data2):.4f}")
        print()
    
    # 1. Test for normality
    if verbose:
        print("1. NORMALITY TESTS")
        print("-" * 30)
        print("Sample 1:")
    normal1 = test_normality(data1, alpha)
    if verbose:
        print("\nSample 2:")
    normal2 = test_normality(data2, alpha)
    
    results['normal1'] = normal1
    results['normal2'] = normal2
    results['both_normal'] = normal1 and normal2
    
    if verbose:
        print("\n2. DISTRIBUTION COMPARISON TESTS")
        print("-" * 40)
    
    # 2. Kolmogorov-Smirnov test (always applicable)
    ks_stat, ks_p = ks_2samp(data1, data2)
    results['ks_test'] = {'statistic': ks_stat, 'p_value': ks_p, 'significant': ks_p < alpha}
    
    if verbose:
        print(f"Kolmogorov-Smirnov Test:")
        print(f"  Statistic: {ks_stat:.4f}")
        print(f"  P-value: {ks_p:.4f}")
        print(f"  Significant difference: {'Yes' if ks_p < alpha else 'No'}")
        print()
    
    # 3. Mann-Whitney U test (non-parametric)
    mw_stat, mw_p = mannwhitneyu(data1, data2, alternative='two-sided')
    results['mann_whitney'] = {'statistic': mw_stat, 'p_value': mw_p, 'significant': mw_p < alpha}
    
    if verbose:
        print(f"Mann-Whitney U Test (non-parametric):")
        print(f"  Statistic: {mw_stat:.4f}")
        print(f"  P-value: {mw_p:.4f}")
        print(f"  Significant difference: {'Yes' if mw_p < alpha else 'No'}")
        print()
    
    # 4. If both distributions appear normal, perform parametric tests
    if normal1 and normal2:
        # Test for equal variances
        levene_stat, levene_p = levene(data1, data2)
        equal_var = levene_p > alpha
        results['equal_variances'] = equal_var
        
        if verbose:
            print(f"Levene's Test for Equal Variances:")
            print(f"  Statistic: {levene_stat:.4f}")
            print(f"  P-value: {levene_p:.4f}")
            print(f"  Equal variances: {'Yes' if equal_var else 'No'}")
            print()
        
        # Perform appropriate t-test
        if equal_var:
            t_stat, t_p = ttest_ind(data1, data2, equal_var=True)
            test_name = "Independent t-test (equal variances)"
        else:
            t_stat, t_p = ttest_ind(data1, data2, equal_var=False)
            test_name = "Welch's t-test (unequal variances)"
        
        results['t_test'] = {'statistic': t_stat, 'p_value': t_p, 'significant': t_p < alpha, 'test_type': test_name}
        
        if verbose:
            print(f"{test_name}:")
            print(f"  Statistic: {t_stat:.4f}")
            print(f"  P-value: {t_p:.4f}")
            print(f"  Significant difference: {'Yes' if t_p < alpha else 'No'}")
            print()
    
    # 5. Effect size (Cohen's d) for practical significance
    pooled_std = np.sqrt(((len(data1)-1)*np.var(data1, ddof=1) + (len(data2)-1)*np.var(data2, ddof=1)) / (len(data1)+len(data2)-2))
    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
    results['cohens_d'] = cohens_d
    
    if verbose:
        print(f"Effect Size (Cohen's d): {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        print(f"Effect size interpretation: {effect_size}")
        print()
    
    # 6. Recommendations
    if verbose:
        print("3. RECOMMENDATIONS")
        print("-" * 20)
        if normal1 and normal2:
            print("✓ Both distributions appear normal")
            print("✓ Recommended: Use t-test results for statistical significance")
            print("✓ Also consider: Kolmogorov-Smirnov test for distribution shape")
        else:
            print("⚠ At least one distribution is non-normal")
            print("✓ Recommended: Use Mann-Whitney U test or Kolmogorov-Smirnov test")
            print("✓ Avoid: t-test (parametric assumptions violated)")
        
        print(f"✓ Consider effect size (Cohen's d = {cohens_d:.3f}) for practical significance")
    
    return results

def plot_distribution_comparison(data1, data2, labels=['Sample 1', 'Sample 2']):
    """
    Create visualization comparing two distributions
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram comparison
    axes[0,0].hist(data1, alpha=0.7, label=labels[0], bins=20, density=True)
    axes[0,0].hist(data2, alpha=0.7, label=labels[1], bins=20, density=True)
    axes[0,0].set_title('Histogram Comparison')
    axes[0,0].set_xlabel('Value')
    axes[0,0].set_ylabel('Density')
    axes[0,0].legend()
    
    # Box plot comparison
    axes[0,1].boxplot([data1, data2], labels=labels)
    axes[0,1].set_title('Box Plot Comparison')
    axes[0,1].set_ylabel('Value')
    
    # Q-Q plots for normality
    stats.probplot(data1, dist="norm", plot=axes[1,0])
    axes[1,0].set_title(f'Q-Q Plot: {labels[0]}')
    
    stats.probplot(data2, dist="norm", plot=axes[1,1])
    axes[1,1].set_title(f'Q-Q Plot: {labels[1]}')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    import pandas as pd
    # Generate example data
    np.random.seed(42)
    
    # Example 1: Two normal distributions with different means
    print("EXAMPLE 1: Two normal distributions")
    sample1_normal = pd.read_csv('PyFstat_example_data/PyFstatExampleSimpleMCMCvsGridComparisonSemi/mismatches-5-10-1000.csv')['Empirical Mismatch (μ)'].values
    sample2_normal = pd.read_csv('PyFstat_example_data/PyFstatExampleSimpleMCMCvsGridComparisonSemi/mismatches-5-6-1000.csv')['Empirical Mismatch (μ)'].values
    
    results1 = compare_distributions(sample1_normal, sample2_normal)
    plot_distribution_comparison(sample1_normal, sample2_normal, ['Normal 1', 'Normal 2'])
    
    print("\n" + "="*80 + "\n")
    
    # # Example 2: One normal, one non-normal distribution
    # print("EXAMPLE 2: Normal vs Non-normal distribution")
    # sample1_normal = np.random.normal(50, 10, 100)
    # sample2_skewed = np.random.exponential(2, 100)
    
    # results2 = compare_distributions(sample1_normal, sample2_skewed)
    # plot_distribution_comparison(sample1_normal, sample2_skewed, ['Normal', 'Exponential'])

# Quick function for simple comparison
def quick_compare(data1, data2, alpha=0.05):
    """
    Quick comparison with minimal output
    Returns: (significant_difference, recommended_test, p_value)
    """
    # Test normality
    normal1 = shapiro(data1)[1] > alpha if len(data1) >= 3 else False
    normal2 = shapiro(data2)[1] > alpha if len(data2) >= 3 else False
    
    if normal1 and normal2:
        # Use t-test
        _, p_val = ttest_ind(data1, data2)
        test_name = "t-test"
    else:
        # Use Mann-Whitney U test
        _, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    return p_val < alpha, test_name, p_val