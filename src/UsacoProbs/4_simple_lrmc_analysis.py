import numpy as np
from scipy import stats

def statistical_analysis():
    """Proper statistical analysis of the results"""

    # From your results (expand to more seeds for rigor)
    baseline_results = [0.8040, 0.7990, 0.8050, 0.8060, 0.8080, 0.7950, 0.8150, 0.8130, 0.8160, 0.8090, 0.8140, 0.8100, 0.8270, 0.8070, 0.8110, 0.8040, 0.8180, 0.8060]
    binary_feature_results = [0.8100, 0.8220, 0.8000, 0.8180, 0.8120, 0.8070, 0.8270, 0.8080, 0.8090, 0.8130, 0.8190, 0.8230, 0.8140, 0.8100, 0.8120, 0.8050, 0.8140, 0.7820]

    print("Statistical Analysis:")
    print(f"Baseline:       {np.mean(baseline_results):.4f} ± {np.std(baseline_results):.4f}")
    print(f"Binary Feature: {np.mean(binary_feature_results):.4f} ± {np.std(binary_feature_results):.4f}")

    # Paired t-test (since we used same seeds)
    t_stat, p_value = stats.ttest_rel(binary_feature_results, baseline_results)
    print("\nPaired t-test:")
    print(f"t-statistic: {t_stat:.3f}")
    print(f"p-value: {p_value:.3f}")
    print(f"Significant at p<0.05: {'Yes' if p_value < 0.05 else 'No'}")

    # Effect size (Cohen's d)
    diff = np.array(binary_feature_results) - np.array(baseline_results)
    effect_size = np.mean(diff) / np.std(diff)
    print(f"Effect size (Cohen's d): {effect_size:.3f}")

    if effect_size > 0.2:
        print("Effect size: Small but meaningful")
    elif effect_size > 0.5:
        print("Effect size: Medium")
    else:
        print("Effect size: Very small")

statistical_analysis()
