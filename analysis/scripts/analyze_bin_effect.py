#!/usr/bin/env python3
"""
Analyze the effect of bin count on MIB values and stability metrics.

This script runs MIB analysis with varying numbers of histogram bins
to understand how discretization affects the mutual information estimates.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import linalg, stats

# Add src to path for imports if package not installed
# After running `pip install -e .` from repo root, this is unnecessary
try:
    from eeg_analysis.config import ANALYSIS_PARAMS
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = REPO_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from eeg_analysis.config import ANALYSIS_PARAMS

from eeg_analysis.analyzers.complexity_analyzer import ComplexityAnalyzer
from eeg_analysis.analyzers.estimators import BinningEstimator
from eeg_analysis.core import EEGDataCache, generate_bipartitions, preprocess_eeg

mne.set_log_level("WARNING")


def gaussian_entropy(cov_matrix):
    """
    Compute entropy of a multivariate Gaussian given its covariance matrix.

    H(X) = 0.5 * n * log(2πe) + 0.5 * log(det(Σ))
         = 0.5 * n * (1 + log(2π)) + 0.5 * log(det(Σ))

    Returns entropy in bits (using log2).
    """
    n = cov_matrix.shape[0]
    # Use slogdet for numerical stability
    sign, logdet = np.linalg.slogdet(cov_matrix)
    if sign <= 0:
        return np.nan  # Covariance matrix is not positive definite

    # Convert to bits (divide by log(2))
    entropy_nats = 0.5 * n * (1 + np.log(2 * np.pi)) + 0.5 * logdet
    entropy_bits = entropy_nats / np.log(2)
    return entropy_bits


def gaussian_integration(data, partition_indices):
    """
    Compute integration (mutual information) for a bipartition assuming Gaussian.

    I(A;B) = H(A) + H(B) - H(A,B)
    """
    n_channels = data.shape[0]
    all_indices = set(range(n_channels))
    subset1 = list(partition_indices)
    subset2 = list(all_indices - set(partition_indices))

    # Compute covariance matrices
    cov_full = np.cov(data)
    cov_1 = np.cov(data[subset1, :])
    cov_2 = np.cov(data[subset2, :])

    # Handle 1D case (single channel subsets)
    if len(subset1) == 1:
        cov_1 = np.array([[cov_1]])
    if len(subset2) == 1:
        cov_2 = np.array([[cov_2]])

    h_full = gaussian_entropy(cov_full)
    h_1 = gaussian_entropy(cov_1)
    h_2 = gaussian_entropy(cov_2)

    if np.isnan(h_full) or np.isnan(h_1) or np.isnan(h_2):
        return np.nan

    return h_1 + h_2 - h_full


def gaussian_mib(data, partitions):
    """
    Compute MIB assuming Gaussian distribution (analytical).

    This serves as the theoretical baseline that binning should converge to.
    """
    integration_values = []
    for p in partitions:
        mi = gaussian_integration(data, p)
        if not np.isnan(mi) and np.isfinite(mi):
            integration_values.append(mi)

    if not integration_values:
        return np.nan

    return min(integration_values)


def compute_gaussian_baseline(epochs_array, n_channels):
    """
    Compute Gaussian MIB baseline for all epochs.
    """
    partitions = generate_bipartitions(n_channels, max_partitions=127, verbose=False)

    mib_values = []
    for epoch_data in epochs_array:
        mib = gaussian_mib(epoch_data, partitions)
        if not np.isnan(mib) and np.isfinite(mib):
            mib_values.append(mib)

    if not mib_values:
        return None

    return {
        'metric_values': np.array(mib_values),
        'mean_mib': float(np.mean(mib_values)),
        'std_mib': float(np.std(mib_values)),
        'cv_mib': float(np.std(mib_values) / np.mean(mib_values)) if np.mean(mib_values) > 0 else np.inf,
    }


def run_mib_with_bins(raw, n_bins: int, n_channels: int = 8, epoch_length: float = 5.0):
    """
    Run MIB analysis with a specific number of bins.

    Returns dict with MIB values and statistics.
    """
    params = {
        **ANALYSIS_PARAMS,
        'n_bins': n_bins,
        'n_channels': n_channels,
        'epoch_length': epoch_length,
        'verbose': False,
    }

    estimator = BinningEstimator(**params)
    analyzer = ComplexityAnalyzer(estimator=estimator, **params)

    # Preprocess and get epochs
    epochs_data, selected_channels = preprocess_eeg(raw, **params)

    if epochs_data is None or len(epochs_data) == 0:
        return None

    # Convert to numpy array if needed
    if isinstance(epochs_data, list):
        epochs_array = np.stack(epochs_data)
    else:
        epochs_array = epochs_data

    # Compute MIB for each epoch
    partitions = analyzer._get_partitions(n_channels)
    metric_values = []

    for epoch_data in epochs_array:
        mib_value = analyzer._calculate_metric_for_epoch(epoch_data, partitions)
        if mib_value is not None and np.isfinite(mib_value):
            metric_values.append(mib_value)

    if not metric_values:
        return None

    metric_values = np.array(metric_values)
    mean_mib = float(np.mean(metric_values))
    std_mib = float(np.std(metric_values))
    cv_mib = std_mib / abs(mean_mib) if abs(mean_mib) > 1e-10 else float('inf')

    return {
        'n_bins': n_bins,
        'n_epochs': len(metric_values),
        'metric_values': metric_values,
        'mean_mib': mean_mib,
        'std_mib': std_mib,
        'cv_mib': cv_mib,
        'min_mib': float(np.min(metric_values)),
        'max_mib': float(np.max(metric_values)),
        'range_mib': float(np.ptp(metric_values)),
    }


def analyze_bin_effect(file_path: str, bin_counts: list, n_channels: int = 8):
    """
    Analyze how different bin counts affect MIB values and stability.
    Returns (binning_results, gaussian_baseline)
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING EFFECT OF BIN COUNT ON MIB")
    print(f"File: {Path(file_path).name}")
    print(f"Channels: {n_channels}")
    print(f"Bin counts to test: {bin_counts}")
    print(f"{'='*70}\n")

    # Load raw data once
    cache = EEGDataCache(max_cache_size=1)
    raw = cache.get_raw_data(file_path, verbose=False)

    # Preprocess once to get epochs for Gaussian baseline
    params = {
        **ANALYSIS_PARAMS,
        'n_channels': n_channels,
        'epoch_length': 5.0,
        'verbose': False,
    }
    epochs_data, selected_channels = preprocess_eeg(raw, **params)

    if isinstance(epochs_data, list):
        epochs_array = np.stack(epochs_data)
    else:
        epochs_array = epochs_data

    # Compute Gaussian baseline
    print("Computing Gaussian (analytical) baseline...", end=" ", flush=True)
    start_time = time.time()
    gaussian_baseline = compute_gaussian_baseline(epochs_array, n_channels)
    elapsed = time.time() - start_time
    if gaussian_baseline:
        print(f"MIB = {gaussian_baseline['mean_mib']:.4f} +/- {gaussian_baseline['std_mib']:.4f} "
              f"(CV = {gaussian_baseline['cv_mib']:.3f}) [{elapsed:.1f}s]")
    else:
        print("Failed")

    print()
    results = []

    for n_bins in bin_counts:
        print(f"Testing n_bins = {n_bins}...", end=" ", flush=True)
        start_time = time.time()

        result = run_mib_with_bins(raw, n_bins=n_bins, n_channels=n_channels)

        elapsed = time.time() - start_time

        if result is not None:
            results.append(result)
            print(f"MIB = {result['mean_mib']:.4f} +/- {result['std_mib']:.4f} "
                  f"(CV = {result['cv_mib']:.3f}) [{elapsed:.1f}s]")
        else:
            print(f"Failed")

    return results, gaussian_baseline


def print_summary(results: list, gaussian_baseline=None):
    """Print a summary table of results."""
    print(f"\n{'='*70}")
    print("SUMMARY: Effect of Bin Count on MIB")
    print(f"{'='*70}")

    if gaussian_baseline:
        print(f"\nGaussian (analytical) baseline: {gaussian_baseline['mean_mib']:.4f} +/- {gaussian_baseline['std_mib']:.4f}")

    print(f"\n{'n_bins':>8} | {'Mean MIB':>10} | {'Std MIB':>10} | {'CV':>8} | {'Range':>10} | {'% of Gauss':>10}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    gauss_mean = gaussian_baseline['mean_mib'] if gaussian_baseline else None

    for r in results:
        pct_gauss = (r['mean_mib'] / gauss_mean * 100) if gauss_mean else 0
        print(f"{r['n_bins']:>8} | {r['mean_mib']:>10.4f} | {r['std_mib']:>10.4f} | "
              f"{r['cv_mib']:>8.4f} | {r['range_mib']:>10.4f} | {pct_gauss:>9.1f}%")

    # Compute sensitivity metrics
    if len(results) >= 2:
        means = [r['mean_mib'] for r in results]
        stds = [r['std_mib'] for r in results]
        cvs = [r['cv_mib'] for r in results]

        print(f"\n{'='*70}")
        print("SENSITIVITY ANALYSIS")
        print(f"{'='*70}")
        print(f"Mean MIB range across bin counts: {min(means):.4f} to {max(means):.4f} (delta = {max(means)-min(means):.4f})")
        print(f"Std MIB range across bin counts:  {min(stds):.4f} to {max(stds):.4f}")
        print(f"CV range across bin counts:       {min(cvs):.4f} to {max(cvs):.4f}")

        # Relative change from smallest to largest bin count
        pct_change = abs(means[-1] - means[0]) / abs(means[0]) * 100 if means[0] != 0 else float('inf')
        print(f"\nRelative change in mean MIB ({results[0]['n_bins']} -> {results[-1]['n_bins']} bins): {pct_change:.1f}%")

        if gauss_mean:
            print(f"Gaussian baseline: {gauss_mean:.4f} bits")
            print(f"Highest bin count ({results[-1]['n_bins']}) reaches {means[-1]/gauss_mean*100:.1f}% of Gaussian estimate")


def plot_results(results: list, gaussian_baseline=None, save_path: str = None):
    """Create visualization of bin count effect with Gaussian baseline."""
    if not results:
        return

    bin_counts = [r['n_bins'] for r in results]
    means = [r['mean_mib'] for r in results]
    stds = [r['std_mib'] for r in results]
    cvs = [r['cv_mib'] for r in results]

    gauss_mean = gaussian_baseline['mean_mib'] if gaussian_baseline else None
    gauss_std = gaussian_baseline['std_mib'] if gaussian_baseline else None
    gauss_cv = gaussian_baseline['cv_mib'] if gaussian_baseline else None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Mean MIB vs bin count with Gaussian baseline
    ax1 = axes[0, 0]
    ax1.errorbar(bin_counts, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8, label='Binning')
    if gauss_mean:
        ax1.axhline(y=gauss_mean, color='red', linestyle='--', linewidth=2, label=f'Gaussian ({gauss_mean:.2f})')
        ax1.fill_between(bin_counts, gauss_mean - gauss_std, gauss_mean + gauss_std, color='red', alpha=0.1)
    ax1.set_xlabel('Number of Bins', fontsize=12)
    ax1.set_ylabel('Mean MIB (bits)', fontsize=12)
    ax1.set_title('Mean MIB vs Bin Count', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Std MIB vs bin count with Gaussian baseline
    ax2 = axes[0, 1]
    ax2.plot(bin_counts, stds, marker='s', linewidth=2, markersize=8, color='orange', label='Binning')
    if gauss_std:
        ax2.axhline(y=gauss_std, color='red', linestyle='--', linewidth=2, label=f'Gaussian ({gauss_std:.2f})')
    ax2.set_xlabel('Number of Bins', fontsize=12)
    ax2.set_ylabel('Std MIB (bits)', fontsize=12)
    ax2.set_title('MIB Standard Deviation vs Bin Count', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: CV vs bin count with Gaussian baseline
    ax3 = axes[1, 0]
    ax3.plot(bin_counts, cvs, marker='^', linewidth=2, markersize=8, color='green', label='Binning')
    if gauss_cv:
        ax3.axhline(y=gauss_cv, color='red', linestyle='--', linewidth=2, label=f'Gaussian ({gauss_cv:.2f})')
    ax3.set_xlabel('Number of Bins', fontsize=12)
    ax3.set_ylabel('Coefficient of Variation', fontsize=12)
    ax3.set_title('MIB Stability (CV) vs Bin Count', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Distribution of MIB values per bin count (boxplot) with Gaussian baseline
    ax4 = axes[1, 1]
    box_data = [r['metric_values'] for r in results]
    bp = ax4.boxplot(box_data, labels=[str(b) for b in bin_counts], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    if gauss_mean:
        ax4.axhline(y=gauss_mean, color='red', linestyle='--', linewidth=2, label=f'Gaussian mean')
    ax4.set_xlabel('Number of Bins', fontsize=12)
    ax4.set_ylabel('MIB (bits)', fontsize=12)
    ax4.set_title('MIB Distribution per Bin Count', fontsize=14)
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def test_gaussianity(epochs_array, channel_names=None):
    """
    Test how Gaussian the data is using multiple methods.

    Returns dict with test results per channel and overall.
    """
    n_epochs, n_channels, n_samples = epochs_array.shape

    results = {
        'per_channel': [],
        'overall': {}
    }

    # Flatten all data for overall tests
    all_data = epochs_array.reshape(-1)

    # Test each channel
    for ch_idx in range(n_channels):
        ch_data = epochs_array[:, ch_idx, :].flatten()
        ch_name = channel_names[ch_idx] if channel_names else f"Ch{ch_idx}"

        # Shapiro-Wilk (use subset if too many samples)
        if len(ch_data) > 5000:
            sample_idx = np.random.choice(len(ch_data), 5000, replace=False)
            sw_stat, sw_p = stats.shapiro(ch_data[sample_idx])
        else:
            sw_stat, sw_p = stats.shapiro(ch_data)

        # D'Agostino-Pearson test for skewness and kurtosis
        if len(ch_data) >= 20:
            dp_stat, dp_p = stats.normaltest(ch_data)
        else:
            dp_stat, dp_p = np.nan, np.nan

        # Skewness and Kurtosis
        skewness = stats.skew(ch_data)
        kurtosis = stats.kurtosis(ch_data)  # Fisher's definition (0 for normal)

        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(ch_data)

        results['per_channel'].append({
            'channel': ch_name,
            'shapiro_stat': sw_stat,
            'shapiro_p': sw_p,
            'dagostino_stat': dp_stat,
            'dagostino_p': dp_p,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_p': jb_p,
            'skewness': skewness,
            'kurtosis': kurtosis,
        })

    # Overall statistics
    if len(all_data) > 5000:
        sample_idx = np.random.choice(len(all_data), 5000, replace=False)
        sw_stat, sw_p = stats.shapiro(all_data[sample_idx])
    else:
        sw_stat, sw_p = stats.shapiro(all_data)

    dp_stat, dp_p = stats.normaltest(all_data)
    jb_stat, jb_p = stats.jarque_bera(all_data)

    results['overall'] = {
        'shapiro_stat': sw_stat,
        'shapiro_p': sw_p,
        'dagostino_stat': dp_stat,
        'dagostino_p': dp_p,
        'jarque_bera_stat': jb_stat,
        'jarque_bera_p': jb_p,
        'skewness': stats.skew(all_data),
        'kurtosis': stats.kurtosis(all_data),
        'n_samples': len(all_data),
    }

    # Summary: average across channels
    avg_skewness = np.mean([r['skewness'] for r in results['per_channel']])
    avg_kurtosis = np.mean([r['kurtosis'] for r in results['per_channel']])
    pct_normal_shapiro = np.mean([r['shapiro_p'] > 0.05 for r in results['per_channel']]) * 100
    pct_normal_jb = np.mean([r['jarque_bera_p'] > 0.05 for r in results['per_channel']]) * 100

    results['summary'] = {
        'avg_skewness': avg_skewness,
        'avg_kurtosis': avg_kurtosis,
        'pct_channels_normal_shapiro': pct_normal_shapiro,
        'pct_channels_normal_jb': pct_normal_jb,
    }

    return results


def plot_gaussianity_comparison(all_gauss_results: dict, all_epochs: dict, save_path: str = None):
    """
    Plot Gaussianity comparison across conditions.
    """
    conditions = list(all_gauss_results.keys())
    n_cond = len(conditions)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = {'Awake EO': 'blue', 'Awake EC': 'green', 'Sedation': 'red'}

    # Row 1: Q-Q plots for each condition
    for idx, cond in enumerate(conditions):
        ax = axes[0, idx]
        epochs = all_epochs[cond]
        # Use first epoch, all channels flattened
        sample_data = epochs[0].flatten()

        # Q-Q plot
        stats.probplot(sample_data, dist="norm", plot=ax)
        ax.set_title(f'{cond} - Q-Q Plot', fontsize=12)
        ax.get_lines()[0].set_color(colors.get(cond, 'gray'))
        ax.get_lines()[0].set_markersize(3)

    # Row 2, Col 1: Skewness comparison
    ax = axes[1, 0]
    skewness_vals = [all_gauss_results[c]['summary']['avg_skewness'] for c in conditions]
    bars = ax.bar(conditions, skewness_vals, color=[colors.get(c, 'gray') for c in conditions], alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=-0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('Skewness', fontsize=12)
    ax.set_title('Average Skewness (0 = Gaussian)', fontsize=12)
    ax.set_ylim(-1, 1)
    for bar, val in zip(bars, skewness_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Row 2, Col 2: Kurtosis comparison
    ax = axes[1, 1]
    kurtosis_vals = [all_gauss_results[c]['summary']['avg_kurtosis'] for c in conditions]
    bars = ax.bar(conditions, kurtosis_vals, color=[colors.get(c, 'gray') for c in conditions], alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=-1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_ylabel('Excess Kurtosis', fontsize=12)
    ax.set_title('Average Kurtosis (0 = Gaussian)', fontsize=12)
    for bar, val in zip(bars, kurtosis_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    # Row 2, Col 3: % channels passing normality tests
    ax = axes[1, 2]
    x = np.arange(len(conditions))
    width = 0.35

    shapiro_pct = [all_gauss_results[c]['summary']['pct_channels_normal_shapiro'] for c in conditions]
    jb_pct = [all_gauss_results[c]['summary']['pct_channels_normal_jb'] for c in conditions]

    bars1 = ax.bar(x - width/2, shapiro_pct, width, label='Shapiro-Wilk', alpha=0.7)
    bars2 = ax.bar(x + width/2, jb_pct, width, label='Jarque-Bera', alpha=0.7)

    ax.set_ylabel('% Channels Passing (p > 0.05)', fontsize=12)
    ax.set_title('Normality Test Pass Rate', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 105)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nGaussianity plot saved to: {save_path}")

    plt.show()


def print_gaussianity_summary(all_gauss_results: dict):
    """Print summary of Gaussianity tests."""
    print(f"\n{'='*80}")
    print("GAUSSIANITY TEST RESULTS")
    print(f"{'='*80}")

    print(f"\n{'Condition':<12} | {'Skewness':>10} | {'Kurtosis':>10} | {'Shapiro p':>10} | {'JB p':>10} | {'% Normal':>10}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for cond, results in all_gauss_results.items():
        s = results['summary']
        o = results['overall']
        print(f"{cond:<12} | {s['avg_skewness']:>10.4f} | {s['avg_kurtosis']:>10.4f} | "
              f"{o['shapiro_p']:>10.2e} | {o['jarque_bera_p']:>10.2e} | {s['pct_channels_normal_shapiro']:>9.1f}%")

    print(f"\nInterpretation:")
    print(f"  - Skewness: 0 = symmetric (Gaussian), |skew| > 0.5 = notable asymmetry")
    print(f"  - Kurtosis: 0 = Gaussian tails, >0 = heavier tails, <0 = lighter tails")
    print(f"  - p-values: p < 0.05 rejects Gaussian hypothesis")
    print(f"  - % Normal: percentage of channels that pass normality test (p > 0.05)")


def plot_multi_condition_results(all_results: dict, save_path: str = None):
    """
    Plot bin count effect for multiple conditions on the same graph.

    all_results: dict mapping condition_name -> (results_list, gaussian_baseline)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    colors = {'Awake EO': 'blue', 'Awake EC': 'green', 'Sedation': 'red'}
    markers = {'Awake EO': 'o', 'Awake EC': 's', 'Sedation': '^'}

    # Plot 1: Mean MIB vs bin count
    ax1 = axes[0, 0]
    for cond_name, (results, gaussian_baseline) in all_results.items():
        bin_counts = [r['n_bins'] for r in results]
        means = [r['mean_mib'] for r in results]
        stds = [r['std_mib'] for r in results]
        color = colors.get(cond_name, 'gray')
        marker = markers.get(cond_name, 'o')

        ax1.errorbar(bin_counts, means, yerr=stds, marker=marker, capsize=3,
                     linewidth=2, markersize=6, label=cond_name, color=color, alpha=0.8)

        # Add Gaussian baseline as horizontal dashed line
        if gaussian_baseline:
            ax1.axhline(y=gaussian_baseline['mean_mib'], color=color, linestyle='--',
                       linewidth=1.5, alpha=0.5)

    ax1.set_xlabel('Number of Bins', fontsize=12)
    ax1.set_ylabel('Mean MIB (bits)', fontsize=12)
    ax1.set_title('Mean MIB vs Bin Count (dashed = Gaussian baseline)', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Std MIB vs bin count
    ax2 = axes[0, 1]
    for cond_name, (results, gaussian_baseline) in all_results.items():
        bin_counts = [r['n_bins'] for r in results]
        stds = [r['std_mib'] for r in results]
        color = colors.get(cond_name, 'gray')
        marker = markers.get(cond_name, 'o')

        ax2.plot(bin_counts, stds, marker=marker, linewidth=2, markersize=6,
                label=cond_name, color=color, alpha=0.8)

        if gaussian_baseline:
            ax2.axhline(y=gaussian_baseline['std_mib'], color=color, linestyle='--',
                       linewidth=1.5, alpha=0.5)

    ax2.set_xlabel('Number of Bins', fontsize=12)
    ax2.set_ylabel('Std MIB (bits)', fontsize=12)
    ax2.set_title('MIB Standard Deviation vs Bin Count', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: CV vs bin count
    ax3 = axes[1, 0]
    for cond_name, (results, gaussian_baseline) in all_results.items():
        bin_counts = [r['n_bins'] for r in results]
        cvs = [r['cv_mib'] for r in results]
        color = colors.get(cond_name, 'gray')
        marker = markers.get(cond_name, 'o')

        ax3.plot(bin_counts, cvs, marker=marker, linewidth=2, markersize=6,
                label=cond_name, color=color, alpha=0.8)

        if gaussian_baseline:
            ax3.axhline(y=gaussian_baseline['cv_mib'], color=color, linestyle='--',
                       linewidth=1.5, alpha=0.5)

    ax3.set_xlabel('Number of Bins', fontsize=12)
    ax3.set_ylabel('Coefficient of Variation', fontsize=12)
    ax3.set_title('MIB Stability (CV) vs Bin Count', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: % of Gaussian estimate
    ax4 = axes[1, 1]
    for cond_name, (results, gaussian_baseline) in all_results.items():
        if not gaussian_baseline:
            continue
        bin_counts = [r['n_bins'] for r in results]
        means = [r['mean_mib'] for r in results]
        pct_gauss = [m / gaussian_baseline['mean_mib'] * 100 for m in means]
        color = colors.get(cond_name, 'gray')
        marker = markers.get(cond_name, 'o')

        ax4.plot(bin_counts, pct_gauss, marker=marker, linewidth=2, markersize=6,
                label=cond_name, color=color, alpha=0.8)

    ax4.axhline(y=100, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='100% (Gaussian)')
    ax4.set_xlabel('Number of Bins', fontsize=12)
    ax4.set_ylabel('% of Gaussian Estimate', fontsize=12)
    ax4.set_title('Convergence to Gaussian Baseline', fontsize=14)
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def print_multi_summary(all_results: dict):
    """Print comparison summary for multiple conditions."""
    print(f"\n{'='*80}")
    print("MULTI-CONDITION COMPARISON")
    print(f"{'='*80}")

    # Print Gaussian baselines
    print("\nGaussian (analytical) baselines:")
    for cond_name, (results, gaussian_baseline) in all_results.items():
        if gaussian_baseline:
            print(f"  {cond_name}: {gaussian_baseline['mean_mib']:.2f} +/- {gaussian_baseline['std_mib']:.2f} bits")

    # Print table header
    print(f"\n{'Condition':<12} | {'n_bins':>6} | {'Mean MIB':>10} | {'Std':>8} | {'CV':>6} | {'% Gauss':>8}")
    print(f"{'-'*12}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")

    for cond_name, (results, gaussian_baseline) in all_results.items():
        gauss_mean = gaussian_baseline['mean_mib'] if gaussian_baseline else None
        for r in results:
            pct = (r['mean_mib'] / gauss_mean * 100) if gauss_mean else 0
            print(f"{cond_name:<12} | {r['n_bins']:>6} | {r['mean_mib']:>10.4f} | {r['std_mib']:>8.4f} | {r['cv_mib']:>6.3f} | {pct:>7.1f}%")
        print(f"{'-'*12}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")


def run_gaussianity_analysis(test_files: dict, n_channels: int = 8):
    """Run Gaussianity tests for multiple conditions."""
    all_gauss_results = {}
    all_epochs = {}

    cache = EEGDataCache(max_cache_size=3)

    for cond_name, file_path in test_files.items():
        print(f"\nTesting Gaussianity for {cond_name}...")

        raw = cache.get_raw_data(file_path, verbose=False)

        params = {
            **ANALYSIS_PARAMS,
            'n_channels': n_channels,
            'epoch_length': 5.0,
            'verbose': False,
        }
        epochs_data, selected_channels = preprocess_eeg(raw, **params)

        if isinstance(epochs_data, list):
            epochs_array = np.stack(epochs_data)
        else:
            epochs_array = epochs_data

        gauss_results = test_gaussianity(epochs_array, selected_channels)
        all_gauss_results[cond_name] = gauss_results
        all_epochs[cond_name] = epochs_array

    return all_gauss_results, all_epochs


def build_test_files(dataset_dir: Path, subject: str) -> dict:
    """Build test file paths for a subject."""
    eeg_dir = dataset_dir / subject / "eeg"
    return {
        'Awake EO': str(eeg_dir / f"{subject}_task-awake_acq-EO_eeg.vhdr"),
        'Awake EC': str(eeg_dir / f"{subject}_task-awake_acq-EC_eeg.vhdr"),
        'Sedation': str(eeg_dir / f"{subject}_task-sed_acq-rest_run-1_eeg.vhdr"),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze effect of bin count on MIB estimates."
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="sub-1067",
        help="Subject ID for test files (default: sub-1067)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ds005620",
        help="Path to dataset directory (default: ds005620)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for plots (default: results)",
    )
    parser.add_argument(
        "--gaussianity",
        action="store_true",
        help="Run only Gaussianity analysis",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=8,
        help="Number of channels to use (default: 8)",
    )
    args = parser.parse_args()

    test_files = build_test_files(Path(args.dataset), args.subject)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.gaussianity:
        print("Running Gaussianity analysis only...")
        all_gauss_results, all_epochs = run_gaussianity_analysis(test_files, n_channels=args.n_channels)
        print_gaussianity_summary(all_gauss_results)
        plot_gaussianity_comparison(all_gauss_results, all_epochs,
                                    save_path=str(output_dir / "gaussianity_comparison.png"))
    else:
        # Bin counts to test
        bin_counts = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]

        # Run analysis for each condition
        all_results = {}
        for cond_name, file_path in test_files.items():
            print(f"\n{'#'*70}")
            print(f"# CONDITION: {cond_name}")
            print(f"{'#'*70}")
            results, gaussian_baseline = analyze_bin_effect(file_path, bin_counts, n_channels=args.n_channels)
            all_results[cond_name] = (results, gaussian_baseline)

        # Print multi-condition summary
        print_multi_summary(all_results)

        # Save multi-condition plot
        plot_multi_condition_results(all_results, save_path=str(output_dir / "bin_count_effect_multi.png"))


if __name__ == "__main__":
    main()
