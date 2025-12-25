#!/usr/bin/env python3
"""
Analyze the effect of number of channels and epoch length on MIB.

Tests how these hyperparameters affect MIB estimates and whether
the relative differences between conditions are preserved.
"""

import sys
from pathlib import Path
import numpy as np
import mne
import time
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eeg_analysis.config import ANALYSIS_PARAMS, BINNING_PARAMS, SPECTRAL_BANDS
from eeg_analysis.analyzers.complexity_analyzer import ComplexityAnalyzer
from eeg_analysis.analyzers.estimators import BinningEstimator
from eeg_analysis.eeg_utils import preprocess_eeg, preprocess_eeg_by_bands, EEGDataCache, generate_bipartitions

mne.set_log_level("WARNING")

# Fixed bin count for these tests
N_BINS = 50


def run_mib_analysis(raw, n_channels: int, epoch_length: float, n_bins: int = N_BINS, band: str = None):
    """
    Run MIB analysis with specific parameters.

    Parameters:
    -----------
    band : str, optional
        If specified, filter to this frequency band (e.g., 'alpha', 'theta')
    """
    params = {
        **ANALYSIS_PARAMS,
        'n_bins': n_bins,
        'n_channels': n_channels,
        'epoch_length': epoch_length,
        'verbose': False,
        'use_all_channels': True,  # Get all channels, we'll select later
    }

    estimator = BinningEstimator(**params)
    analyzer = ComplexityAnalyzer(estimator=estimator, **params)

    # Preprocess and get epochs
    try:
        if band:
            # Use band-filtered preprocessing
            band_data, all_channels = preprocess_eeg_by_bands(raw, bands=[band], **params)
            if band not in band_data or band_data[band] is None:
                return None
            epochs_array = band_data[band]

            # Select channels if we have more than requested
            if epochs_array.shape[1] > n_channels:
                # Use first n_channels (they should be the standard ones)
                epochs_array = epochs_array[:, :n_channels, :]
            elif epochs_array.shape[1] < n_channels:
                print(f"    Only {epochs_array.shape[1]} channels available")
                return None
        else:
            # Broadband
            epochs_data, selected_channels = preprocess_eeg(raw, **params)
            if epochs_data is None or len(epochs_data) == 0:
                return None
            if isinstance(epochs_data, list):
                epochs_array = np.stack(epochs_data)
            else:
                epochs_array = epochs_data
    except Exception as e:
        print(f"    Error: {e}")
        return None

    if epochs_array is None or len(epochs_array) == 0:
        return None

    n_epochs = epochs_array.shape[0]
    if n_epochs < 3:
        return None

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
        'n_channels': n_channels,
        'epoch_length': epoch_length,
        'n_epochs': len(metric_values),
        'metric_values': metric_values,
        'mean_mib': mean_mib,
        'std_mib': std_mib,
        'cv_mib': cv_mib,
        'band': band or 'broadband',
    }


def test_channel_count(test_files: dict, channel_counts: list, epoch_length: float = 5.0, band: str = None):
    """
    Test how MIB changes with different numbers of channels.
    """
    band_str = band.upper() if band else "BROADBAND"
    print(f"\n{'='*70}")
    print(f"TESTING EFFECT OF NUMBER OF CHANNELS ({band_str})")
    print(f"Epoch length: {epoch_length}s, Bins: {N_BINS}")
    print(f"Channel counts: {channel_counts}")
    print(f"{'='*70}\n")

    cache = EEGDataCache(max_cache_size=3)
    all_results = {}

    for cond_name, file_path in test_files.items():
        print(f"\n--- {cond_name} ---")
        raw = cache.get_raw_data(file_path, verbose=False)

        results = []
        for n_ch in channel_counts:
            print(f"  n_channels = {n_ch}...", end=" ", flush=True)
            start = time.time()

            result = run_mib_analysis(raw, n_channels=n_ch, epoch_length=epoch_length, band=band)

            if result:
                results.append(result)
                print(f"MIB = {result['mean_mib']:.4f} +/- {result['std_mib']:.4f} "
                      f"(CV={result['cv_mib']:.3f}, n_epochs={result['n_epochs']}) [{time.time()-start:.1f}s]")
            else:
                print("Failed")

        all_results[cond_name] = results

    return all_results


def test_epoch_length(test_files: dict, epoch_lengths: list, n_channels: int = 8):
    """
    Test how MIB changes with different epoch lengths.
    """
    print(f"\n{'='*70}")
    print(f"TESTING EFFECT OF EPOCH LENGTH")
    print(f"Channels: {n_channels}, Bins: {N_BINS}")
    print(f"Epoch lengths: {epoch_lengths}")
    print(f"{'='*70}\n")

    cache = EEGDataCache(max_cache_size=3)
    all_results = {}

    for cond_name, file_path in test_files.items():
        print(f"\n--- {cond_name} ---")
        raw = cache.get_raw_data(file_path, verbose=False)

        results = []
        for ep_len in epoch_lengths:
            print(f"  epoch_length = {ep_len}s...", end=" ", flush=True)
            start = time.time()

            result = run_mib_analysis(raw, n_channels=n_channels, epoch_length=ep_len)

            if result:
                results.append(result)
                print(f"MIB = {result['mean_mib']:.4f} +/- {result['std_mib']:.4f} "
                      f"(CV={result['cv_mib']:.3f}, n_epochs={result['n_epochs']}) [{time.time()-start:.1f}s]")
            else:
                print("Failed")

        all_results[cond_name] = results

    return all_results


def plot_channel_count_results(all_results: dict, save_path: str = None):
    """Plot effect of channel count on MIB."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    colors = {'Awake EO': 'blue', 'Awake EC': 'green', 'Sedation': 'red'}
    markers = {'Awake EO': 'o', 'Awake EC': 's', 'Sedation': '^'}

    # Plot 1: Mean MIB vs channel count
    ax1 = axes[0, 0]
    for cond_name, results in all_results.items():
        if not results:
            continue
        ch_counts = [r['n_channels'] for r in results]
        means = [r['mean_mib'] for r in results]
        stds = [r['std_mib'] for r in results]
        color = colors.get(cond_name, 'gray')
        marker = markers.get(cond_name, 'o')

        ax1.errorbar(ch_counts, means, yerr=stds, marker=marker, capsize=3,
                     linewidth=2, markersize=8, label=cond_name, color=color, alpha=0.8)

    ax1.set_xlabel('Number of Channels', fontsize=12)
    ax1.set_ylabel('Mean MIB (bits)', fontsize=12)
    ax1.set_title('Mean MIB vs Number of Channels', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: CV vs channel count
    ax2 = axes[0, 1]
    for cond_name, results in all_results.items():
        if not results:
            continue
        ch_counts = [r['n_channels'] for r in results]
        cvs = [r['cv_mib'] for r in results]
        color = colors.get(cond_name, 'gray')
        marker = markers.get(cond_name, 'o')

        ax2.plot(ch_counts, cvs, marker=marker, linewidth=2, markersize=8,
                label=cond_name, color=color, alpha=0.8)

    ax2.set_xlabel('Number of Channels', fontsize=12)
    ax2.set_ylabel('Coefficient of Variation', fontsize=12)
    ax2.set_title('MIB Stability (CV) vs Number of Channels', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: MIB per channel (normalized)
    ax3 = axes[1, 0]
    for cond_name, results in all_results.items():
        if not results:
            continue
        ch_counts = [r['n_channels'] for r in results]
        means = [r['mean_mib'] for r in results]
        mib_per_ch = [m / n for m, n in zip(means, ch_counts)]
        color = colors.get(cond_name, 'gray')
        marker = markers.get(cond_name, 'o')

        ax3.plot(ch_counts, mib_per_ch, marker=marker, linewidth=2, markersize=8,
                label=cond_name, color=color, alpha=0.8)

    ax3.set_xlabel('Number of Channels', fontsize=12)
    ax3.set_ylabel('MIB per Channel (bits)', fontsize=12)
    ax3.set_title('MIB per Channel vs Number of Channels', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Relative difference (Sedation - Awake) vs channel count
    ax4 = axes[1, 1]
    if 'Awake EO' in all_results and 'Sedation' in all_results:
        awake_eo = all_results['Awake EO']
        sedation = all_results['Sedation']

        if awake_eo and sedation:
            ch_counts = [r['n_channels'] for r in awake_eo]
            awake_means = {r['n_channels']: r['mean_mib'] for r in awake_eo}
            sed_means = {r['n_channels']: r['mean_mib'] for r in sedation}

            diff_eo = []
            for n_ch in ch_counts:
                if n_ch in awake_means and n_ch in sed_means:
                    diff_eo.append(sed_means[n_ch] - awake_means[n_ch])
                else:
                    diff_eo.append(np.nan)

            ax4.plot(ch_counts, diff_eo, marker='o', linewidth=2, markersize=8,
                    label='Sedation - Awake EO', color='purple', alpha=0.8)

    if 'Awake EC' in all_results and 'Sedation' in all_results:
        awake_ec = all_results['Awake EC']
        sedation = all_results['Sedation']

        if awake_ec and sedation:
            ch_counts = [r['n_channels'] for r in awake_ec]
            awake_means = {r['n_channels']: r['mean_mib'] for r in awake_ec}
            sed_means = {r['n_channels']: r['mean_mib'] for r in sedation}

            diff_ec = []
            for n_ch in ch_counts:
                if n_ch in awake_means and n_ch in sed_means:
                    diff_ec.append(sed_means[n_ch] - awake_means[n_ch])
                else:
                    diff_ec.append(np.nan)

            ax4.plot(ch_counts, diff_ec, marker='s', linewidth=2, markersize=8,
                    label='Sedation - Awake EC', color='orange', alpha=0.8)

    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Number of Channels', fontsize=12)
    ax4.set_ylabel('MIB Difference (bits)', fontsize=12)
    ax4.set_title('Condition Difference vs Number of Channels', fontsize=14)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def plot_epoch_length_results(all_results: dict, save_path: str = None):
    """Plot effect of epoch length on MIB."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    colors = {'Awake EO': 'blue', 'Awake EC': 'green', 'Sedation': 'red'}
    markers = {'Awake EO': 'o', 'Awake EC': 's', 'Sedation': '^'}

    # Plot 1: Mean MIB vs epoch length
    ax1 = axes[0, 0]
    for cond_name, results in all_results.items():
        if not results:
            continue
        ep_lens = [r['epoch_length'] for r in results]
        means = [r['mean_mib'] for r in results]
        stds = [r['std_mib'] for r in results]
        color = colors.get(cond_name, 'gray')
        marker = markers.get(cond_name, 'o')

        ax1.errorbar(ep_lens, means, yerr=stds, marker=marker, capsize=3,
                     linewidth=2, markersize=8, label=cond_name, color=color, alpha=0.8)

    ax1.set_xlabel('Epoch Length (seconds)', fontsize=12)
    ax1.set_ylabel('Mean MIB (bits)', fontsize=12)
    ax1.set_title('Mean MIB vs Epoch Length', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: CV vs epoch length
    ax2 = axes[0, 1]
    for cond_name, results in all_results.items():
        if not results:
            continue
        ep_lens = [r['epoch_length'] for r in results]
        cvs = [r['cv_mib'] for r in results]
        color = colors.get(cond_name, 'gray')
        marker = markers.get(cond_name, 'o')

        ax2.plot(ep_lens, cvs, marker=marker, linewidth=2, markersize=8,
                label=cond_name, color=color, alpha=0.8)

    ax2.set_xlabel('Epoch Length (seconds)', fontsize=12)
    ax2.set_ylabel('Coefficient of Variation', fontsize=12)
    ax2.set_title('MIB Stability (CV) vs Epoch Length', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Number of epochs vs epoch length
    ax3 = axes[1, 0]
    for cond_name, results in all_results.items():
        if not results:
            continue
        ep_lens = [r['epoch_length'] for r in results]
        n_epochs = [r['n_epochs'] for r in results]
        color = colors.get(cond_name, 'gray')
        marker = markers.get(cond_name, 'o')

        ax3.plot(ep_lens, n_epochs, marker=marker, linewidth=2, markersize=8,
                label=cond_name, color=color, alpha=0.8)

    ax3.set_xlabel('Epoch Length (seconds)', fontsize=12)
    ax3.set_ylabel('Number of Epochs', fontsize=12)
    ax3.set_title('Available Epochs vs Epoch Length', fontsize=14)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Relative difference vs epoch length
    ax4 = axes[1, 1]
    if 'Awake EO' in all_results and 'Sedation' in all_results:
        awake_eo = all_results['Awake EO']
        sedation = all_results['Sedation']

        if awake_eo and sedation:
            ep_lens = [r['epoch_length'] for r in awake_eo]
            awake_means = {r['epoch_length']: r['mean_mib'] for r in awake_eo}
            sed_means = {r['epoch_length']: r['mean_mib'] for r in sedation}

            diff_eo = []
            for ep in ep_lens:
                if ep in awake_means and ep in sed_means:
                    diff_eo.append(sed_means[ep] - awake_means[ep])
                else:
                    diff_eo.append(np.nan)

            ax4.plot(ep_lens, diff_eo, marker='o', linewidth=2, markersize=8,
                    label='Sedation - Awake EO', color='purple', alpha=0.8)

    if 'Awake EC' in all_results and 'Sedation' in all_results:
        awake_ec = all_results['Awake EC']
        sedation = all_results['Sedation']

        if awake_ec and sedation:
            ep_lens = [r['epoch_length'] for r in awake_ec]
            awake_means = {r['epoch_length']: r['mean_mib'] for r in awake_ec}
            sed_means = {r['epoch_length']: r['mean_mib'] for r in sedation}

            diff_ec = []
            for ep in ep_lens:
                if ep in awake_means and ep in sed_means:
                    diff_ec.append(sed_means[ep] - awake_means[ep])
                else:
                    diff_ec.append(np.nan)

            ax4.plot(ep_lens, diff_ec, marker='s', linewidth=2, markersize=8,
                    label='Sedation - Awake EC', color='orange', alpha=0.8)

    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Epoch Length (seconds)', fontsize=12)
    ax4.set_ylabel('MIB Difference (bits)', fontsize=12)
    ax4.set_title('Condition Difference vs Epoch Length', fontsize=14)
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def print_summary(all_results: dict, param_name: str):
    """Print summary table."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: Effect of {param_name}")
    print(f"{'='*80}")

    for cond_name, results in all_results.items():
        print(f"\n{cond_name}:")
        if not results:
            print("  No results")
            continue

        if param_name == "Channel Count":
            print(f"  {'n_ch':>6} | {'Mean MIB':>10} | {'Std':>8} | {'CV':>6} | {'n_epochs':>8}")
            print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")
            for r in results:
                print(f"  {r['n_channels']:>6} | {r['mean_mib']:>10.4f} | {r['std_mib']:>8.4f} | "
                      f"{r['cv_mib']:>6.3f} | {r['n_epochs']:>8}")
        else:
            print(f"  {'epoch_len':>9} | {'Mean MIB':>10} | {'Std':>8} | {'CV':>6} | {'n_epochs':>8}")
            print(f"  {'-'*9}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}-+-{'-'*8}")
            for r in results:
                print(f"  {r['epoch_length']:>9.1f} | {r['mean_mib']:>10.4f} | {r['std_mib']:>8.4f} | "
                      f"{r['cv_mib']:>6.3f} | {r['n_epochs']:>8}")


if __name__ == "__main__":
    # Files to compare
    test_files = {
        'Awake EO': "/home/rk/Desktop/projects/EEG/ds005620/sub-1067/eeg/sub-1067_task-awake_acq-EO_eeg.vhdr",
        'Awake EC': "/home/rk/Desktop/projects/EEG/ds005620/sub-1067/eeg/sub-1067_task-awake_acq-EC_eeg.vhdr",
        'Sedation': "/home/rk/Desktop/projects/EEG/ds005620/sub-1067/eeg/sub-1067_task-sed_acq-rest_run-1_eeg.vhdr",
    }

    output_dir = Path("/home/rk/Desktop/projects/EEG/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Number of channels - ALPHA BAND (10s epochs)
    print("\n" + "#"*70)
    print("# TEST: NUMBER OF CHANNELS - ALPHA BAND")
    print("#"*70)

    channel_counts = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
    channel_results = test_channel_count(test_files, channel_counts, epoch_length=10.0, band='alpha')
    print_summary(channel_results, "Channel Count (Alpha)")
    plot_channel_count_results(channel_results, save_path=str(output_dir / "channel_count_effect_alpha.png"))
