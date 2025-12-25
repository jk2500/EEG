"""
Plotting Functions
==================

Standardized visualization functions for EEG analysis results.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from ..utils.helpers import log_print


def plot_results(results_list, method_name, save_path=None, verbose=True):
    """
    Create a standardized visualization for analysis results.

    Parameters
    ----------
    results_list : List[Dict]
        List of result dictionaries with keys: 'condition', 'mean_metric',
        'std_metric', 'metric_values'.
    method_name : str
        Name of the analysis method for the title.
    save_path : str, optional
        Path to save the figure.
    verbose : bool
        Whether to print messages.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Analysis: {method_name}', fontsize=18, fontweight='bold')

    conditions = [r['condition'] for r in results_list]
    means = [r['mean_metric'] for r in results_list]
    stds = [r['std_metric'] for r in results_list]

    axes[0].bar(conditions, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    axes[0].set_title('Mean Result by Condition')
    axes[0].set_ylabel('Metric Value (bits)')

    axes[1].boxplot([r['metric_values'] for r in results_list], labels=conditions, patch_artist=True)
    axes[1].set_title('Distribution of Metric Values')

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_print(f"Plot saved to: {save_path}", verbose)
    plt.show()


def plot_spectral_complexity_results(results, method_name, save_path=None):
    """
    Create visualizations for spectral band complexity analysis.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict]]
        Nested dictionary: {condition: {band: {'mean_metric': ..., 'std_metric': ...}}}
    method_name : str
        Name of the analysis method for the title.
    save_path : str, optional
        Path to save the figure.
    """
    conditions = list(results.keys())
    if not results or not results[conditions[0]]:
        return

    bands = list(results[conditions[0]].keys())

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, condition in enumerate(conditions):
        means = [results[condition].get(band, {}).get('mean_metric', 0) for band in bands]
        stds = [results[condition].get(band, {}).get('std_metric', 0) for band in bands]
        ax.errorbar(bands, means, yerr=stds, marker='o', label=condition)

    ax.set_title(f'Spectral Analysis - {method_name}')
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Metric Value (bits)')
    ax.legend()
    ax.grid(True)

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()


def print_spectral_summary(results, verbose=True):
    """
    Print a summary of spectral complexity analysis results.

    Parameters
    ----------
    results : Dict[str, Dict[str, Dict]]
        Nested dictionary: {condition: {band: {'mean_metric': ...}}}
    verbose : bool
        Whether to print output.
    """
    if not verbose or not results:
        return
    log_print("\n--- Spectral Analysis Summary ---", True)
    df = pd.DataFrame({
        cond: {band: res['mean_metric'] for band, res in data.items()}
        for cond, data in results.items()
    }).T
    log_print(df.to_string(float_format="%.4f"), True)
    log_print("---------------------------------\n", True)
