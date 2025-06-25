#!/usr/bin/env python3
"""
EEG Analysis Utilities
======================

This file provides common utility functions for EEG analysis, including
preprocessing, bipartition generation for complexity analysis, and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import mne
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import time

def log_print(message, verbose=True):
    """Prints a message if verbose is True."""
    if verbose:
        print(message)

def _extract_condition(file_path):
    """Extract experimental condition from file path."""
    if 'awake' in file_path:
        if 'EO' in file_path:
            return 'awake_eyes_open'
        elif 'EC' in file_path:
            return 'awake_eyes_closed'
        else:
            return 'awake'
    elif 'sed' in file_path:
        return 'sedation'
    else:
        return 'unknown'

def generate_bipartitions(n_channels, max_partitions=None, verbose=True):
    """
    Generate all possible bipartitions of channels.
    
    Parameters:
    -----------
    n_channels : int
        Number of channels
    max_partitions : int, optional
        Maximum number of partitions to generate
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    partitions : list of tuples
        List of bipartitions
    """
    all_partitions = []
    
    # Generate all possible non-trivial bipartitions
    # We only need to go up to n_channels//2 to avoid duplicates
    # since partition (A, B) is the same as partition (B, A)
    for subset_size in range(1, n_channels // 2 + 1):
        for partition in combinations(range(n_channels), subset_size):
            all_partitions.append(partition)
    
    if max_partitions and len(all_partitions) > max_partitions:
        log_print(f"Limiting to {max_partitions} random partitions from {len(all_partitions)}.", verbose)
        indices = np.random.choice(len(all_partitions), max_partitions, replace=False)
        all_partitions = [all_partitions[i] for i in indices]
        
    log_print(f"Generated {len(all_partitions)} bipartitions.", verbose)
    return all_partitions


def preprocess_eeg(raw, freq_min=1, freq_max=40, epoch_length=5.0, 
                   n_channels=None, channel_selection='variance', 
                   subsample_factor=1, verbose=True):
    """
    Preprocess EEG data using a standardized MNE-based pipeline.
    """
    log_print("Preprocessing EEG data...", verbose)
    raw_copy = raw.copy()
    
    # Get EEG channels
    eeg_channel_indices = mne.pick_types(raw_copy.info, eeg=True)
    eeg_channel_names = [raw_copy.ch_names[i] for i in eeg_channel_indices]
    
    if len(eeg_channel_names) == 0:
        log_print("No EEG channels found, using all channels.", verbose)
        eeg_channel_names = raw_copy.ch_names

    # Select a subset of channels if specified
    if n_channels and n_channels < len(eeg_channel_names):
        if channel_selection == 'variance':
            data = raw_copy.get_data(picks=eeg_channel_names)
            variances = np.var(data, axis=1)
            top_indices = np.argsort(variances)[-n_channels:]
            selected_channels = [eeg_channel_names[i] for i in top_indices]
        elif channel_selection == 'random':
            indices = np.random.choice(len(eeg_channel_names), n_channels, replace=False)
            selected_channels = [eeg_channel_names[i] for i in indices]
        else: # Default to first n channels
            selected_channels = eeg_channel_names[:n_channels]
    else:
        selected_channels = eeg_channel_names
    
    log_print(f"Selected {len(selected_channels)} channels.", verbose)
    raw_copy.pick(selected_channels)
    
    # Band-pass filter
    log_print(f"Filtering {freq_min}-{freq_max} Hz...", verbose)
    raw_copy.filter(freq_min, freq_max, fir_design='firwin', verbose=False)
    
    # Subsample for computational efficiency
    if subsample_factor > 1:
        log_print(f"Subsampling by factor {subsample_factor}", verbose)
        raw_copy.resample(raw_copy.info['sfreq'] / subsample_factor, verbose=False)
    
    # Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(raw_copy, duration=epoch_length, preload=True, verbose=False)
    epochs_data = epochs.get_data()
    
    log_print(f"Created {len(epochs_data)} epochs of {epoch_length}s each.", verbose)
    log_print(f"Preprocessed data shape: {epochs_data.shape}", verbose)
    
    return epochs_data, selected_channels


def plot_results(results_list, method_name, save_path=None, verbose=True):
    """
    Create a standardized visualization for neural complexity results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f'Neural Complexity Analysis - {method_name} Method', fontsize=18, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # --- Plot 1: Mean complexity comparison ---
    ax1 = axes[0]
    conditions = [r['condition'] for r in results_list]
    means = [r['mean_complexity'] for r in results_list]
    stds = [r['std_complexity'] for r in results_list]
    
    bars = ax1.bar(conditions, means, yerr=stds, capsize=5,
                  color=colors[:len(conditions)], alpha=0.8)
    ax1.set_title('Mean Neural Complexity by Condition', fontsize=14)
    ax1.set_ylabel('Neural Complexity (bits)', fontsize=12)
    ax1.tick_params(axis='x', rotation=15, labelsize=11)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # --- Plot 2: Complexity distributions ---
    ax2 = axes[1]
    all_values = [r['complexity_values'] for r in results_list]
    
    box = ax2.boxplot(all_values, labels=conditions, patch_artist=True,
                      medianprops=dict(color="black"))
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax2.set_title('Distribution of Neural Complexity Values', fontsize=14)
    ax2.set_ylabel('Neural Complexity (bits)', fontsize=12)
    ax2.tick_params(axis='x', rotation=15, labelsize=11)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_print(f"Plot saved to: {save_path}", verbose)
    
    plt.show()


def preprocess_eeg_by_bands(raw, epoch_length=5.0, n_channels=None, 
                           channel_selection='variance', subsample_factor=1, 
                           bands=None, verbose=True):
    """
    Preprocess EEG data by spectral bands for neural complexity analysis.
    
    Parameters:
    -----------
    raw : mne.Raw
        Raw EEG data
    epoch_length : float
        Length of epochs in seconds
    n_channels : int, optional
        Number of channels to select
    channel_selection : str
        Method for channel selection ('variance', 'random', 'first')
    subsample_factor : int
        Factor for downsampling
    bands : dict, optional
        Dictionary of frequency bands. Default includes all major bands.
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    band_data : dict
        Dictionary with band names as keys and preprocessed data as values
    selected_channels : list
        List of selected channel names
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100),
            'broadband': (1, 100)  # Full spectrum for comparison
        }
    
    log_print("Preprocessing EEG data by spectral bands...", verbose)
    log_print(f"Frequency bands: {bands}", verbose)
    
    raw_copy = raw.copy()
    
    # Get EEG channels
    eeg_channel_indices = mne.pick_types(raw_copy.info, eeg=True)
    eeg_channel_names = [raw_copy.ch_names[i] for i in eeg_channel_indices]
    
    if len(eeg_channel_names) == 0:
        log_print("No EEG channels found, using all channels.", verbose)
        eeg_channel_names = raw_copy.ch_names

    # Select a subset of channels if specified
    if n_channels and n_channels < len(eeg_channel_names):
        if channel_selection == 'variance':
            data = raw_copy.get_data(picks=eeg_channel_names)
            variances = np.var(data, axis=1)
            top_indices = np.argsort(variances)[-n_channels:]
            selected_channels = [eeg_channel_names[i] for i in top_indices]
        elif channel_selection == 'random':
            indices = np.random.choice(len(eeg_channel_names), n_channels, replace=False)
            selected_channels = [eeg_channel_names[i] for i in indices]
        else: # Default to first n channels
            selected_channels = eeg_channel_names[:n_channels]
    else:
        selected_channels = eeg_channel_names
    
    log_print(f"Selected {len(selected_channels)} channels.", verbose)
    raw_copy.pick(selected_channels)
    
    # Subsample for computational efficiency
    if subsample_factor > 1:
        log_print(f"Subsampling by factor {subsample_factor}", verbose)
        raw_copy.resample(raw_copy.info['sfreq'] / subsample_factor, verbose=False)
    
    # Process each frequency band
    band_data = {}
    for band_name, (freq_min, freq_max) in bands.items():
        log_print(f"Processing {band_name} band ({freq_min}-{freq_max} Hz)...", verbose)
        
        # Create a copy for this band
        band_raw = raw_copy.copy()
        
        # Apply band-pass filter
        try:
            band_raw.filter(freq_min, freq_max, fir_design='firwin', verbose=False)
            
            # Create fixed-length epochs
            epochs = mne.make_fixed_length_epochs(band_raw, duration=epoch_length, 
                                                preload=True, verbose=False)
            epochs_data = epochs.get_data()
            
            band_data[band_name] = epochs_data
            log_print(f"{band_name}: {epochs_data.shape} epochs created.", verbose)
            
        except Exception as e:
            log_print(f"Warning: Could not process {band_name} band: {e}", verbose)
            continue
    
    return band_data, selected_channels


def analyze_spectral_complexity(file_paths, complexity_func, n_channels=8, 
                               subsample_factor=10, verbose=True, **kwargs):
    """
    Analyze neural complexity across different spectral bands.
    
    Parameters:
    -----------
    file_paths : list or dict
        Paths to EEG files or dict with condition names as keys
    complexity_func : callable
        Function to calculate complexity (e.g., calculate_cn_ksg)
    n_channels : int
        Number of channels to analyze
    subsample_factor : int
        Downsampling factor
    verbose : bool
        Print progress messages
    **kwargs : dict
        Additional arguments for complexity function
        
    Returns:
    --------
    results : dict
        Results organized by condition and frequency band
    """
    if isinstance(file_paths, list):
        file_paths = {f"condition_{i}": path for i, path in enumerate(file_paths)}
    
    results = {}
    
    for condition, file_path in file_paths.items():
        log_print(f"\nAnalyzing {condition}: {os.path.basename(file_path)}", verbose)
        
        # Load data
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        condition_name = _extract_condition(file_path)
        
        # Preprocess by bands
        band_data, channels = preprocess_eeg_by_bands(
            raw, n_channels=n_channels, subsample_factor=subsample_factor, 
            verbose=verbose
        )
        
        results[condition_name] = {}
        
        # Analyze each frequency band
        for band_name, epochs_data in band_data.items():
            log_print(f"Calculating complexity for {band_name} band...", verbose)
            
            band_complexities = []
            for epoch_idx, epoch in enumerate(epochs_data):
                try:
                    complexity = complexity_func(epoch, verbose=False, **kwargs)
                    band_complexities.append(complexity)
                except Exception as e:
                    if verbose:
                        log_print(f"Error in {band_name} epoch {epoch_idx}: {e}", verbose)
                    continue
                    
            if band_complexities:
                results[condition_name][band_name] = {
                    'complexity_values': band_complexities,
                    'mean_complexity': np.mean(band_complexities),
                    'std_complexity': np.std(band_complexities),
                    'n_epochs': len(band_complexities)
                }
                log_print(f"{band_name}: {np.mean(band_complexities):.4f} ± {np.std(band_complexities):.4f} bits", verbose)
    
    return results 

def plot_spectral_complexity_results(results, method_name, save_path=None, verbose=True):
    """
    Create visualizations for spectral band complexity analysis.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_spectral_complexity function
    method_name : str
        Name of the complexity method used
    save_path : str, optional
        Path to save the plot
    verbose : bool
        Whether to print messages
    """
    # Extract data for plotting
    conditions = list(results.keys())
    bands = list(results[conditions[0]].keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f'Neural Complexity Across Frequency Bands - {method_name} Method', 
                 fontsize=18, fontweight='bold')
    
    # Colors for conditions and bands
    condition_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    band_colors = ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#9467bd']
    
    # --- Plot 1: Bar chart by frequency bands ---
    ax1 = axes[0, 0]
    x = np.arange(len(bands))
    width = 0.35
    
    for i, condition in enumerate(conditions):
        means = [results[condition][band]['mean_complexity'] for band in bands]
        stds = [results[condition][band]['std_complexity'] for band in bands]
        ax1.bar(x + i*width, means, width, yerr=stds, label=condition, 
                color=condition_colors[i], alpha=0.8, capsize=5)
    
    ax1.set_xlabel('Frequency Bands', fontsize=12)
    ax1.set_ylabel('Neural Complexity (bits)', fontsize=12)
    ax1.set_title('Mean Complexity by Frequency Band', fontsize=14)
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(bands, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # --- Plot 2: Line plot showing band progression ---
    ax2 = axes[0, 1]
    for i, condition in enumerate(conditions):
        means = [results[condition][band]['mean_complexity'] for band in bands]
        stds = [results[condition][band]['std_complexity'] for band in bands]
        ax2.errorbar(bands, means, yerr=stds, marker='o', linewidth=2, 
                    markersize=8, label=condition, color=condition_colors[i], capsize=5)
    
    ax2.set_xlabel('Frequency Bands', fontsize=12)
    ax2.set_ylabel('Neural Complexity (bits)', fontsize=12)
    ax2.set_title('Complexity Progression Across Bands', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.tick_params(axis='x', rotation=45)
    
    # --- Plot 3: Heatmap of complexity values ---
    ax3 = axes[1, 0]
    complexity_matrix = []
    for condition in conditions:
        condition_values = [results[condition][band]['mean_complexity'] for band in bands]
        complexity_matrix.append(condition_values)
    
    im = ax3.imshow(complexity_matrix, cmap='viridis', aspect='auto')
    ax3.set_xticks(range(len(bands)))
    ax3.set_xticklabels(bands, rotation=45)
    ax3.set_yticks(range(len(conditions)))
    ax3.set_yticklabels(conditions)
    ax3.set_title('Complexity Heatmap', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Neural Complexity (bits)', fontsize=12)
    
    # Add text annotations
    for i in range(len(conditions)):
        for j in range(len(bands)):
            text = ax3.text(j, i, f'{complexity_matrix[i][j]:.3f}',
                           ha="center", va="center", color="white", fontweight='bold')
    
    # --- Plot 4: Box plots for gamma band (most important for consciousness) ---
    ax4 = axes[1, 1]
    if 'gamma' in bands:
        gamma_data = []
        gamma_labels = []
        for condition in conditions:
            if 'gamma' in results[condition]:
                gamma_data.append(results[condition]['gamma']['complexity_values'])
                gamma_labels.append(f"{condition}\n(n={results[condition]['gamma']['n_epochs']})")
        
        if gamma_data:
            box = ax4.boxplot(gamma_data, labels=gamma_labels, patch_artist=True)
            for patch, color in zip(box['boxes'], condition_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            
            ax4.set_title('Gamma Band Complexity Distribution\n(Key for Consciousness)', fontsize=14)
            ax4.set_ylabel('Neural Complexity (bits)', fontsize=12)
            ax4.grid(axis='y', linestyle='--', alpha=0.6)
    else:
        ax4.text(0.5, 0.5, 'Gamma band data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Gamma Band Analysis', fontsize=14)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_print(f"Spectral complexity plot saved to: {save_path}", verbose)
    
    plt.show()


def print_spectral_summary(results, verbose=True):
    """
    Print a summary of spectral complexity analysis results.
    """
    if not verbose:
        return
        
    print("\n" + "="*80)
    print("SPECTRAL NEURAL COMPLEXITY ANALYSIS SUMMARY")
    print("="*80)
    
    conditions = list(results.keys())
    bands = list(results[conditions[0]].keys())
    
    # Print results table
    print(f"\n{'Band':<12} ", end="")
    for condition in conditions:
        print(f"{condition:<20}", end="")
    print()
    print("-" * (12 + 20 * len(conditions)))
    
    for band in bands:
        print(f"{band:<12} ", end="")
        for condition in conditions:
            if band in results[condition]:
                mean_val = results[condition][band]['mean_complexity']
                std_val = results[condition][band]['std_complexity']
                print(f"{mean_val:>8.4f}±{std_val:<8.4f} ", end="")
            else:
                print(f"{'N/A':<18} ", end="")
        print()
    
    # Highlight key findings
    print(f"\n{'='*50}")
    print("KEY FINDINGS:")
    print(f"{'='*50}")
    
    # Compare gamma band between conditions (most important for consciousness)
    if 'gamma' in bands and len(conditions) >= 2:
        gamma_values = {}
        for condition in conditions:
            if 'gamma' in results[condition]:
                gamma_values[condition] = results[condition]['gamma']['mean_complexity']
        
        if len(gamma_values) >= 2:
            sorted_gamma = sorted(gamma_values.items(), key=lambda x: x[1], reverse=True)
            highest = sorted_gamma[0]
            lowest = sorted_gamma[-1]
            
            print(f"🧠 GAMMA BAND (Consciousness Marker):")
            print(f"   Highest: {highest[0]} = {highest[1]:.4f} bits")
            print(f"   Lowest:  {lowest[0]} = {lowest[1]:.4f} bits")
            
            if highest[1] != 0:
                diff_pct = ((highest[1] - lowest[1]) / abs(lowest[1])) * 100
                print(f"   Difference: {diff_pct:+.1f}%")
    
    print(f"\n{'='*50}") 