#!/usr/bin/env python3
"""
EEG Analysis Utilities
======================

This file provides common utility functions for EEG analysis, including
preprocessing, bipartition generation, plotting, and data handling. It is
designed to be used by the various analysis modules in the project.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import mne
import os
import pandas as pd

# Local imports
from .config import ANALYSIS_PARAMS, SPECTRAL_BANDS, BAND_CONFIGS

def log_print(message, verbose=True):
    """Prints a message if verbose is True."""
    if verbose:
        print(message)

def _extract_condition(file_path):
    """Extract experimental condition from a BIDS-compliant file path."""
    filename = os.path.basename(file_path)
    if 'task-awake' in filename:
        return 'awake_eyes_open' if 'acq-EO' in filename else 'awake_eyes_closed'
    elif 'task-sed' in filename:
        return 'sedation'
    return 'unknown'

def generate_bipartitions(n_channels, max_partitions=None, verbose=True, random_state=None):
    """
    Generate all possible non-trivial bipartitions of channels deterministically.
    """
    all_partitions = []
    for subset_size in range(1, n_channels // 2 + 1):
        for partition in combinations(range(n_channels), subset_size):
            if n_channels % 2 == 0 and subset_size == n_channels // 2:
                if partition[0] == 0:
                    all_partitions.append(partition)
            else:
                all_partitions.append(partition)
    
    total_partitions = len(all_partitions)
    if max_partitions and max_partitions < total_partitions:
        log_print(f"Limiting to {max_partitions} partitions sampled from {total_partitions}.", verbose)
        rng = np.random.default_rng(random_state)
        indices = rng.choice(total_partitions, max_partitions, replace=False)
        return [all_partitions[i] for i in indices]
    return all_partitions

def preprocess_eeg(raw, **kwargs):
    """
    Preprocess EEG data using a standardized MNE-based pipeline.
    """
    params = {**ANALYSIS_PARAMS, **kwargs}
    raw_copy = raw.copy()
    
    if raw_copy.info['sfreq'] != params['target_sfreq']:
        raw_copy.resample(params['target_sfreq'], verbose=False)
    
    # Reference to A2 if available, otherwise average reference
    ref_ch = params.get('reference_channel', None)
    if ref_ch and ref_ch in raw_copy.ch_names:
        try:
            raw_copy.set_eeg_reference(ref_channels=[ref_ch], verbose=False)
        except Exception:
            raw_copy.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)
    else:
        raw_copy.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)

    # Deterministic channel selection: prefer named list
    eeg_channels = mne.pick_types(raw_copy.info, eeg=True)
    selection_mode = params.get('channel_selection', 'named')
    if selection_mode == 'named' and 'channels_list' in params:
        desired = [ch for ch in params['channels_list'] if ch in raw_copy.ch_names]
        if len(desired) >= params['n_channels']:
            raw_copy.pick(desired[:params['n_channels']])
        else:
            # Fallback to first-N if not all desired present
            raw_copy.pick(raw_copy.ch_names[:params['n_channels']])
    else:
        if len(eeg_channels) > params['n_channels']:
            if selection_mode == 'random':
                indices = np.random.choice(eeg_channels, params['n_channels'], replace=False)
                raw_copy.pick([raw_copy.ch_names[i] for i in indices])
            else:
                raw_copy.pick(raw_copy.ch_names[:params['n_channels']])

    raw_copy.filter(l_freq=1, h_freq=40, fir_design='firwin', verbose=False)
    
    if params.get('subsample_factor_broadband', 1) > 1:
        raw_copy.resample(raw_copy.info['sfreq'] / params['subsample_factor_broadband'], verbose=False)
    
    epochs = mne.make_fixed_length_epochs(raw_copy, duration=params['epoch_length'], preload=True, verbose=False)
    epochs_data = epochs.get_data()
    
    for i, epoch in enumerate(epochs_data):
        mean, std = np.mean(epoch, axis=1, keepdims=True), np.std(epoch, axis=1, keepdims=True)
        std[std == 0] = 1
        epochs_data[i] = (epoch - mean) / std
                
    return epochs_data, raw_copy.ch_names

def preprocess_eeg_by_bands(raw, **kwargs):
    """
    Preprocess EEG data by spectral bands.
    """
    params = {**ANALYSIS_PARAMS, **kwargs}
    band_data = {}
    
    for band_name, (l_freq, h_freq) in SPECTRAL_BANDS.items():
        raw_copy = raw.copy()
        if raw_copy.info['sfreq'] != params['target_sfreq']:
            raw_copy.resample(params['target_sfreq'], verbose=False)
            
        # Reference to A2 if available, otherwise average reference
        ref_ch = params.get('reference_channel', None)
        if ref_ch and ref_ch in raw_copy.ch_names:
            try:
                raw_copy.set_eeg_reference(ref_channels=[ref_ch], verbose=False)
            except Exception:
                raw_copy.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)
        else:
            raw_copy.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)

        # Deterministic channel selection: prefer named list
        eeg_channels = mne.pick_types(raw_copy.info, eeg=True)
        selection_mode = params.get('channel_selection', 'named')
        if selection_mode == 'named' and 'channels_list' in params:
            desired = [ch for ch in params['channels_list'] if ch in raw_copy.ch_names]
            if len(desired) >= params['n_channels']:
                raw_copy.pick(desired[:params['n_channels']])
            else:
                raw_copy.pick(raw_copy.ch_names[:params['n_channels']])
        else:
            if len(eeg_channels) > params['n_channels']:
                if selection_mode == 'random':
                    indices = np.random.choice(eeg_channels, params['n_channels'], replace=False)
                    raw_copy.pick([raw_copy.ch_names[i] for i in indices])
                else:
                    raw_copy.pick(raw_copy.ch_names[:params['n_channels']])
        raw_copy.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
        
        subsample_factor = BAND_CONFIGS.get(band_name, {}).get('subsample_factor', params.get('subsample_factor_spectral', 2))
        if subsample_factor > 1:
            raw_copy.resample(raw_copy.info['sfreq'] / subsample_factor, verbose=False)
            
        epochs = mne.make_fixed_length_epochs(raw_copy, duration=params['epoch_length'], preload=True, verbose=False)
        epochs_data = epochs.get_data()
        
        for i, epoch in enumerate(epochs_data):
            mean, std = np.mean(epoch, axis=1, keepdims=True), np.std(epoch, axis=1, keepdims=True)
            std[std == 0] = 1
            epochs_data[i] = (epoch - mean) / std
            
        band_data[band_name] = epochs_data
        
    return band_data, raw_copy.ch_names

def plot_results(results_list, method_name, save_path=None, verbose=True):
    """
    Create a standardized visualization for analysis results.
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_print(f"Plot saved to: {save_path}", verbose)
    plt.show()

def plot_spectral_complexity_results(results, method_name, save_path=None):
    """
    Create visualizations for spectral band complexity analysis.
    """
    conditions = list(results.keys())
    if not results or not results[conditions[0]]: return
    
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
        plt.savefig(save_path, dpi=300)
    plt.show()

def print_spectral_summary(results, verbose=True):
    """Print a summary of spectral complexity analysis results."""
    if not verbose or not results: return
    log_print("\n--- Spectral Analysis Summary ---", True)
    df = pd.DataFrame({
        cond: {band: res['mean_metric'] for band, res in data.items()}
        for cond, data in results.items()
    }).T
    log_print(df.to_string(float_format="%.4f"), True)
    log_print("---------------------------------\n", True)

def create_subject_file_map(dataset_dir, verbose=True):
    """
    Create an optimized file mapping for batch analysis.
    """
    condition_patterns = {
        'awake_eyes_closed': '*_task-awake_acq-EC_*.vhdr',
        'awake_eyes_open': '*_task-awake_acq-EO_*.vhdr',
        'sedation_1': '*_task-sed_acq-rest_*.vhdr',
        'sedation_2': '*_task-sed2_acq-rest_*.vhdr',
    }
    
    subject_paths = {}
    subject_dirs = [d for d in os.listdir(dataset_dir) if d.startswith('sub-')]
    
    for subject_dir in subject_dirs:
        eeg_path = os.path.join(dataset_dir, subject_dir, 'eeg')
        if not os.path.exists(eeg_path): continue
        
        found_files = {}
        for cond, pattern in condition_patterns.items():
            import glob
            matches = glob.glob(os.path.join(eeg_path, pattern))
            if matches:
                found_files[cond] = matches[0]
        
        if len(found_files) >= 2: # Require at least two conditions
            subject_paths[subject_dir] = found_files
            
    log_print(f"Found {len(subject_paths)} subjects with sufficient data for analysis.", verbose)
    return subject_paths

class EEGDataCache:
    """
    Efficient caching system for EEG data to avoid redundant file loading.
    """
    def __init__(self, max_cache_size=3):
        self.cache = {}
        self.access_order = []
        self.max_cache_size = max_cache_size
    
    def get_raw_data(self, file_path, verbose=True):
        if file_path in self.cache:
            self.access_order.remove(file_path)
            self.access_order.append(file_path)
            return self.cache[file_path]
        
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        
        if len(self.cache) >= self.max_cache_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
            
        self.cache[file_path] = raw
        self.access_order.append(file_path)
        return raw
        
    def clear(self):
        self.cache.clear()
