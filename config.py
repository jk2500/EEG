#!/usr/bin/env python3
"""
Central Configuration File for EEG Neural Complexity Analysis
=============================================================

This file centralizes all default parameters and settings for the project,
making it easier to manage and configure different analysis methods.
"""

import os

# ============================================================================
# 🧠 GENERAL ANALYSIS PARAMETERS
# ============================================================================

ANALYSIS_PARAMS = {
    'n_channels': 8,              # Number of EEG channels to analyze
    'epoch_length': 5.0,          # Length of each epoch in seconds
    'max_partitions': 127,        # Max bipartitions (2^(n-1)-1 for n=8)
    'subsample_factor_broadband': 10, # Subsampling for broadband analysis
    'subsample_factor_spectral': 2,   # Subsampling for spectral analysis
    'n_jobs': -1,                 # Number of parallel jobs (-1 for all cores)
    'verbose': True,              # Enable verbose output
    # Deterministic channel selection: fixed set used by default
    'channel_selection': 'named', # Options: 'named', 'first', 'random'
    'channels_list': ['Fp1', 'Fp2', 'F3', 'F4', 'P3', 'P4', 'T7', 'T8'],
    # Reference all EEG channels to the right earlobe (A2) by default
    'reference_channel': 'A2',
    'target_sfreq': 500.0,        # Target sampling frequency for consistency
    'max_subset_size': 8,         # Maximum subset size for IIT complexes calculation
    'top_complexes': 30           # Number of top complexes to return per epoch
}

# ============================================================================
# 🔬 METHOD-SPECIFIC PARAMETERS
# ============================================================================

KSG_PARAMS = {
    'k': 6,                       # Number of nearest neighbors for KSG
}

BINNING_PARAMS = {
    'n_bins': 10,                 # Number of bins for histogram discretization
}

GAUSSIAN_PARAMS = {
    'alpha': 0.05,                # Significance level for Gaussianity test
}

# ============================================================================
# 📁 FILE PATHS & DIRECTORIES
# ============================================================================

# Default dataset directory
DATASET_DIR = 'ds005620'

# Default file paths for quick analysis
DEFAULT_FILE_PATHS = {
    'awake': os.path.join(DATASET_DIR, 'sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr'),
    'sedation': os.path.join(DATASET_DIR, 'sub-1010/eeg/sub-1010_task-sed2_acq-rest_run-1_eeg.vhdr')
}

# Default output directory
DEFAULT_OUTPUT_DIR = 'results'

# ============================================================================
# 📊 PLOTTING & OUTPUT PARAMETERS
# ============================================================================

# Default output file names (can be customized)
OUTPUT_FILES = {
    'broadband_plot': "neural_complexity_{method}_broadband_results.png",
    'spectral_plot': "neural_complexity_{method}_spectral_results.png",
    'spectral_csv': "neural_complexity_{method}_spectral_summary.csv",
    'mib_broadband_plot': "mib_{method}_broadband_results.png",
    'mib_spectral_plot': "mib_{method}_spectral_results.png",
    'mib_spectral_csv': "mib_{method}_spectral_summary.csv"
}

# ============================================================================
# ⚙️ ADVANCED & NUMERICAL PARAMETERS
# ============================================================================

# Numerical constants for stability
NUMERICAL_PARAMS = {
    'epsilon_ksg': 1e-15,         # Small value for KSG to avoid log(0)
    'epsilon_binning': 1e-10      # Small value for handling constant channels in binning
}

# UI/Display parameters
UI_PARAMS = {
    'header_width': 60            # Width of header separators in console output
}

# ============================================================================
# 🎵 SPECTRAL ANALYSIS PARAMETERS
# ============================================================================

SPECTRAL_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100),
    'broadband': (1, 100)  # Full spectrum for comparison
}

# Band-specific configurations (e.g., different subsample factors)
# This allows for optimizing analysis for different frequency ranges.
BAND_CONFIGS = {
    'gamma': {'subsample_factor': 2}, # Less subsampling for high-frequency content
    'beta': {'subsample_factor': 4},
    'alpha': {'subsample_factor': 8},
    'theta': {'subsample_factor': 10},
    'delta': {'subsample_factor': 10}
}
