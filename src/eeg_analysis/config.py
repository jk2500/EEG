#!/usr/bin/env python3
"""
Central Configuration - Edit defaults here or override via kwargs.

Key configs:
- ANALYSIS_PARAMS: n_channels=8, epoch_length=5s, target_sfreq=500Hz
- BINNING_PARAMS: n_bins=10 (increase to 50 for production)
- SPECTRAL_BANDS: delta/theta/alpha/beta/gamma/broadband freq ranges
"""

import os

# ============================================================================
# 🧠 GENERAL ANALYSIS PARAMETERS
# ============================================================================

ANALYSIS_PARAMS = {
    'n_channels': 8,              # Number of EEG channels to analyze
    'epoch_length': 5.0,          # Length of each epoch in seconds
    'max_partitions': 127,        # Max bipartitions (2^(n-1)-1 for n=8)
    'partition_seed': 42,         # Ensures reproducible partition sampling
    'raw_cache_size': 3,          # Number of raw files to keep in memory
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
    # Channels to exclude from analysis (non-EEG aux channels)
    'exclude_channels': ['VEOG', 'HEOG', 'EMG'],
}

# ============================================================================
# 🔬 METHOD-SPECIFIC PARAMETERS
# ============================================================================

BINNING_PARAMS = {
    'n_bins': 10,                 # Number of bins for histogram discretization
}

# ============================================================================
# 📁 FILE PATHS & DIRECTORIES
# ============================================================================

# Default dataset directory (BIDS format - BrainVision files)
DATASET_DIR = 'ds005620'
DS005620_DATASET_DIR = DATASET_DIR  # Alias for mib_analysis.py

# Common 10-20 channels for DS005620 (shared with ANALYSIS_PARAMS['channels_list'])
DS005620_COMMON_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'P3', 'P4', 'T7', 'T8']

# Secondary dataset directory (EEGLAB format - Sedation-RestingState)
SEDATION_DATASET_DIR = 'Sedation-RestingState'

# Default file paths for quick analysis
DEFAULT_FILE_PATHS = {
    'awake': os.path.join(DATASET_DIR, 'sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr'),
    'sedation': os.path.join(DATASET_DIR, 'sub-1010/eeg/sub-1010_task-sed2_acq-rest_run-1_eeg.vhdr')
}

# Sedation-RestingState condition mapping
SEDATION_CONDITIONS = {
    1: 'baseline',
    2: 'light_sedation',
    3: 'deep_sedation',
    4: 'recovery',
}

# Common 10-20 channels available in the Sedation-RestingState dataset
# (subset that overlaps with standard 10-20 system)
SEDATION_COMMON_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',
    'C3', 'C4', 'Cz', 'T3', 'T4', 'T5', 'T6',
    'P3', 'P4', 'Pz', 'O1', 'O2', 'Oz'
]

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
