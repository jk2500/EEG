"""
EEG Analysis Library
====================

A modular library for EEG neural complexity analysis using Mutual Information
Based (MIB) metrics.

Modules
-------
core
    Core preprocessing and partitioning functions.
analyzers
    Analysis engines (ComplexityAnalyzer) and MI estimators.
loaders
    Data loaders for different EEG formats (BrainVision, EEGLAB).
visualization
    Plotting and visualization functions.
utils
    Helper functions and utilities.
config
    Central configuration parameters.

Quick Start
-----------
>>> from eeg_analysis import ComplexityAnalyzer, BinningEstimator
>>> from eeg_analysis import preprocess_eeg, EEGDataCache
>>> from eeg_analysis.config import ANALYSIS_PARAMS
>>>
>>> # Load and preprocess data
>>> cache = EEGDataCache()
>>> raw = cache.get_raw_data('path/to/file.vhdr')
>>> epochs, channels = preprocess_eeg(raw)
>>>
>>> # Run MIB analysis
>>> estimator = BinningEstimator(n_bins=50)
>>> analyzer = ComplexityAnalyzer(estimator=estimator)
>>> results = analyzer.analyze(epochs)
"""

__version__ = '1.0.0'

# Core functionality
from .core import (
    preprocess_eeg,
    preprocess_eeg_by_bands,
    generate_bipartitions,
    EEGDataCache,
)

# Analyzers
from .analyzers import (
    ComplexityAnalyzer,
    BinningEstimator,
)

# Configuration
from .config import (
    ANALYSIS_PARAMS,
    BINNING_PARAMS,
    SPECTRAL_BANDS,
    BAND_CONFIGS,
    NUMERICAL_PARAMS,
)

# Loaders
from .loaders import (
    create_subject_file_map,
    extract_condition,
    SedationDataCache,
    create_sedation_subject_file_map,
    get_sedation_epochs_data,
)

# Visualization
from .visualization import (
    plot_results,
    plot_spectral_complexity_results,
    print_spectral_summary,
)

# Utils
from .utils import (
    log_print,
    check_gaussianity,
)

__all__ = [
    # Version
    '__version__',
    # Core
    'preprocess_eeg',
    'preprocess_eeg_by_bands',
    'generate_bipartitions',
    'EEGDataCache',
    # Analyzers
    'ComplexityAnalyzer',
    'BinningEstimator',
    # Config
    'ANALYSIS_PARAMS',
    'BINNING_PARAMS',
    'SPECTRAL_BANDS',
    'BAND_CONFIGS',
    'NUMERICAL_PARAMS',
    # Loaders
    'create_subject_file_map',
    'extract_condition',
    'SedationDataCache',
    'create_sedation_subject_file_map',
    'get_sedation_epochs_data',
    # Visualization
    'plot_results',
    'plot_spectral_complexity_results',
    'print_spectral_summary',
    # Utils
    'log_print',
    'check_gaussianity',
]
