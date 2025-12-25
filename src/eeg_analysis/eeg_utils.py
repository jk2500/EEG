"""
EEG Analysis Utilities (Backward Compatibility)
================================================

This module provides backward compatibility for existing code that imports
from eeg_utils. All functionality has been moved to dedicated submodules.

For new code, please import directly from the submodules:
- eeg_analysis.core: preprocessing, partitions, cache
- eeg_analysis.loaders: data loaders
- eeg_analysis.visualization: plotting functions
- eeg_analysis.utils: helper functions
"""

import warnings

warnings.warn(
    "Importing from eeg_utils is deprecated. "
    "Please import from eeg_analysis.core, eeg_analysis.loaders, etc. instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility
from .core import (
    preprocess_eeg,
    preprocess_eeg_by_bands,
    generate_bipartitions,
    EEGDataCache,
)

from .loaders import (
    create_subject_file_map,
    extract_condition as _extract_condition,
)

from .visualization import (
    plot_results,
    plot_spectral_complexity_results,
    print_spectral_summary,
)

from .utils import (
    log_print,
    check_gaussianity,
)

from .config import ANALYSIS_PARAMS, SPECTRAL_BANDS, BAND_CONFIGS

__all__ = [
    'preprocess_eeg',
    'preprocess_eeg_by_bands',
    'generate_bipartitions',
    'EEGDataCache',
    'create_subject_file_map',
    '_extract_condition',
    'plot_results',
    'plot_spectral_complexity_results',
    'print_spectral_summary',
    'log_print',
    'check_gaussianity',
    'ANALYSIS_PARAMS',
    'SPECTRAL_BANDS',
    'BAND_CONFIGS',
]
