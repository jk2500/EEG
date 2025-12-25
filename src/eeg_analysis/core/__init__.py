"""
Core EEG processing functionality.
"""

from .preprocessing import preprocess_eeg, preprocess_eeg_by_bands
from .partitions import generate_bipartitions
from .cache import EEGDataCache

__all__ = [
    'preprocess_eeg',
    'preprocess_eeg_by_bands',
    'generate_bipartitions',
    'EEGDataCache',
]
