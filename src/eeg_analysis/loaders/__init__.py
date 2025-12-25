"""
Data Loaders for different EEG formats.
"""

from .brainvision import create_subject_file_map, extract_condition
from .sedation import (
    SedationDataCache,
    create_sedation_subject_file_map,
    load_sedation_epochs,
    preprocess_sedation_epochs,
    preprocess_sedation_epochs_by_bands,
    get_sedation_epochs_data,
    load_datainfo,
    SEDATION_CONDITIONS,
    SEDATION_COMMON_CHANNELS,
)

__all__ = [
    # BrainVision loaders
    'create_subject_file_map',
    'extract_condition',
    # Sedation loaders
    'SedationDataCache',
    'create_sedation_subject_file_map',
    'load_sedation_epochs',
    'preprocess_sedation_epochs',
    'preprocess_sedation_epochs_by_bands',
    'get_sedation_epochs_data',
    'load_datainfo',
    'SEDATION_CONDITIONS',
    'SEDATION_COMMON_CHANNELS',
]
