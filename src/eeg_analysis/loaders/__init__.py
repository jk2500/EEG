"""
Data Loaders for different EEG formats.
"""

from .brainvision import create_subject_file_map, extract_condition
from .ds005620 import (
    create_ds005620_subject_file_map,
    load_ds005620_epochs,
    preprocess_ds005620_epochs,
    preprocess_ds005620_epochs_by_bands,
    get_ds005620_epochs_data,
    DS005620_COMMON_CHANNELS,
    CONDITION_PATTERNS as DS005620_CONDITION_PATTERNS,
)
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
    # DS005620 loaders
    'create_ds005620_subject_file_map',
    'load_ds005620_epochs',
    'preprocess_ds005620_epochs',
    'preprocess_ds005620_epochs_by_bands',
    'get_ds005620_epochs_data',
    'DS005620_COMMON_CHANNELS',
    'DS005620_CONDITION_PATTERNS',
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
