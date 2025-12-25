"""
BrainVision Data Loader
=======================

Utilities for loading and organizing BrainVision format EEG data (BIDS structure).
"""

import os
import glob

from ..utils.helpers import log_print


def extract_condition(file_path):
    """
    Extract experimental condition from a BIDS-compliant file path.

    Parameters
    ----------
    file_path : str
        Path to EEG file.

    Returns
    -------
    str
        Condition name (e.g., 'awake_eyes_open', 'sedation_1').
    """
    filename = os.path.basename(file_path)
    if 'task-awake' in filename:
        return 'awake_eyes_open' if 'acq-EO' in filename else 'awake_eyes_closed'
    elif 'task-sed2' in filename:
        return 'sedation_2'
    elif 'task-sed' in filename:
        return 'sedation_1'
    return 'unknown'


def create_subject_file_map(dataset_dir, verbose=True):
    """
    Create an optimized file mapping for batch analysis.

    Scans a BIDS-formatted dataset directory and creates a mapping of
    subject IDs to their available condition files.

    Parameters
    ----------
    dataset_dir : str
        Path to the dataset root directory.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    Dict[str, Dict[str, str]]
        Nested dictionary: {subject_id: {condition_name: filepath}}
    """
    condition_patterns = {
        'awake_eyes_closed': '*_task-awake_acq-EC_*.vhdr',
        'awake_eyes_open': '*_task-awake_acq-EO_*.vhdr',
        'sedation_1': '*_task-sed_acq-rest_*.vhdr',
        'sedation_2': '*_task-sed2_acq-rest_*.vhdr',
    }

    subject_paths = {}
    subject_dirs = sorted(d for d in os.listdir(dataset_dir) if d.startswith('sub-'))

    for subject_dir in subject_dirs:
        eeg_path = os.path.join(dataset_dir, subject_dir, 'eeg')
        if not os.path.exists(eeg_path):
            continue

        found_files = {}
        for cond, pattern in condition_patterns.items():
            matches = sorted(glob.glob(os.path.join(eeg_path, pattern)))
            if matches:
                found_files[cond] = matches[0]

        if len(found_files) >= 2:  # Require at least two conditions
            subject_paths[subject_dir] = found_files

    log_print(f"Found {len(subject_paths)} subjects with sufficient data for analysis.", verbose)
    return subject_paths
