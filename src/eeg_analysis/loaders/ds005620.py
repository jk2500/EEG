#!/usr/bin/env python3
"""
DS005620 Dataset Loader (BrainVision/.vhdr format)
==================================================

Propofol sedation study: awake (EO/EC) vs sedation conditions.
BIDS structure: ds005620/sub-XXXX/eeg/*.vhdr

Key functions:
- load_ds005620_epochs(): Load .vhdr -> fixed-length epochs
- preprocess_ds005620_epochs(): Channel select, filter, normalize
- preprocess_ds005620_epochs_by_bands(): Per-band preprocessing
- create_ds005620_subject_file_map(): Discover all subject files
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import mne
import numpy as np

from ..config import ANALYSIS_PARAMS, BAND_CONFIGS, SPECTRAL_BANDS

if TYPE_CHECKING:
    from mne import Epochs

# Common 10-20 channels available in the DS005620 dataset
DS005620_COMMON_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',
    'C3', 'C4', 'Cz', 'T7', 'T8',
    'P3', 'P4', 'Pz', 'P7', 'P8',
    'O1', 'O2', 'Oz',
    'FC1', 'FC2', 'FC5', 'FC6',
    'CP1', 'CP2', 'CP5', 'CP6',
]

# Condition mapping patterns to descriptive names
CONDITION_PATTERNS = {
    'awake_eyes_closed': '*_task-awake_acq-EC_*.vhdr',
    'awake_eyes_open': '*_task-awake_acq-EO_*.vhdr',
    'sedation_1': '*_task-sed_acq-rest_*.vhdr',
    'sedation_2': '*_task-sed2_acq-rest_*.vhdr',
}


def create_ds005620_subject_file_map(
    dataset_dir: str,
    verbose: bool = True
) -> Dict[str, Dict[str, str]]:
    """
    Create a file mapping for the DS005620 dataset.

    Parameters
    ----------
    dataset_dir : str
        Path to the DS005620 dataset directory.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    Dict[str, Dict[str, str]]
        Nested dictionary: {subject_id: {condition_name: filepath}}
    """
    import glob

    subject_paths = {}
    subject_dirs = sorted(d for d in os.listdir(dataset_dir) if d.startswith('sub-'))

    for subject_dir in subject_dirs:
        eeg_path = os.path.join(dataset_dir, subject_dir, 'eeg')
        if not os.path.exists(eeg_path):
            continue

        found_files = {}
        for cond, pattern in CONDITION_PATTERNS.items():
            matches = sorted(glob.glob(os.path.join(eeg_path, pattern)))
            if matches:
                found_files[cond] = matches[0]

        if len(found_files) >= 2:  # Require at least two conditions
            subject_paths[subject_dir] = found_files

    if verbose:
        print(f"Found {len(subject_paths)} subjects with sufficient data for analysis.")
        conditions = set()
        for conds in subject_paths.values():
            conditions.update(conds.keys())
        print(f"Conditions available: {sorted(conditions)}")

    return subject_paths


def load_ds005620_epochs(
    vhdr_file: str,
    epoch_length: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[Epochs, Dict[str, Any]]:
    """
    Load raw EEG data from a BrainVision .vhdr file and create fixed-length epochs.

    Parameters
    ----------
    vhdr_file : str
        Path to the .vhdr file.
    epoch_length : float, optional
        Length of epochs in seconds. Defaults to ANALYSIS_PARAMS['epoch_length'].
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    Tuple[mne.Epochs, Dict[str, Any]]
        MNE Epochs object and metadata dictionary.
    """
    if epoch_length is None:
        epoch_length = float(ANALYSIS_PARAMS.get('epoch_length', 5.0))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw = mne.io.read_raw_brainvision(vhdr_file, preload=True, verbose=False)

    # Create fixed-length events
    sfreq = raw.info['sfreq']
    samples_per_epoch = int(epoch_length * sfreq)
    n_samples = raw.n_times
    n_epochs = n_samples // samples_per_epoch

    # Create events array: (sample, 0, event_id)
    events = np.array([
        [i * samples_per_epoch, 0, 1]
        for i in range(n_epochs)
    ])

    # Create epochs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        epochs = mne.Epochs(
            raw,
            events,
            event_id={'epoch': 1},
            tmin=0,
            tmax=epoch_length - 1/sfreq,
            baseline=None,
            preload=True,
            verbose=False,
        )

    meta = {
        'sfreq': epochs.info['sfreq'],
        'n_channels': len(epochs.ch_names),
        'n_epochs': len(epochs),
        'epoch_duration': epoch_length,
        'channel_names': epochs.ch_names,
        'original_n_samples': n_samples,
    }

    if verbose:
        print(f"Loaded: {os.path.basename(vhdr_file)}")
        print(f"  Epochs: {meta['n_epochs']}, Channels: {meta['n_channels']}")
        print(f"  Duration: {meta['epoch_duration']:.2f}s, Sfreq: {meta['sfreq']} Hz")

    return epochs, meta


def preprocess_ds005620_epochs(
    epochs: mne.Epochs,
    **kwargs: Any
) -> Tuple[np.ndarray, List[str]]:
    """
    Preprocess epoched EEG data from the DS005620 dataset.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object.
    **kwargs : dict
        Override parameters from ANALYSIS_PARAMS:
        - target_sfreq: Target sampling frequency (default: 500.0)
        - n_channels: Number of channels to select (default: 8)
        - channel_selection: 'named', 'first', or 'random' (default: 'named')
        - channels_list: List of channel names to select (for 'named' mode)
        - reference_channel: Reference channel name (default: 'A2')
        - exclude_channels: Channels to exclude (default: ['VEOG', 'HEOG', 'EMG'])

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Preprocessed epochs data (n_epochs, n_channels, n_samples) and channel names.
    """
    params = {**ANALYSIS_PARAMS, **kwargs}
    epochs_copy = epochs.copy()

    # Drop excluded channels
    exclude = params.get('exclude_channels', ['VEOG', 'HEOG', 'EMG'])
    to_drop = [ch for ch in exclude if ch in epochs_copy.ch_names]
    if to_drop:
        epochs_copy.drop_channels(to_drop)

    # Resample if needed
    target_sfreq = params.get('target_sfreq', 500.0)
    if epochs_copy.info['sfreq'] != target_sfreq:
        epochs_copy.resample(target_sfreq, verbose=False)

    # Apply reference
    ref_channel = params.get('reference_channel', 'A2')
    if ref_channel and ref_channel in epochs_copy.ch_names:
        epochs_copy.set_eeg_reference([ref_channel], verbose=False)
        epochs_copy.drop_channels([ref_channel])
    else:
        epochs_copy.set_eeg_reference('average', projection=True, verbose=False)
        epochs_copy.apply_proj(verbose=False)

    # Channel selection
    eeg_channels = mne.pick_types(epochs_copy.info, eeg=True)
    eeg_channel_names = [epochs_copy.ch_names[i] for i in eeg_channels]

    if not eeg_channel_names:
        raise ValueError("No EEG channels available for analysis.")

    # Check if use_all_channels is set
    use_all_channels = params.get('use_all_channels', False)
    if use_all_channels:
        epochs_copy.pick(eeg_channel_names)
    else:
        n_channels = params.get('n_channels', 8)
        if n_channels > len(eeg_channel_names):
            raise ValueError(
                f"Requested {n_channels} EEG channels but only {len(eeg_channel_names)} available."
            )

        selection_mode = params.get('channel_selection', 'named')
        if selection_mode == 'named' and 'channels_list' in params:
            desired = params.get('channels_list', DS005620_COMMON_CHANNELS)
            selected = [ch for ch in desired if ch in eeg_channel_names]
            if len(selected) >= n_channels:
                epochs_copy.pick(selected[:n_channels])
            else:
                epochs_copy.pick(eeg_channel_names[:n_channels])
        elif selection_mode == 'random':
            rng = np.random.default_rng(params.get('partition_seed', 42))
            indices = rng.choice(len(eeg_channel_names), n_channels, replace=False)
            epochs_copy.pick([eeg_channel_names[i] for i in indices])
        else:
            epochs_copy.pick(eeg_channel_names[:n_channels])

    # Filter (broadband: 1-100 Hz)
    l_freq = params.get('l_freq', 1)
    h_freq = params.get('h_freq', 100)
    epochs_copy.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)

    # Get data and normalize
    epochs_data = epochs_copy.get_data()
    for i, epoch in enumerate(epochs_data):
        mean = np.mean(epoch, axis=1, keepdims=True)
        std = np.std(epoch, axis=1, keepdims=True)
        std[std == 0] = 1
        epochs_data[i] = (epoch - mean) / std

    return epochs_data, list(epochs_copy.ch_names)


def preprocess_ds005620_epochs_by_bands(
    epochs: mne.Epochs,
    bands: Optional[List[str]] = None,
    **kwargs: Any
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Preprocess epoched EEG data by spectral bands.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object.
    bands : Optional[List[str]]
        List of band names to process. If None, uses all bands from SPECTRAL_BANDS.
    **kwargs : dict
        Override parameters from ANALYSIS_PARAMS.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], List[str]]
        Dictionary mapping band names to epoch data arrays, and channel names.
    """
    params = {**ANALYSIS_PARAMS, **kwargs}
    use_all_channels = params.get('use_all_channels', False)
    band_data: Dict[str, np.ndarray] = {}
    ch_names: List[str] = []

    requested_bands = [b.lower() for b in bands] if bands else list(SPECTRAL_BANDS.keys())
    missing = [b for b in requested_bands if b not in SPECTRAL_BANDS]
    if missing:
        raise ValueError(f"Requested bands not defined in SPECTRAL_BANDS: {missing}")

    for band_name in requested_bands:
        l_freq, h_freq = SPECTRAL_BANDS[band_name]
        epochs_copy = epochs.copy()

        # Drop excluded channels
        exclude = params.get('exclude_channels', ['VEOG', 'HEOG', 'EMG'])
        to_drop = [ch for ch in exclude if ch in epochs_copy.ch_names]
        if to_drop:
            epochs_copy.drop_channels(to_drop)

        # Resample if needed
        target_sfreq = params.get('target_sfreq', 500.0)
        if epochs_copy.info['sfreq'] != target_sfreq:
            epochs_copy.resample(target_sfreq, verbose=False)

        # Apply reference
        ref_channel = params.get('reference_channel', 'A2')
        if ref_channel and ref_channel in epochs_copy.ch_names:
            epochs_copy.set_eeg_reference([ref_channel], verbose=False)
            epochs_copy.drop_channels([ref_channel])
        else:
            epochs_copy.set_eeg_reference('average', projection=True, verbose=False)
            epochs_copy.apply_proj(verbose=False)

        # Channel selection
        eeg_channels = mne.pick_types(epochs_copy.info, eeg=True)
        eeg_channel_names = [epochs_copy.ch_names[i] for i in eeg_channels]

        if not eeg_channel_names:
            raise ValueError("No EEG channels available for analysis.")

        if use_all_channels:
            epochs_copy.pick(eeg_channel_names)
        else:
            n_channels = params.get('n_channels', 8)
            if n_channels > len(eeg_channel_names):
                raise ValueError(
                    f"Requested {n_channels} EEG channels but only {len(eeg_channel_names)} available."
                )

            selection_mode = params.get('channel_selection', 'named')
            if selection_mode == 'named' and 'channels_list' in params:
                desired = params.get('channels_list', DS005620_COMMON_CHANNELS)
                selected = [ch for ch in desired if ch in eeg_channel_names]
                if len(selected) >= n_channels:
                    epochs_copy.pick(selected[:n_channels])
                else:
                    epochs_copy.pick(eeg_channel_names[:n_channels])
            elif selection_mode == 'random':
                rng = np.random.default_rng(params.get('partition_seed', 42))
                indices = rng.choice(len(eeg_channel_names), n_channels, replace=False)
                epochs_copy.pick([eeg_channel_names[i] for i in indices])
            else:
                epochs_copy.pick(eeg_channel_names[:n_channels])

        # Band-pass filter
        epochs_copy.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)

        # Subsample based on band configuration
        subsample_source = BAND_CONFIGS.get(band_name, {}).get(
            'subsample_factor',
            params.get('subsample_factor_spectral', 2)
        )
        subsample_factor = float(subsample_source if subsample_source is not None else 1.0)
        if subsample_factor > 1:
            epochs_copy.resample(epochs_copy.info['sfreq'] / subsample_factor, verbose=False)

        # Get data and normalize
        epochs_data = epochs_copy.get_data()
        for i, epoch in enumerate(epochs_data):
            mean = np.mean(epoch, axis=1, keepdims=True)
            std = np.std(epoch, axis=1, keepdims=True)
            std[std == 0] = 1
            epochs_data[i] = (epoch - mean) / std

        band_data[band_name] = epochs_data

        if not ch_names:
            ch_names = list(epochs_copy.ch_names)

    if not ch_names:
        raise ValueError("No bands were processed; check requested band list and data.")

    return band_data, ch_names


def get_ds005620_epochs_data(
    vhdr_file: str,
    mode: str = 'broadband',
    bands: Optional[List[str]] = None,
    epoch_length: Optional[float] = None,
    verbose: bool = False,
    **kwargs: Any
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    High-level function to load and preprocess DS005620 data.

    Parameters
    ----------
    vhdr_file : str
        Path to the .vhdr file.
    mode : str
        'broadband' or 'spectral'.
    bands : Optional[List[str]]
        Spectral bands to process (for spectral mode).
    epoch_length : float, optional
        Epoch length in seconds.
    verbose : bool
        Whether to print progress messages.
    **kwargs : dict
        Additional preprocessing parameters.

    Returns
    -------
    Tuple[Any, List[str], Dict[str, Any]]
        - For broadband: (epochs_data array, channel_names, metadata)
        - For spectral: (band_data dict, channel_names, metadata)
    """
    epochs, meta = load_ds005620_epochs(vhdr_file, epoch_length=epoch_length, verbose=verbose)

    if mode == 'broadband':
        epochs_data, ch_names = preprocess_ds005620_epochs(epochs, **kwargs)
        return epochs_data, ch_names, meta
    else:  # spectral
        band_data, ch_names = preprocess_ds005620_epochs_by_bands(epochs, bands=bands, **kwargs)
        return band_data, ch_names, meta
