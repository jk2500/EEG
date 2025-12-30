#!/usr/bin/env python3
"""
Sedation-RestingState Dataset Loader (EEGLAB/.set format)
=========================================================

Propofol sedation study with 4 conditions per subject.
Conditions: baseline -> light_sedation -> deep_sedation -> recovery

Key functions:
- load_sedation_epochs(): Load .set -> MNE Epochs
- preprocess_sedation_epochs(): Channel select, filter, normalize
- preprocess_sedation_epochs_by_bands(): Per-band preprocessing
- create_sedation_subject_file_map(): Discover files via datainfo.mat
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mne
import numpy as np
import scipy.io as sio

from ..config import ANALYSIS_PARAMS, BAND_CONFIGS, SPECTRAL_BANDS

if TYPE_CHECKING:
    from mne import Epochs


# Condition mapping: numeric labels to descriptive names
SEDATION_CONDITIONS = {
    1: "baseline",
    2: "light_sedation",
    3: "deep_sedation",
    4: "recovery",
}

# Reverse mapping
CONDITION_TO_LABEL = {v: k for k, v in SEDATION_CONDITIONS.items()}

# Common 10-20 channels available in the Sedation-RestingState dataset
SEDATION_COMMON_CHANNELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz',
    'C3', 'C4', 'Cz', 'T3', 'T4', 'T5', 'T6',
    'P3', 'P4', 'Pz', 'O1', 'O2', 'Oz'
]


def load_datainfo(dataset_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load and parse the datainfo.mat file containing metadata for all recordings.
    
    Parameters
    ----------
    dataset_dir : str
        Path to the Sedation-RestingState directory.
        
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dictionary mapping filenames to their metadata:
        - condition: int (1-4)
        - condition_name: str
        - start_sample: int
        - end_sample: int
        - n_channels: int
    """
    mat_path = os.path.join(dataset_dir, 'datainfo.mat')
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"datainfo.mat not found in {dataset_dir}")
    
    data = sio.loadmat(mat_path)
    datainfo = data['datainfo']
    
    result = {}
    for row in datainfo:
        filename = row[0].flat[0]
        condition = int(row[1].flat[0])
        start_sample = int(row[2].flat[0])
        end_sample_raw = row[3].flat[0]
        
        # Handle NaN values in end_sample
        if isinstance(end_sample_raw, float) and np.isnan(end_sample_raw):
            end_sample = None
        else:
            end_sample = int(end_sample_raw)
            
        n_channels = int(row[4].flat[0])
        
        result[filename] = {
            'condition': condition,
            'condition_name': SEDATION_CONDITIONS.get(condition, f'unknown_{condition}'),
            'start_sample': start_sample,
            'end_sample': end_sample,
            'n_channels': n_channels,
        }
    
    return result


def create_sedation_subject_file_map(
    dataset_dir: str,
    verbose: bool = True
) -> Dict[str, Dict[str, str]]:
    """
    Create a file mapping for the Sedation-RestingState dataset.
    
    This function mimics the structure of create_subject_file_map from eeg_utils.py
    but adapted for the EEGLAB format files.
    
    Parameters
    ----------
    dataset_dir : str
        Path to the Sedation-RestingState directory.
    verbose : bool
        Whether to print progress messages.
        
    Returns
    -------
    Dict[str, Dict[str, str]]
        Nested dictionary: {subject_id: {condition_name: filepath}}
    """
    datainfo = load_datainfo(dataset_dir)
    
    subject_map = {}
    for filename, meta in datainfo.items():
        # Extract subject ID from filename (e.g., '25-2010' from '25-2010-anest 20100422 133.003')
        subject_id = filename.split('-anest')[0]
        if subject_id.endswith('-'):
            subject_id = subject_id[:-1]  # Handle typos like '02-2010-anest-'
        
        # Normalize subject_id format
        subject_id = f"sub-{subject_id.replace('-', '')}"
        
        condition_name = meta['condition_name']
        filepath = os.path.join(dataset_dir, f"{filename}.set")
        
        if not os.path.exists(filepath):
            if verbose:
                print(f"Warning: File not found: {filepath}")
            continue
        
        if subject_id not in subject_map:
            subject_map[subject_id] = {}
        
        # Only keep one file per condition per subject (first one found)
        if condition_name not in subject_map[subject_id]:
            subject_map[subject_id][condition_name] = filepath
    
    if verbose:
        print(f"Found {len(subject_map)} subjects with data.")
        conditions = set()
        for conds in subject_map.values():
            conditions.update(conds.keys())
        print(f"Conditions available: {sorted(conditions)}")
    
    return subject_map


def load_sedation_epochs(
    set_file: str,
    verbose: bool = False,
) -> tuple[Epochs, dict[str, Any]]:
    """
    Load epoched EEG data from an EEGLAB .set file.
    
    Parameters
    ----------
    set_file : str
        Path to the .set file.
    verbose : bool
        Whether to print progress messages.
        
    Returns
    -------
    Tuple[mne.Epochs, Dict[str, Any]]
        MNE Epochs object and metadata dictionary.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        epochs = mne.io.read_epochs_eeglab(set_file, verbose=False)
    
    meta = {
        'sfreq': epochs.info['sfreq'],
        'n_channels': len(epochs.ch_names),
        'n_epochs': len(epochs),
        'epoch_duration': epochs.times[-1] - epochs.times[0],
        'channel_names': epochs.ch_names,
    }
    
    if verbose:
        print(f"Loaded: {os.path.basename(set_file)}")
        print(f"  Epochs: {meta['n_epochs']}, Channels: {meta['n_channels']}")
        print(f"  Duration: {meta['epoch_duration']:.2f}s, Sfreq: {meta['sfreq']} Hz")
    
    return epochs, meta


def preprocess_sedation_epochs(
    epochs: mne.Epochs,
    **kwargs
) -> Tuple[np.ndarray, List[str]]:
    """
    Preprocess epoched EEG data from the Sedation-RestingState dataset.
    
    This function applies preprocessing steps similar to preprocess_eeg from
    eeg_utils.py but adapted for already-epoched data.
    
    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object loaded from EEGLAB file.
    **kwargs : dict
        Override parameters from ANALYSIS_PARAMS:
        - target_sfreq: Target sampling frequency (default: 250.0)
        - n_channels: Number of channels to select (default: 8)
        - channel_selection: 'named', 'first', or 'random' (default: 'named')
        - channels_list: List of channel names to select (for 'named' mode)
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Preprocessed epochs data (n_epochs, n_channels, n_samples) and channel names.
    """
    params = {**ANALYSIS_PARAMS, **kwargs}
    epochs_copy = epochs.copy()
    
    # Resample if needed
    target_sfreq = params.get('target_sfreq', 250.0)
    if epochs_copy.info['sfreq'] != target_sfreq:
        epochs_copy.resample(target_sfreq, verbose=False)
    
    # Apply average reference
    epochs_copy.set_eeg_reference('average', projection=True, verbose=False)
    epochs_copy.apply_proj(verbose=False)
    
    # Channel selection
    eeg_channels = mne.pick_types(epochs_copy.info, eeg=True)
    eeg_channel_names = [epochs_copy.ch_names[i] for i in eeg_channels]
    
    if not eeg_channel_names:
        raise ValueError("No EEG channels available for analysis.")
    
    n_channels = params.get('n_channels', 8)
    if n_channels > len(eeg_channel_names):
        raise ValueError(
            f"Requested {n_channels} EEG channels but only {len(eeg_channel_names)} available."
        )
    
    selection_mode = params.get('channel_selection', 'named')
    if selection_mode == 'named' and 'channels_list' in params:
        # Use SEDATION_COMMON_CHANNELS as fallback
        desired = params.get('channels_list', SEDATION_COMMON_CHANNELS)
        selected = [ch for ch in desired if ch in eeg_channel_names]
        if len(selected) >= n_channels:
            epochs_copy.pick(selected[:n_channels])
        else:
            # Fallback to first-N EEG channels
            epochs_copy.pick(eeg_channel_names[:n_channels])
    elif selection_mode == 'random':
        rng = np.random.default_rng(params.get('partition_seed', 42))
        indices = rng.choice(len(eeg_channel_names), n_channels, replace=False)
        epochs_copy.pick([eeg_channel_names[i] for i in indices])
    else:
        epochs_copy.pick(eeg_channel_names[:n_channels])
    
    # Filter (optional - data may already be filtered)
    l_freq = params.get('l_freq', 1)
    h_freq = params.get('h_freq', 40)
    epochs_copy.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    
    # Get data and normalize
    epochs_data = epochs_copy.get_data()
    for i, epoch in enumerate(epochs_data):
        mean = np.mean(epoch, axis=1, keepdims=True)
        std = np.std(epoch, axis=1, keepdims=True)
        std[std == 0] = 1
        epochs_data[i] = (epoch - mean) / std
    
    return epochs_data, list(epochs_copy.ch_names)


def preprocess_sedation_epochs_by_bands(
    epochs: mne.Epochs,
    bands: Optional[List[str]] = None,
    **kwargs
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Preprocess epoched EEG data by spectral bands.
    
    Similar to preprocess_eeg_by_bands from eeg_utils.py but for pre-epoched data.
    
    Parameters
    ----------
    epochs : mne.Epochs
        MNE Epochs object loaded from EEGLAB file.
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
    band_data = {}
    ch_names = []
    
    requested_bands = [b.lower() for b in bands] if bands else list(SPECTRAL_BANDS.keys())
    missing = [b for b in requested_bands if b not in SPECTRAL_BANDS]
    if missing:
        raise ValueError(f"Requested bands not defined in SPECTRAL_BANDS: {missing}")
    
    for band_name in requested_bands:
        l_freq, h_freq = SPECTRAL_BANDS[band_name]
        epochs_copy = epochs.copy()
        
        # Resample if needed
        target_sfreq = params.get('target_sfreq', 250.0)
        if epochs_copy.info['sfreq'] != target_sfreq:
            epochs_copy.resample(target_sfreq, verbose=False)
        
        # Apply average reference
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
                desired = params.get('channels_list', SEDATION_COMMON_CHANNELS)
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


def get_sedation_epochs_data(
    set_file: str,
    mode: str = 'broadband',
    bands: Optional[List[str]] = None,
    verbose: bool = False,
    **kwargs
) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    High-level function to load and preprocess Sedation-RestingState data.
    
    This provides a unified interface similar to the BrainVision loading workflow.
    
    Parameters
    ----------
    set_file : str
        Path to the .set file.
    mode : str
        'broadband' or 'spectral'.
    bands : Optional[List[str]]
        Spectral bands to process (for spectral mode).
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
    epochs, meta = load_sedation_epochs(set_file, verbose=verbose)
    
    if mode == 'broadband':
        epochs_data, ch_names = preprocess_sedation_epochs(epochs, **kwargs)
        return epochs_data, ch_names, meta
    else:  # spectral
        band_data, ch_names = preprocess_sedation_epochs_by_bands(epochs, bands=bands, **kwargs)
        return band_data, ch_names, meta


class SedationDataCache:
    """
    Efficient caching system for Sedation-RestingState EEG data.
    
    Similar to EEGDataCache in eeg_utils.py but for EEGLAB format.
    """
    
    def __init__(self, max_cache_size: int = 3):
        self.cache: Dict[str, mne.Epochs] = {}
        self.access_order: List[str] = []
        self.max_cache_size = max_cache_size
    
    def get_epochs(self, file_path: str, verbose: bool = False) -> mne.Epochs:
        """Get epochs from cache or load from file."""
        if file_path in self.cache:
            self.access_order.remove(file_path)
            self.access_order.append(file_path)
            return self.cache[file_path]
        
        epochs, _ = load_sedation_epochs(file_path, verbose=verbose)
        
        if len(self.cache) >= self.max_cache_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[file_path] = epochs
        self.access_order.append(file_path)
        return epochs
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()


