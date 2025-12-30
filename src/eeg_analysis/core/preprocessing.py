"""
EEG Preprocessing Functions
===========================

Provides standardized preprocessing pipelines for EEG data using MNE.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mne
import numpy as np

from ..config import ANALYSIS_PARAMS, BAND_CONFIGS, SPECTRAL_BANDS

if TYPE_CHECKING:
    from mne.io import Raw


def preprocess_eeg(raw: Raw, **kwargs: Any) -> tuple[np.ndarray, list[str]]:
    """
    Preprocess EEG data using a standardized MNE-based pipeline.

    Pipeline: resample -> reference -> channel select -> filter -> epoch -> normalize
    """
    params = {**ANALYSIS_PARAMS, **kwargs}  # Merge defaults with overrides
    raw_copy = raw.copy()

    # Step 1: Drop non-EEG auxiliary channels (VEOG, HEOG, EMG)
    exclude = params.get('exclude_channels', [])
    if exclude:
        to_drop = [ch for ch in exclude if ch in raw_copy.ch_names]
        if len(to_drop) > 0:
            raw_copy.drop_channels(to_drop)

    # Step 2: Resample to target frequency (default 500 Hz)
    if raw_copy.info['sfreq'] != params['target_sfreq']:
        raw_copy.resample(params['target_sfreq'], verbose=False)

    # Step 3: Apply reference (A2 ear lobe preferred, fallback to average)
    ref_ch = params.get('reference_channel', None)
    if ref_ch and ref_ch in raw_copy.ch_names:
        try:
            raw_copy.set_eeg_reference(ref_channels=[ref_ch], verbose=False)
        except (ValueError, RuntimeError):
            # Fall back to average reference if specified channel fails
            raw_copy.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)
    else:
        raw_copy.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)

    # Step 4: Channel selection (named > first-N > random)
    eeg_channels = mne.pick_types(raw_copy.info, eeg=True)
    eeg_channel_names = [raw_copy.ch_names[i] for i in eeg_channels]
    if not eeg_channel_names:
        raise ValueError("No EEG channels available for analysis.")
    n_channels = params['n_channels']
    if n_channels > len(eeg_channel_names):
        raise ValueError(
            f"Requested {n_channels} EEG channels but only {len(eeg_channel_names)} available."
        )
    selection_mode = params.get('channel_selection', 'named')
    if selection_mode == 'named' and 'channels_list' in params:
        desired = [ch for ch in params['channels_list'] if ch in eeg_channel_names]
        if len(desired) >= n_channels:
            raw_copy.pick(desired[:n_channels])
        else:
            # Fallback to first-N EEG channels if not all desired present
            raw_copy.pick(eeg_channel_names[:n_channels])
    else:
        if selection_mode == 'random':
            indices = np.random.choice(eeg_channels, n_channels, replace=False)
            raw_copy.pick([raw_copy.ch_names[i] for i in indices])
        else:
            raw_copy.pick(eeg_channel_names[:n_channels])

    # Step 5: Bandpass filter (1-40 Hz for broadband)
    raw_copy.filter(l_freq=1, h_freq=40, fir_design='firwin', verbose=False)

    # Step 6: Optional subsampling to reduce computation
    if params.get('subsample_factor_broadband', 1) > 1:
        raw_copy.resample(raw_copy.info['sfreq'] / params['subsample_factor_broadband'], verbose=False)

    # Step 7: Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(raw_copy, duration=params['epoch_length'], preload=True, verbose=False)
    epochs_data = epochs.get_data()

    # Step 8: Z-score normalize each channel within each epoch
    for i, epoch in enumerate(epochs_data):
        mean, std = np.mean(epoch, axis=1, keepdims=True), np.std(epoch, axis=1, keepdims=True)
        std[std == 0] = 1  # Prevent division by zero for flat channels
        epochs_data[i] = (epoch - mean) / std

    return epochs_data, raw_copy.ch_names


def preprocess_eeg_by_bands(
    raw: Raw,
    bands: list[str] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Preprocess EEG by spectral bands (delta, theta, alpha, beta, gamma, broadband).

    Same pipeline as preprocess_eeg but with band-specific filtering and subsampling.
    Returns dict mapping band names to epoch arrays.
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
        raw_copy = raw.copy()
        # Exclude specified auxiliary channels before further processing
        exclude = params.get('exclude_channels', [])
        if exclude:
            to_drop = [ch for ch in exclude if ch in raw_copy.ch_names]
            if len(to_drop) > 0:
                raw_copy.drop_channels(to_drop)
        if raw_copy.info['sfreq'] != params['target_sfreq']:
            raw_copy.resample(params['target_sfreq'], verbose=False)

        # Reference to A2 if available, otherwise average reference
        ref_ch = params.get('reference_channel', None)
        if ref_ch and ref_ch in raw_copy.ch_names:
            try:
                raw_copy.set_eeg_reference(ref_channels=[ref_ch], verbose=False)
            except (ValueError, RuntimeError):
                # Fall back to average reference if specified channel fails
                raw_copy.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)
        else:
            raw_copy.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)

        # Deterministic channel selection: prefer named list
        eeg_channels = mne.pick_types(raw_copy.info, eeg=True)
        eeg_channel_names = [raw_copy.ch_names[i] for i in eeg_channels]
        if not eeg_channel_names:
            raise ValueError("No EEG channels available for analysis.")
        if use_all_channels:
            # Keep all EEG channels (minus excluded), no limiting by n_channels
            raw_copy.pick(eeg_channel_names)
        else:
            n_channels = params['n_channels']
            if n_channels > len(eeg_channel_names):
                raise ValueError(
                    f"Requested {n_channels} EEG channels but only {len(eeg_channel_names)} available."
                )
            selection_mode = params.get('channel_selection', 'named')
            if selection_mode == 'named' and 'channels_list' in params:
                desired = [ch for ch in params['channels_list'] if ch in eeg_channel_names]
                if len(desired) >= n_channels:
                    raw_copy.pick(desired[:n_channels])
                else:
                    raw_copy.pick(eeg_channel_names[:n_channels])
            else:
                if selection_mode == 'random':
                    indices = np.random.choice(eeg_channels, n_channels, replace=False)
                    raw_copy.pick([raw_copy.ch_names[i] for i in indices])
                else:
                    raw_copy.pick(eeg_channel_names[:n_channels])
        raw_copy.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)

        subsample_source = BAND_CONFIGS.get(band_name, {}).get('subsample_factor', params.get('subsample_factor_spectral', 2))
        subsample_factor = float(subsample_source if subsample_source is not None else 1.0)
        if subsample_factor > 1:
            raw_copy.resample(raw_copy.info['sfreq'] / subsample_factor, verbose=False)

        epochs = mne.make_fixed_length_epochs(raw_copy, duration=params['epoch_length'], preload=True, verbose=False)
        epochs_data = epochs.get_data()

        for i, epoch in enumerate(epochs_data):
            mean, std = np.mean(epoch, axis=1, keepdims=True), np.std(epoch, axis=1, keepdims=True)
            std[std == 0] = 1
            epochs_data[i] = (epoch - mean) / std

        band_data[band_name] = epochs_data

        # Track channel names from the first processed band for return.
        if not ch_names:
            ch_names = list(raw_copy.ch_names)

    if not ch_names:
        raise ValueError("No bands were processed; check requested band list and raw data.")

    return band_data, ch_names
