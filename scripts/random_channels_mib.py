#!/usr/bin/env python3
"""
Single-Purpose Random-Channel MIB Runner
========================================

This script is the only public entrypoint for the project. It repeatedly samples
random subsets of EEG channels and computes the Minimum Information
Bipartition (MIB) on:
- Broadband data
- Spectral bands

It reports two stability views (mean/std):
- Across epochs for a fixed channel set
- Across channel draws (random subsets)

It supports sweeping over subjects and epoch lengths.

For every combination it reports mean and standard deviation of the per-epoch
MIB within each random channel draw, as well as across draws.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mne
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eeg_analysis.config import (  # type: ignore
    ANALYSIS_PARAMS,
    BINNING_PARAMS,
    DATASET_DIR,
    DEFAULT_FILE_PATHS,
    DEFAULT_OUTPUT_DIR,
    SPECTRAL_BANDS,
)
from eeg_analysis.analyzers.complexity_analyzer import ComplexityAnalyzer  # type: ignore
from eeg_analysis.analyzers.estimators import BinningEstimator  # type: ignore
from eeg_analysis.eeg_utils import (  # type: ignore
    create_subject_file_map,
    log_print,
    preprocess_eeg_by_bands,
)

mne.set_log_level("WARNING")


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------


def _get_binning_estimator():
    return BinningEstimator, BINNING_PARAMS


def _build_analyzer(
    n_channels: int,
    epoch_length: float,
    verbose: bool,
    n_jobs: Optional[int] = None,
) -> ComplexityAnalyzer:
    EstimatorClass, method_params = _get_binning_estimator()
    params: Dict[str, Any] = {
        **ANALYSIS_PARAMS,
        **method_params,
        "n_channels": int(n_channels),
        "epoch_length": float(epoch_length),
        "verbose": verbose,
    }
    if n_jobs is not None:
        params["n_jobs"] = int(n_jobs)
    estimator = EstimatorClass(**params)
    return ComplexityAnalyzer(estimator=estimator, **params)


def _select_channel_indices(
    all_channel_names: Sequence[str],
    n_channels: int,
    preferred: Optional[Sequence[str]],
) -> Tuple[np.ndarray, List[str]]:
    """Pick a deterministic subset of channels, honoring user preference when possible."""
    if n_channels > len(all_channel_names):
        raise ValueError(f"Requested {n_channels} channels but only {len(all_channel_names)} available.")

    selected: List[str] = []
    if preferred:
        seen = set()
        for ch in preferred:
            if ch in all_channel_names and ch not in seen:
                selected.append(ch)
                seen.add(ch)

    if len(selected) < n_channels:
        fallback = [ch for ch in all_channel_names if ch not in selected]
        needed = n_channels - len(selected)
        selected.extend(fallback[:needed])
    else:
        selected = selected[:n_channels]

    if len(selected) < n_channels:
        raise ValueError(
            f"Unable to select {n_channels} channels from the available list. "
            f"Available: {all_channel_names}"
        )

    indices = np.array([all_channel_names.index(ch) for ch in selected], dtype=int)
    return indices, selected


def compute_single_random_channel_mib(
    *,
    epochs_data: np.ndarray,
    channel_names: List[str],
    analyzer: ComplexityAnalyzer,
    n_channels: int,
    rng: np.random.Generator,
    repeat_index: int = 0,
    progress_label: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Compute MIB for a single random channel subset.

    This is the core single-sample calculation that can be run independently
    or called repeatedly in a loop for random sampling.

    Parameters
    ----------
    epochs_data : np.ndarray
        EEG epochs of shape (n_epochs, n_channels, n_samples).
    channel_names : List[str]
        Names of all available channels.
    analyzer : ComplexityAnalyzer
        Pre-configured analyzer instance.
    n_channels : int
        Number of channels to randomly select.
    rng : np.random.Generator
        Random number generator (allows external control of randomness).
    repeat_index : int
        Index of this sample (for tracking/logging purposes).
    progress_label : str
        Label for progress reporting.

    Returns
    -------
    Optional[Dict[str, Any]]
        Result dictionary with selected channels, indices, and MIB values,
        or None if no valid MIB values could be computed.
    """
    n_all_channels = len(channel_names)
    if n_channels > n_all_channels:
        raise ValueError(f"Requested {n_channels} channels but only {n_all_channels} available.")

    selected_indices = rng.choice(n_all_channels, size=n_channels, replace=False)
    selected_indices = np.sort(selected_indices)
    selected_channels = [channel_names[i] for i in selected_indices]
    subset_epochs = epochs_data[:, selected_indices, :]

    metric_values = analyzer._evaluate_epochs(  # pylint: disable=protected-access
        subset_epochs,
        progress_label=progress_label,
        verbose_override=False,
    )
    metric_values = [v for v in metric_values if v is not None]

    if not metric_values:
        return None

    mean_mib = float(np.mean(metric_values))
    std_mib = float(np.std(metric_values))

    return {
        "repeat_index": repeat_index,
        "selected_channels": selected_channels,
        "selected_indices": selected_indices.tolist(),
        "n_epochs": len(metric_values),
        "mean_mib": mean_mib,
        "std_mib": std_mib,
        "metric_values": [float(x) for x in metric_values],
    }


def _aggregate_repeat_results(
    repeats_payload: List[Dict[str, Any]],
    repeats_requested: int,
    n_channels: int,
    n_all_channels: int,
) -> Dict[str, Any]:
    """
    Aggregate results from multiple single-sample runs into summary statistics.
    """
    repeats_completed = len(repeats_payload)
    per_repeat_means = np.array([r["mean_mib"] for r in repeats_payload], dtype=np.float64)
    per_repeat_stds = np.array([r["std_mib"] for r in repeats_payload], dtype=np.float64)

    mean_of_means = float(np.mean(per_repeat_means)) if len(per_repeat_means) > 0 else 0.0
    std_of_means = float(np.std(per_repeat_means)) if len(per_repeat_means) > 0 else 0.0
    mean_of_stds = float(np.mean(per_repeat_stds)) if len(per_repeat_stds) > 0 else 0.0
    cv_of_means = float(std_of_means / abs(mean_of_means)) if abs(mean_of_means) > 1e-10 else float("inf")

    return {
        "repeats": int(repeats_completed),
        "repeats_requested": int(repeats_requested),
        "n_channels_selected": int(n_channels),
        "n_channels_total": int(n_all_channels),
        "overall_stats": {
            "mean_of_per_repeat_means": mean_of_means,
            "std_of_per_repeat_means": std_of_means,
            "cv_of_per_repeat_means": cv_of_means,
            "mean_of_per_repeat_stds": mean_of_stds,
            "min_per_repeat_mean": float(np.min(per_repeat_means)) if len(per_repeat_means) > 0 else 0.0,
            "max_per_repeat_mean": float(np.max(per_repeat_means)) if len(per_repeat_means) > 0 else 0.0,
            "range_per_repeat_means": float(np.ptp(per_repeat_means)) if len(per_repeat_means) > 0 else 0.0,
        },
        "repeats_payload": repeats_payload,
    }


def _compute_random_channel_stats(
    *,
    epochs_data: np.ndarray,
    channel_names: List[str],
    analyzer: ComplexityAnalyzer,
    repeats: int,
    n_channels: int,
    rng_seed: int,
    progress_prefix: str,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Core loop: sample random channel subsets, evaluate MIB, summarize variability.

    This function loops over compute_single_random_channel_mib() and aggregates results.
    """
    n_all_channels = len(channel_names)
    if n_channels > n_all_channels:
        raise ValueError(f"Requested {n_channels} channels but only {n_all_channels} available.")

    rng = np.random.default_rng(seed=rng_seed)
    repeats_payload: List[Dict[str, Any]] = []

    for rep_idx in tqdm(range(repeats), desc=f"{progress_prefix} repeats", disable=not verbose):
        result = compute_single_random_channel_mib(
            epochs_data=epochs_data,
            channel_names=channel_names,
            analyzer=analyzer,
            n_channels=n_channels,
            rng=rng,
            repeat_index=rep_idx,
            progress_label=f"{progress_prefix} {rep_idx + 1}/{repeats}",
        )

        if result is None:
            log_print(f"[{rep_idx + 1}/{repeats}] No valid MIB values computed.", verbose)
            continue

        repeats_payload.append(result)

        if verbose:
            log_print(
                f"[{rep_idx + 1}/{repeats}] mean={result['mean_mib']:.4f} std={result['std_mib']:.4f} "
                f"channels={','.join(result['selected_channels'])}",
                True,
            )

    return _aggregate_repeat_results(repeats_payload, repeats, n_channels, n_all_channels)


def _compute_epoch_stability(
    *,
    epochs_data: np.ndarray,
    channel_names: List[str],
    analyzer: ComplexityAnalyzer,
    n_channels: int,
    fixed_channels: Optional[List[str]],
    verbose: bool,
) -> Dict[str, Any]:
    """Stability across epochs for a fixed channel set."""
    indices, selected = _select_channel_indices(channel_names, n_channels, fixed_channels)
    subset_epochs = epochs_data[:, indices, :]
    metric_values = analyzer._evaluate_epochs(  # pylint: disable=protected-access
        subset_epochs,
        progress_label="Fixed-channel epochs",
        verbose_override=False,
    )
    metric_values = [v for v in metric_values if v is not None]

    if not metric_values:
        return {
            "selected_channels": selected,
            "selected_indices": indices.tolist(),
            "n_epochs": 0,
            "mean_mib": 0.0,
            "std_mib": 0.0,
            "metric_values": [],
        }

    mean_mib = float(np.mean(metric_values))
    std_mib = float(np.std(metric_values))
    if verbose:
        log_print(
            f"Fixed-channel stability | mean={mean_mib:.4f} std={std_mib:.4f} "
            f"channels={','.join(selected)}",
            True,
        )
    return {
        "selected_channels": selected,
        "selected_indices": indices.tolist(),
        "n_epochs": len(metric_values),
        "mean_mib": mean_mib,
        "std_mib": std_mib,
        "metric_values": [float(x) for x in metric_values],
    }


# -----------------------------------------------------------------------------
# Broadband workflow
# -----------------------------------------------------------------------------


def _load_and_preprocess_full_eeg(
    vhdr_path: Path,
    *,
    epoch_length: float,
    verbose: bool,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load raw EEG, preprocess ALL channels, and return normalized epochs.
    """
    log_print(f"Loading broadband EEG from {vhdr_path.name}...", verbose)
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    # Count EEG channels excluding auxiliary channels that will be dropped
    exclude_channels = ANALYSIS_PARAMS.get('exclude_channels', [])
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    eeg_channel_names = [raw.ch_names[i] for i in eeg_picks]
    eeg_channel_count = len([ch for ch in eeg_channel_names if ch not in exclude_channels])
    if eeg_channel_count == 0:
        raise ValueError("No EEG channels found in the recording.")
    from eeg_analysis.eeg_utils import preprocess_eeg  # Local import to avoid cycle

    epochs_data, channel_names = preprocess_eeg(
        raw,
        epoch_length=float(epoch_length),
        n_channels=eeg_channel_count,
        channel_selection="first",
        verbose=False,
    )
    return epochs_data, list(channel_names)


def run_random_channel_mib_broadband(
    *,
    vhdr_path: Path,
    repeats: int,
    n_channels: int,
    epoch_length: float,
    rng_seed: int,
    fixed_channels: Optional[List[str]],
    output_dir: Path,
    verbose: bool,
    n_jobs: Optional[int],
) -> Path:
    analyzer = _build_analyzer(n_channels, epoch_length, verbose=False, n_jobs=n_jobs)
    epochs_data, channel_names = _load_and_preprocess_full_eeg(
        vhdr_path,
        epoch_length=epoch_length,
        verbose=verbose,
    )
    epoch_stability = _compute_epoch_stability(
        epochs_data=epochs_data,
        channel_names=channel_names,
        analyzer=analyzer,
        n_channels=n_channels,
        fixed_channels=fixed_channels,
        verbose=verbose,
    )
    stats_payload = _compute_random_channel_stats(
        epochs_data=epochs_data,
        channel_names=channel_names,
        analyzer=analyzer,
        repeats=repeats,
        n_channels=n_channels,
        rng_seed=rng_seed,
        progress_prefix="Broadband",
        verbose=verbose,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"mib_random_channels_broadband_binning_{timestamp}.json"
    payload: Dict[str, Any] = {
        "mode": "broadband",
        "vhdr_path": str(vhdr_path),
        "epoch_length": float(epoch_length),
        "estimator": "binning",
        "all_channel_names": channel_names,
        "epoch_stability": epoch_stability,
        **stats_payload,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    log_print(
        f"\nSaved broadband random-channel MIB summary to: {out_json}\n"
        f"  Mean of per-repeat means: {payload['overall_stats']['mean_of_per_repeat_means']:.4f} "
        f"± {payload['overall_stats']['std_of_per_repeat_means']:.4f}",
        verbose,
    )
    return out_json


# -----------------------------------------------------------------------------
# Spectral workflow
# -----------------------------------------------------------------------------


def run_random_channel_mib_spectral(
    *,
    vhdr_path: Path,
    repeats: int,
    n_channels: int,
    epoch_length: float,
    rng_seed: int,
    fixed_channels: Optional[List[str]],
    bands: Optional[List[str]],
    output_dir: Path,
    verbose: bool,
    n_jobs: Optional[int],
) -> Path:
    analyzer = _build_analyzer(n_channels, epoch_length, verbose=False, n_jobs=n_jobs)
    band_list = bands or list(SPECTRAL_BANDS.keys())
    log_print(f"Loading EEG and preparing spectral-band epochs from {vhdr_path.name}...", verbose)
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    band_data, channel_names = preprocess_eeg_by_bands(
        raw,
        epoch_length=float(epoch_length),
        target_sfreq=ANALYSIS_PARAMS.get("target_sfreq", 500.0),
        reference_channel=ANALYSIS_PARAMS.get("reference_channel", "A2"),
        use_all_channels=True,
        bands=band_list,
        verbose=False,
    )

    per_band_results: Dict[str, Any] = {}
    for band_name, epochs_data in band_data.items():
        epoch_stability = _compute_epoch_stability(
            epochs_data=epochs_data,
            channel_names=channel_names,
            analyzer=analyzer,
            n_channels=n_channels,
            fixed_channels=fixed_channels,
            verbose=verbose,
        )
        band_stats = _compute_random_channel_stats(
            epochs_data=epochs_data,
            channel_names=channel_names,
            analyzer=analyzer,
            repeats=repeats,
            n_channels=n_channels,
            rng_seed=rng_seed,
            progress_prefix=f"{band_name}",
            verbose=verbose,
        )
        if band_stats["repeats_payload"]:
            per_band_results[band_name] = {**band_stats, "epoch_stability": epoch_stability}

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"mib_random_channels_spectral_binning_{timestamp}.json"
    payload: Dict[str, Any] = {
        "mode": "spectral",
        "vhdr_path": str(vhdr_path),
        "epoch_length": float(epoch_length),
        "estimator": "binning",
        "all_channel_names": channel_names,
        "selected_bands": band_list,
        "bands": per_band_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    log_print(f"\nSaved spectral random-channel MIB summary to: {out_json}", verbose)
    return out_json


# -----------------------------------------------------------------------------
# Single-sample workflow
# -----------------------------------------------------------------------------


def run_single_sample_mib(
    *,
    vhdr_path: Path,
    n_channels: int,
    epoch_length: float,
    rng_seed: int,
    output_dir: Optional[Path],
    verbose: bool,
    n_jobs: Optional[int],
    mode: str = "broadband",
    bands: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a single random-channel MIB sample (one channel subset, all epochs).

    Parameters
    ----------
    vhdr_path : Path
        Path to BrainVision .vhdr file.
    n_channels : int
        Number of channels to randomly select.
    epoch_length : float
        Epoch length in seconds.
    rng_seed : int
        Random seed for reproducibility.
    output_dir : Optional[Path]
        Directory to save results (None to skip saving).
    verbose : bool
        Whether to print progress.
    n_jobs : Optional[int]
        Number of parallel jobs.
    mode : str
        'broadband' or 'spectral'.
    bands : Optional[List[str]]
        Spectral bands to use (only for spectral mode).

    Returns
    -------
    Dict[str, Any]
        Result dictionary with selected channels and MIB values.
    """
    analyzer = _build_analyzer(n_channels, epoch_length, verbose=False, n_jobs=n_jobs)
    rng = np.random.default_rng(seed=rng_seed)

    if mode == "broadband":
        epochs_data, channel_names = _load_and_preprocess_full_eeg(
            vhdr_path,
            epoch_length=epoch_length,
            verbose=verbose,
        )
        result = compute_single_random_channel_mib(
            epochs_data=epochs_data,
            channel_names=channel_names,
            analyzer=analyzer,
            n_channels=n_channels,
            rng=rng,
            repeat_index=0,
            progress_label="Single-sample broadband",
        )
        if result is None:
            log_print("No valid MIB values computed.", verbose)
            return {"error": "No valid MIB values computed", "mode": mode}

        result["mode"] = "broadband"
        result["vhdr_path"] = str(vhdr_path)
        result["epoch_length"] = epoch_length
        result["estimator"] = "binning"
        result["rng_seed"] = rng_seed
        result["all_channel_names"] = channel_names

    else:  # spectral
        band_list = bands or list(SPECTRAL_BANDS.keys())
        log_print(f"Loading EEG and preparing spectral-band epochs from {vhdr_path.name}...", verbose)
        raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
        band_data, channel_names = preprocess_eeg_by_bands(
            raw,
            epoch_length=float(epoch_length),
            target_sfreq=ANALYSIS_PARAMS.get("target_sfreq", 500.0),
            reference_channel=ANALYSIS_PARAMS.get("reference_channel", "A2"),
            use_all_channels=True,
            bands=band_list,
            verbose=False,
        )

        per_band_results: Dict[str, Any] = {}
        for band_name, epochs_data in band_data.items():
            band_result = compute_single_random_channel_mib(
                epochs_data=epochs_data,
                channel_names=channel_names,
                analyzer=analyzer,
                n_channels=n_channels,
                rng=rng,
                repeat_index=0,
                progress_label=f"Single-sample {band_name}",
            )
            if band_result is not None:
                per_band_results[band_name] = band_result

        result = {
            "mode": "spectral",
            "vhdr_path": str(vhdr_path),
            "epoch_length": epoch_length,
            "estimator": "binning",
            "rng_seed": rng_seed,
            "all_channel_names": channel_names,
            "selected_bands": band_list,
            "bands": per_band_results,
        }

    if verbose:
        if mode == "broadband":
            log_print(
                f"Single-sample result: mean_mib={result['mean_mib']:.4f} "
                f"std_mib={result['std_mib']:.4f} "
                f"channels={','.join(result['selected_channels'])}",
                True,
            )
        else:
            for band_name, band_res in result.get("bands", {}).items():
                log_print(
                    f"  {band_name}: mean_mib={band_res['mean_mib']:.4f} "
                    f"std_mib={band_res['std_mib']:.4f}",
                    True,
                )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_json = output_dir / f"mib_single_sample_{mode}_binning_{timestamp}.json"
        with out_json.open("w") as f:
            json.dump(result, f, indent=2)
        log_print(f"Saved single-sample result to: {out_json}", verbose)
        result["output_path"] = str(out_json)

    return result


# -----------------------------------------------------------------------------
# Dataset orchestration
# -----------------------------------------------------------------------------


def _iter_subjects(dataset_dir: Path, subjects: Optional[Iterable[str]], verbose: bool):
    subject_map = create_subject_file_map(dataset_dir=dataset_dir, verbose=verbose)
    if subjects:
        subject_set = {s.strip() for s in subjects}
        subject_map = {sid: paths for sid, paths in subject_map.items() if sid in subject_set}
    # Prefer 'sedation_1' over 'sedation_2' if both exist
    for sid, cond_map in subject_map.items():
        if "sedation_1" in cond_map and "sedation_2" in cond_map:
            cond_map.pop("sedation_2", None)
    return subject_map


def _compose_output_dir(base: Path, mode: str, epoch_length: float, subject_id: str, condition: str) -> Path:
    epoch_component = f"epoch-{epoch_length:.2f}s".replace(".", "p")
    return base / mode / "binning" / epoch_component / subject_id / condition


def run_dataset(
    *,
    dataset_dir: Path,
    subjects: Optional[Iterable[str]],
    epoch_lengths: List[float],
    repeats: int,
    n_channels: int,
    rng_seed: int,
    fixed_channels: Optional[List[str]],
    mode: str,
    output_dir: Path,
    verbose: bool,
    included_conditions: Optional[Iterable[str]] = None,
    n_jobs: Optional[int] = None,
    bands: Optional[List[str]] = None,
    sampling: str = "random",
) -> List[Path]:
    generated_files: List[Path] = []
    subject_map = _iter_subjects(dataset_dir, subjects, verbose)
    if not subject_map:
        log_print("No subjects matched the criteria.", True)
        return generated_files

    mode_options = {"broadband", "spectral"}
    if mode not in mode_options:
        raise ValueError(f"mode must be one of {mode_options}")

    allowed_conditions = {c.strip() for c in included_conditions} if included_conditions else None
    sampling_label = "single-sample" if sampling == "single" else f"repeats={repeats}"

    for epoch_length in epoch_lengths:
        log_print(
            f"\n=== Running dataset sweep | mode={mode} | {sampling_label} | "
            f"epoch_length={epoch_length}s | n_channels={n_channels} ===",
            verbose,
        )
        for subject_id, cond_map in subject_map.items():
            filtered_cond_map = (
                {cond: path for cond, path in cond_map.items() if cond in allowed_conditions}
                if allowed_conditions
                else cond_map
            )
            if not filtered_cond_map:
                continue

            for condition, vhdr_path in sorted(filtered_cond_map.items()):
                condition_dir = _compose_output_dir(
                    output_dir,
                    mode,
                    epoch_length,
                    subject_id,
                    condition,
                )

                if sampling == "single":
                    # Single sample mode
                    result = run_single_sample_mib(
                        vhdr_path=Path(vhdr_path),
                        n_channels=n_channels,
                        epoch_length=epoch_length,
                        rng_seed=rng_seed,
                        output_dir=condition_dir,
                        verbose=verbose,
                        n_jobs=n_jobs,
                        mode=mode,
                        bands=bands,
                    )
                    if "output_path" in result:
                        generated_files.append(Path(result["output_path"]))
                else:
                    # Random sampling mode
                    if mode == "broadband":
                        generated_files.append(
                            run_random_channel_mib_broadband(
                                vhdr_path=Path(vhdr_path),
                                repeats=repeats,
                                n_channels=n_channels,
                                epoch_length=epoch_length,
                                rng_seed=rng_seed,
                                fixed_channels=fixed_channels,
                                output_dir=condition_dir,
                                verbose=verbose,
                                n_jobs=n_jobs,
                            )
                        )
                    else:  # spectral
                        generated_files.append(
                            run_random_channel_mib_spectral(
                                vhdr_path=Path(vhdr_path),
                                repeats=repeats,
                                n_channels=n_channels,
                                epoch_length=epoch_length,
                                rng_seed=rng_seed,
                                fixed_channels=fixed_channels,
                                bands=bands,
                                output_dir=condition_dir,
                                verbose=verbose,
                                n_jobs=n_jobs,
                            )
                        )
    return generated_files


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments common to multiple subcommands."""
    parser.add_argument(
        "--vhdr",
        type=str,
        required=True,
        help="Path to BrainVision .vhdr file.",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=int(ANALYSIS_PARAMS.get("n_channels", 8)),
        help="Number of channels per random draw (default: %(default)s).",
    )
    parser.add_argument(
        "--epoch-length",
        type=float,
        default=float(ANALYSIS_PARAMS.get("epoch_length", 5.0)),
        help="Epoch length in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=["broadband", "spectral"],
        default="broadband",
        help="Analysis mode (default: %(default)s).",
    )
    parser.add_argument(
        "--bands",
        nargs="*",
        choices=list(SPECTRAL_BANDS.keys()),
        default=list(SPECTRAL_BANDS.keys()),
        help="Spectral bands (only for spectral mode, default: all). Use --bands without args for all.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(ANALYSIS_PARAMS.get("n_jobs", -1)),
        help="Number of parallel jobs (-1 for all cores, default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to save results (default: current directory).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging.",
    )


def _prompt_choice(prompt: str, choices: List[str], default: str) -> str:
    """Prompt user for a choice from a list."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = " [default]" if choice == default else ""
        print(f"  {i}. {choice}{marker}")
    while True:
        user_input = input(f"Enter choice (1-{len(choices)}) or press Enter for default: ").strip()
        if not user_input:
            return default
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
        except ValueError:
            if user_input in choices:
                return user_input
        print(f"Invalid choice. Please enter 1-{len(choices)} or one of: {', '.join(choices)}")


def _prompt_number(prompt: str, default: float, min_val: float = 0, max_val: float = float("inf")) -> float:
    """Prompt user for a numeric value."""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            return default
        try:
            val = float(user_input)
            if min_val <= val <= max_val:
                return val
            print(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid number.")


def _prompt_int(prompt: str, default: int, min_val: int = 0, max_val: int = 10000) -> int:
    """Prompt user for an integer value."""
    return int(_prompt_number(prompt, float(default), float(min_val), float(max_val)))


def _prompt_path(prompt: str, must_exist: bool = True) -> Path:
    """Prompt user for a file path."""
    while True:
        user_input = input(f"{prompt}: ").strip()
        if not user_input:
            print("Path is required.")
            continue
        path = Path(user_input).expanduser().resolve()
        if must_exist and not path.exists():
            print(f"File not found: {path}")
            continue
        return path


def _prompt_yes_no(prompt: str, default: bool = True) -> bool:
    """Prompt user for yes/no."""
    default_str = "Y/n" if default else "y/N"
    user_input = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not user_input:
        return default
    return user_input in ("y", "yes", "1", "true")


def _run_interactive() -> None:
    """Run the interactive CLI mode."""
    print("\n" + "=" * 60)
    print("  MIB Random Channel Analysis - Interactive Mode")
    print("=" * 60)

    # Choose sampling mode
    sampling_mode = _prompt_choice(
        "Select sampling mode:",
        ["single", "random"],
        "single",
    )

    # Get common parameters
    vhdr_path = _prompt_path("Enter path to .vhdr file")

    n_channels = _prompt_int("Number of channels to select", 8, 2, 64)
    epoch_length = _prompt_number("Epoch length (seconds)", 5.0, 0.5, 60.0)
    rng_seed = _prompt_int("Random seed", 42, 0, 2**31 - 1)

    analysis_mode = _prompt_choice(
        "Select analysis mode:",
        ["broadband", "spectral"],
        "broadband",
    )

    bands = None
    if analysis_mode == "spectral":
        available_bands = list(SPECTRAL_BANDS.keys())
        print(f"\nAvailable bands: {', '.join(available_bands)}")
        use_all = _prompt_yes_no("Use all bands?", True)
        if not use_all:
            bands_input = input("Enter bands (space-separated): ").strip().split()
            bands = [b for b in bands_input if b in available_bands]
            if not bands:
                bands = None
                print("No valid bands specified, using all.")

    n_jobs = _prompt_int("Number of parallel jobs (-1 for all cores)", -1, -1, 128)

    save_output = _prompt_yes_no("Save results to file?", True)
    output_dir = None
    if save_output:
        output_path = input("Output directory [./output]: ").strip()
        output_dir = Path(output_path) if output_path else Path("./output")

    verbose = not _prompt_yes_no("Quiet mode (suppress logging)?", False)

    # Mode-specific parameters
    if sampling_mode == "random":
        repeats = _prompt_int("Number of random samples (repeats)", 50, 1, 10000)
        compute_stability = _prompt_yes_no("Compute fixed-channel epoch stability?", True)
        fixed_channels = None
        if compute_stability:
            use_default_channels = _prompt_yes_no("Use default channel selection?", True)
            if not use_default_channels:
                channels_input = input("Enter channel names (space-separated): ").strip().split()
                fixed_channels = channels_input if channels_input else None

    # Confirm and run
    print("\n" + "-" * 60)
    print("Configuration Summary:")
    print("-" * 60)
    print(f"  Sampling mode:   {sampling_mode}")
    print(f"  VHDR file:       {vhdr_path}")
    print(f"  Estimator:       binning")
    print(f"  N channels:      {n_channels}")
    print(f"  Epoch length:    {epoch_length}s")
    print(f"  RNG seed:        {rng_seed}")
    print(f"  Analysis mode:   {analysis_mode}")
    if analysis_mode == "spectral":
        print(f"  Bands:           {bands or 'all'}")
    print(f"  Parallel jobs:   {n_jobs}")
    print(f"  Output dir:      {output_dir or 'None (no save)'}")
    if sampling_mode == "random":
        print(f"  Repeats:         {repeats}")
    print("-" * 60)

    if not _prompt_yes_no("\nProceed with analysis?", True):
        print("Aborted.")
        return

    print("\nStarting analysis...\n")

    if sampling_mode == "single":
        result = run_single_sample_mib(
            vhdr_path=vhdr_path,
            n_channels=n_channels,
            epoch_length=epoch_length,
            rng_seed=rng_seed,
            output_dir=output_dir,
            verbose=verbose,
            n_jobs=n_jobs,
            mode=analysis_mode,
            bands=bands,
        )
        print("\n" + "=" * 60)
        print("Single-Sample Result:")
        print("=" * 60)
        if "error" in result:
            print(f"  Error: {result['error']}")
        elif analysis_mode == "broadband":
            print(f"  Mean MIB:     {result['mean_mib']:.4f}")
            print(f"  Std MIB:      {result['std_mib']:.4f}")
            print(f"  N epochs:     {result['n_epochs']}")
            print(f"  Channels:     {', '.join(result['selected_channels'])}")
        else:
            for band_name, band_res in result.get("bands", {}).items():
                print(f"  {band_name}:")
                print(f"    Mean MIB: {band_res['mean_mib']:.4f}")
                print(f"    Std MIB:  {band_res['std_mib']:.4f}")
    else:
        # Random sampling mode
        if analysis_mode == "broadband":
            out_path = run_random_channel_mib_broadband(
                vhdr_path=vhdr_path,
                repeats=repeats,
                n_channels=n_channels,
                epoch_length=epoch_length,
                rng_seed=rng_seed,
                fixed_channels=fixed_channels if compute_stability else None,
                output_dir=output_dir or Path("./output"),
                verbose=verbose,
                n_jobs=n_jobs,
            )
        else:
            out_path = run_random_channel_mib_spectral(
                vhdr_path=vhdr_path,
                repeats=repeats,
                n_channels=n_channels,
                epoch_length=epoch_length,
                rng_seed=rng_seed,
                fixed_channels=fixed_channels if compute_stability else None,
                bands=bands,
                output_dir=output_dir or Path("./output"),
                verbose=verbose,
                n_jobs=n_jobs,
            )
        print(f"\nResults saved to: {out_path}")

    print("\nDone!")


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="MIB Random Channel Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (guided prompts)
  python scripts/random_channels_mib.py interactive

  # Single sample (one random channel subset)
  python scripts/random_channels_mib.py single --vhdr ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr

  # Random sampling (multiple repeats)
  python scripts/random_channels_mib.py random --vhdr ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr --repeats 50

  # Dataset sweep (multiple subjects)
  python scripts/random_channels_mib.py dataset --dataset ds005620 --subjects sub-1010 sub-1011
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Interactive mode
    interactive_parser = subparsers.add_parser(
        "interactive",
        aliases=["i"],
        help="Interactive mode with guided prompts",
    )

    # Single sample mode
    single_parser = subparsers.add_parser(
        "single",
        aliases=["s"],
        help="Run a single random-channel MIB sample",
    )
    _add_common_args(single_parser)

    # Random sampling mode
    random_parser = subparsers.add_parser(
        "random",
        aliases=["r"],
        help="Run multiple random-channel samples (Monte Carlo)",
    )
    _add_common_args(random_parser)
    random_parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of random samples to evaluate (default: %(default)s).",
    )
    random_parser.add_argument(
        "--fixed-channels",
        nargs="+",
        default=None,
        help="Fixed channel list for epoch stability checks.",
    )
    random_parser.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip fixed-channel epoch stability computation.",
    )

    # Dataset sweep mode
    dataset_parser = subparsers.add_parser(
        "dataset",
        aliases=["d"],
        help="Run analysis across a dataset of subjects",
    )
    dataset_parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_DIR,
        help="Path to the dataset directory.",
    )
    dataset_parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subject IDs to include (e.g., sub-1010 sub-1011).",
    )
    dataset_parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help="Conditions to include (e.g., awake_eyes_closed sedation_1).",
    )
    dataset_parser.add_argument(
        "--two-sample",
        nargs=2,
        metavar=("SUBJECT_A", "SUBJECT_B"),
        help="Run a fixed two-subject sweep.",
    )
    dataset_parser.add_argument(
        "--epoch-lengths",
        nargs="+",
        type=float,
        default=[float(ANALYSIS_PARAMS.get("epoch_length", 5.0))],
        help="Epoch length(s) in seconds.",
    )
    dataset_parser.add_argument(
        "--n-channels",
        type=int,
        default=int(ANALYSIS_PARAMS.get("n_channels", 8)),
        help="Number of channels per random draw.",
    )
    dataset_parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of random samples per combination.",
    )
    dataset_parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    dataset_parser.add_argument(
        "--fixed-channels",
        nargs="+",
        default=None,
        help="Fixed channel list for epoch stability.",
    )
    dataset_parser.add_argument(
        "--mode",
        choices=["broadband", "spectral"],
        default="spectral",
        help="Analysis mode (default: spectral, which includes broadband as a band).",
    )
    dataset_parser.add_argument(
        "--bands",
        nargs="*",
        choices=list(SPECTRAL_BANDS.keys()),
        default=list(SPECTRAL_BANDS.keys()),
        help="Spectral bands to include (default: all). Use --bands without args for all.",
    )
    dataset_parser.add_argument(
        "--sampling",
        choices=["random", "single"],
        default="random",
        help="Sampling mode: 'random' for Monte Carlo, 'single' for one sample (default: random).",
    )
    dataset_parser.add_argument(
        "--jobs",
        type=int,
        default=int(ANALYSIS_PARAMS.get("n_jobs", -1)),
        help="Number of parallel jobs.",
    )
    dataset_parser.add_argument(
        "--output",
        type=str,
        default=str(Path(DEFAULT_OUTPUT_DIR) / "ds005620" / "mib_random_channels"),
        help="Directory to save results.",
    )
    dataset_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging.",
    )

    return parser


def cli_main() -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\nTip: Use 'interactive' command for guided mode.")
        return

    if args.command in ("interactive", "i"):
        _run_interactive()
        return

    if args.command in ("single", "s"):
        vhdr_path = Path(args.vhdr)
        if not vhdr_path.exists():
            raise FileNotFoundError(f"VHDR file not found: {vhdr_path}")

        output_dir = Path(args.output) if args.output else None
        result = run_single_sample_mib(
            vhdr_path=vhdr_path,
            n_channels=args.n_channels,
            epoch_length=args.epoch_length,
            rng_seed=args.rng_seed,
            output_dir=output_dir,
            verbose=not args.quiet,
            n_jobs=args.jobs,
            mode=args.mode,
            bands=args.bands,
        )
        if not args.quiet:
            if "error" not in result:
                if args.mode == "broadband":
                    print(f"\nMean MIB: {result['mean_mib']:.4f} +/- {result['std_mib']:.4f}")
                else:
                    print("\nSpectral results:")
                    for band, res in result.get("bands", {}).items():
                        print(f"  {band}: {res['mean_mib']:.4f} +/- {res['std_mib']:.4f}")
        return

    if args.command in ("random", "r"):
        vhdr_path = Path(args.vhdr)
        if not vhdr_path.exists():
            raise FileNotFoundError(f"VHDR file not found: {vhdr_path}")

        output_dir = Path(args.output) if args.output else Path("./output")
        verbose = not args.quiet
        fixed_channels = None if args.skip_stability else (args.fixed_channels or ANALYSIS_PARAMS.get("channels_list"))

        if args.mode == "broadband":
            run_random_channel_mib_broadband(
                vhdr_path=vhdr_path,
                repeats=args.repeats,
                n_channels=args.n_channels,
                epoch_length=args.epoch_length,
                rng_seed=args.rng_seed,
                fixed_channels=fixed_channels,
                output_dir=output_dir,
                verbose=verbose,
                n_jobs=args.jobs,
            )
        else:
            run_random_channel_mib_spectral(
                vhdr_path=vhdr_path,
                repeats=args.repeats,
                n_channels=args.n_channels,
                epoch_length=args.epoch_length,
                rng_seed=args.rng_seed,
                fixed_channels=fixed_channels,
                bands=args.bands,
                output_dir=output_dir,
                verbose=verbose,
                n_jobs=args.jobs,
            )
        return

    if args.command in ("dataset", "d"):
        verbose = not args.quiet
        output_dir = Path(args.output)
        fixed_channels = args.fixed_channels or ANALYSIS_PARAMS.get("channels_list")

        two_sample_conditions = ["awake_eyes_closed", "awake_eyes_open", "sedation_1"]
        subject_list: Optional[List[str]] = args.subjects
        included_conditions: Optional[List[str]] = args.conditions
        if args.two_sample:
            subject_list = list(args.two_sample)
            included_conditions = two_sample_conditions

        # Handle --bands with no arguments (nargs="*" returns [] when flag used without args)
        bands = args.bands if args.bands else list(SPECTRAL_BANDS.keys())

        generated = run_dataset(
            dataset_dir=Path(args.dataset),
            subjects=subject_list,
            epoch_lengths=[float(e) for e in args.epoch_lengths],
            repeats=args.repeats,
            n_channels=args.n_channels,
            rng_seed=args.rng_seed,
            fixed_channels=fixed_channels,
            mode=args.mode,
            output_dir=output_dir,
            verbose=verbose,
            included_conditions=included_conditions,
            n_jobs=args.jobs,
            bands=bands,
            sampling=args.sampling,
        )
        if verbose:
            log_print(f"\nGenerated {len(generated)} result files.", True)
        return


if __name__ == "__main__":
    cli_main()
