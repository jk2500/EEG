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

It supports sweeping over subjects, epoch lengths, and estimators.

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

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eeg_analysis.config import (  # type: ignore
    ANALYSIS_PARAMS,
    BINNING_PARAMS,
    DATASET_DIR,
    DEFAULT_FILE_PATHS,
    DEFAULT_OUTPUT_DIR,
    GAUSSIAN_PARAMS,
    KSG_PARAMS,
    SPECTRAL_BANDS,
)
from eeg_analysis.analyzers.complexity_analyzer import ComplexityAnalyzer  # type: ignore
from eeg_analysis.analyzers.estimators import (  # type: ignore
    BinningEstimator,
    GaussianEstimator,
    KSGEstimator,
)
from eeg_analysis.eeg_utils import (  # type: ignore
    create_subject_file_map,
    log_print,
    preprocess_eeg_by_bands,
)

mne.set_log_level("WARNING")


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------


def _estimator_factory(name: str):
    name_l = name.lower()
    if name_l == "ksg":
        return KSGEstimator, KSG_PARAMS
    if name_l == "binning":
        return BinningEstimator, BINNING_PARAMS
    if name_l == "gaussian":
        return GaussianEstimator, GAUSSIAN_PARAMS
    raise ValueError(f"Unknown estimator: {name}. Choose from ['ksg', 'binning', 'gaussian'].")


def _build_analyzer(
    estimator_name: str,
    n_channels: int,
    epoch_length: float,
    verbose: bool,
    n_jobs: Optional[int] = None,
) -> ComplexityAnalyzer:
    EstimatorClass, method_params = _estimator_factory(estimator_name)
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
    """
    n_all_channels = len(channel_names)
    if n_channels > n_all_channels:
        raise ValueError(f"Requested {n_channels} channels but only {n_all_channels} available.")

    rng = np.random.default_rng(seed=rng_seed)
    repeats_payload: List[Dict[str, Any]] = []

    for rep_idx in tqdm(range(repeats), desc=f"{progress_prefix} repeats", disable=not verbose):
        selected_indices = rng.choice(n_all_channels, size=n_channels, replace=False)
        selected_indices = np.sort(selected_indices)
        selected_channels = [channel_names[i] for i in selected_indices]
        subset_epochs = epochs_data[:, selected_indices, :]

        metric_values = analyzer._evaluate_epochs(  # pylint: disable=protected-access
            subset_epochs,
            progress_label=f"{progress_prefix} {rep_idx + 1}/{repeats}",
            verbose_override=False,
        )
        metric_values = [v for v in metric_values if v is not None]
        if not metric_values:
            log_print(f"[{rep_idx + 1}/{repeats}] No valid MIB values computed.", verbose)
            continue

        mean_mib = float(np.mean(metric_values))
        std_mib = float(np.std(metric_values))

        repeats_payload.append(
            {
                "repeat_index": rep_idx,
                "selected_channels": selected_channels,
                "selected_indices": selected_indices.tolist(),
                "n_epochs": len(metric_values),
                "mean_mib": mean_mib,
                "std_mib": std_mib,
                "metric_values": [float(x) for x in metric_values],
            }
        )

        if verbose:
            log_print(
                f"[{rep_idx + 1}/{repeats}] mean={mean_mib:.4f} std={std_mib:.4f} "
                f"channels={','.join(selected_channels)}",
                True,
            )

    per_repeat_means = np.array([r["mean_mib"] for r in repeats_payload], dtype=np.float64)
    per_repeat_stds = np.array([r["std_mib"] for r in repeats_payload], dtype=np.float64)

    mean_of_means = float(np.mean(per_repeat_means)) if len(per_repeat_means) > 0 else 0.0
    std_of_means = float(np.std(per_repeat_means)) if len(per_repeat_means) > 0 else 0.0
    mean_of_stds = float(np.mean(per_repeat_stds)) if len(per_repeat_stds) > 0 else 0.0
    cv_of_means = float(std_of_means / abs(mean_of_means)) if abs(mean_of_means) > 1e-10 else float("inf")

    return {
        "repeats": int(repeats),
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
    from eeg_analysis.eeg_utils import preprocess_eeg  # Local import to avoid cycle

    epochs_data, channel_names = preprocess_eeg(
        raw,
        epoch_length=float(epoch_length),
        n_channels=len(raw.ch_names),
        channel_selection="first",
        verbose=False,
    )
    return epochs_data, list(channel_names)


def run_random_channel_mib_broadband(
    *,
    vhdr_path: Path,
    repeats: int,
    estimator_name: str,
    n_channels: int,
    epoch_length: float,
    rng_seed: int,
    fixed_channels: Optional[List[str]],
    bands: Optional[List[str]],
    output_dir: Path,
    verbose: bool,
    n_jobs: Optional[int],
) -> Path:
    analyzer = _build_analyzer(estimator_name, n_channels, epoch_length, verbose=False, n_jobs=n_jobs)
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
    out_json = output_dir / f"mib_random_channels_broadband_{estimator_name}_{timestamp}.json"
    payload: Dict[str, Any] = {
        "mode": "broadband",
        "vhdr_path": str(vhdr_path),
        "epoch_length": float(epoch_length),
        "estimator": estimator_name,
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
    estimator_name: str,
    n_channels: int,
    epoch_length: float,
    rng_seed: int,
    fixed_channels: Optional[List[str]],
    bands: Optional[List[str]],
    output_dir: Path,
    verbose: bool,
    n_jobs: Optional[int],
) -> Path:
    analyzer = _build_analyzer(estimator_name, n_channels, epoch_length, verbose=False, n_jobs=n_jobs)
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
    out_json = output_dir / f"mib_random_channels_spectral_{estimator_name}_{timestamp}.json"
    payload: Dict[str, Any] = {
        "mode": "spectral",
        "vhdr_path": str(vhdr_path),
        "epoch_length": float(epoch_length),
        "estimator": estimator_name,
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


def _compose_output_dir(base: Path, mode: str, estimator: str, epoch_length: float, subject_id: str, condition: str) -> Path:
    epoch_component = f"epoch-{epoch_length:.2f}s".replace(".", "p")
    return base / mode / estimator / epoch_component / subject_id / condition


def run_dataset(
    *,
    dataset_dir: Path,
    subjects: Optional[Iterable[str]],
    epoch_lengths: List[float],
    estimators: List[str],
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
) -> List[Path]:
    generated_files: List[Path] = []
    subject_map = _iter_subjects(dataset_dir, subjects, verbose)
    if not subject_map:
        log_print("No subjects matched the criteria.", True)
        return generated_files

    mode_options = {"broadband", "spectral", "both"}
    if mode not in mode_options:
        raise ValueError(f"mode must be one of {mode_options}")

    allowed_conditions = {c.strip() for c in included_conditions} if included_conditions else None

    for epoch_length in epoch_lengths:
        for estimator in estimators:
            log_print(
                f"\n=== Running dataset sweep | mode={mode} | estimator={estimator} | "
                f"epoch_length={epoch_length}s | repeats={repeats} | n_channels={n_channels} ===",
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
                        "broadband" if mode == "both" else mode,
                        estimator,
                        epoch_length,
                        subject_id,
                        condition,
                    )
                    if mode in ("broadband", "both"):
                        generated_files.append(
                            run_random_channel_mib_broadband(
                                vhdr_path=Path(vhdr_path),
                                repeats=repeats,
                                estimator_name=estimator,
                                n_channels=n_channels,
                                epoch_length=epoch_length,
                                rng_seed=rng_seed,
                                fixed_channels=fixed_channels,
                                output_dir=condition_dir,
                                verbose=verbose,
                                n_jobs=n_jobs,
                            )
                        )
                    if mode in ("spectral", "both"):
                        spectral_dir = (
                            condition_dir if mode == "spectral" else _compose_output_dir(
                                output_dir,
                                "spectral",
                                estimator,
                                epoch_length,
                                subject_id,
                                condition,
                            )
                        )
                        generated_files.append(
                            run_random_channel_mib_spectral(
                                vhdr_path=Path(vhdr_path),
                                repeats=repeats,
                                estimator_name=estimator,
                                n_channels=n_channels,
                                epoch_length=epoch_length,
                                rng_seed=rng_seed,
                                fixed_channels=fixed_channels,
                                bands=bands,
                                output_dir=spectral_dir,
                                verbose=verbose,
                                n_jobs=n_jobs,
                            )
                        )
    return generated_files


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MIB repeatedly using random subsets of EEG channels (broadband and/or spectral).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_DIR,
        help="Path to the dataset directory (ignored if --vhdr is provided).",
    )
    parser.add_argument(
        "--vhdr",
        type=str,
        default=None,
        help="Run on a single BrainVision .vhdr file instead of sweeping subjects.",
    )
    parser.add_argument(
        "--two-sample",
        nargs=2,
        metavar=("SUBJECT_A", "SUBJECT_B"),
        help="Run a fixed two-subject sweep (eyes-closed, eyes-open, sedation_1 conditions).",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional list of subject IDs to include (e.g., sub-1010 sub-1011).",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=["ksg", "binning", "gaussian"],
        default=["ksg"],
        help="Mutual information estimator(s) to use.",
    )
    parser.add_argument(
        "--epoch-lengths",
        nargs="+",
        type=float,
        default=[float(ANALYSIS_PARAMS.get("epoch_length", 5.0))],
        help="Epoch length(s) in seconds.",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=int(ANALYSIS_PARAMS.get("n_channels", 8)),
        help="Number of channels per random draw.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of random channel subsets to evaluate per combination.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(ANALYSIS_PARAMS.get("n_jobs", -1)),
        help="Number of parallel jobs for epoch evaluation (-1 for all cores).",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Seed for channel subset reproducibility.",
    )
    parser.add_argument(
        "--fixed-channels",
        nargs="+",
        default=None,
        help="Optional fixed channel list for epoch stability checks (uses defaults if omitted).",
    )
    parser.add_argument(
        "--mode",
        choices=["broadband", "spectral", "both"],
        default="both",
        help="Whether to run broadband, spectral, or both analyses.",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        choices=list(SPECTRAL_BANDS.keys()),
        default=None,
        help="Subset of spectral bands to include (default: all).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(DEFAULT_OUTPUT_DIR) / "mib_random_channels"),
        help="Directory to save results.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging.",
    )
    return parser.parse_args()


def cli_main() -> None:
    args = _parse_args()
    verbose = not args.quiet
    output_dir = Path(args.output)
    fixed_channels = args.fixed_channels or ANALYSIS_PARAMS.get("channels_list")

    if args.vhdr:
        vhdr_path = Path(args.vhdr)
        if not vhdr_path.exists():
            raise FileNotFoundError(f"VHDR file not found: {vhdr_path}")

        for epoch_length in args.epoch_lengths:
            for estimator in args.estimators:
                base_dir = output_dir / ("broadband" if args.mode == "both" else args.mode) / estimator
                if args.mode in ("broadband", "both"):
                    run_random_channel_mib_broadband(
                        vhdr_path=vhdr_path,
                        repeats=int(args.repeats),
                        estimator_name=str(estimator),
                        n_channels=int(args.n_channels),
                        epoch_length=float(epoch_length),
                        rng_seed=int(args.rng_seed),
                        fixed_channels=fixed_channels,
                        output_dir=base_dir,
                        verbose=verbose,
                        n_jobs=int(args.jobs),
                    )
                if args.mode in ("spectral", "both"):
                    spectral_dir = output_dir / "spectral" / estimator
                    run_random_channel_mib_spectral(
                        vhdr_path=vhdr_path,
                        repeats=int(args.repeats),
                        estimator_name=str(estimator),
                        n_channels=int(args.n_channels),
                        epoch_length=float(epoch_length),
                        rng_seed=int(args.rng_seed),
                        fixed_channels=fixed_channels,
                        bands=args.bands,
                        output_dir=spectral_dir,
                        verbose=verbose,
                        n_jobs=int(args.jobs),
                    )
        return

    two_sample_conditions = ["awake_eyes_closed", "awake_eyes_open", "sedation_1"]
    subject_list: Optional[List[str]] = args.subjects
    included_conditions: Optional[List[str]] = None
    if args.two_sample:
        subject_list = list(args.two_sample)
        included_conditions = two_sample_conditions

    generated = run_dataset(
        dataset_dir=Path(args.dataset),
        subjects=subject_list,
        epoch_lengths=[float(e) for e in args.epoch_lengths],
        estimators=[str(e) for e in args.estimators],
        repeats=int(args.repeats),
        n_channels=int(args.n_channels),
        rng_seed=int(args.rng_seed),
        fixed_channels=fixed_channels,
        mode=args.mode,
        output_dir=output_dir,
        verbose=verbose,
        included_conditions=included_conditions,
        n_jobs=int(args.jobs),
        bands=args.bands,
    )
    if verbose:
        log_print(f"\nGenerated {len(generated)} result files.", True)


if __name__ == "__main__":
    cli_main()
