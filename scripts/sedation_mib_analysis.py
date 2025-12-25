#!/usr/bin/env python3
"""
Sedation-RestingState MIB Analysis
==================================

This script runs Minimum Information Bipartition (MIB) analysis on the
Sedation-RestingState dataset (EEGLAB format .set/.fdt files).

The dataset contains 20 subjects with 4 conditions each:
- baseline: Pre-sedation awake state
- light_sedation: Light propofol sedation
- deep_sedation: Deep propofol sedation  
- recovery: Post-sedation recovery

Usage:
    python scripts/sedation_mib_analysis.py --help
    python scripts/sedation_mib_analysis.py dataset --conditions baseline deep_sedation
    python scripts/sedation_mib_analysis.py single --file "path/to/file.set"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mne
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eeg_analysis.config import (
    ANALYSIS_PARAMS,
    BINNING_PARAMS,
    SEDATION_DATASET_DIR,
    SPECTRAL_BANDS,
    DEFAULT_OUTPUT_DIR,
)
from eeg_analysis.analyzers.complexity_analyzer import ComplexityAnalyzer
from eeg_analysis.analyzers.estimators import BinningEstimator
from eeg_analysis.sedation_loader import (
    create_sedation_subject_file_map,
    load_sedation_epochs,
    preprocess_sedation_epochs,
    preprocess_sedation_epochs_by_bands,
    SEDATION_CONDITIONS,
    SEDATION_COMMON_CHANNELS,
)
from eeg_analysis.eeg_utils import log_print

mne.set_log_level("WARNING")


def _build_analyzer(
    n_channels: int,
    epoch_length: float,
    verbose: bool,
    n_jobs: Optional[int] = None,
) -> ComplexityAnalyzer:
    """Build a complexity analyzer with binning estimator."""
    params: Dict[str, Any] = {
        **ANALYSIS_PARAMS,
        **BINNING_PARAMS,
        "n_channels": int(n_channels),
        "epoch_length": float(epoch_length),
        "verbose": verbose,
    }
    if n_jobs is not None:
        params["n_jobs"] = int(n_jobs)
    estimator = BinningEstimator(**params)
    return ComplexityAnalyzer(estimator=estimator, **params)


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
    """Compute MIB for a single random channel subset."""
    n_all_channels = len(channel_names)
    if n_channels > n_all_channels:
        raise ValueError(f"Requested {n_channels} channels but only {n_all_channels} available.")

    selected_indices = rng.choice(n_all_channels, size=n_channels, replace=False)
    selected_indices = np.sort(selected_indices)
    selected_channels = [channel_names[i] for i in selected_indices]
    subset_epochs = epochs_data[:, selected_indices, :]

    metric_values = analyzer._evaluate_epochs(
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
    """Aggregate results from multiple random channel samples."""
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
    """Sample random channel subsets and compute MIB statistics."""
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

    return _aggregate_repeat_results(repeats_payload, repeats, n_channels, n_all_channels)


def _select_channel_indices(
    all_channel_names: List[str],
    n_channels: int,
    preferred: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Select channel indices, preferring named channels if available.

    Falls back to first-N EEG channels if preferred channels are unavailable.
    """
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


def _compute_epoch_stability(
    *,
    epochs_data: np.ndarray,
    channel_names: List[str],
    analyzer: ComplexityAnalyzer,
    n_channels: int,
    fixed_channels: Optional[List[str]],
    verbose: bool,
) -> Dict[str, Any]:
    """
    Compute stability across epochs for a fixed channel set.

    This measures epoch-to-epoch variability with a consistent channel selection,
    separating temporal effects from channel sampling effects.
    """
    indices, selected = _select_channel_indices(channel_names, n_channels, fixed_channels)
    subset_epochs = epochs_data[:, indices, :]
    metric_values = analyzer._evaluate_epochs(
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

    log_print(
        f"Fixed-channel stability | mean={mean_mib:.4f} std={std_mib:.4f} "
        f"channels={','.join(selected)}",
        verbose,
    )

    return {
        "selected_channels": selected,
        "selected_indices": indices.tolist(),
        "n_epochs": len(metric_values),
        "mean_mib": mean_mib,
        "std_mib": std_mib,
        "metric_values": [float(x) for x in metric_values],
    }


def run_sedation_mib_spectral(
    *,
    set_file: Path,
    repeats: int,
    n_channels: int,
    rng_seed: int,
    fixed_channels: Optional[List[str]],
    bands: Optional[List[str]],
    output_dir: Path,
    verbose: bool,
    n_jobs: Optional[int],
) -> Path:
    """Run spectral MIB analysis on a Sedation-RestingState file."""
    # Load epochs - they already have 10-second duration
    epochs, meta = load_sedation_epochs(str(set_file), verbose=verbose)
    epoch_length = meta['epoch_duration']

    analyzer = _build_analyzer(n_channels, epoch_length, verbose=False, n_jobs=n_jobs)
    band_list = bands or list(SPECTRAL_BANDS.keys())

    log_print(f"Processing spectral bands: {band_list}", verbose)

    band_data, channel_names = preprocess_sedation_epochs_by_bands(
        epochs,
        bands=band_list,
        use_all_channels=True,
        verbose=False,
    )

    per_band_results: Dict[str, Any] = {}
    for band_name, epochs_data in band_data.items():
        # Compute fixed-channel epoch stability first
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
            band_stats["epoch_stability"] = epoch_stability
            per_band_results[band_name] = band_stats

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = output_dir / f"sedation_mib_spectral_{timestamp}.json"
    payload: Dict[str, Any] = {
        "mode": "spectral",
        "dataset": "sedation_resting_state",
        "set_file": str(set_file),
        "epoch_length": float(epoch_length),
        "estimator": "binning",
        "all_channel_names": channel_names,
        "selected_bands": band_list,
        "original_sfreq": meta['sfreq'],
        "n_original_epochs": meta['n_epochs'],
        "bands": per_band_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    log_print(f"\nSaved spectral MIB results to: {out_json}", verbose)
    return out_json


def run_sedation_mib_broadband(
    *,
    set_file: Path,
    repeats: int,
    n_channels: int,
    rng_seed: int,
    fixed_channels: Optional[List[str]],
    output_dir: Path,
    verbose: bool,
    n_jobs: Optional[int],
) -> Path:
    """Run broadband MIB analysis on a Sedation-RestingState file."""
    epochs, meta = load_sedation_epochs(str(set_file), verbose=verbose)
    epoch_length = meta['epoch_duration']

    analyzer = _build_analyzer(n_channels, epoch_length, verbose=False, n_jobs=n_jobs)

    epochs_data, channel_names = preprocess_sedation_epochs(
        epochs,
        use_all_channels=True,
        verbose=False,
    )

    # Compute fixed-channel epoch stability
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
    out_json = output_dir / f"sedation_mib_broadband_{timestamp}.json"
    payload: Dict[str, Any] = {
        "mode": "broadband",
        "dataset": "sedation_resting_state",
        "set_file": str(set_file),
        "epoch_length": float(epoch_length),
        "estimator": "binning",
        "all_channel_names": channel_names,
        "original_sfreq": meta['sfreq'],
        "n_original_epochs": meta['n_epochs'],
        "epoch_stability": epoch_stability,
        **stats_payload,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(payload, f, indent=2)

    log_print(f"\nSaved broadband MIB results to: {out_json}", verbose)
    return out_json


def _compose_output_dir(base: Path, mode: str, subject_id: str, condition: str) -> Path:
    """Compose output directory path."""
    return base / "sedation" / mode / "binning" / subject_id / condition


def run_sedation_dataset(
    *,
    dataset_dir: Path,
    subjects: Optional[Iterable[str]],
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
    """Run MIB analysis across the Sedation-RestingState dataset."""
    generated_files: List[Path] = []
    subject_map = create_sedation_subject_file_map(str(dataset_dir), verbose=verbose)

    if not subject_map:
        log_print("No subjects found in dataset.", True)
        return generated_files

    # Filter subjects if specified
    if subjects:
        subject_set = {s.strip() for s in subjects}
        subject_map = {sid: paths for sid, paths in subject_map.items() if sid in subject_set}

    if not subject_map:
        log_print("No subjects matched the filter criteria.", True)
        return generated_files

    allowed_conditions = {c.strip() for c in included_conditions} if included_conditions else None

    log_print(
        f"\n=== Running Sedation-RestingState sweep | mode={mode} | "
        f"repeats={repeats} | n_channels={n_channels} ===",
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

        for condition, set_path in sorted(filtered_cond_map.items()):
            log_print(f"\nProcessing: {subject_id} / {condition}", verbose)

            condition_dir = _compose_output_dir(output_dir, mode, subject_id, condition)

            if mode == "broadband":
                generated_files.append(
                    run_sedation_mib_broadband(
                        set_file=Path(set_path),
                        repeats=repeats,
                        n_channels=n_channels,
                        rng_seed=rng_seed,
                        fixed_channels=fixed_channels,
                        output_dir=condition_dir,
                        verbose=verbose,
                        n_jobs=n_jobs,
                    )
                )
            else:  # spectral
                generated_files.append(
                    run_sedation_mib_spectral(
                        set_file=Path(set_path),
                        repeats=repeats,
                        n_channels=n_channels,
                        rng_seed=rng_seed,
                        fixed_channels=fixed_channels,
                        bands=bands,
                        output_dir=condition_dir,
                        verbose=verbose,
                        n_jobs=n_jobs,
                    )
                )

    log_print(f"\nGenerated {len(generated_files)} result files.", verbose)
    return generated_files


def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="MIB Analysis for Sedation-RestingState Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all subjects, baseline vs deep_sedation
  python scripts/sedation_mib_analysis.py dataset --conditions baseline deep_sedation

  # Run spectral analysis on specific subjects
  python scripts/sedation_mib_analysis.py dataset --subjects sub-022010 sub-032010 --mode spectral

  # Run on a single file
  python scripts/sedation_mib_analysis.py single --file "Sedation-RestingState/25-2010-anest 20100422 133.003.set"
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Dataset sweep mode
    dataset_parser = subparsers.add_parser(
        "dataset",
        aliases=["d"],
        help="Run analysis across the dataset",
    )
    dataset_parser.add_argument(
        "--dataset",
        type=str,
        default=SEDATION_DATASET_DIR,
        help="Path to the Sedation-RestingState directory.",
    )
    dataset_parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subject IDs to include (e.g., sub-022010 sub-032010).",
    )
    dataset_parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["baseline", "light_sedation", "deep_sedation", "recovery"],
        default=None,
        help="Conditions to include.",
    )
    dataset_parser.add_argument(
        "--mode",
        choices=["broadband", "spectral"],
        default="spectral",
        help="Analysis mode (default: spectral).",
    )
    dataset_parser.add_argument(
        "--bands",
        nargs="+",
        choices=list(SPECTRAL_BANDS.keys()),
        default=list(SPECTRAL_BANDS.keys()),
        help="Spectral bands to include (default: all).",
    )
    dataset_parser.add_argument(
        "--n-channels",
        type=int,
        default=8,
        help="Number of channels per random draw (default: 8).",
    )
    dataset_parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of random samples per recording (default: 50).",
    )
    dataset_parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Seed for reproducibility (default: 42).",
    )
    dataset_parser.add_argument(
        "--fixed-channels",
        nargs="+",
        default=SEDATION_COMMON_CHANNELS[:8],
        help="Fixed channel list for epoch stability (default: first 8 common 10-20 channels).",
    )
    dataset_parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all cores).",
    )
    dataset_parser.add_argument(
        "--output",
        type=str,
        default=str(Path(DEFAULT_OUTPUT_DIR) / "sedation_resting_state" / "mib_sedation"),
        help="Directory to save results.",
    )
    dataset_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging.",
    )

    # Single file mode
    single_parser = subparsers.add_parser(
        "single",
        aliases=["s"],
        help="Run analysis on a single file",
    )
    single_parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the .set file.",
    )
    single_parser.add_argument(
        "--mode",
        choices=["broadband", "spectral"],
        default="spectral",
        help="Analysis mode (default: spectral).",
    )
    single_parser.add_argument(
        "--bands",
        nargs="+",
        choices=list(SPECTRAL_BANDS.keys()),
        default=list(SPECTRAL_BANDS.keys()),
        help="Spectral bands to include.",
    )
    single_parser.add_argument(
        "--n-channels",
        type=int,
        default=8,
        help="Number of channels per random draw.",
    )
    single_parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of random samples.",
    )
    single_parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )
    single_parser.add_argument(
        "--fixed-channels",
        nargs="+",
        default=SEDATION_COMMON_CHANNELS[:8],
        help="Fixed channel list for epoch stability.",
    )
    single_parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs.",
    )
    single_parser.add_argument(
        "--output",
        type=str,
        default=str(Path(DEFAULT_OUTPUT_DIR) / "sedation_resting_state" / "mib_sedation" / "single"),
        help="Directory to save results.",
    )
    single_parser.add_argument(
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
        return

    if args.command in ("dataset", "d"):
        run_sedation_dataset(
            dataset_dir=Path(args.dataset),
            subjects=args.subjects,
            repeats=args.repeats,
            n_channels=args.n_channels,
            rng_seed=args.rng_seed,
            fixed_channels=args.fixed_channels,
            mode=args.mode,
            output_dir=Path(args.output),
            verbose=not args.quiet,
            included_conditions=args.conditions,
            n_jobs=args.jobs,
            bands=args.bands,
        )
        return

    if args.command in ("single", "s"):
        set_file = Path(args.file)
        if not set_file.exists():
            raise FileNotFoundError(f"File not found: {set_file}")

        output_dir = Path(args.output)
        if args.mode == "broadband":
            run_sedation_mib_broadband(
                set_file=set_file,
                repeats=args.repeats,
                n_channels=args.n_channels,
                rng_seed=args.rng_seed,
                fixed_channels=args.fixed_channels,
                output_dir=output_dir,
                verbose=not args.quiet,
                n_jobs=args.jobs,
            )
        else:
            run_sedation_mib_spectral(
                set_file=set_file,
                repeats=args.repeats,
                n_channels=args.n_channels,
                rng_seed=args.rng_seed,
                fixed_channels=args.fixed_channels,
                bands=args.bands,
                output_dir=output_dir,
                verbose=not args.quiet,
                n_jobs=args.jobs,
            )
        return


if __name__ == "__main__":
    cli_main()
