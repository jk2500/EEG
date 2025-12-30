#!/usr/bin/env python3
"""
Unified MIB Analysis Script
===========================

This script runs Minimum Information Bipartition (MIB) analysis on EEG datasets.
Supports multiple dataset formats through a unified interface.

Supported Datasets:
- ds005620: BrainVision format (.vhdr) - propofol sedation study
- sedation: EEGLAB format (.set/.fdt) - Sedation-RestingState dataset

Usage:
    # DS005620 dataset
    python scripts/mib_analysis.py ds005620 dataset --subjects sub-1010
    python scripts/mib_analysis.py ds005620 single --file path/to/file.vhdr

    # Sedation-RestingState dataset
    python scripts/mib_analysis.py sedation dataset --conditions baseline deep_sedation
    python scripts/mib_analysis.py sedation single --file path/to/file.set
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import mne

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eeg_analysis.config import (
    DEFAULT_OUTPUT_DIR,
    SPECTRAL_BANDS,
)
from eeg_analysis.utils import log_print

# Import shared MIB core utilities
from utils.mib_core import (
    build_mib_analyzer as _build_analyzer,
    compute_random_channel_stats as _compute_random_channel_stats,
    compute_epoch_stability as _compute_epoch_stability,
)

mne.set_log_level("WARNING")


# =============================================================================
# Dataset-specific loaders
# =============================================================================

def _load_ds005620_data(
    file_path: Path,
    bands: Optional[List[str]],
    use_all_channels: bool,
    verbose: bool,
) -> tuple[Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    """Load DS005620 BrainVision data."""
    from eeg_analysis.config import DS005620_DATASET_DIR, DS005620_COMMON_CHANNELS
    from eeg_analysis.loaders.ds005620 import (
        load_ds005620_epochs,
        preprocess_ds005620_epochs,
        preprocess_ds005620_epochs_by_bands,
    )

    epochs, meta = load_ds005620_epochs(str(file_path), verbose=verbose)
    epoch_length = meta['epoch_duration']

    if bands:
        band_data, channel_names = preprocess_ds005620_epochs_by_bands(
            epochs,
            bands=bands,
            use_all_channels=use_all_channels,
            verbose=False,
        )
    else:
        epochs_data, channel_names = preprocess_ds005620_epochs(
            epochs,
            use_all_channels=use_all_channels,
            verbose=False,
        )
        band_data = {"broadband": epochs_data}

    return band_data, meta, channel_names, {
        "default_channels": DS005620_COMMON_CHANNELS[:8],
        "dataset_dir": DS005620_DATASET_DIR,
    }


def _load_sedation_data(
    file_path: Path,
    bands: Optional[List[str]],
    use_all_channels: bool,
    verbose: bool,
) -> tuple[Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    """Load Sedation-RestingState EEGLAB data."""
    from eeg_analysis.config import SEDATION_DATASET_DIR
    from eeg_analysis.loaders.sedation import (
        load_sedation_epochs,
        preprocess_sedation_epochs,
        preprocess_sedation_epochs_by_bands,
        SEDATION_COMMON_CHANNELS,
    )

    epochs, meta = load_sedation_epochs(str(file_path), verbose=verbose)
    epoch_length = meta['epoch_duration']

    if bands:
        band_data, channel_names = preprocess_sedation_epochs_by_bands(
            epochs,
            bands=bands,
            use_all_channels=use_all_channels,
            verbose=False,
        )
    else:
        epochs_data, channel_names = preprocess_sedation_epochs(
            epochs,
            use_all_channels=use_all_channels,
            verbose=False,
        )
        band_data = {"broadband": epochs_data}

    return band_data, meta, channel_names, {
        "default_channels": SEDATION_COMMON_CHANNELS[:8],
        "dataset_dir": SEDATION_DATASET_DIR,
    }


DATASET_LOADERS = {
    "ds005620": _load_ds005620_data,
    "sedation": _load_sedation_data,
}


# =============================================================================
# Core analysis functions
# =============================================================================

def run_mib_spectral(
    *,
    dataset: str,
    file_path: Path,
    repeats: int,
    n_channels: int,
    rng_seed: int,
    fixed_channels: Optional[List[str]],
    bands: Optional[List[str]],
    output_dir: Path,
    verbose: bool,
    n_jobs: Optional[int],
) -> Path:
    """Run spectral MIB analysis on a single file."""
    loader = DATASET_LOADERS[dataset]
    band_list = bands or list(SPECTRAL_BANDS.keys())

    band_data, meta, channel_names, dataset_info = loader(
        file_path,
        bands=band_list,
        use_all_channels=True,
        verbose=verbose,
    )

    epoch_length = meta['epoch_duration']
    analyzer = _build_analyzer(n_channels, epoch_length, verbose=False, n_jobs=n_jobs)

    if fixed_channels is None:
        fixed_channels = dataset_info["default_channels"]

    log_print(f"Processing spectral bands: {band_list}", verbose)

    per_band_results: Dict[str, Any] = {}
    for band_name, epochs_data in band_data.items():
        if epochs_data is None:
            continue

        # Compute fixed-channel epoch stability
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
    out_json = output_dir / f"mib_spectral_{timestamp}.json"
    payload: Dict[str, Any] = {
        "mode": "spectral",
        "dataset": dataset,
        "file_path": str(file_path),
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


def run_mib_broadband(
    *,
    dataset: str,
    file_path: Path,
    repeats: int,
    n_channels: int,
    rng_seed: int,
    fixed_channels: Optional[List[str]],
    output_dir: Path,
    verbose: bool,
    n_jobs: Optional[int],
) -> Path:
    """Run broadband MIB analysis on a single file."""
    loader = DATASET_LOADERS[dataset]

    band_data, meta, channel_names, dataset_info = loader(
        file_path,
        bands=None,
        use_all_channels=True,
        verbose=verbose,
    )

    epochs_data = band_data["broadband"]
    epoch_length = meta['epoch_duration']
    analyzer = _build_analyzer(n_channels, epoch_length, verbose=False, n_jobs=n_jobs)

    if fixed_channels is None:
        fixed_channels = dataset_info["default_channels"]

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
    out_json = output_dir / f"mib_broadband_{timestamp}.json"
    payload: Dict[str, Any] = {
        "mode": "broadband",
        "dataset": dataset,
        "file_path": str(file_path),
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


# =============================================================================
# Dataset sweep functions
# =============================================================================

def _get_ds005620_file_map(dataset_dir: str, verbose: bool) -> Dict[str, Dict[str, str]]:
    """Get file map for DS005620 dataset."""
    from eeg_analysis.loaders.ds005620 import create_ds005620_subject_file_map
    return create_ds005620_subject_file_map(dataset_dir, verbose=verbose)


def _get_sedation_file_map(dataset_dir: str, verbose: bool) -> Dict[str, Dict[str, str]]:
    """Get file map for Sedation-RestingState dataset."""
    from eeg_analysis.loaders.sedation import create_sedation_subject_file_map
    return create_sedation_subject_file_map(dataset_dir, verbose=verbose)


DATASET_FILE_MAPS = {
    "ds005620": _get_ds005620_file_map,
    "sedation": _get_sedation_file_map,
}


def run_dataset_sweep(
    *,
    dataset: str,
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
    """Run MIB analysis across a dataset."""
    generated_files: List[Path] = []
    file_map_fn = DATASET_FILE_MAPS[dataset]
    subject_map = file_map_fn(str(dataset_dir), verbose=verbose)

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
        f"\n=== Running {dataset} sweep | mode={mode} | "
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

        for condition, file_path in sorted(filtered_cond_map.items()):
            log_print(f"\nProcessing: {subject_id} / {condition}", verbose)

            condition_dir = output_dir / dataset / mode / "binning" / subject_id / condition

            if mode == "broadband":
                generated_files.append(
                    run_mib_broadband(
                        dataset=dataset,
                        file_path=Path(file_path),
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
                    run_mib_spectral(
                        dataset=dataset,
                        file_path=Path(file_path),
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


# =============================================================================
# CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified MIB Analysis for EEG Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # DS005620 dataset
  python scripts/mib_analysis.py ds005620 dataset --subjects sub-1010 sub-1019
  python scripts/mib_analysis.py ds005620 single --file ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr

  # Sedation-RestingState dataset
  python scripts/mib_analysis.py sedation dataset --conditions baseline deep_sedation
  python scripts/mib_analysis.py sedation single --file "Sedation-RestingState/25-2010-anest 20100422 133.003.set"
""",
    )

    # Dataset subparsers
    dataset_subparsers = parser.add_subparsers(dest="dataset_type", help="Dataset type")

    for ds_name in ["ds005620", "sedation"]:
        ds_parser = dataset_subparsers.add_parser(ds_name, help=f"Analyze {ds_name} dataset")
        mode_subparsers = ds_parser.add_subparsers(dest="command", help="Available commands")

        # Dataset sweep mode
        sweep_parser = mode_subparsers.add_parser("dataset", aliases=["d"], help="Run analysis across the dataset")
        sweep_parser.add_argument("--dataset-dir", type=str, default=None, help="Path to the dataset directory.")
        sweep_parser.add_argument("--subjects", nargs="+", default=None, help="Subject IDs to include.")
        sweep_parser.add_argument("--conditions", nargs="+", default=None, help="Conditions to include.")
        sweep_parser.add_argument("--mode", choices=["broadband", "spectral"], default="spectral", help="Analysis mode.")
        sweep_parser.add_argument("--bands", nargs="+", choices=list(SPECTRAL_BANDS.keys()), default=None, help="Spectral bands.")
        sweep_parser.add_argument("--n-channels", type=int, default=8, help="Number of channels per random draw.")
        sweep_parser.add_argument("--repeats", type=int, default=50, help="Number of random samples.")
        sweep_parser.add_argument("--rng-seed", type=int, default=42, help="Seed for reproducibility.")
        sweep_parser.add_argument("--fixed-channels", nargs="+", default=None, help="Fixed channel list for epoch stability.")
        sweep_parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs.")
        sweep_parser.add_argument("--output", type=str, default=None, help="Directory to save results.")
        sweep_parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging.")

        # Single file mode
        single_parser = mode_subparsers.add_parser("single", aliases=["s"], help="Run analysis on a single file")
        single_parser.add_argument("--file", type=str, required=True, help="Path to the EEG file.")
        single_parser.add_argument("--mode", choices=["broadband", "spectral"], default="spectral", help="Analysis mode.")
        single_parser.add_argument("--bands", nargs="+", choices=list(SPECTRAL_BANDS.keys()), default=None, help="Spectral bands.")
        single_parser.add_argument("--n-channels", type=int, default=8, help="Number of channels per random draw.")
        single_parser.add_argument("--repeats", type=int, default=50, help="Number of random samples.")
        single_parser.add_argument("--rng-seed", type=int, default=42, help="Seed for reproducibility.")
        single_parser.add_argument("--fixed-channels", nargs="+", default=None, help="Fixed channel list for epoch stability.")
        single_parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs.")
        single_parser.add_argument("--output", type=str, default=None, help="Directory to save results.")
        single_parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging.")

    return parser


def cli_main() -> None:
    """Main CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.dataset_type is None:
        parser.print_help()
        return

    dataset = args.dataset_type

    # Set default paths based on dataset
    if dataset == "ds005620":
        from eeg_analysis.config import DS005620_DATASET_DIR
        default_dataset_dir = DS005620_DATASET_DIR
        default_output_dir = Path(DEFAULT_OUTPUT_DIR) / "ds005620" / "mib_analysis"
    else:
        from eeg_analysis.config import SEDATION_DATASET_DIR
        default_dataset_dir = SEDATION_DATASET_DIR
        default_output_dir = Path(DEFAULT_OUTPUT_DIR) / "sedation_resting_state" / "mib_analysis"

    if args.command in ("dataset", "d"):
        dataset_dir = Path(args.dataset_dir) if args.dataset_dir else Path(default_dataset_dir)
        output_dir = Path(args.output) if args.output else default_output_dir

        run_dataset_sweep(
            dataset=dataset,
            dataset_dir=dataset_dir,
            subjects=args.subjects,
            repeats=args.repeats,
            n_channels=args.n_channels,
            rng_seed=args.rng_seed,
            fixed_channels=args.fixed_channels,
            mode=args.mode,
            output_dir=output_dir,
            verbose=not args.quiet,
            included_conditions=args.conditions,
            n_jobs=args.jobs,
            bands=args.bands or list(SPECTRAL_BANDS.keys()),
        )
        return

    if args.command in ("single", "s"):
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        output_dir = Path(args.output) if args.output else default_output_dir / "single"

        if args.mode == "broadband":
            run_mib_broadband(
                dataset=dataset,
                file_path=file_path,
                repeats=args.repeats,
                n_channels=args.n_channels,
                rng_seed=args.rng_seed,
                fixed_channels=args.fixed_channels,
                output_dir=output_dir,
                verbose=not args.quiet,
                n_jobs=args.jobs,
            )
        else:
            run_mib_spectral(
                dataset=dataset,
                file_path=file_path,
                repeats=args.repeats,
                n_channels=args.n_channels,
                rng_seed=args.rng_seed,
                fixed_channels=args.fixed_channels,
                bands=args.bands or list(SPECTRAL_BANDS.keys()),
                output_dir=output_dir,
                verbose=not args.quiet,
                n_jobs=args.jobs,
            )
        return


if __name__ == "__main__":
    cli_main()
