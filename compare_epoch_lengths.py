#!/usr/bin/env python3
"""
Compare MIB Stability Across Different Epoch Lengths
=====================================================

This script tests how MIB variability changes with different epoch lengths.
Longer epochs should capture more stable neural dynamics and reduce temporal variability.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import mne
import numpy as np
from tqdm import tqdm

# Ensure local src is importable
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Project imports
from eeg_analysis.config import (  # type: ignore
    ANALYSIS_PARAMS,
    KSG_PARAMS,
    DEFAULT_FILE_PATHS,
    DEFAULT_OUTPUT_DIR,
)
from eeg_analysis.analyzers.complexity_analyzer import ComplexityAnalyzer  # type: ignore
from eeg_analysis.analyzers.estimators import KSGEstimator  # type: ignore
from eeg_analysis.eeg_utils import log_print  # type: ignore

mne.set_log_level('WARNING')


def load_and_preprocess_full_eeg(
    vhdr_path: Path,
    epoch_length: float,
    target_sfreq: float = 500.0,
    reference_channel: str = "A2",
    n_channels: int = 8,
    verbose: bool = True,
) -> tuple[np.ndarray, List[str]]:
    """Load and preprocess EEG with specified epoch length."""
    log_print(f"  Loading with {epoch_length}s epochs...", verbose)
    
    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)
    
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    
    if reference_channel and reference_channel in raw.ch_names:
        try:
            raw.set_eeg_reference(ref_channels=reference_channel, verbose=False)
        except Exception:
            raw.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)
    else:
        raw.set_eeg_reference('average', projection=True, verbose=False).apply_proj(verbose=False)
    
    # Exclude specified auxiliary channels (e.g., VEOG/HEOG/EMG) if present
    exclude = ANALYSIS_PARAMS.get("exclude_channels", [])
    if exclude:
        to_drop = [ch for ch in exclude if ch in raw.ch_names]
        if len(to_drop) > 0:
            raw.drop_channels(to_drop)
    
    # Pick EEG channels
    eeg_channels = mne.pick_types(raw.info, eeg=True)
    raw.pick([raw.ch_names[i] for i in eeg_channels])
    
    # Select first n_channels for consistency
    if len(raw.ch_names) > n_channels:
        raw.pick(raw.ch_names[:n_channels])
    
    channel_names = raw.ch_names.copy()
    
    raw.filter(l_freq=1, h_freq=40, fir_design='firwin', verbose=False)
    
    subsample_factor = ANALYSIS_PARAMS.get('subsample_factor_broadband', 10)
    if subsample_factor > 1:
        raw.resample(raw.info['sfreq'] / subsample_factor, verbose=False)
    
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_length, preload=True, verbose=False)
    epochs_data = epochs.get_data()
    
    # Normalize per-epoch
    for i in range(epochs_data.shape[0]):
        epoch = epochs_data[i]
        mean = np.mean(epoch, axis=1, keepdims=True)
        std = np.std(epoch, axis=1, keepdims=True)
        std[std == 0] = 1
        epochs_data[i] = (epoch - mean) / std
    
    return epochs_data, channel_names


def analyze_epoch_length(
    vhdr_path: Path,
    epoch_length: float,
    n_channels: int,
    estimator,
    analyzer,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Analyze MIB for a specific epoch length."""
    
    epochs_data, channel_names = load_and_preprocess_full_eeg(
        vhdr_path,
        epoch_length=epoch_length,
        target_sfreq=ANALYSIS_PARAMS.get('target_sfreq', 500.0),
        reference_channel=ANALYSIS_PARAMS.get('reference_channel', 'A2'),
        n_channels=n_channels,
        verbose=verbose,
    )
    
    n_epochs = epochs_data.shape[0]
    
    # Evaluate MIB on all epochs
    metric_values = analyzer._evaluate_epochs(
        epochs_data,
        progress_label=f"Epoch length {epoch_length}s",
        verbose_override=False,
    )
    metric_values = [v for v in metric_values if v is not None]
    
    if not metric_values:
        return None
    
    mean_mib = float(np.mean(metric_values))
    std_mib = float(np.std(metric_values))
    cv_mib = float(std_mib / abs(mean_mib)) if abs(mean_mib) > 1e-10 else float('inf')
    
    return {
        "epoch_length": float(epoch_length),
        "n_epochs": int(n_epochs),
        "n_channels": int(n_channels),
        "channels": channel_names,
        "mean_mib": mean_mib,
        "std_mib": std_mib,
        "cv_mib": cv_mib,
        "min_mib": float(np.min(metric_values)),
        "max_mib": float(np.max(metric_values)),
        "metric_values": [float(x) for x in metric_values],
    }


def compare_epoch_lengths(
    vhdr_path: Path,
    epoch_lengths: List[float],
    n_channels: int,
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compare MIB stability across different epoch lengths."""
    
    log_print(f"\nComparing MIB stability across {len(epoch_lengths)} epoch lengths", verbose)
    log_print(f"File: {vhdr_path.name}", verbose)
    log_print(f"Channels: {n_channels} (first N from file)", verbose)
    log_print(f"Epoch lengths: {epoch_lengths} seconds\n", verbose)
    
    # Setup analyzer once
    base_params: Dict[str, Any] = {
        **ANALYSIS_PARAMS,
        **KSG_PARAMS,
        "n_channels": int(n_channels),
        "verbose": False,
    }
    
    estimator = KSGEstimator(**base_params)
    analyzer = ComplexityAnalyzer(estimator=estimator, **base_params)
    
    results = []
    
    for epoch_len in tqdm(epoch_lengths, desc="Testing epoch lengths", disable=not verbose):
        result = analyze_epoch_length(
            vhdr_path,
            epoch_len,
            n_channels,
            estimator,
            analyzer,
            verbose=False,
        )
        if result:
            results.append(result)
            if verbose:
                log_print(
                    f"  {epoch_len:>5.1f}s: mean={result['mean_mib']:>7.4f} ± {result['std_mib']:.4f} "
                    f"(CV={result['cv_mib']:.3f}, n_epochs={result['n_epochs']})",
                    True,
                )
    
    # Summary
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "vhdr_path": str(vhdr_path),
        "n_channels": int(n_channels),
        "epoch_lengths_tested": epoch_lengths,
        "results": results,
    }
    
    out_json = output_dir / f"epoch_length_comparison.json"
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary table
    log_print(f"\n{'='*85}", verbose)
    log_print("EPOCH LENGTH COMPARISON SUMMARY", verbose)
    log_print(f"{'='*85}", verbose)
    log_print(
        f"{'Epoch (s)':<12} {'N Epochs':<10} {'Mean MIB':<12} {'Std MIB':<12} "
        f"{'CV':<10} {'Range':<20}",
        verbose,
    )
    log_print(f"{'-'*85}", verbose)
    
    for r in results:
        range_str = f"[{r['min_mib']:.3f}, {r['max_mib']:.3f}]"
        log_print(
            f"{r['epoch_length']:<12.1f} {r['n_epochs']:<10} {r['mean_mib']:<12.4f} "
            f"{r['std_mib']:<12.4f} {r['cv_mib']:<10.3f} {range_str:<20}",
            verbose,
        )
    
    log_print(f"{'='*85}", verbose)
    log_print(f"\nResults saved to: {out_json}", verbose)
    log_print(
        f"\nInterpretation:\n"
        f"  - Lower CV (coefficient of variation) = more stable MIB across time\n"
        f"  - Longer epochs typically reduce temporal variability\n"
        f"  - However, too long epochs reduce sample size (fewer epochs)\n",
        verbose,
    )
    
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MIB temporal stability across different epoch lengths.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--vhdr",
        type=str,
        default=str(DEFAULT_FILE_PATHS.get("awake", "")),
        help="Path to a BrainVision .vhdr file.",
    )
    parser.add_argument(
        "--epoch-lengths",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 5.0, 10.0, 20.0],
        help="List of epoch lengths to test (in seconds). Default: 1 2 5 10 20",
    )
    parser.add_argument(
        "--n-channels",
        type=int,
        default=8,
        help="Number of channels to analyze (first N from file). Default: 8",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(DEFAULT_OUTPUT_DIR) / "epoch_length_comparison"),
        help="Output directory. Default: results/epoch_length_comparison",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vhdr_path = Path(args.vhdr)
    if not vhdr_path.exists():
        raise FileNotFoundError(f"VHDR file not found: {vhdr_path}")
    
    output_dir = Path(args.output)
    compare_epoch_lengths(
        vhdr_path=vhdr_path,
        epoch_lengths=args.epoch_lengths,
        n_channels=args.n_channels,
        output_dir=output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

