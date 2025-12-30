"""
Core MIB computation utilities shared across analysis scripts.

Key functions:
- build_mib_analyzer(): Create ComplexityAnalyzer with binning estimator
- compute_random_channel_stats(): Sample N random channel subsets, compute MIB
- compute_epoch_stability(): Fixed-channel baseline (isolate temporal variance)
- select_channel_indices(): Prefer named channels, fallback to first-N
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

# Add src to path for imports if package not installed
# After running `pip install -e .`, this is unnecessary
try:
    from eeg_analysis.config import ANALYSIS_PARAMS, BINNING_PARAMS
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    SRC_DIR = REPO_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from eeg_analysis.config import ANALYSIS_PARAMS, BINNING_PARAMS

from eeg_analysis.analyzers.complexity_analyzer import ComplexityAnalyzer
from eeg_analysis.analyzers.estimators import BinningEstimator
from eeg_analysis.utils import log_print

if TYPE_CHECKING:
    from typing import Any, Sequence


def build_mib_analyzer(
    n_channels: int,
    epoch_length: float,
    verbose: bool = False,
    n_jobs: Optional[int] = None,
) -> ComplexityAnalyzer:
    """Create analyzer with BinningEstimator. Uses config defaults + overrides."""
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


def select_channel_indices(
    all_channel_names: Sequence[str],
    n_channels: int,
    preferred: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Select n_channels from available. Prefers named channels, falls back to first-N."""
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
            f"Available: {list(all_channel_names)}"
        )

    indices = np.array([list(all_channel_names).index(ch) for ch in selected], dtype=int)
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
    """Compute MIB for one random channel subset. Returns dict with mean, std, channels."""
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


def aggregate_repeat_results(
    repeats_payload: List[Dict[str, Any]],
    repeats_requested: int,
    n_channels: int,
    n_all_channels: int,
) -> Dict[str, Any]:
    """Aggregate multiple random samples into summary stats (mean, std, CV)."""
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


def compute_random_channel_stats(
    *,
    epochs_data: np.ndarray,
    channel_names: List[str],
    analyzer: ComplexityAnalyzer,
    repeats: int,
    n_channels: int,
    rng_seed: int,
    progress_prefix: str = "",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run MIB on `repeats` random channel subsets and aggregate results."""
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

    return aggregate_repeat_results(repeats_payload, repeats, n_channels, n_all_channels)


def compute_epoch_stability(
    *,
    epochs_data: np.ndarray,
    channel_names: List[str],
    analyzer: ComplexityAnalyzer,
    n_channels: int,
    fixed_channels: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Fixed-channel baseline: MIB variability across epochs (no channel sampling).
    Isolates temporal variance from channel selection variance.
    """
    indices, selected = select_channel_indices(channel_names, n_channels, fixed_channels)
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
