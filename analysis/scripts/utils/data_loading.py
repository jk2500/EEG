"""Data loading utilities for MIB analysis results."""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import BANDS, COMPOSITE_BANDS


def load_mib_results(
    root_path: Path,
    glob_pattern: str = "sub-*/*/mib_*.json",
    bands: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load MIB results from JSON files matching a glob pattern.

    Parameters
    ----------
    root_path : Path
        Root directory to search for JSON files.
    glob_pattern : str
        Glob pattern to match JSON files.
    bands : list[str], optional
        List of bands to extract. Defaults to all standard bands.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject, condition, band, mean, std, cv,
        repeats, n_channels_selected, n_channels_total, source_file.
    """
    if bands is None:
        bands = BANDS

    rows = []
    for path in root_path.glob(glob_pattern):
        parts = path.parts
        # Try to extract subject and condition from path
        # Common patterns: sub-*/condition/file.json or sub-*/file.json
        subject = None
        condition = None

        for i, part in enumerate(parts):
            if part.startswith("sub-"):
                subject = part
                # Check if next part is the condition directory
                if i + 1 < len(parts) - 1:  # Not the filename
                    condition = parts[i + 1]
                break

        if subject is None:
            continue

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, IOError):
            continue

        # Handle both "bands" structure and direct band data
        bands_data = data.get("bands", {})
        if not bands_data:
            continue

        for band in bands:
            if band not in bands_data:
                continue
            band_data = bands_data[band]
            overall_stats = band_data.get("overall_stats", {})

            rows.append({
                "subject": subject,
                "condition": condition,
                "band": band,
                "mean": overall_stats.get("mean_of_per_repeat_means"),
                "std": overall_stats.get("std_of_per_repeat_means"),
                "cv": overall_stats.get("cv_of_per_repeat_means"),
                "repeats": band_data.get("repeats"),
                "n_channels_selected": band_data.get("n_channels_selected"),
                "n_channels_total": band_data.get("n_channels_total"),
                "source_file": str(path),
            })

    return pd.DataFrame(rows)


def extract_band_records(
    json_path: Path,
    bands: Optional[list[str]] = None,
) -> list[dict]:
    """
    Extract per-band records from a single JSON result file.

    Parameters
    ----------
    json_path : Path
        Path to JSON file.
    bands : list[str], optional
        List of bands to extract. Defaults to all standard bands.

    Returns
    -------
    list[dict]
        List of dictionaries with band statistics.
    """
    if bands is None:
        bands = BANDS

    data = json.loads(json_path.read_text())
    bands_data = data.get("bands", {})

    records = []
    for band in bands:
        if band not in bands_data:
            continue
        band_data = bands_data[band]
        overall_stats = band_data.get("overall_stats", {})
        records.append({
            "band": band,
            "mean": overall_stats.get("mean_of_per_repeat_means"),
            "std": overall_stats.get("std_of_per_repeat_means"),
            "cv": overall_stats.get("cv_of_per_repeat_means"),
            "repeats": band_data.get("repeats"),
            "n_channels_selected": band_data.get("n_channels_selected"),
            "n_channels_total": band_data.get("n_channels_total"),
        })

    return records


def aggregate_subject_results(
    df: pd.DataFrame,
    add_composites: bool = True,
) -> pd.DataFrame:
    """
    Pivot long-form band results to wide format with optional composite bands.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form DataFrame with columns: subject, condition, band, mean.
    add_composites : bool
        Whether to add composite band metrics (default True).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with one row per subject-condition and
        bands as columns.
    """
    wide = (
        df.pivot_table(index=["subject", "condition"], columns="band", values="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    if add_composites:
        for name, band_list in COMPOSITE_BANDS.items():
            # Only add if all constituent bands are present
            if all(b in wide.columns for b in band_list):
                wide[name] = wide[band_list].mean(axis=1)

        # Add high_over_low if both components are present
        if "high_abc" in wide.columns and "low_dt" in wide.columns:
            wide["high_over_low"] = wide["high_abc"] - wide["low_dt"]

    return wide
