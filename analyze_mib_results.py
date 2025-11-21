#!/usr/bin/env python3
"""
Analyze Random-Channel MIB Output JSONs
---------------------------------------

Reads the JSON files produced by `random_channels_mib.py` and reports:
- Stability per condition (and band) via coefficient of variation (CV) of per-repeat means.
- Within-subject ordering check: eyes-open > eyes-closed > sedation_1 (per band).
- Cross-subject differences for two-subject sweeps when available.

Example:
    python analyze_mib_results.py --root results/mib_random_channels --mode broadband --estimator ksg --cv-threshold 0.15 --delta-threshold 0.05
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import math

import pandas as pd


def _infer_subject(vhdr_path: str) -> str:
    """Extract a subject ID from the VHDR path (e.g., sub-1010)."""
    match = re.search(r"(sub-[A-Za-z0-9]+)", vhdr_path)
    return match.group(1) if match else "unknown"


def _infer_condition(vhdr_path: str) -> str:
    """Mirror the repo's condition naming for awareness of eyes-open/closed and sedation."""
    name = vhdr_path.lower()
    if "task-awake" in name and "acq-eo" in name:
        return "awake_eyes_open"
    if "task-awake" in name and "acq-ec" in name:
        return "awake_eyes_closed"
    if "task-sed" in name and "acq-rest" in name:
        return "sedation_1"
    if "task-sed2" in name and "acq-rest" in name:
        return "sedation_2"
    return "unknown"


def _load_jsons(root: Path, mode: str) -> List[Path]:
    pattern = f"**/mib_random_channels_{mode}_*.json"
    return sorted(root.glob(pattern))


def _extract_broadband_records(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)

    subject = _infer_subject(payload.get("vhdr_path", ""))
    condition = _infer_condition(payload.get("vhdr_path", ""))
    overall = payload.get("overall_stats", {})
    repeats = payload.get("repeats", 0)
    return [
        {
            "file": str(path),
            "band": "broadband",
            "subject": subject,
            "condition": condition,
            "epoch_length": float(payload.get("epoch_length", 0.0)),
            "estimator": payload.get("estimator", "unknown"),
            "mean": float(overall.get("mean_of_per_repeat_means", 0.0)),
            "std": float(overall.get("std_of_per_repeat_means", 0.0)),
            "cv": float(overall.get("cv_of_per_repeat_means", float("inf"))),
            "repeats": int(repeats),
        }
    ]


def _extract_spectral_records(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)

    subject = _infer_subject(payload.get("vhdr_path", ""))
    condition = _infer_condition(payload.get("vhdr_path", ""))
    epoch_length = float(payload.get("epoch_length", 0.0))
    estimator = payload.get("estimator", "unknown")

    records: List[Dict[str, Any]] = []
    for band, band_payload in payload.get("bands", {}).items():
        overall = band_payload.get("overall_stats", {})
        repeats = band_payload.get("repeats", 0)
        records.append(
            {
                "file": str(path),
                "band": band,
                "subject": subject,
                "condition": condition,
                "epoch_length": epoch_length,
                "estimator": estimator,
                "mean": float(overall.get("mean_of_per_repeat_means", 0.0)),
                "std": float(overall.get("std_of_per_repeat_means", 0.0)),
                "cv": float(overall.get("cv_of_per_repeat_means", float("inf"))),
                "repeats": int(repeats),
            }
        )
    return records


def _summarize(
    df: pd.DataFrame,
    cv_threshold: float,
    delta_threshold: float,
) -> None:
    if df.empty:
        print("No records matched the filters.")
        return

    df = df.copy()
    df["stable"] = (df["cv"] <= cv_threshold) & (df["repeats"] > 0)

    keep_cols = ["band", "subject", "condition", "mean", "std", "cv", "repeats", "stable", "file"]
    print("\nPer-condition stability (lower CV is better):")
    print(df[keep_cols].sort_values(by=["band", "condition", "subject"]).to_string(index=False, float_format="%.4f"))

    # Within-subject ordering: eyes-open > eyes-closed > sedation_1
    expected_order = ["awake_eyes_open", "awake_eyes_closed", "sedation_1"]
    order_rows: List[Dict[str, Any]] = []
    for subject in sorted(df["subject"].unique()):
        for band in sorted(df["band"].unique()):
            sub = df[(df["subject"] == subject) & (df["band"] == band)]
            means = {}
            for cond in expected_order:
                cond_values = sub.loc[sub["condition"] == cond, "mean"]
                if not cond_values.empty:
                    means[cond] = float(cond_values.iloc[0])

            open_mean = means.get("awake_eyes_open", math.nan)
            closed_mean = means.get("awake_eyes_closed", math.nan)
            sed_mean = means.get("sedation_1", math.nan)
            complete = all(math.isfinite(x) for x in [open_mean, closed_mean, sed_mean])
            delta_oc = open_mean - closed_mean if complete else math.nan
            delta_cs = closed_mean - sed_mean if complete else math.nan
            min_delta = min(delta_oc, delta_cs) if complete else math.nan
            order_ok = complete and (open_mean > closed_mean > sed_mean)
            order_rows.append(
                {
                    "band": band,
                    "subject": subject,
                    "eyes_open": open_mean,
                    "eyes_closed": closed_mean,
                    "sedation_1": sed_mean,
                    "delta_open_closed": delta_oc,
                    "delta_closed_sedation": delta_cs,
                    "min_delta": min_delta,
                    "order_ok": order_ok,
                }
            )

    if order_rows:
        order_df = pd.DataFrame(order_rows)
        print("\nWithin-subject ordering (expect eyes_open > eyes_closed > sedation_1):")
        print(
            order_df.sort_values(by=["band", "subject"])
            .to_string(index=False, float_format="%.4f")
        )

        # Band-level separation score: average min_delta where ordering holds
        valid_band_rows = order_df[order_df["order_ok"] & order_df["min_delta"].notna()]
        if not valid_band_rows.empty:
            band_strength = (
                valid_band_rows.groupby("band")["min_delta"]
                .mean()
                .reset_index()
                .rename(columns={"min_delta": "mean_min_delta"})
                .sort_values("mean_min_delta", ascending=False)
            )
            print("\nBand separation strength (higher mean_min_delta = clearer drop):")
            print(band_strength.to_string(index=False, float_format="%.4f"))
        else:
            print("\nBand separation strength skipped (no bands met ordering criteria).")

    unique_subjects = sorted(df["subject"].unique())
    if len(unique_subjects) == 2:
        pivot = (
            df.pivot_table(
                index=["band", "condition"],
                columns="subject",
                values="mean",
                aggfunc="first",
            )
            .reset_index()
        )
        subj_a, subj_b = unique_subjects
        pivot["abs_delta"] = (pivot[subj_a] - pivot[subj_b]).abs()
        pivot["diff_flag"] = pivot["abs_delta"] >= delta_threshold
        print("\nCross-subject differences (absolute mean delta):")
        print(
            pivot[["band", "condition", subj_a, subj_b, "abs_delta", "diff_flag"]]
            .sort_values(by=["band", "condition"])
            .to_string(index=False, float_format="%.4f")
        )
    else:
        print("\nCross-subject difference table skipped (need exactly two subjects).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize random-channel MIB JSON outputs.")
    parser.add_argument("--root", type=str, default="results/mib_random_channels", help="Root directory containing JSON outputs.")
    parser.add_argument("--mode", choices=["broadband", "spectral"], default="broadband", help="Which output mode to summarize.")
    parser.add_argument("--estimator", type=str, default=None, help="Optional estimator filter (e.g., ksg).")
    parser.add_argument("--epoch-length", type=float, default=None, help="Optional epoch length filter (seconds).")
    parser.add_argument("--cv-threshold", type=float, default=0.15, help="CV threshold for calling a result stable.")
    parser.add_argument("--delta-threshold", type=float, default=0.1, help="Absolute mean difference threshold between two subjects.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    json_paths = _load_jsons(root, args.mode)
    if not json_paths:
        print(f"No JSON files found under {root} for mode={args.mode}.")
        return

    records: List[Dict[str, Any]] = []
    for path in json_paths:
        if args.mode == "broadband":
            records.extend(_extract_broadband_records(path))
        else:
            records.extend(_extract_spectral_records(path))

    df: pd.DataFrame = pd.DataFrame(records)
    if args.estimator:
        df = df[df["estimator"] == args.estimator]
    if args.epoch_length is not None:
        df = df[df["epoch_length"].round(3) == round(args.epoch_length, 3)]

    _summarize(df, args.cv_threshold, args.delta_threshold)


if __name__ == "__main__":
    main()
