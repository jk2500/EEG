#!/usr/bin/env python3
"""Aggregate MIB results from multiple datasets into summary CSVs.

Usage:
    python analysis/scripts/aggregate_mib_results.py
    python analysis/scripts/aggregate_mib_results.py --ds005620 path/to/results --sedation path/to/results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def process_results(output_dir: str, dataset_name: str) -> pd.DataFrame:
    """Process JSON results from a dataset directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return pd.DataFrame()

    json_files = list(output_path.rglob("*.json"))
    if not json_files:
        return pd.DataFrame()

    records = []
    for f in json_files:
        with open(f) as fp:
            data = json.load(fp)

        # Extract subject/condition from path
        parts = f.relative_to(output_path).parts
        subject = parts[3] if len(parts) > 3 else "unknown"
        condition = parts[4] if len(parts) > 4 else "unknown"

        base_record = {
            "dataset": dataset_name,
            "subject": subject,
            "condition": condition,
            "file": str(f),
            "mode": data.get("mode", "unknown"),
            "epoch_length": data.get("epoch_length", 0),
            "n_bins": data.get("n_bins", 10),
            "n_channels": data.get("n_channels", 8),
        }

        # Add band stats if spectral
        if "bands" in data:
            for band, stats in data["bands"].items():
                if "overall_stats" in stats:
                    record = base_record.copy()
                    record["band"] = band
                    record["mean_mib"] = stats["overall_stats"].get(
                        "mean_of_per_repeat_means", np.nan
                    )
                    record["std_mib"] = stats["overall_stats"].get(
                        "std_of_per_repeat_means", np.nan
                    )
                    record["cv_mib"] = stats["overall_stats"].get(
                        "cv_of_per_repeat_means", np.nan
                    )
                    records.append(record)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Aggregate MIB results")
    parser.add_argument(
        "--ds005620",
        default="results/ds005620/mib_analysis_optimal",
        help="Path to DS005620 results",
    )
    parser.add_argument(
        "--sedation",
        default="results/sedation_resting_state/mib_analysis_optimal",
        help="Path to Sedation results",
    )
    parser.add_argument(
        "--output",
        default="analysis/outputs/optimal_analysis",
        help="Output directory for CSVs",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process both datasets
    ds005620_df = process_results(args.ds005620, "ds005620")
    sedation_df = process_results(args.sedation, "sedation")

    # Combine
    all_df = pd.concat([ds005620_df, sedation_df], ignore_index=True)

    if all_df.empty:
        print("No results found to analyze.")
        return

    # Save all results
    all_df.to_csv(output_dir / "all_results.csv", index=False)
    print(f"Combined results: {len(all_df)} records from {all_df['dataset'].nunique()} datasets")

    # Summary statistics by dataset and condition
    summary = (
        all_df.groupby(["dataset", "condition", "band"])
        .agg({"mean_mib": ["mean", "std", "count"], "cv_mib": "mean"})
        .round(4)
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "summary_by_condition.csv", index=False)
    print("Summary by condition saved.")

    # Pivot for comparison
    pivot = all_df.pivot_table(
        index=["dataset", "subject", "band"], columns="condition", values="mean_mib"
    ).reset_index()
    pivot.to_csv(output_dir / "pivot_by_subject.csv", index=False)
    print("Subject-level pivot saved.")

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
