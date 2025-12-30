#!/usr/bin/env python3
"""EDA for all subjects MIB results.

Supports different epoch lengths via CLI argument.

Usage:
    python analysis/scripts/eda_results_all_subjects.py          # Default 5s epochs
    python analysis/scripts/eda_results_all_subjects.py --epoch 10
    python analysis/scripts/eda_results_all_subjects.py --epoch 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    summarize_diff,
    COMPOSITE_BANDS,
    load_mib_results,
    aggregate_subject_results,
    format_epoch_path,
)


def run_eda(epoch_length: float) -> None:
    """Run EDA analysis for the specified epoch length."""
    epoch_str = format_epoch_path(epoch_length)
    epoch_label = f"{int(epoch_length)}s"

    root = Path(f"results/ds005620/mib_random_channels/spectral/binning/epoch-{epoch_str}")
    out_dir = Path(f"analysis/outputs/ds005620/all_subjects_results_epoch{int(epoch_length)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_mib_results(
        root,
        glob_pattern="sub-*/*/mib_random_channels_spectral_binning_*.json",
    )
    if df.empty:
        raise SystemExit(f"No results found under {root}")

    df.to_csv(out_dir / "band_means.csv", index=False)

    wide = aggregate_subject_results(df, add_composites=True)
    wide.to_csv(out_dir / "composites.csv", index=False)

    metrics = list(df["band"].unique()) + list(COMPOSITE_BANDS.keys()) + ["high_over_low"]
    comparisons = []

    for metric in metrics:
        if metric not in wide.columns:
            continue
        pivot = wide.pivot(index="subject", columns="condition", values=metric)
        subset = pivot.dropna(
            subset=["sedation_1", "awake_eyes_open", "awake_eyes_closed"]
        )
        if subset.empty:
            continue

        sed = subset["sedation_1"]
        eo = subset["awake_eyes_open"]
        ec = subset["awake_eyes_closed"]
        awake_avg = (eo + ec) / 2

        for label, awake in [
            ("awake_eyes_open", eo),
            ("awake_eyes_closed", ec),
            ("awake_avg", awake_avg),
        ]:
            diff = (awake - sed).to_numpy()
            stats = summarize_diff(diff)
            comparisons.append({"metric": metric, "awake_label": label, **stats})

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(out_dir / "awake_vs_sedation.csv", index=False)

    band_list = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]
    comp_band = comp_df[
        (comp_df["metric"].isin(band_list)) & (comp_df["awake_label"] == "awake_avg")
    ].set_index("metric")

    plt.figure(figsize=(9, 5))
    x = np.arange(len(band_list))
    means = [comp_band.loc[b, "mean_diff"] for b in band_list]
    ci_low = [comp_band.loc[b, "ci_low"] for b in band_list]
    ci_high = [comp_band.loc[b, "ci_high"] for b in band_list]
    yerr = [
        [m - l for m, l in zip(means, ci_low)],
        [h - m for h, m in zip(ci_high, means)],
    ]
    plt.bar(x, means, color="#4c72b0")
    plt.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#333333", capsize=4)
    plt.axhline(0, color="#666666", linewidth=1)
    plt.xticks(x, band_list)
    plt.ylabel("Awake_avg - Sedation (mean diff)")
    plt.title(f"Epoch {epoch_label}: Awake_avg vs Sedation by Band (95% CI)")
    plt.tight_layout()
    plt.savefig(out_dir / "awake_avg_vs_sedation_by_band.png", dpi=150)
    plt.close()

    comp_cond = comp_df[comp_df["metric"].isin(band_list)]
    labels = ["awake_eyes_open", "awake_eyes_closed", "awake_avg"]
    width = 0.25
    plt.figure(figsize=(11, 5))
    for idx, label in enumerate(labels):
        sub = comp_cond[comp_cond["awake_label"] == label].set_index("metric")
        means = [sub.loc[b, "mean_diff"] for b in band_list]
        plt.bar(x + idx * width, means, width=width, label=label)
    plt.axhline(0, color="#666666", linewidth=1)
    plt.xticks(x + width, band_list)
    plt.ylabel("Awake - Sedation (mean diff)")
    plt.title(f"Epoch {epoch_label}: Awake vs Sedation by Band")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "awake_vs_sedation_by_band.png", dpi=150)
    plt.close()

    comp_metric = comp_df[
        comp_df["metric"].isin(
            ["high_abc", "high_bg", "mid_alpha_beta", "gamma_only", "alpha_only", "high_over_low"]
        )
        & (comp_df["awake_label"] == "awake_avg")
    ].set_index("metric")
    metric_order = [
        "high_abc",
        "high_bg",
        "mid_alpha_beta",
        "alpha_only",
        "gamma_only",
        "high_over_low",
    ]
    plt.figure(figsize=(10, 5))
    x2 = np.arange(len(metric_order))
    means = [comp_metric.loc[m, "mean_diff"] for m in metric_order]
    plt.bar(x2, means, color="#55a868")
    plt.axhline(0, color="#666666", linewidth=1)
    plt.xticks(x2, metric_order, rotation=20, ha="right")
    plt.ylabel("Awake_avg - Sedation (mean diff)")
    plt.title(f"Epoch {epoch_label}: Composite Metrics (Awake_avg vs Sedation)")
    plt.tight_layout()
    plt.savefig(out_dir / "awake_avg_vs_sedation_composites.png", dpi=150)
    plt.close()

    print(f"EDA complete for epoch length {epoch_label}. Results saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EDA on all subjects MIB results for a given epoch length."
    )
    parser.add_argument(
        "--epoch",
        type=float,
        default=5.0,
        help="Epoch length in seconds (default: 5.0)",
    )
    args = parser.parse_args()
    run_eda(args.epoch)


if __name__ == "__main__":
    main()
