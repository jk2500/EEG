#!/usr/bin/env python3
"""Plot four-way sedation comparison across all frequency bands.

Combines functionality from plot_sedation_fourway_all_bands.py and
plot_sedation_fourway_band_ranges.py into a single configurable script.

Usage:
    python analysis/scripts/plot_sedation_fourway.py                    # All plots
    python analysis/scripts/plot_sedation_fourway.py --plot trajectories
    python analysis/scripts/plot_sedation_fourway.py --plot heatmap
    python analysis/scripts/plot_sedation_fourway.py --plot ranges
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

from utils import mean_ci, BANDS, SEDATION_CONDITIONS, SEDATION_CONDITION_PAIRS


def load_band_means(root: Path) -> pd.DataFrame:
    """Load band means from JSON result files."""
    rows = []
    for subject_dir in sorted(root.glob("sub-*")):
        subject = subject_dir.name
        for condition in SEDATION_CONDITIONS:
            for path in subject_dir.glob(f"{condition}/sedation_mib_spectral_*.json"):
                data = json.loads(path.read_text())
                for band in BANDS:
                    if band not in data["bands"]:
                        continue
                    mean_val = data["bands"][band]["overall_stats"]["mean_of_per_repeat_means"]
                    rows.append(
                        {
                            "subject": subject,
                            "condition": condition,
                            "band": band,
                            "mean": mean_val,
                            "source_file": str(path),
                        }
                    )
    return pd.DataFrame(rows)


def compute_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute summary statistics and pairwise comparisons."""
    summary_rows = []
    pairwise_rows = []

    for band in BANDS:
        band_df = df[df["band"] == band]
        band_wide = band_df.pivot_table(index="subject", columns="condition", values="mean")
        paired = band_wide.dropna(subset=SEDATION_CONDITIONS)
        if paired.empty:
            continue

        for condition in SEDATION_CONDITIONS:
            vals = paired[condition].to_numpy()
            ci_low, ci_high = mean_ci(vals)
            summary_rows.append(
                {
                    "band": band,
                    "condition": condition,
                    "n_subjects": vals.size,
                    "mean": vals.mean(),
                    "median": np.median(vals),
                    "std": vals.std(ddof=1),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

        for a, b in SEDATION_CONDITION_PAIRS:
            diff = paired[a] - paired[b]
            ci_low, ci_high = mean_ci(diff.to_numpy())
            t_stat, p_val = ttest_rel(paired[a], paired[b])
            pairwise_rows.append(
                {
                    "band": band,
                    "pair": f"{a} - {b}",
                    "n_subjects": diff.size,
                    "mean_diff": diff.mean(),
                    "median_diff": np.median(diff),
                    "std_diff": diff.std(ddof=1),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "t_stat": t_stat,
                    "t_p": p_val,
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(pairwise_rows)


def plot_trajectories(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot subject trajectories per band."""
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7), sharey=False)
    axes = axes.flatten()
    x = np.arange(len(SEDATION_CONDITIONS))

    for ax, band in zip(axes, BANDS):
        band_df = df[df["band"] == band]
        band_wide = band_df.pivot_table(index="subject", columns="condition", values="mean")
        paired = band_wide.dropna(subset=SEDATION_CONDITIONS)
        if paired.empty:
            ax.set_axis_off()
            continue

        vals = paired[SEDATION_CONDITIONS].to_numpy()
        for row in vals:
            ax.plot(x, row, color="#999999", linewidth=1, alpha=0.6)
            ax.scatter(x, row, color="#777777", s=18, alpha=0.7)

        means = [paired[c].mean() for c in SEDATION_CONDITIONS]
        ci_lows = [mean_ci(paired[c].to_numpy())[0] for c in SEDATION_CONDITIONS]
        ci_highs = [mean_ci(paired[c].to_numpy())[1] for c in SEDATION_CONDITIONS]
        yerr = [
            [m - l for m, l in zip(means, ci_lows)],
            [h - m for h, m in zip(ci_highs, means)],
        ]
        ax.errorbar(x, means, yerr=yerr, fmt="o", color="#000000", capsize=4)

        ax.set_title(f"{band} (n={paired.shape[0]})")
        ax.set_xticks(x)
        ax.set_xticklabels(SEDATION_CONDITIONS, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes[len(BANDS):]:
        ax.set_axis_off()

    fig.suptitle("All Bands: baseline vs light vs deep vs recovery", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fourway_all_bands.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'fourway_all_bands.png'}")


def plot_heatmap(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Plot heatmap of mean values by band/condition."""
    heat = summary_df.pivot(index="band", columns="condition", values="mean").reindex(BANDS)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(heat.values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(SEDATION_CONDITIONS)))
    ax.set_xticklabels(SEDATION_CONDITIONS, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(BANDS)))
    ax.set_yticklabels(BANDS)
    for i in range(len(BANDS)):
        for j in range(len(SEDATION_CONDITIONS)):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_title("Mean MIB by Band and Condition")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "fourway_band_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'fourway_band_heatmap.png'}")


def plot_ranges(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot mean with min-max ranges per band."""
    summary = (
        df.groupby(["band", "condition"])["mean"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    summary.to_csv(out_dir / "fourway_band_mean_range.csv", index=False)

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7), sharey=False)
    axes = axes.flatten()
    x = np.arange(len(SEDATION_CONDITIONS))

    for ax, band in zip(axes, BANDS):
        band_df = summary[summary["band"] == band].set_index("condition").reindex(SEDATION_CONDITIONS)
        means = band_df["mean"].to_numpy()
        mins = band_df["min"].to_numpy()
        maxs = band_df["max"].to_numpy()

        yerr = np.vstack([means - mins, maxs - means])
        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="-o",
            color="#0072b2",
            ecolor="#999999",
            capsize=4,
            linewidth=1.5,
        )
        ax.set_title(band)
        ax.set_xticks(x)
        ax.set_xticklabels(SEDATION_CONDITIONS, rotation=20, ha="right")
        ax.set_ylabel("Mean MIB (min-max)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes[len(BANDS):]:
        ax.set_axis_off()

    fig.suptitle("Sedation Dataset: Mean +/- Range by Band and Condition", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fourway_band_mean_range.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'fourway_band_mean_range.png'}")


def run_analysis(root: Path, out_dir: Path, plot_types: list[str]) -> None:
    """Run the full analysis pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_band_means(root)
    if df.empty:
        raise SystemExit(f"No results found under {root}")

    df.to_csv(out_dir / "fourway_band_means_raw.csv", index=False)
    print(f"Saved: {out_dir / 'fourway_band_means_raw.csv'}")

    summary_df, pairwise_df = compute_summaries(df)
    summary_df.to_csv(out_dir / "fourway_band_summary.csv", index=False)
    pairwise_df.to_csv(out_dir / "fourway_band_pairwise.csv", index=False)
    print(f"Saved: {out_dir / 'fourway_band_summary.csv'}")
    print(f"Saved: {out_dir / 'fourway_band_pairwise.csv'}")

    if "trajectories" in plot_types or "all" in plot_types:
        plot_trajectories(df, out_dir)

    if "heatmap" in plot_types or "all" in plot_types:
        plot_heatmap(summary_df, out_dir)

    if "ranges" in plot_types or "all" in plot_types:
        plot_ranges(df, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot four-way sedation comparison across frequency bands."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="results/sedation_resting_state/mib_sedation/sedation/spectral/binning",
        help="Root directory containing MIB results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/outputs/sedation_resting_state",
        help="Output directory for plots and CSVs",
    )
    parser.add_argument(
        "--plot",
        type=str,
        nargs="+",
        choices=["trajectories", "heatmap", "ranges", "all"],
        default=["all"],
        help="Which plots to generate (default: all)",
    )
    args = parser.parse_args()

    run_analysis(Path(args.root), Path(args.output), args.plot)


if __name__ == "__main__":
    main()
