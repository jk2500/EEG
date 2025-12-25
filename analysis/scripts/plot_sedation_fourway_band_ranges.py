#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUT_DIR = Path("analysis/outputs/sedation_resting_state")
DATA_PATH = OUT_DIR / "fourway_band_means_raw.csv"
CONDITIONS = ["baseline", "light_sedation", "deep_sedation", "recovery"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing data file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    summary = (
        df.groupby(["band", "condition"])["mean"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "fourway_band_mean_range.csv", index=False)

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7), sharey=False)
    axes = axes.flatten()
    x = np.arange(len(CONDITIONS))

    for ax, band in zip(axes, BANDS):
        band_df = summary[summary["band"] == band].set_index("condition").reindex(CONDITIONS)
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
        ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
        ax.set_ylabel("Mean MIB (min–max)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes[len(BANDS):]:
        ax.set_axis_off()

    fig.suptitle("Sedation Dataset: Mean ± Range by Band and Condition", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourway_band_mean_range.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
