#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_rel


ROOT = Path("results/sedation_resting_state/mib_sedation/sedation/spectral/binning")
OUT_DIR = Path("analysis/outputs/sedation_resting_state")
CONDITIONS = ["baseline", "light_sedation", "deep_sedation", "recovery"]
BANDS = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size < 2:
        return (np.nan, np.nan)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half_width = t.ppf(1 - alpha / 2, df=values.size - 1) * std / np.sqrt(values.size)
    return (mean - half_width, mean + half_width)


def load_band_means() -> pd.DataFrame:
    rows = []
    for subject_dir in sorted(ROOT.glob("sub-*")):
        subject = subject_dir.name
        for condition in CONDITIONS:
            for path in subject_dir.glob(f"{condition}/sedation_mib_spectral_*.json"):
                data = json.loads(path.read_text())
                for band in BANDS:
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_band_means()
    if df.empty:
        raise SystemExit(f"No results found under {ROOT}")

    df.to_csv(OUT_DIR / "fourway_band_means_raw.csv", index=False)

    summary_rows = []
    pairwise_rows = []

    for band in BANDS:
        band_df = df[df["band"] == band]
        band_wide = band_df.pivot_table(index="subject", columns="condition", values="mean")
        paired = band_wide.dropna(subset=CONDITIONS)
        if paired.empty:
            continue

        for condition in CONDITIONS:
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

        pairs = [
            ("baseline", "light_sedation"),
            ("baseline", "deep_sedation"),
            ("baseline", "recovery"),
            ("light_sedation", "deep_sedation"),
            ("light_sedation", "recovery"),
            ("deep_sedation", "recovery"),
        ]
        for a, b in pairs:
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

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "fourway_band_summary.csv", index=False)

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_df.to_csv(OUT_DIR / "fourway_band_pairwise.csv", index=False)

    # Plot: subject trajectories per band
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7), sharey=False)
    axes = axes.flatten()
    x = np.arange(len(CONDITIONS))

    for ax, band in zip(axes, BANDS):
        band_df = df[df["band"] == band]
        band_wide = band_df.pivot_table(index="subject", columns="condition", values="mean")
        paired = band_wide.dropna(subset=CONDITIONS)
        if paired.empty:
            ax.set_axis_off()
            continue

        vals = paired[CONDITIONS].to_numpy()
        for row in vals:
            ax.plot(x, row, color="#999999", linewidth=1, alpha=0.6)
            ax.scatter(x, row, color="#777777", s=18, alpha=0.7)

        means = [paired[c].mean() for c in CONDITIONS]
        ci_lows = [mean_ci(paired[c].to_numpy())[0] for c in CONDITIONS]
        ci_highs = [mean_ci(paired[c].to_numpy())[1] for c in CONDITIONS]
        yerr = [
            [m - l for m, l in zip(means, ci_lows)],
            [h - m for h, m in zip(ci_highs, means)],
        ]
        ax.errorbar(x, means, yerr=yerr, fmt="o", color="#000000", capsize=4)

        ax.set_title(f"{band} (n={paired.shape[0]})")
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes[len(BANDS):]:
        ax.set_axis_off()

    fig.suptitle("Alpha and All Bands: baseline vs light vs deep vs recovery", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourway_all_bands.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Heatmap of mean values by band/condition
    heat = summary_df.pivot(index="band", columns="condition", values="mean").reindex(BANDS)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(heat.values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(BANDS)))
    ax.set_yticklabels(BANDS)
    for i in range(len(BANDS)):
        for j in range(len(CONDITIONS)):
            ax.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_title("Mean MIB by Band and Condition")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fourway_band_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
