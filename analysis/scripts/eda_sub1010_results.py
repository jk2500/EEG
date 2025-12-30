#!/usr/bin/env python3
"""Detailed EDA for subject MIB results.

Supports different subjects and epoch lengths via CLI arguments.

Usage:
    python analysis/scripts/eda_sub1010_results.py                    # Default: sub-1010, 5s
    python analysis/scripts/eda_sub1010_results.py --subject sub-1010 --epoch 10
    python analysis/scripts/eda_sub1010_results.py --subject sub-1019 --epoch 5
"""

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind

from utils import mean_ci, cohen_d, fdr_bh, jaccard_mean, format_epoch_path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_eda(subject: str, epoch_length: float) -> None:
    """Run detailed EDA for the specified subject and epoch length."""
    epoch_str = format_epoch_path(epoch_length)

    results_dir = Path(
        f"results/ds005620/mib_random_channels/spectral/binning/epoch-{epoch_str}/{subject}"
    )
    out_dir = Path(f"analysis/outputs/ds005620/{subject}_results_epoch{int(epoch_length)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(results_dir.glob("*/mib_random_channels_spectral_binning_*.json"))
    if not json_paths:
        raise SystemExit(f"No results found in {results_dir}")

    summary_rows = []
    repeat_rows = []
    epoch_rows = []
    stability_rows = []
    selection_rows = []
    jaccard_rows = []

    for path in json_paths:
        condition = path.parent.name
        data = load_json(path)
        all_channels = data["all_channel_names"]
        n_channels_total = len(all_channels)
        epoch_length = data.get("epoch_length")
        vhdr_path = data.get("vhdr_path")

        for band, band_data in data["bands"].items():
            repeats_payload = band_data["repeats_payload"]
            per_repeat_mean = np.array([r["mean_mib"] for r in repeats_payload], dtype=float)
            per_repeat_std = np.array([r["std_mib"] for r in repeats_payload], dtype=float)
            n_epochs = np.array([r["n_epochs"] for r in repeats_payload], dtype=int)

            epoch_values = np.concatenate([r["metric_values"] for r in repeats_payload])

            ci_low, ci_high = mean_ci(per_repeat_mean)

            summary_rows.append(
                {
                    "condition": condition,
                    "band": band,
                    "epoch_length": epoch_length,
                    "vhdr_path": vhdr_path,
                    "repeats": band_data["repeats"],
                    "n_channels_selected": band_data["n_channels_selected"],
                    "n_channels_total": band_data["n_channels_total"],
                    "mean_repeat_mean": per_repeat_mean.mean(),
                    "std_repeat_mean": per_repeat_mean.std(ddof=1),
                    "cv_repeat_mean": per_repeat_mean.std(ddof=1) / per_repeat_mean.mean(),
                    "median_repeat_mean": np.median(per_repeat_mean),
                    "repeat_mean_ci_low": ci_low,
                    "repeat_mean_ci_high": ci_high,
                    "mean_repeat_std": per_repeat_std.mean(),
                    "epoch_mean": epoch_values.mean(),
                    "epoch_std": epoch_values.std(ddof=1),
                    "epoch_cv": epoch_values.std(ddof=1) / epoch_values.mean(),
                    "overall_mean_of_repeat_means": band_data["overall_stats"][
                        "mean_of_per_repeat_means"
                    ],
                    "overall_std_of_repeat_means": band_data["overall_stats"][
                        "std_of_per_repeat_means"
                    ],
                    "overall_cv_of_repeat_means": band_data["overall_stats"][
                        "cv_of_per_repeat_means"
                    ],
                    "overall_mean_of_repeat_stds": band_data["overall_stats"][
                        "mean_of_per_repeat_stds"
                    ],
                    "overall_min_repeat_mean": band_data["overall_stats"][
                        "min_per_repeat_mean"
                    ],
                    "overall_max_repeat_mean": band_data["overall_stats"][
                        "max_per_repeat_mean"
                    ],
                    "overall_range_repeat_means": band_data["overall_stats"][
                        "range_per_repeat_means"
                    ],
                }
            )

            for repeat in repeats_payload:
                repeat_rows.append(
                    {
                        "condition": condition,
                        "band": band,
                        "repeat_index": repeat["repeat_index"],
                        "mean_mib": repeat["mean_mib"],
                        "std_mib": repeat["std_mib"],
                        "n_epochs": repeat["n_epochs"],
                    }
                )

            epoch_rows.append(
                {
                    "condition": condition,
                    "band": band,
                    "epoch_mean": epoch_values.mean(),
                    "epoch_std": epoch_values.std(ddof=1),
                    "epoch_cv": epoch_values.std(ddof=1) / epoch_values.mean(),
                    "epoch_count": epoch_values.size,
                }
            )

            stability = band_data["epoch_stability"]
            stability_rows.append(
                {
                    "condition": condition,
                    "band": band,
                    "stability_mean": stability["mean_mib"],
                    "stability_std": stability["std_mib"],
                    "stability_cv": stability["std_mib"] / stability["mean_mib"],
                    "stability_epochs": stability["n_epochs"],
                }
            )

            counts = Counter()
            selection_sets = []
            for repeat in repeats_payload:
                counts.update(repeat["selected_channels"])
                selection_sets.append(set(repeat["selected_channels"]))

            expected = band_data["repeats"] * band_data["n_channels_selected"] / n_channels_total
            for channel in all_channels:
                selection_rows.append(
                    {
                        "condition": condition,
                        "band": band,
                        "channel": channel,
                        "count": counts.get(channel, 0),
                        "expected_count": expected,
                        "count_diff": counts.get(channel, 0) - expected,
                    }
                )

            jaccard_rows.append(
                {
                    "condition": condition,
                    "band": band,
                    "mean_jaccard": jaccard_mean(selection_sets),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "results_summary.csv", index=False)

    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(out_dir / "repeat_means.csv", index=False)

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_df.to_csv(out_dir / "epoch_distribution.csv", index=False)

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(out_dir / "epoch_stability.csv", index=False)

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(out_dir / "channel_selection_counts.csv", index=False)

    jaccard_df = pd.DataFrame(jaccard_rows)
    jaccard_df.to_csv(out_dir / "selection_jaccard.csv", index=False)

    comparisons = []
    for band in sorted(summary_df["band"].unique()):
        band_df = repeat_df[repeat_df["band"] == band]
        conditions = sorted(band_df["condition"].unique())
        for cond_a, cond_b in itertools.combinations(conditions, 2):
            a_vals = band_df[band_df["condition"] == cond_a]["mean_mib"].to_numpy()
            b_vals = band_df[band_df["condition"] == cond_b]["mean_mib"].to_numpy()
            tstat, pval = ttest_ind(a_vals, b_vals, equal_var=False)
            ustat, upval = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            comparisons.append(
                {
                    "band": band,
                    "cond_a": cond_a,
                    "cond_b": cond_b,
                    "mean_a": a_vals.mean(),
                    "mean_b": b_vals.mean(),
                    "mean_diff": a_vals.mean() - b_vals.mean(),
                    "cohen_d": cohen_d(a_vals, b_vals),
                    "t_stat": tstat,
                    "t_pvalue": pval,
                    "mw_u": ustat,
                    "mw_pvalue": upval,
                }
            )

    comp_df = pd.DataFrame(comparisons)
    if not comp_df.empty:
        comp_df["t_pvalue_fdr"] = fdr_bh(comp_df["t_pvalue"].to_numpy())
        comp_df["mw_pvalue_fdr"] = fdr_bh(comp_df["mw_pvalue"].to_numpy())
        comp_df.to_csv(out_dir / "condition_comparisons.csv", index=False)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Repeat mean distributions by condition (faceted by band).
    bands = sorted(repeat_df["band"].unique())
    conditions = sorted(repeat_df["condition"].unique())
    ncols = 3
    nrows = int(np.ceil(len(bands) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), sharey=False)
    axes = np.atleast_1d(axes).flatten()
    for ax, band in zip(axes, bands):
        band_df = repeat_df[repeat_df["band"] == band]
        data = [
            band_df[band_df["condition"] == cond]["mean_mib"].to_numpy()
            for cond in conditions
        ]
        ax.boxplot(data, tick_labels=conditions)
        ax.set_title(band)
        ax.set_ylabel("Mean MIB (per repeat)")
        ax.tick_params(axis="x", rotation=20)
    for ax in axes[len(bands):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "repeat_means_boxplot.png", dpi=150)
    plt.close()

    # Mean MIB across bands by condition.
    band_means = repeat_df.groupby(["band", "condition"])["mean_mib"].mean().reset_index()
    x = np.arange(len(bands))
    width = 0.8 / len(conditions)
    plt.figure(figsize=(10, 5))
    for idx, cond in enumerate(conditions):
        vals = (
            band_means[band_means["condition"] == cond]
            .set_index("band")
            .reindex(bands)["mean_mib"]
            .to_numpy()
        )
        plt.bar(x + idx * width, vals, width=width, label=cond)
    plt.xticks(x + width * (len(conditions) - 1) / 2, bands)
    plt.title(f"{subject}: Mean MIB by Band and Condition")
    plt.ylabel("Mean MIB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "band_means.png", dpi=150)
    plt.close()

    # Variability of repeat means (CV).
    cv_df = summary_df[["condition", "band", "cv_repeat_mean"]]
    plt.figure(figsize=(10, 5))
    for idx, cond in enumerate(conditions):
        vals = (
            cv_df[cv_df["condition"] == cond]
            .set_index("band")
            .reindex(bands)["cv_repeat_mean"]
            .to_numpy()
        )
        plt.bar(x + idx * width, vals, width=width, label=cond)
    plt.xticks(x + width * (len(conditions) - 1) / 2, bands)
    plt.title(f"{subject}: CV of Per-Repeat Mean MIB by Band")
    plt.ylabel("CV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "repeat_mean_cv.png", dpi=150)
    plt.close()

    # Epoch stability summary.
    plt.figure(figsize=(10, 5))
    for idx, cond in enumerate(conditions):
        vals = (
            stability_df[stability_df["condition"] == cond]
            .set_index("band")
            .reindex(bands)["stability_cv"]
            .to_numpy()
        )
        plt.bar(x + idx * width, vals, width=width, label=cond)
    plt.xticks(x + width * (len(conditions) - 1) / 2, bands)
    plt.title(f"{subject}: Epoch Stability CV by Band")
    plt.ylabel("CV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "epoch_stability_cv.png", dpi=150)
    plt.close()

    print(f"EDA complete for {subject} (epoch {int(epoch_length)}s). Results saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run detailed EDA on subject MIB results."
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="sub-1010",
        help="Subject ID (default: sub-1010)",
    )
    parser.add_argument(
        "--epoch",
        type=float,
        default=5.0,
        help="Epoch length in seconds (default: 5.0)",
    )
    args = parser.parse_args()
    run_eda(args.subject, args.epoch)


if __name__ == "__main__":
    main()
