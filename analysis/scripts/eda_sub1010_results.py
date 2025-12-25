#!/usr/bin/env python3
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind, t


RESULTS_DIR = Path(
    "results/ds005620/mib_random_channels/spectral/binning/epoch-5p00s/sub-1010"
)
OUT_DIR = Path("analysis/outputs/ds005620/sub1010_results")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size < 2:
        return (np.nan, np.nan)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half_width = t.ppf(1 - alpha / 2, df=values.size - 1) * std / np.sqrt(values.size)
    return (mean - half_width, mean + half_width)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return np.nan
    s1 = a.std(ddof=1)
    s2 = b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def fdr_bh(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    ranks = np.arange(1, len(pvalues) + 1)
    p_sorted = pvalues[order]
    adj_sorted = p_sorted * len(pvalues) / ranks
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adjusted = np.empty_like(pvalues, dtype=float)
    adjusted[order] = adj_sorted
    return np.clip(adjusted, 0.0, 1.0)


def jaccard_mean(sets: list[set[str]]) -> float:
    if len(sets) < 2:
        return np.nan
    total = 0.0
    count = 0
    for a, b in itertools.combinations(sets, 2):
        denom = len(a | b)
        total += len(a & b) / denom if denom else 0.0
        count += 1
    return total / count if count else np.nan


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(RESULTS_DIR.glob("*/mib_random_channels_spectral_binning_*.json"))
    if not json_paths:
        raise SystemExit(f"No results found in {RESULTS_DIR}")

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
    summary_df.to_csv(OUT_DIR / "sub1010_results_summary.csv", index=False)

    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(OUT_DIR / "sub1010_repeat_means.csv", index=False)

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_df.to_csv(OUT_DIR / "sub1010_epoch_distribution.csv", index=False)

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(OUT_DIR / "sub1010_epoch_stability.csv", index=False)

    selection_df = pd.DataFrame(selection_rows)
    selection_df.to_csv(OUT_DIR / "sub1010_channel_selection_counts.csv", index=False)

    jaccard_df = pd.DataFrame(jaccard_rows)
    jaccard_df.to_csv(OUT_DIR / "sub1010_selection_jaccard.csv", index=False)

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
        comp_df.to_csv(OUT_DIR / "sub1010_condition_comparisons.csv", index=False)

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
    for ax in axes[len(bands) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sub1010_repeat_means_boxplot.png", dpi=150)
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
    plt.title("Mean MIB by Band and Condition (per-repeat means)")
    plt.ylabel("Mean MIB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sub1010_band_means.png", dpi=150)
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
    plt.title("CV of Per-Repeat Mean MIB by Band")
    plt.ylabel("CV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sub1010_repeat_mean_cv.png", dpi=150)
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
    plt.title("Epoch Stability CV by Band")
    plt.ylabel("CV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "sub1010_epoch_stability_cv.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
