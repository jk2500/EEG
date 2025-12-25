#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_1samp, wilcoxon


ROOT = Path("results/ds005620/mib_random_channels/spectral/binning/epoch-5p00s")
OUT_DIR = Path("analysis/outputs/ds005620/all_subjects_results")


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size < 2:
        return (np.nan, np.nan)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half_width = t.ppf(1 - alpha / 2, df=values.size - 1) * std / np.sqrt(values.size)
    return (mean - half_width, mean + half_width)


def summarize_diff(diff: np.ndarray) -> dict:
    n = diff.size
    mean = diff.mean()
    median = np.median(diff)
    std = diff.std(ddof=1)
    pos_frac = float((diff > 0).mean())
    t_stat, t_p = ttest_1samp(diff, 0.0)
    d = mean / std if std > 0 else np.nan
    try:
        w_stat, w_p = wilcoxon(diff)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    ci_low, ci_high = mean_ci(diff)
    return {
        "n": n,
        "mean_diff": mean,
        "median_diff": median,
        "std_diff": std,
        "pos_frac": pos_frac,
        "t_stat": t_stat,
        "t_p": t_p,
        "cohen_d": d,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in ROOT.glob("sub-*/*/mib_random_channels_spectral_binning_*.json"):
        subject = path.parts[-3]
        condition = path.parts[-2]
        data = json.loads(path.read_text())
        for band, band_data in data["bands"].items():
            rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "band": band,
                    "mean": band_data["overall_stats"]["mean_of_per_repeat_means"],
                    "std": band_data["overall_stats"]["std_of_per_repeat_means"],
                    "cv": band_data["overall_stats"]["cv_of_per_repeat_means"],
                    "repeats": band_data["repeats"],
                    "n_channels_selected": band_data["n_channels_selected"],
                    "n_channels_total": band_data["n_channels_total"],
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No results found under {ROOT}")

    df.to_csv(OUT_DIR / "all_subjects_band_means.csv", index=False)

    wide = (
        df.pivot_table(index=["subject", "condition"], columns="band", values="mean")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    composites = {
        "high_abc": ["alpha", "beta", "gamma"],
        "high_bg": ["beta", "gamma"],
        "mid_alpha_beta": ["alpha", "beta"],
        "gamma_only": ["gamma"],
        "alpha_only": ["alpha"],
        "low_dt": ["delta", "theta"],
    }
    for name, bands in composites.items():
        wide[name] = wide[bands].mean(axis=1)
    wide["high_over_low"] = wide["high_abc"] - wide["low_dt"]

    wide.to_csv(OUT_DIR / "all_subjects_composites.csv", index=False)

    metrics = list(df["band"].unique()) + list(composites.keys()) + ["high_over_low"]
    comparisons = []

    for metric in metrics:
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
    comp_df.to_csv(OUT_DIR / "all_subjects_awake_vs_sedation.csv", index=False)

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
    plt.title("Awake_avg vs Sedation by Band (95% CI)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "awake_avg_vs_sedation_by_band.png", dpi=150)
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
    plt.title("Awake vs Sedation by Band")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "awake_vs_sedation_by_band.png", dpi=150)
    plt.close()

    comp_metric = comp_df[
        comp_df["metric"].isin(["high_abc", "high_bg", "mid_alpha_beta", "gamma_only", "alpha_only", "high_over_low"])
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
    x = np.arange(len(metric_order))
    means = [comp_metric.loc[m, "mean_diff"] for m in metric_order]
    plt.bar(x, means, color="#55a868")
    plt.axhline(0, color="#666666", linewidth=1)
    plt.xticks(x, metric_order, rotation=20, ha="right")
    plt.ylabel("Awake_avg - Sedation (mean diff)")
    plt.title("Composite Metrics: Awake_avg vs Sedation")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "awake_avg_vs_sedation_composites.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
