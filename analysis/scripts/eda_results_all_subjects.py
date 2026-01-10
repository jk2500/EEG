#!/usr/bin/env python3
"""EDA for all subjects MIB results.

Supports different epoch lengths and configurable results roots.

Usage:
    # Default: epoch=5s, default results root
    python analysis/scripts/eda_results_all_subjects.py

    # 10s sensitivity analysis (separate results root recommended)
    python analysis/scripts/eda_results_all_subjects.py --epoch 10 --root path/to/results_root
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
)


def _default_results_root(epoch_length: float) -> Path:
    """
    Pick a sensible default root for ds005620 spectral results.

    This defaults to the output layout used by scripts/mib_analysis.py:
        results/ds005620/mib_analysis_optimal[/epochXX]/ds005620/spectral/binning/
    """
    base = Path("results/ds005620/mib_analysis_optimal")
    if int(epoch_length) == 10:
        base = Path("results/ds005620/mib_analysis_optimal_epoch10")
    elif int(epoch_length) == 5:
        base = Path("results/ds005620/mib_analysis_optimal")

    return base / "ds005620" / "spectral" / "binning"


def _default_output_dir(epoch_length: float) -> Path:
    """
    Match the folder names referenced by paper/paper.tex.
    """
    if int(epoch_length) == 5:
        return Path("analysis/outputs/ds005620/all_subjects_results")
    return Path(f"analysis/outputs/ds005620/all_subjects_results_epoch{int(epoch_length)}")


def run_eda(*, epoch_length: float, root: Path, out_dir: Path) -> None:
    """Run EDA analysis for the specified epoch length."""
    epoch_label = f"{int(epoch_length)}s"

    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_mib_results(
        root,
        glob_pattern="sub-*/*/mib_*.json",
    )
    if df.empty:
        raise SystemExit(f"No results found under {root}")

    # Restrict to subjects with all three primary conditions available.
    required = ["awake_eyes_open", "awake_eyes_closed", "sedation_1"]
    coverage = (
        df[df["condition"].isin(required)]
        .groupby(["subject", "condition"])["band"]
        .nunique()
        .unstack("condition")
    )
    keep_subjects = coverage.dropna(subset=required).index.tolist()
    df = df[df["subject"].isin(keep_subjects)].copy()
    # If multiple result files exist per subject/condition, keep only the most recent record.
    df["mtime"] = df["source_file"].map(lambda p: Path(p).stat().st_mtime)
    df = (
        df.sort_values("mtime")
        .drop_duplicates(subset=["subject", "condition", "band"], keep="last")
        .drop(columns=["mtime"])
    )

    # Persist the long-form subject/condition/band means as source data.
    long_name = "all_subjects_band_means_long.csv" if int(epoch_length) == 5 else f"epoch{int(epoch_length)}_band_means_long.csv"
    df.to_csv(out_dir / long_name, index=False)

    wide = aggregate_subject_results(df, add_composites=True)
    composites_name = (
        "all_subjects_composites.csv"
        if int(epoch_length) == 5
        else f"epoch{int(epoch_length)}_composites.csv"
    )
    wide.to_csv(out_dir / composites_name, index=False)

    # Condition-level means by band (mean ± SD across subjects).
    band_list = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]
    cond_means = (
        df[df["band"].isin(band_list)]
        .groupby(["band", "condition"])["mean"]
        .agg(["mean", "std"])
        .reset_index()
    )
    cond_means = cond_means[cond_means["condition"].isin(required)]
    means_pivot = cond_means.pivot(index="band", columns="condition", values="mean")
    std_pivot = cond_means.pivot(index="band", columns="condition", values="std")
    band_means_table = pd.DataFrame(
        {
            "band": band_list,
            "awake_eo_mean": means_pivot.reindex(band_list)["awake_eyes_open"].to_numpy(),
            "awake_eo_std": std_pivot.reindex(band_list)["awake_eyes_open"].to_numpy(),
            "awake_ec_mean": means_pivot.reindex(band_list)["awake_eyes_closed"].to_numpy(),
            "awake_ec_std": std_pivot.reindex(band_list)["awake_eyes_closed"].to_numpy(),
            "sedation_1_mean": means_pivot.reindex(band_list)["sedation_1"].to_numpy(),
            "sedation_1_std": std_pivot.reindex(band_list)["sedation_1"].to_numpy(),
        }
    )
    band_means_name = (
        "all_subjects_band_means.csv"
        if int(epoch_length) == 5
        else f"epoch{int(epoch_length)}_band_means.csv"
    )
    band_means_table.to_csv(out_dir / band_means_name, index=False)

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
    awake_vs_sed_name = (
        "all_subjects_awake_vs_sedation.csv"
        if int(epoch_length) == 5
        else f"epoch{int(epoch_length)}_awake_vs_sedation.csv"
    )
    comp_df.to_csv(out_dir / awake_vs_sed_name, index=False)

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
    fig_name = (
        "awake_avg_vs_sedation_by_band.png"
        if int(epoch_length) == 5
        else f"epoch{int(epoch_length)}_awake_avg_vs_sedation_by_band.png"
    )
    plt.savefig(out_dir / fig_name, dpi=150)
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
    fig_name = (
        "awake_vs_sedation_by_band.png"
        if int(epoch_length) == 5
        else f"epoch{int(epoch_length)}_awake_vs_sedation_by_band.png"
    )
    plt.savefig(out_dir / fig_name, dpi=150)
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
    fig_name = (
        "awake_avg_vs_sedation_composites.png"
        if int(epoch_length) == 5
        else f"epoch{int(epoch_length)}_awake_avg_vs_sedation_composites.png"
    )
    plt.savefig(out_dir / fig_name, dpi=150)
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
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory containing ds005620 spectral JSON results (default: inferred).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for CSVs/figures (default: inferred).",
    )
    args = parser.parse_args()
    root = Path(args.root) if args.root else _default_results_root(args.epoch)
    out_dir = Path(args.output) if args.output else _default_output_dir(args.epoch)
    run_eda(epoch_length=args.epoch, root=root, out_dir=out_dir)


if __name__ == "__main__":
    main()
