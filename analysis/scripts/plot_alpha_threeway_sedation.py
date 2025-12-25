#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_rel


ROOT = Path("results/sedation_resting_state/mib_sedation/sedation/spectral/binning")
OUT_DIR = Path("analysis/outputs/sedation_resting_state")
CONDITIONS = ["baseline", "light_sedation", "deep_sedation"]


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size < 2:
        return (np.nan, np.nan)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half_width = t.ppf(1 - alpha / 2, df=values.size - 1) * std / np.sqrt(values.size)
    return (mean - half_width, mean + half_width)


def load_alpha_means() -> pd.DataFrame:
    rows = []
    for subject_dir in sorted(ROOT.glob("sub-*")):
        subject = subject_dir.name
        for condition in CONDITIONS:
            for path in subject_dir.glob(f"{condition}/sedation_mib_spectral_*.json"):
                data = json.loads(path.read_text())
                alpha_mean = data["bands"]["alpha"]["overall_stats"]["mean_of_per_repeat_means"]
                rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "alpha_mean": alpha_mean,
                        "source_file": str(path),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_alpha_means()
    if df.empty:
        raise SystemExit(f"No alpha results found under {ROOT}")

    df.to_csv(OUT_DIR / "alpha_threeway_raw.csv", index=False)

    wide = df.pivot_table(index="subject", columns="condition", values="alpha_mean")
    paired = wide.dropna(subset=CONDITIONS)

    if paired.empty:
        raise SystemExit("No subjects with baseline/light/deep all present.")

    paired.to_csv(OUT_DIR / "alpha_threeway_per_subject.csv")

    values = {cond: paired[cond].to_numpy() for cond in CONDITIONS}
    n = paired.shape[0]

    summary_rows = []
    for cond in CONDITIONS:
        vals = values[cond]
        ci_low, ci_high = mean_ci(vals)
        summary_rows.append(
            {
                "condition": cond,
                "n_subjects": vals.size,
                "mean": vals.mean(),
                "median": np.median(vals),
                "std": vals.std(ddof=1),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "alpha_threeway_summary.csv", index=False)

    diff_rows = []
    diff_pairs = [
        ("baseline", "light_sedation"),
        ("baseline", "deep_sedation"),
        ("light_sedation", "deep_sedation"),
    ]
    for a, b in diff_pairs:
        diff = values[a] - values[b]
        ci_low, ci_high = mean_ci(diff)
        t_stat, p_val = ttest_rel(values[a], values[b])
        diff_rows.append(
            {
                "pair": f"{a} - {b}",
                "mean_diff": diff.mean(),
                "median_diff": np.median(diff),
                "std_diff": diff.std(ddof=1),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "t_stat": t_stat,
                "t_p": p_val,
            }
        )
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(OUT_DIR / "alpha_threeway_pairwise.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Left: paired subject trajectories
    x = np.arange(len(CONDITIONS))
    for i in range(n):
        subj_vals = [values[cond][i] for cond in CONDITIONS]
        axes[0].plot(x, subj_vals, color="#999999", linewidth=1, alpha=0.6)
        axes[0].scatter(x, subj_vals, color="#666666", s=18, alpha=0.8)

    means = [values[cond].mean() for cond in CONDITIONS]
    ci_lows = [mean_ci(values[cond])[0] for cond in CONDITIONS]
    ci_highs = [mean_ci(values[cond])[1] for cond in CONDITIONS]
    yerr = [
        [m - l for m, l in zip(means, ci_lows)],
        [h - m for h, m in zip(ci_highs, means)],
    ]
    axes[0].errorbar(x, means, yerr=yerr, fmt="o", color="#000000", capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(CONDITIONS, rotation=20, ha="right")
    axes[0].set_ylabel("Alpha mean MIB (per subject)")
    axes[0].set_title("Subject trajectories with mean ± 95% CI")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Right: pairwise differences
    diff_labels = [row["pair"] for row in diff_rows]
    diff_means = [row["mean_diff"] for row in diff_rows]
    diff_ci_lows = [row["ci_low"] for row in diff_rows]
    diff_ci_highs = [row["ci_high"] for row in diff_rows]
    x2 = np.arange(len(diff_labels))
    yerr2 = [
        [m - l for m, l in zip(diff_means, diff_ci_lows)],
        [h - m for h, m in zip(diff_ci_highs, diff_means)],
    ]
    axes[1].bar(x2, diff_means, color="#4c72b0")
    axes[1].errorbar(x2, diff_means, yerr=yerr2, fmt="none", ecolor="#333333", capsize=4)
    axes[1].axhline(0, color="#666666", linewidth=1)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(diff_labels, rotation=20, ha="right")
    axes[1].set_title("Pairwise mean differences (95% CI)")
    axes[1].set_ylabel("Mean difference")

    for idx, row in enumerate(diff_rows):
        axes[1].text(
            idx,
            diff_means[idx],
            f"p={row['t_p']:.3g}",
            ha="center",
            va="bottom" if diff_means[idx] >= 0 else "top",
            fontsize=8,
        )

    fig.suptitle(f"Alpha MIB: baseline vs light vs deep sedation (n={n})", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_threeway_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
