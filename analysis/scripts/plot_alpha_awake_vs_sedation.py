#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_1samp


ROOT = Path("results/ds005620/mib_random_channels/spectral/binning/epoch-5p00s")
OUT_DIR = Path("analysis/outputs/ds005620/all_subjects_results")


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    if values.size < 2:
        return (np.nan, np.nan)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half_width = t.ppf(1 - alpha / 2, df=values.size - 1) * std / np.sqrt(values.size)
    return (mean - half_width, mean + half_width)


def load_alpha_means() -> pd.DataFrame:
    rows = []
    for path in ROOT.glob("sub-*/*/mib_random_channels_spectral_binning_*.json"):
        subject = path.parts[-3]
        condition = path.parts[-2]
        data = json.loads(path.read_text())
        alpha_mean = data["bands"]["alpha"]["overall_stats"]["mean_of_per_repeat_means"]
        rows.append({"subject": subject, "condition": condition, "alpha_mean": alpha_mean})
    df = pd.DataFrame(rows)
    return df


def summarize_diff(diff: np.ndarray) -> dict:
    mean = float(diff.mean())
    median = float(np.median(diff))
    std = float(diff.std(ddof=1))
    t_stat, p_val = ttest_1samp(diff, 0.0)
    ci_low, ci_high = mean_ci(diff)
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "t_stat": t_stat,
        "p_val": p_val,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_alpha_means()
    wide = df.pivot_table(index="subject", columns="condition", values="alpha_mean")
    paired = wide.dropna(subset=["sedation_1", "awake_eyes_open", "awake_eyes_closed"])

    if paired.empty:
        raise SystemExit("No subjects with both awake and sedation alpha results.")

    sed = paired["sedation_1"].to_numpy()
    eo = paired["awake_eyes_open"].to_numpy()
    ec = paired["awake_eyes_closed"].to_numpy()
    avg = (eo + ec) / 2
    n = sed.size

    comparisons = {
        "Awake EC vs Sedation": ec - sed,
        "Awake EO vs Sedation": eo - sed,
        "Awake Avg vs Sedation": avg - sed,
    }

    stats = {label: summarize_diff(diff) for label, diff in comparisons.items()}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), sharey=True)
    for ax, (label, diff) in zip(axes, comparisons.items()):
        if "EC" in label:
            awake = ec
            awake_label = "awake_eyes_closed"
        elif "EO" in label:
            awake = eo
            awake_label = "awake_eyes_open"
        else:
            awake = avg
            awake_label = "awake_avg"

        x = np.array([0, 1])
        for i in range(n):
            ax.plot(
                x,
                [sed[i], awake[i]],
                color="#999999",
                linewidth=1,
                alpha=0.6,
            )

        ax.scatter(
            np.zeros(n),
            sed,
            color="#d55e00",
            s=20,
            label="sedation_1" if label.startswith("Awake EC") else None,
            zorder=3,
        )
        ax.scatter(
            np.ones(n),
            awake,
            color="#0072b2",
            s=20,
            label=awake_label if label.startswith("Awake EC") else None,
            zorder=3,
        )

        mean_sed = sed.mean()
        mean_awake = awake.mean()
        ax.scatter([0, 1], [mean_sed, mean_awake], color="#000000", s=60, zorder=4)

        stat = stats[label]
        ax.set_title(
            f"{label}\\nmean diff={stat['mean']:.3f}, p={stat['p_val']:.3g}",
            fontsize=10,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["sedation_1", awake_label], rotation=20, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Alpha mean MIB (per subject)")
    axes[0].legend(loc="upper left", frameon=False)

    fig.suptitle(
        f"Alpha Band MIB: Awake vs Sedation (n={n} subjects)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_awake_vs_sedation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
