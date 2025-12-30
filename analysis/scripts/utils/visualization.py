"""Visualization utilities for EEG analysis."""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from .stats import mean_ci


# Color schemes for consistency across plots
BAND_COLORS = {
    "delta": "#1f77b4",
    "theta": "#ff7f0e",
    "alpha": "#2ca02c",
    "beta": "#d62728",
    "gamma": "#9467bd",
    "broadband": "#8c564b",
}

CONDITION_COLORS = {
    # DS005620 conditions
    "awake_eyes_open": "#4c72b0",
    "awake_eyes_closed": "#55a868",
    "sedation_1": "#c44e52",
    # Sedation resting state conditions
    "baseline": "#4c72b0",
    "light_sedation": "#55a868",
    "deep_sedation": "#c44e52",
    "recovery": "#8172b2",
}

CONDITION_MARKERS = {
    "awake_eyes_open": "o",
    "awake_eyes_closed": "s",
    "sedation_1": "^",
    "baseline": "o",
    "light_sedation": "s",
    "deep_sedation": "^",
    "recovery": "D",
}


def plot_paired_trajectories(
    ax: plt.Axes,
    data: np.ndarray,
    conditions: list[str],
    title: str = "",
    show_mean: bool = True,
    line_color: str = "#999999",
    line_alpha: float = 0.6,
    mean_color: str = "#000000",
) -> None:
    """
    Plot subject trajectories across conditions with mean and CI overlay.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on.
    data : np.ndarray
        2D array of shape (n_subjects, n_conditions).
    conditions : list[str]
        List of condition names for x-axis labels.
    title : str
        Plot title.
    show_mean : bool
        Whether to show mean with error bars.
    line_color : str
        Color for individual subject lines.
    line_alpha : float
        Alpha for individual subject lines.
    mean_color : str
        Color for mean markers and error bars.
    """
    x = np.arange(len(conditions))

    # Plot individual subject trajectories
    for row in data:
        ax.plot(x, row, color=line_color, linewidth=1, alpha=line_alpha)
        ax.scatter(x, row, color=line_color, s=18, alpha=line_alpha + 0.1)

    # Plot mean with CI
    if show_mean:
        means = data.mean(axis=0)
        cis = [mean_ci(data[:, i]) for i in range(data.shape[1])]
        ci_lows = [ci[0] for ci in cis]
        ci_highs = [ci[1] for ci in cis]
        yerr = [
            [m - l for m, l in zip(means, ci_lows)],
            [h - m for h, m in zip(ci_highs, means)],
        ]
        ax.errorbar(x, means, yerr=yerr, fmt="o", color=mean_color, capsize=4,
                    markersize=8, linewidth=2, zorder=10)

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def plot_condition_bars(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    x_labels: list[str],
    title: str = "",
    ylabel: str = "Value",
    show_ci: bool = True,
    bar_width: float = 0.25,
) -> None:
    """
    Plot grouped bar chart comparing conditions.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on.
    data : dict[str, np.ndarray]
        Dictionary mapping condition names to arrays of values per x-label.
    x_labels : list[str]
        Labels for x-axis (e.g., band names).
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    show_ci : bool
        Whether to show 95% CI error bars.
    bar_width : float
        Width of each bar.
    """
    x = np.arange(len(x_labels))
    n_conditions = len(data)

    for idx, (cond_name, values) in enumerate(data.items()):
        color = CONDITION_COLORS.get(cond_name, f"C{idx}")

        if show_ci and values.ndim == 2:
            # values is (n_subjects, n_x_labels) - compute mean and CI
            means = values.mean(axis=0)
            cis = [mean_ci(values[:, i]) for i in range(values.shape[1])]
            yerr = [
                [means[i] - cis[i][0] for i in range(len(means))],
                [cis[i][1] - means[i] for i in range(len(means))],
            ]
            ax.bar(x + idx * bar_width, means, width=bar_width, label=cond_name,
                   color=color)
            ax.errorbar(x + idx * bar_width, means, yerr=yerr, fmt="none",
                        ecolor="#333333", capsize=3)
        else:
            # values is 1D - just plot directly
            ax.bar(x + idx * bar_width, values, width=bar_width, label=cond_name,
                   color=color)

    ax.set_xticks(x + bar_width * (n_conditions - 1) / 2)
    ax.set_xticklabels(x_labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.axhline(0, color="#666666", linewidth=1)


def setup_faceted_subplots(
    n_panels: int,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (4.5, 3.5),
    sharey: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with faceted subplots.

    Parameters
    ----------
    n_panels : int
        Number of panels to create.
    ncols : int
        Number of columns.
    figsize_per_panel : tuple[float, float]
        (width, height) per panel in inches.
    sharey : bool
        Whether to share y-axis across panels.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and flattened array of axes.
    """
    nrows = int(np.ceil(n_panels / ncols))
    figsize = (figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)
    axes = np.atleast_1d(axes).flatten()

    # Turn off unused axes
    for ax in axes[n_panels:]:
        ax.axis("off")

    return fig, axes
