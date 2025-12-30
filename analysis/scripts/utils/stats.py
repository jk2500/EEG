"""Statistical utility functions for EEG analysis."""

import itertools
import numpy as np
from scipy.stats import t, ttest_1samp, wilcoxon


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """
    Compute the mean and 95% confidence interval using t-distribution.

    Parameters
    ----------
    values : np.ndarray
        Array of values to compute statistics for.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of confidence interval.
    """
    if values.size < 2:
        return (np.nan, np.nan)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half_width = t.ppf(1 - alpha / 2, df=values.size - 1) * std / np.sqrt(values.size)
    return (mean - half_width, mean + half_width)


def summarize_diff(diff: np.ndarray) -> dict:
    """
    Compute comprehensive statistics for a difference array.

    Includes t-test, Wilcoxon test, and Cohen's d effect size.

    Parameters
    ----------
    diff : np.ndarray
        Array of difference values (e.g., condition A - condition B).

    Returns
    -------
    dict
        Dictionary with keys: n, mean_diff, median_diff, std_diff, pos_frac,
        t_stat, t_p, cohen_d, wilcoxon_stat, wilcoxon_p, ci_low, ci_high.
    """
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


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for two independent samples.

    Parameters
    ----------
    a : np.ndarray
        First sample array.
    b : np.ndarray
        Second sample array.

    Returns
    -------
    float
        Cohen's d effect size, or NaN if computation is not possible.
    """
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
    """
    Apply Benjamini-Hochberg FDR correction to p-values.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of raw p-values.

    Returns
    -------
    np.ndarray
        Array of FDR-adjusted p-values.
    """
    order = np.argsort(pvalues)
    ranks = np.arange(1, len(pvalues) + 1)
    p_sorted = pvalues[order]
    adj_sorted = p_sorted * len(pvalues) / ranks
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adjusted = np.empty_like(pvalues, dtype=float)
    adjusted[order] = adj_sorted
    return np.clip(adjusted, 0.0, 1.0)


def jaccard_mean(sets: list[set[str]]) -> float:
    """
    Compute mean pairwise Jaccard similarity across a list of sets.

    Parameters
    ----------
    sets : list[set[str]]
        List of sets to compare.

    Returns
    -------
    float
        Mean Jaccard similarity across all pairs, or NaN if fewer than 2 sets.
    """
    if len(sets) < 2:
        return np.nan
    total = 0.0
    count = 0
    for a, b in itertools.combinations(sets, 2):
        denom = len(a | b)
        total += len(a & b) / denom if denom else 0.0
        count += 1
    return total / count if count else np.nan
