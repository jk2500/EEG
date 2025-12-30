"""
Helper Functions
================

Miscellaneous utility functions used across the project.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import normaltest


def log_print(message: str, verbose: bool = True) -> None:
    """
    Print a message if verbose mode is enabled.

    Parameters
    ----------
    message : str
        Message to print.
    verbose : bool
        Whether to print the message.
    """
    if verbose:
        print(message)


def check_gaussianity(
    signal: np.ndarray,
    alpha: float | None = None,
    axis: int = -1,
    min_samples: int = 8,
    verbose: bool = True,
    return_p_values: bool = False,
) -> bool | tuple[bool, np.ndarray]:
    """
    Check Gaussianity using D'Agostino-Pearson normality test.

    Parameters
    ----------
    signal : array-like
        Input signal(s) to test.
    alpha : float, optional
        Significance level (default: 0.05).
    axis : int
        Axis along which to test (default: -1).
    min_samples : int
        Minimum number of samples required for the test (default: 8).
    verbose : bool
        Whether to print results.
    return_p_values : bool
        Whether to return p-values array.

    Returns
    -------
    bool or Tuple[bool, np.ndarray]
        Whether all signals passed the normality test. If return_p_values=True,
        also returns array of p-values.
    """
    if alpha is None:
        alpha = 0.05

    data = np.asarray(signal)
    if data.ndim == 0:
        raise ValueError("Signal must be at least 1D.")

    if data.ndim == 1:
        samples = data.reshape(1, -1)
        output_shape = ()
    else:
        data_moved = np.moveaxis(data, axis, -1)
        output_shape = data_moved.shape[:-1]
        samples = data_moved.reshape(-1, data_moved.shape[-1])

    p_values = np.full(samples.shape[0], np.nan, dtype=float)
    invalid = 0
    for i, row in enumerate(samples):
        row = row[np.isfinite(row)]
        if row.size < min_samples or np.std(row) <= 0:
            invalid += 1
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                p_values[i] = normaltest(row)[1]
            except ValueError:
                invalid += 1

    passed = int(np.sum(p_values > alpha))
    total = p_values.size
    is_gaussian = passed == total

    if verbose:
        log_print(
            f"Gaussianity Test (normaltest): {passed}/{total} signals passed (alpha={alpha}).",
            verbose
        )
        if invalid:
            log_print(
                f"{invalid}/{total} signals could not be tested (insufficient samples/variance) "
                "and were treated as non-Gaussian.",
                verbose
            )
        if not is_gaussian:
            log_print("WARNING: Data appears non-Gaussian. Results are unreliable.", verbose)

    if return_p_values:
        return is_gaussian, p_values.reshape(output_shape)
    return is_gaussian
