#!/usr/bin/env python3
"""
Mutual Information Estimators (Numba-optimized)
===============================================

Binning-based MI estimation for MIB analysis. Uses histogram discretization
to estimate entropy: H(X) = -sum(p * log2(p)).

Integration (mutual info) for partition: I = H(A) + H(B) - H(A,B)
MIB = min(I) across all bipartitions.
"""

import numpy as np
from numba import njit

from ..config import BINNING_PARAMS


@njit(cache=True, fastmath=True)
def _compute_joint_entropy_fast(data, n_bins):
    """
    Fast joint entropy computation using uniform binning.

    Combines digitization and entropy calculation in a single pass
    where possible, minimizing memory allocations.
    """
    n_channels, n_samples = data.shape

    # Digitize all channels
    digitized = np.empty((n_channels, n_samples), dtype=np.int64)

    for i in range(n_channels):
        ch = data[i]
        mn = ch[0]
        mx = ch[0]
        for j in range(1, n_samples):
            if ch[j] < mn:
                mn = ch[j]
            if ch[j] > mx:
                mx = ch[j]

        rng = mx - mn
        if rng < 1e-10:
            for j in range(n_samples):
                digitized[i, j] = 0
        else:
            scale = (n_bins - 1e-10) / rng
            for j in range(n_samples):
                bin_idx = int((ch[j] - mn) * scale)
                if bin_idx < 0:
                    bin_idx = 0
                elif bin_idx >= n_bins:
                    bin_idx = n_bins - 1
                digitized[i, j] = bin_idx

    # Compute joint index
    joint_indices = np.zeros(n_samples, dtype=np.int64)
    multiplier = 1
    for i in range(n_channels):
        for j in range(n_samples):
            joint_indices[j] += digitized[i, j] * multiplier
        multiplier *= n_bins

    # Sort and count unique values
    joint_indices_sorted = np.sort(joint_indices)

    # Compute entropy from sorted counts
    entropy = 0.0
    count = 1
    for i in range(1, n_samples):
        if joint_indices_sorted[i] == joint_indices_sorted[i - 1]:
            count += 1
        else:
            prob = count / n_samples
            entropy -= prob * np.log2(prob)
            count = 1

    # Last group
    prob = count / n_samples
    entropy -= prob * np.log2(prob)

    return entropy


@njit(cache=True, fastmath=True)
def _compute_entropy_1d_fast(data, n_bins):
    """Fast 1D entropy using direct bincount."""
    n_samples = len(data)

    # Find min/max
    mn = data[0]
    mx = data[0]
    for i in range(1, n_samples):
        if data[i] < mn:
            mn = data[i]
        if data[i] > mx:
            mx = data[i]

    rng = mx - mn
    if rng < 1e-10:
        return 0.0

    # Count directly into bins
    counts = np.zeros(n_bins, dtype=np.int64)
    scale = (n_bins - 1e-10) / rng

    for i in range(n_samples):
        bin_idx = int((data[i] - mn) * scale)
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= n_bins:
            bin_idx = n_bins - 1
        counts[bin_idx] += 1

    # Compute entropy
    entropy = 0.0
    for i in range(n_bins):
        if counts[i] > 0:
            prob = counts[i] / n_samples
            entropy -= prob * np.log2(prob)

    return entropy


@njit(cache=True, fastmath=True)
def _binning_entropy_numba(data, n_bins):
    """
    Numba-optimized entropy computation.

    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_channels, n_samples), must be contiguous and finite.
    n_bins : int
        Number of bins for discretization.

    Returns
    -------
    float
        Entropy in bits.
    """
    n_channels = data.shape[0]

    if n_channels == 1:
        return _compute_entropy_1d_fast(data[0], n_bins)

    return _compute_joint_entropy_fast(data, n_bins)


@njit(cache=True, fastmath=True)
def _calculate_integration_numba(data, subset1_indices, subset2_indices, n_bins):
    """
    Calculate integration for a single bipartition.

    This is the hot path - called once per partition per epoch.
    """
    n_samples = data.shape[1]

    # Extract subsets directly
    len1 = len(subset1_indices)
    len2 = len(subset2_indices)

    data1 = np.empty((len1, n_samples), dtype=np.float64)
    for i in range(len1):
        data1[i] = data[subset1_indices[i]]

    data2 = np.empty((len2, n_samples), dtype=np.float64)
    for i in range(len2):
        data2[i] = data[subset2_indices[i]]

    h1 = _binning_entropy_numba(data1, n_bins)
    h2 = _binning_entropy_numba(data2, n_bins)
    h_total = _binning_entropy_numba(data, n_bins)

    return h1 + h2 - h_total


class BinningEstimator:
    """
    Histogram-based MI estimator for MIB computation.

    Key param: n_bins (default 10) - higher = more precision, more computation.
    Typical: n_bins=50 for analysis, n_bins=10 for quick tests.
    """

    def __init__(self, **kwargs):
        self.params = {**BINNING_PARAMS, **kwargs}
        self.name = 'Binning'
        # Pre-compute complement indices for common channel counts
        self._complement_cache = {}

    def _get_complement(self, n_channels, partition_indices):
        """Get complement indices, with caching."""
        key = (n_channels, partition_indices)
        if key not in self._complement_cache:
            all_set = set(range(n_channels))
            self._complement_cache[key] = tuple(sorted(all_set - set(partition_indices)))
        return self._complement_cache[key]

    def _binning_entropy(self, data):
        """Compute entropy using Numba-optimized implementation."""
        n_bins = self.params.get('n_bins', BINNING_PARAMS['n_bins'])

        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Clean data - remove non-finite columns
        finite_mask = np.all(np.isfinite(data), axis=0)
        if not np.any(finite_mask):
            return 0.0

        data_clean = np.ascontiguousarray(data[:, finite_mask])
        return _binning_entropy_numba(data_clean, n_bins)

    def calculate_integration(self, data, partition_indices):
        """Calculate integration (mutual information) for a bipartition."""
        n_bins = self.params.get('n_bins', BINNING_PARAMS['n_bins'])
        n_channels = data.shape[0]

        # Get complement indices
        subset2_indices = self._get_complement(n_channels, tuple(partition_indices))

        # Convert to numpy arrays for Numba
        subset1 = np.array(partition_indices, dtype=np.int64)
        subset2 = np.array(subset2_indices, dtype=np.int64)

        # Ensure contiguous
        data = np.ascontiguousarray(data)

        return _calculate_integration_numba(data, subset1, subset2, n_bins)
