"""Tests for estimators in eeg_analysis.analyzers.estimators."""

import numpy as np
import pytest

from eeg_analysis.analyzers.estimators import BinningEstimator


def sample_epoch():
    """Create a sample 2-channel epoch for testing."""
    base = np.linspace(0, 1, 32)
    channel_one = base
    channel_two = base + 0.5
    return np.vstack([channel_one, channel_two])


class TestBinningEstimator:
    """Tests for BinningEstimator class."""

    def test_calculate_integration_returns_float(self):
        data = sample_epoch()
        estimator = BinningEstimator(n_bins=5)
        value = estimator.calculate_integration(data, (0,))
        assert isinstance(value, float)

    def test_calculate_integration_returns_nonnegative(self):
        data = sample_epoch()
        estimator = BinningEstimator(n_bins=10)
        value = estimator.calculate_integration(data, (0,))
        assert value >= 0.0

    def test_different_bin_counts(self):
        data = sample_epoch()
        results = []
        for n_bins in [5, 10, 20]:
            estimator = BinningEstimator(n_bins=n_bins)
            results.append(estimator.calculate_integration(data, (0,)))
        # All should return valid floats
        assert all(isinstance(r, float) for r in results)

    def test_estimator_name(self):
        estimator = BinningEstimator()
        assert estimator.name == "Binning"

    def test_params_stored(self):
        estimator = BinningEstimator(n_bins=50)
        assert "n_bins" in estimator.params
        assert estimator.params["n_bins"] == 50
