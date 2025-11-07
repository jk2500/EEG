import numpy as np

from eeg_analysis.analyzers.estimators import (
    KSGEstimator,
    BinningEstimator,
    GaussianEstimator
)


def sample_epoch():
    base = np.linspace(0, 1, 32)
    channel_one = base
    channel_two = base + 0.5
    return np.vstack([channel_one, channel_two])


def test_ksg_estimator_returns_float():
    data = sample_epoch()
    estimator = KSGEstimator(k=2)
    value = estimator.calculate_integration(data, (0,))
    assert isinstance(value, float)


def test_binning_estimator_returns_float():
    data = sample_epoch()
    estimator = BinningEstimator(n_bins=5)
    value = estimator.calculate_integration(data, (0,))
    assert isinstance(value, float)


def test_gaussian_estimator_returns_float():
    data = sample_epoch()
    estimator = GaussianEstimator()
    value = estimator.calculate_integration(data, (0,))
    assert isinstance(value, float)
