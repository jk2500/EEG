#!/usr/bin/env python3
"""
Mutual Information Estimators
=============================

This file provides different classes for estimating entropy and mutual
information, which are the core components for calculating neural complexity
and MIB. Each class implements a different estimation strategy.
"""

import numpy as np
from scipy import spatial
from scipy.special import digamma, gammaln
from scipy.stats import normaltest

from ..config import KSG_PARAMS, BINNING_PARAMS, GAUSSIAN_PARAMS, NUMERICAL_PARAMS
from ..eeg_utils import log_print

# --- Base Estimator (for type hinting and structure) ---

class BaseMIEstimator:
    """Base class for MI estimators."""
    def __init__(self, name, params):
        self.name = name
        self.params = params

    def calculate_integration(self, data, partition_indices):
        """Abstract method for calculating integration for a bipartition."""
        raise NotImplementedError

# --- KSG Estimator ---

class KSGEstimator(BaseMIEstimator):
    """
    Calculates Mutual Information using the KSG k-nearest neighbor method.
    """
    def __init__(self, **kwargs):
        params = {**KSG_PARAMS, **kwargs}
        super().__init__('KSG', params)

    def _ksg_entropy(self, data):
        k = self.params.get('k', KSG_PARAMS['k'])
        if data.ndim == 1: data = data.reshape(1, -1)
        n_channels, n_samples = data.shape

        if n_channels == 1:
            data_1d = data[0][np.isfinite(data[0])]
            n_samples_1d = len(data_1d)
            if n_samples_1d < k + 1: return 0.0
            
            sorted_data = np.sort(data_1d)
            distances = np.maximum(np.array([sorted_data[i+k] - sorted_data[i] for i in range(n_samples_1d - k)]), NUMERICAL_PARAMS['epsilon_ksg'])
            
            entropy_nats = digamma(n_samples_1d) - digamma(k) + np.log(n_samples_1d - k) + np.mean(np.log(distances))
            return entropy_nats / np.log(2)

        X = data.T[np.all(np.isfinite(data.T), axis=1)]
        n_samples = X.shape[0]
        if n_samples < k + 1: return 0.0
        
        tree = spatial.cKDTree(X)
        distances, _ = tree.query(X, k=k+1)
        kth_distances = np.maximum(distances[:, k], NUMERICAL_PARAMS['epsilon_ksg'])
        
        log_volume_d = (n_channels / 2.0) * np.log(np.pi) - gammaln(n_channels / 2.0 + 1)
        entropy_nats = -digamma(k) + digamma(n_samples) + log_volume_d + (n_channels / n_samples) * np.sum(np.log(kth_distances))
        return entropy_nats / np.log(2)

    def calculate_integration(self, data, partition_indices):
        all_indices = set(range(data.shape[0]))
        subset1_indices = list(partition_indices)
        subset2_indices = list(all_indices - set(partition_indices))
        
        h_subset1 = self._ksg_entropy(data[subset1_indices, :])
        h_subset2 = self._ksg_entropy(data[subset2_indices, :])
        h_total = self._ksg_entropy(data)
        
        return h_subset1 + h_subset2 - h_total

# --- Binning Estimator ---

class BinningEstimator(BaseMIEstimator):
    """
    Calculates Mutual Information using the binning (histogram) method.
    """
    def __init__(self, **kwargs):
        params = {**BINNING_PARAMS, **kwargs}
        super().__init__('Binning', params)

    def _binning_entropy(self, data):
        n_bins = self.params.get('n_bins', BINNING_PARAMS['n_bins'])
        if data.ndim == 1: data = data.reshape(1, -1)
        n_channels, _ = data.shape

        data_clean = data[:, np.all(np.isfinite(data), axis=0)]
        if data_clean.shape[1] == 0: return 0.0

        if n_channels == 1:
            counts, _ = np.histogram(data_clean[0], bins=n_bins)
            probs = counts / np.sum(counts)
            return -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))

        bin_edges = [np.linspace(np.min(ch), np.max(ch), n_bins + 1) if np.std(ch) > NUMERICAL_PARAMS['epsilon_binning'] else np.linspace(ch[0] - NUMERICAL_PARAMS['epsilon_binning'], ch[0] + NUMERICAL_PARAMS['epsilon_binning'], n_bins + 1) for ch in data_clean]
        digitized = np.vstack([np.clip(np.digitize(data_clean[i], bin_edges[i]) - 1, 0, n_bins - 1) for i in range(n_channels)])
        
        joint_indices = np.sum(digitized * (n_bins**np.arange(n_channels)).reshape(-1, 1), axis=0)
        _, counts = np.unique(joint_indices, return_counts=True)
        probs = counts / np.sum(counts)
        return -np.sum(probs * np.log2(probs))

    def calculate_integration(self, data, partition_indices):
        all_indices = set(range(data.shape[0]))
        subset1_indices = list(partition_indices)
        subset2_indices = list(all_indices - set(partition_indices))
        
        h_subset1 = self._binning_entropy(data[subset1_indices, :])
        h_subset2 = self._binning_entropy(data[subset2_indices, :])
        h_total = self._binning_entropy(data)
        
        return h_subset1 + h_subset2 - h_total

# --- Gaussian Estimator ---

class GaussianEstimator(BaseMIEstimator):
    """
    Calculates Mutual Information assuming a multivariate Gaussian distribution.
    WARNING: Scientifically invalid for EEG data. For comparison only.
    """
    def __init__(self, **kwargs):
        params = {**GAUSSIAN_PARAMS, **kwargs}
        super().__init__('Gaussian', params)

    def _gaussian_entropy(self, data):
        if data.ndim == 1: data = data.reshape(1, -1)
        n_channels, _ = data.shape
        if n_channels == 0: return 0.0

        if n_channels == 1:
            variance = np.var(data[0])
            return (0.5 * (1 + np.log(2 * np.pi * variance)) / np.log(2)) if variance > 0 else -np.inf
        
        cov_matrix = np.cov(data)
        sign, log_det = np.linalg.slogdet(cov_matrix)
        if sign <= 0: return -np.inf
        
        entropy_nats = 0.5 * (n_channels * (1 + np.log(2 * np.pi)) + log_det)
        return entropy_nats / np.log(2)

    def calculate_integration(self, data, partition_indices):
        all_indices = set(range(data.shape[0]))
        subset1_indices = list(partition_indices)
        subset2_indices = list(all_indices - set(partition_indices))
        
        h_subset1 = self._gaussian_entropy(data[subset1_indices, :])
        h_subset2 = self._gaussian_entropy(data[subset2_indices, :])
        h_total = self._gaussian_entropy(data)
        
        if any(np.isneginf([h_subset1, h_subset2, h_total])): return 0.0
        return h_subset1 + h_subset2 - h_total

    def test_gaussianity(self, data, verbose=True):
        alpha = self.params.get('alpha', GAUSSIAN_PARAMS['alpha'])
        n_channels = data.shape[0]
        passed = sum(1 for i in range(n_channels) if normaltest(data[i, :])[1] > alpha)
        log_print(f"Gaussianity Test: {passed}/{n_channels} channels passed.", verbose)
        if passed < n_channels:
            log_print("WARNING: Data appears non-Gaussian. Results are unreliable.", verbose)
        return passed == n_channels
