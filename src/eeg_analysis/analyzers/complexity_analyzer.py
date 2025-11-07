#!/usr/bin/env python3
"""
Unified MIB Analyzer
====================

This file defines the ComplexityAnalyzer, which encapsulates the logic for
running Minimum Information Bipartition (MIB) analyses. It is designed to work
with any of the provided MI estimators (KSG, Binning, etc.) while focusing on a
single metric.
"""

import numpy as np
import time
import os
import mne
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

# Local imports
from ..eeg_utils import (
    preprocess_eeg,
    preprocess_eeg_by_bands,
    generate_bipartitions,
    _extract_condition,
    plot_results,
    plot_spectral_complexity_results,
    print_spectral_summary,
    log_print,
    EEGDataCache
)
from ..config import ANALYSIS_PARAMS, UI_PARAMS

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')


@dataclass
class BroadbandResult:
    """Container for broadband metric summaries per file/condition."""
    file_path: str
    condition: str
    metric: str
    estimator: str
    n_channels: int
    selected_channels: List[str]
    n_epochs: int
    epoch_length: float
    metric_values: List[float]
    mean_metric: float
    std_metric: float
    params: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        """Return a plotting/serialization friendly dict."""
        return {
            'file_path': self.file_path,
            'condition': self.condition,
            'metric': self.metric,
            'estimator': self.estimator,
            'n_channels': self.n_channels,
            'selected_channels': self.selected_channels,
            'n_epochs': self.n_epochs,
            'epoch_length': self.epoch_length,
            'metric_values': self.metric_values,
            'mean_metric': self.mean_metric,
            'std_metric': self.std_metric,
            'params': self.params
        }

class ComplexityAnalyzer:
    """
    Analyzer dedicated to computing the Minimum Information Bipartition (MIB).
    
    1. Load and preprocess EEG data.
    2. Run epoch-wise analysis in parallel using the specified estimator.
    3. Calculate the MIB for each epoch and summarize the results.
    4. Handle both broadband and spectral analysis.
    """
    
    def __init__(self, estimator, verbose=True, **kwargs):
        """
        Initialize the analyzer for MIB computation.
        
        Parameters:
        -----------
        estimator : BaseMIEstimator
            An instance of an MI estimator (e.g., KSGEstimator).
        verbose : bool
            Whether to print progress messages.
        **kwargs : dict
            Method-specific parameters.
        """
        self.estimator = estimator
        self.metric = 'mib'
        self.method_name = f"{self.metric.upper()}-{self.estimator.name}"
        self.verbose = verbose
        self.params = {**ANALYSIS_PARAMS, **self.estimator.params, **kwargs}
        self.results = {}
        self._partition_cache: Dict[tuple, List[tuple]] = {}
        self._raw_cache = EEGDataCache(max_cache_size=self.params.get('raw_cache_size', 3))

    def _calculate_metric_for_epoch(self, data, partitions=None, **kwargs):
        """
        Calculate the Minimum Information Bipartition (MIB) for a single epoch.
        
        Parameters:
        -----------
        data : np.ndarray
            The data for a single epoch, shape (n_channels, n_samples).
        **kwargs : dict
            Additional parameters (not used here but kept for compatibility).
            
        Returns:
        --------
        float or None
            Minimum integration value across all bipartitions, or None if unavailable.
        """
        n_channels, _ = data.shape
        partitions = partitions or self._get_partitions(n_channels)

        integration_values = [self.estimator.calculate_integration(data, p) for p in partitions]
        integration_values = [v for v in integration_values if v is not None and np.isfinite(v)]
        
        if not integration_values:
            return None
                
        return np.min(integration_values)

    def analyze_eeg_file(self, file_path):
        """
        Complete analysis pipeline for a single EEG file (broadband).
        """
        header = f"\n{'='*UI_PARAMS['header_width']}\nANALYZING ({self.method_name}): {os.path.basename(file_path)}\n{'='*UI_PARAMS['header_width']}"
        log_print(header, self.verbose)

        epochs_data, selected_channels = self._load_epochs(file_path)
        metric_values = self._evaluate_epochs(
            epochs_data,
            progress_label=f"Analyzing {os.path.basename(file_path)}"
        )

        metric_values = [v for v in metric_values if v is not None]
        result = BroadbandResult(
            file_path=file_path,
            condition=_extract_condition(file_path),
            metric=self.metric,
            estimator=self.estimator.name,
            n_channels=self.params['n_channels'],
            selected_channels=list(selected_channels),
            n_epochs=len(metric_values),
            epoch_length=self.params['epoch_length'],
            metric_values=metric_values,
            mean_metric=float(np.mean(metric_values)) if metric_values else 0.0,
            std_metric=float(np.std(metric_values)) if metric_values else 0.0,
            params=self.params
        )

        log_print(
            f"\nRESULTS:\nMean {self.metric.upper()}: {result.mean_metric:.4f} \u00b1 {result.std_metric:.4f}",
            self.verbose
        )
        return result

    def run_analysis(self, analysis_type, file_paths, output_dir):
        """
        Run a full analysis, either broadband, spectral, or both.
        """
        log_print(f"Starting {self.method_name} analysis...", self.verbose)

        normalized_type = analysis_type.lower()
        analysis_results: Dict[str, Any] = {}

        if normalized_type in ('broadband', 'both'):
            analysis_results['broadband'] = self._run_broadband(file_paths, output_dir)

        if normalized_type in ('spectral', 'both'):
            analysis_results['spectral'] = self._run_spectral(file_paths, output_dir)

        log_print(f"\n{self.method_name} analysis complete.", self.verbose)
        return analysis_results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_partitions(self, n_channels: int) -> List[tuple]:
        """Return (and cache) bipartitions for the given channel count."""
        max_partitions = self.params.get('max_partitions')
        seed = self.params.get('partition_seed')
        cache_key = (n_channels, max_partitions, seed)

        if cache_key not in self._partition_cache:
            partitions = generate_bipartitions(
                n_channels,
                max_partitions,
                verbose=self.verbose,
                random_state=seed
            )
            self._partition_cache[cache_key] = partitions
        return self._partition_cache[cache_key]

    def _get_raw(self, file_path: str):
        """Load (and cache) raw EEG data."""
        return self._raw_cache.get_raw_data(file_path, verbose=self.verbose)

    def _load_epochs(self, file_path: str):
        """Load raw data and return normalized epochs plus channel names."""
        raw = self._get_raw(file_path)
        return preprocess_eeg(raw, **self.params)

    def _iter_file_items(self, file_paths: Dict[str, str]):
        """Iterate over provided files in a deterministic order."""
        return sorted(file_paths.items(), key=lambda item: item[0])

    def _evaluate_epochs(self, epochs_data, progress_label: str, verbose_override: Optional[bool] = None):
        """Evaluate the MIB metric for a batch of epochs."""
        if epochs_data is None:
            return []

        if isinstance(epochs_data, np.ndarray):
            epochs_array = epochs_data
        else:
            if not epochs_data:
                return []
            epochs_array = np.stack(epochs_data)
        num_epochs = epochs_array.shape[0]
        if num_epochs == 0:
            return []

        n_jobs = self.params.get('n_jobs', -1)
        num_cores = os.cpu_count() or 1
        n_jobs_to_use = min(n_jobs, num_cores) if n_jobs != -1 else num_cores
        use_verbose = self.verbose if verbose_override is None else verbose_override
        log_print(f"\nProcessing {num_epochs} epochs using {n_jobs_to_use} parallel jobs...", use_verbose)

        partitions = self._get_partitions(epochs_array.shape[1])
        start_time = time.time()
        with Parallel(n_jobs=n_jobs_to_use) as parallel:
            metric_values = parallel(
                delayed(self._calculate_metric_for_epoch)(epoch_data, partitions)
                for epoch_data in tqdm(
                    epochs_array,
                    desc=progress_label,
                    disable=not use_verbose
                )
            )
        total_time = time.time() - start_time
        log_print(f"Epoch processing completed in {total_time:.2f} seconds.", use_verbose)
        return metric_values

    def _run_broadband(self, file_paths: Dict[str, str], output_dir: str) -> List[Dict[str, Any]]:
        """Execute broadband analysis for each provided file path."""
        log_print(f"\n--- Running Broadband Analysis ---", self.verbose)
        broadband_results: List[BroadbandResult] = []
        for condition_key, file_path in self._iter_file_items(file_paths):
            try:
                broadband_results.append(self.analyze_eeg_file(file_path))
            except Exception as exc:
                log_print(f"Error analyzing {condition_key} ({file_path}): {exc}", self.verbose)

        result_payload = [result.as_dict() for result in broadband_results]
        if len(result_payload) >= 2:
            plot_results(
                result_payload,
                self.method_name,
                save_path=os.path.join(
                    output_dir, "plots", f"broadband_{self.method_name.lower()}.png"
                ),
                verbose=self.verbose
            )
        return result_payload

    def _run_spectral(self, file_paths: Dict[str, str], output_dir: str) -> Dict[str, Any]:
        """Execute spectral analysis workflow."""
        log_print(f"\n--- Running Spectral Analysis ---", self.verbose)
        spectral_results: Dict[str, Dict[str, Any]] = {}

        for condition_key, file_path in self._iter_file_items(file_paths):
            try:
                raw = self._get_raw(file_path)
                band_data, _ = preprocess_eeg_by_bands(raw, **self.params)
            except Exception as exc:
                log_print(f"Error preprocessing spectral data for {condition_key}: {exc}", self.verbose)
                continue

            condition_results = {}
            for band_name, epochs_data in band_data.items():
                metrics = self._evaluate_epochs(
                    epochs_data,
                    progress_label=f"{condition_key} | {band_name}"
                )
                metrics = [v for v in metrics if v is not None]
                if not metrics:
                    continue

                condition_results[band_name] = {
                    'metric_values': metrics,
                    'mean_metric': float(np.mean(metrics)),
                    'std_metric': float(np.std(metrics))
                }

            if condition_results:
                spectral_results[condition_key] = condition_results

        if spectral_results:
            print_spectral_summary(spectral_results, self.verbose)
            plot_spectral_complexity_results(
                spectral_results,
                self.method_name,
                save_path=os.path.join(
                    output_dir, "plots", f"spectral_{self.method_name.lower()}.png"
                )
            )
        else:
            log_print("No spectral results were generated.", self.verbose)
        return spectral_results
