#!/usr/bin/env python3
"""
Unified Complexity Analyzer
===========================

This file defines the ComplexityAnalyzer, which encapsulates the logic for
running neural complexity and MIB analyses. It is designed to work with
any of the provided MI estimators (KSG, Binning, etc.) and can calculate
different metrics (Neural Complexity, MIB).
"""

import numpy as np
import pandas as pd
import time
import os
import mne
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings

# Local imports
from eeg_utils import (
    preprocess_eeg,
    generate_bipartitions,
    generate_all_subsets,
    find_local_maxima,
    _extract_condition,
    plot_results,
    log_print,
    preprocess_eeg_by_bands,
    analyze_spectral_complexity,
    plot_spectral_complexity_results,
    print_spectral_summary
)
from config import ANALYSIS_PARAMS, UI_PARAMS

warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

class ComplexityAnalyzer:
    """
    A flexible analyzer for calculating Neural Complexity and MIB.
    
    This class orchestrates the analysis pipeline by taking a specific 
    MI estimator and a metric, allowing for modular and clear configuration.
    
    1. Load and preprocess EEG data.
    2. Run analysis across epochs in parallel using the specified estimator.
    3. Calculate the final metric (e.g., mean for complexity, min for MIB).
    4. Handle both broadband and spectral analysis.
    """
    
    def __init__(self, estimator, metric='complexity', verbose=True, **kwargs):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        estimator : BaseMIEstimator
            An instance of an MI estimator (e.g., KSGEstimator).
        metric : str
            The metric to calculate: 'complexity' (mean), 'mib' (min), or 'complexes' (IIT complexes).
        verbose : bool
            Whether to print progress messages.
        **kwargs : dict
            Method-specific parameters.
        """
        self.estimator = estimator
        self.metric = metric.lower()
        self.method_name = f"{self.metric.upper()}-{self.estimator.name}"
        self.verbose = verbose
        self.params = {**ANALYSIS_PARAMS, **self.estimator.params, **kwargs}
        self.results = {}

    def _calculate_metric_for_epoch(self, data, **kwargs):
        """
        Calculate the desired metric (Complexity, MIB, or Complexes) for a single epoch.
        
        This method generates all bipartitions (for complexity/MIB) or all subsets
        (for complexes), calculates integration for each using the provided estimator,
        and then computes the final metric.
        
        Parameters:
        -----------
        data : np.ndarray
            The data for a single epoch, shape (n_channels, n_samples).
        **kwargs : dict
            Additional parameters (not used here but kept for compatibility).
            
        Returns:
        --------
        metric_value : float or list
            For complexity/MIB: single float value
            For complexes: list of (subset, phi_value) tuples representing top complexes
        """
        n_channels, _ = data.shape
        
        if self.metric == 'complexes':
            # IIT Complexes: Calculate Φ(S) for all subsets S, find local maxima
            return self._calculate_iit_complexes(data, **kwargs)
        else:
            # Traditional approach: all bipartitions of the full system
            max_partitions = self.params.get('max_partitions')
            partitions = generate_bipartitions(n_channels, max_partitions, verbose=False)
            
            integration_values = [self.estimator.calculate_integration(data, p) for p in partitions]
            integration_values = [v for v in integration_values if v is not None and np.isfinite(v)]
            
            if not integration_values:
                return 0.0
                
            if self.metric == 'mib':
                return np.min(integration_values)
            else: # Default to neural complexity (mean)
                return np.mean(integration_values)
    
    def _calculate_iit_complexes(self, data, **kwargs):
        """
        Calculate IIT complexes - subsets with locally maximal Φ values.
        
        This is the correct implementation of IIT:
        1. For each subset S ⊆ X, calculate Φ(S) = MIB(S)
        2. Find subsets whose Φ is a local maximum
        3. Return the top complexes
        
        Parameters:
        -----------
        data : np.ndarray
            The data for a single epoch, shape (n_channels, n_samples).
        **kwargs : dict
            Additional parameters.
            
        Returns:
        --------
        complexes : list
            List of (subset, phi_value) tuples representing top complexes
        """
        n_channels, _ = data.shape
        max_subset_size = self.params.get('max_subset_size', n_channels)
        top_n = self.params.get('top_complexes', 30)
        
        # 1. Generate all possible subsets
        all_subsets = generate_all_subsets(n_channels, min_size=2, max_size=max_subset_size)
        
        # 2. Calculate Φ(S) for each subset S
        phi_values = {}
        for subset in all_subsets:
            if len(subset) >= 2:  # Need at least 2 channels for bipartition
                subset_data = data[list(subset), :]
                
                # Generate bipartitions of this specific subset
                subset_partitions = generate_bipartitions(len(subset), verbose=False)
                
                # Calculate integration for each bipartition of this subset
                integration_values = []
                for partition in subset_partitions:
                    # Calculate integration using the subset data
                    integration = self.estimator.calculate_integration(subset_data, partition)
                    if integration is not None and np.isfinite(integration):
                        integration_values.append(integration)
                
                # MIB of this subset (minimum integration)
                if integration_values:
                    phi_values[subset] = np.min(integration_values)
        
        # 3. Find complexes (local maxima)
        complexes = find_local_maxima(phi_values, top_n=top_n)
        
        return complexes

    def analyze_eeg_file(self, file_path):
        """
        Complete analysis pipeline for a single EEG file (broadband).
        """
        log_print(f"\n{'='*UI_PARAMS['header_width']}\nANALYZING ({self.method_name}): {os.path.basename(file_path)}\n{'='*UI_PARAMS['header_width']}", self.verbose)
        
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        epochs_data, selected_channels = preprocess_eeg(raw, **self.params)
        
        n_jobs = self.params.get('n_jobs', -1)
        num_cores = os.cpu_count() or 1
        n_jobs_to_use = min(n_jobs, num_cores) if n_jobs != -1 else num_cores

        log_print(f"\nProcessing {len(epochs_data)} epochs using {n_jobs_to_use} parallel jobs...", self.verbose)
        start_time = time.time()

        with Parallel(n_jobs=n_jobs_to_use) as parallel:
            metric_values = parallel(
                delayed(self._calculate_metric_for_epoch)(epoch_data)
                for epoch_data in tqdm(epochs_data, desc=f"Analyzing {os.path.basename(file_path)}", disable=not self.verbose)
            )
        
        total_time = time.time() - start_time
        log_print(f"Epoch processing completed in {total_time:.2f} seconds.", self.verbose)
        
        metric_values = [v for v in metric_values if v is not None]
        
        if self.metric == 'complexes':
            # For complexes, metric_values is a list of lists of (subset, phi_value) tuples
            results = {
                'file_path': file_path,
                'condition': _extract_condition(file_path),
                'metric': self.metric,
                'estimator': self.estimator.name,
                'n_channels': self.params['n_channels'],
                'selected_channels': selected_channels,
                'n_epochs': len(metric_values),
                'epoch_length': self.params['epoch_length'],
                'complexes_per_epoch': metric_values,  # List of complexes for each epoch
                'params': self.params
            }
            
            # Calculate aggregate statistics for complexes
            if metric_values:
                all_phi_values = []
                for epoch_complexes in metric_values:
                    for subset, phi_value in epoch_complexes:
                        all_phi_values.append(phi_value)
                
                results['mean_phi'] = np.mean(all_phi_values) if all_phi_values else 0
                results['std_phi'] = np.std(all_phi_values) if all_phi_values else 0
                results['total_complexes'] = len(all_phi_values)
                
                log_print(f"\nRESULTS:\nTotal complexes found: {results['total_complexes']}", self.verbose)
                log_print(f"Mean Φ across all complexes: {results['mean_phi']:.4f} \u00b1 {results['std_phi']:.4f}", self.verbose)
                log_print(f"Average complexes per epoch: {results['total_complexes'] / len(metric_values):.1f}", self.verbose)
            else:
                results['mean_phi'] = 0
                results['std_phi'] = 0
                results['total_complexes'] = 0
                log_print(f"\nRESULTS:\nNo complexes found.", self.verbose)
        else:
            # Traditional complexity/MIB metrics
            results = {
                'file_path': file_path,
                'condition': _extract_condition(file_path),
                'metric': self.metric,
                'estimator': self.estimator.name,
                'n_channels': self.params['n_channels'],
                'selected_channels': selected_channels,
                'n_epochs': len(metric_values),
                'epoch_length': self.params['epoch_length'],
                'metric_values': metric_values,
                'mean_metric': np.mean(metric_values) if metric_values else 0,
                'std_metric': np.std(metric_values) if metric_values else 0,
                'params': self.params
            }
            
            log_print(f"\nRESULTS:\nMean {self.metric.upper()}: {results['mean_metric']:.4f} \u00b1 {results['std_metric']:.4f}", self.verbose)
        
        return results

    def run_analysis(self, analysis_type, file_paths, output_dir):
        """
        Run a full analysis, either broadband, spectral, or both.
        """
        log_print(f"Starting {self.method_name} analysis...", self.verbose)
        
        analysis_results = {}
        
        if analysis_type in ['broadband', 'both']:
            log_print(f"\n--- Running Broadband Analysis ---", self.verbose)
            broadband_results_list = []
            for condition, file_path in file_paths.items():
                try:
                    res = self.analyze_eeg_file(file_path)
                    broadband_results_list.append(res)
                except Exception as e:
                    log_print(f"Error analyzing {file_path}: {e}", self.verbose)
            
            if len(broadband_results_list) >= 2:
                plot_results(broadband_results_list, self.method_name, 
                             save_path=os.path.join(output_dir, "plots", f"broadband_{self.method_name.lower()}.png"),
                             verbose=self.verbose)
            
            analysis_results['broadband'] = broadband_results_list

        if analysis_type in ['spectral', 'both']:
            log_print(f"\n--- Running Spectral Analysis ---", self.verbose)
            try:
                spectral_results = analyze_spectral_complexity(
                    file_paths=file_paths,
                    complexity_func=self._calculate_metric_for_epoch,
                    n_channels=self.params['n_channels'],
                    verbose=self.verbose,
                    n_jobs=self.params['n_jobs'],
                    subsample_factor=self.params.get('subsample_factor_spectral', 2)
                )
                
                print_spectral_summary(spectral_results, self.verbose)
                plot_spectral_complexity_results(
                    spectral_results, self.method_name,
                    save_path=os.path.join(output_dir, "plots", f"spectral_{self.method_name.lower()}.png")
                )
                
                analysis_results['spectral'] = spectral_results
            except Exception as e:
                log_print(f"Error in spectral analysis: {e}", self.verbose)

        log_print(f"\n{self.method_name} analysis complete.", self.verbose)
        return analysis_results
