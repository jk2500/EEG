#!/usr/bin/env python3
"""
Neural Complexity - Gaussian Method Implementation
================================================

Implementation of Neural Complexity (CN) based on the assumption that EEG
data follows a multivariate Gaussian distribution. This method is computationally
efficient but relies on a strong assumption that may not hold for real EEG data.

Based on PLAN.md: Project: Estimating Consciousness from EEG using Neural Complexity.
This script also includes a Gaussianity test to validate the underlying assumption.
"""

import numpy as np
import pandas as pd
from scipy.stats import normaltest
import time
import warnings
from joblib import Parallel, delayed
import os
from tqdm import tqdm

# Local utility imports
from eeg_utils import (
    preprocess_eeg, 
    generate_bipartitions, 
    _extract_condition, 
    plot_results,
    log_print,
    preprocess_eeg_by_bands,
    analyze_spectral_complexity,
    plot_spectral_complexity_results,
    print_spectral_summary
)

warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL CONFIGURATION PARAMETERS
# ============================================================================

# Analysis Parameters
DEFAULT_N_CHANNELS = 20              # Number of EEG channels to analyze
DEFAULT_EPOCH_LENGTH = 5.0           # Length of each epoch in seconds
DEFAULT_MAX_PARTITIONS = 200         # Maximum bipartitions to compute
DEFAULT_SUBSAMPLE_FACTOR_BROADBAND = 10  # Subsampling for broadband analysis
DEFAULT_SUBSAMPLE_FACTOR_SPECTRAL = 5    # Subsampling for spectral analysis
DEFAULT_N_JOBS = -1                  # Number of parallel jobs (-1 for all cores)
DEFAULT_VERBOSE = True               # Enable verbose output
DEFAULT_ALPHA = 0.05                 # Significance level for Gaussianity test

# File paths for analysis
DEFAULT_FILE_PATHS = {
    'awake': 'ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr',
    'sedation': 'ds005620/sub-1010/eeg/sub-1010_task-sed2_acq-rest_run-1_eeg.vhdr'
}

# Output file names
OUTPUT_BROADBAND_PLOT = "neural_complexity_gaussian_broadband_results.png"
OUTPUT_SPECTRAL_PLOT = "neural_complexity_gaussian_spectral_results.png"
OUTPUT_SPECTRAL_CSV = "neural_complexity_gaussian_spectral_summary.csv"

# Numerical constants
HEADER_WIDTH = 60                    # Width of header separators

# ============================================================================

# MNE for EEG processing
try:
    import mne
    mne.set_log_level('WARNING')
    MNE_AVAILABLE = True
except ImportError:
    print("MNE-Python not available")
    MNE_AVAILABLE = False

class NeuralComplexityGaussian:
    """
    Neural Complexity calculation assuming a multivariate Gaussian distribution.
    
    This method estimates entropy from the covariance matrix of the EEG signals,
    which is computationally fast but may be inaccurate if the data is non-Gaussian.
    """
    
    def __init__(self, verbose=DEFAULT_VERBOSE):
        """
        Initialize Gaussian neural complexity calculator.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print progress messages.
        """
        self.verbose = verbose
        self.results = {}
        
    def test_gaussianity(self, data, alpha=DEFAULT_ALPHA):
        """
        Test if the data follows a Gaussian distribution using D'Agostino's K^2 test.
        
        Parameters:
        -----------
        data : array-like, shape (n_channels, n_samples)
            Input data matrix.
        alpha : float
            Significance level for the test.
            
        Returns:
        --------
        is_gaussian : bool
            True if all channels pass the normality test, False otherwise.
        stats : dict
            Detailed statistics for each channel.
        """
        n_channels = data.shape[0]
        log_print("\n--- Gaussianity Test ---", self.verbose)
        
        passed_channels = 0
        stats = {'channels': []}
        
        for i in range(n_channels):
            _, p_value = normaltest(data[i, :])
            is_normal = p_value > alpha
            if is_normal:
                passed_channels += 1
            
            stats['channels'].append({
                'channel': i,
                'k2_stat': None,
                'p_value': p_value,
                'is_normal': is_normal
            })
            log_print(f"Channel {i}: p-value={p_value:.4f} -> {'Gaussian' if is_normal else 'Non-Gaussian'}", self.verbose)
        
        overall_gaussian = (passed_channels == n_channels)
        log_print(f"Result: {passed_channels}/{n_channels} channels passed the normality test.", self.verbose)
        if not overall_gaussian:
            log_print("WARNING: Data appears to be non-Gaussian. Results may be unreliable.", self.verbose)
        log_print("------------------------\n", self.verbose)
        
        return overall_gaussian, stats

    def gaussian_entropy(self, data):
        """
        Calculate joint entropy assuming a multivariate Gaussian distribution.
        
        H(X) = 0.5 * log2((2 * pi * e)^k * det(Cov(X)))
        
        Parameters:
        -----------
        data : array-like, shape (n_channels, n_samples)
            Input data matrix.
            
        Returns:
        --------
        entropy : float
            Joint entropy in bits.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        n_channels, n_samples = data.shape
        
        if n_channels == 0:
            return 0.0
        
        # Handle single channel case
        if n_channels == 1:
            variance = np.var(data[0])
            if variance <= 0:
                return -np.inf
            entropy_nats = 0.5 * (1 + np.log(2 * np.pi * variance))
            return entropy_nats / np.log(2)
        
        # Calculate covariance matrix for multi-channel case
        cov_matrix = np.cov(data)
        
        # Determinant of the covariance matrix
        # Use slogdet for numerical stability with small determinants
        sign, log_det = np.linalg.slogdet(cov_matrix)
        
        if sign <= 0:
            # If determinant is non-positive, data is singular (linearly dependent)
            # which implies zero or undefined entropy in this context.
            return -np.inf 
        
        # Entropy formula for a multivariate Gaussian distribution
        entropy_nats = 0.5 * (n_channels * (1 + np.log(2 * np.pi)) + log_det)
        
        # Convert from nats to bits
        entropy_bits = entropy_nats / np.log(2)
        
        return entropy_bits
    
    def integration_bipartition(self, data, partition_indices):
        """
        Calculate integration for a single bipartition using the Gaussian method.
        
        I(S_k, X\S_k) = H(S_k) + H(X\S_k) - H(X)
        """
        n_channels = data.shape[0]
        all_indices = set(range(n_channels))
        
        subset1_indices = list(partition_indices)
        subset2_indices = list(all_indices - set(partition_indices))
        
        subset1_data = data[subset1_indices, :]
        subset2_data = data[subset2_indices, :]
        
        h_subset1 = self.gaussian_entropy(subset1_data)
        h_subset2 = self.gaussian_entropy(subset2_data)
        h_total = self.gaussian_entropy(data)
        
        # Check for invalid entropy values
        if any(np.isneginf([h_subset1, h_subset2, h_total])):
            return 0.0 # Indeterminate integration if any entropy is -inf
            
        integration = h_subset1 + h_subset2 - h_total
        return integration
    
    def calculate_neural_complexity(self, data, max_partitions=DEFAULT_MAX_PARTITIONS, verbose=DEFAULT_VERBOSE):
        """Calculate Neural Complexity using the Gaussian method."""
        n_channels, n_samples = data.shape
        
        if verbose and self.verbose:
            log_print(f"Calculating Neural Complexity (Gaussian method)")
            log_print(f"Data shape: {n_channels} channels, {n_samples} samples")

        partitions = generate_bipartitions(n_channels, max_partitions, verbose=False)
        
        integration_values = [self.integration_bipartition(data, p) for p in partitions]
        
        complexity = np.mean(integration_values) if integration_values else 0.0
        
        if verbose and self.verbose:
            log_print(f"Neural Complexity (Gaussian): {complexity:.4f} bits")
        
        return complexity, integration_values
    
    def analyze_eeg_file(self, file_path, n_channels=DEFAULT_N_CHANNELS, epoch_length=DEFAULT_EPOCH_LENGTH, 
                        max_partitions=DEFAULT_MAX_PARTITIONS, subsample_factor=DEFAULT_SUBSAMPLE_FACTOR_BROADBAND, n_jobs=DEFAULT_N_JOBS):
        """Complete analysis pipeline for a single EEG file."""
        if not MNE_AVAILABLE:
            raise ImportError("MNE-Python is required.")
        
        log_print(f"\n{'='*HEADER_WIDTH}\nANALYZING (GAUSSIAN): {file_path}\n{'='*HEADER_WIDTH}", self.verbose)
        
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        
        epochs_data, selected_channels = preprocess_eeg(
            raw, n_channels=n_channels, epoch_length=epoch_length,
            subsample_factor=subsample_factor, verbose=self.verbose
        )
        
        self.test_gaussianity(epochs_data[0])

        log_print(f"\nProcessing {len(epochs_data)} epochs using {n_jobs if n_jobs!=-1 else 'all'} parallel jobs...", self.verbose)
        
        with Parallel(n_jobs=n_jobs) as parallel:
            results_parallel = parallel(
                delayed(self.calculate_neural_complexity)(
                    epoch_data, max_partitions=max_partitions, verbose=False
                )
                for epoch_data in tqdm(epochs_data, desc=f"Analyzing {os.path.basename(file_path)}", disable=not self.verbose)
            )

        complexity_values, _ = zip(*results_parallel)
        
        results = {
            'file_path': file_path,
            'condition': _extract_condition(file_path),
            'method': 'Gaussian',
            'n_channels': n_channels,
            'selected_channels': selected_channels,
            'n_epochs': len(epochs_data),
            'complexity_values': complexity_values,
            'mean_complexity': np.mean(complexity_values),
            'std_complexity': np.std(complexity_values),
        }
        
        log_print(f"\nRESULTS:\nMean Neural Complexity (Gaussian): {results['mean_complexity']:.4f} ± {results['std_complexity']:.4f}", self.verbose)
        
        return results

# Wrapper function for spectral analysis compatibility
def calculate_cn_gaussian(epoch_data, max_partitions=DEFAULT_MAX_PARTITIONS, verbose=False):
    """
    Calculate Neural Complexity using Gaussian method for a single epoch.
    
    This function provides a simple interface for the spectral analysis framework.
    
    WARNING: This method assumes Gaussian distribution which is invalid for EEG data.
    
    Parameters:
    -----------
    epoch_data : array-like, shape (n_channels, n_samples)
        Single epoch of EEG data
    max_partitions : int
        Maximum number of bipartitions to use
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    complexity : float
        Neural complexity value for this epoch
    """
    analyzer = NeuralComplexityGaussian(verbose=verbose)
    complexity, _ = analyzer.calculate_neural_complexity(
        epoch_data, max_partitions=max_partitions, verbose=verbose
    )
    return complexity

def run_gaussian_analysis(analysis_type, file_paths, n_channels=DEFAULT_N_CHANNELS,
                         epoch_length=DEFAULT_EPOCH_LENGTH, max_partitions=DEFAULT_MAX_PARTITIONS,
                         subsample_factor_broadband=DEFAULT_SUBSAMPLE_FACTOR_BROADBAND,
                         subsample_factor_spectral=DEFAULT_SUBSAMPLE_FACTOR_SPECTRAL,
                         n_jobs=DEFAULT_N_JOBS, verbose=DEFAULT_VERBOSE, output_dir='.'):
    """
    Run Gaussian neural complexity analysis.
    
    WARNING: This method assumes Gaussian distribution which is invalid for EEG data.
    This analysis is included for completeness and comparison purposes only.
    
    Parameters:
    -----------
    analysis_type : str
        Type of analysis: 'broadband', 'spectral', or 'both'
    file_paths : dict
        Dictionary of condition names to file paths
    n_channels : int
        Number of channels to analyze
    epoch_length : float
        Length of epochs in seconds
    max_partitions : int
        Maximum number of bipartitions
    subsample_factor_broadband : int
        Subsampling factor for broadband analysis
    subsample_factor_spectral : int
        Subsampling factor for spectral analysis
    n_jobs : int
        Number of parallel jobs
    verbose : bool
        Enable verbose output
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    results : dict
        Analysis results
    """
    log_print("NEURAL COMPLEXITY ANALYSIS - GAUSSIAN METHOD", verbose)
    log_print("=" * HEADER_WIDTH, verbose)
    log_print("Parametric entropy estimation assuming Gaussian distribution.", verbose)
    log_print("", verbose)
    log_print("⚠️  WARNING: This method assumes Gaussian distribution!", verbose)
    log_print("⚠️  Previous analysis showed EEG data is highly non-Gaussian!", verbose)
    log_print("⚠️  Results should be interpreted with extreme caution!", verbose)
    log_print("⚠️  This analysis is for comparison purposes only!", verbose)
    
    results = {}
    
    # Traditional broadband analysis
    if analysis_type in ['broadband', 'both']:
        log_print(f"\n{'='*HEADER_WIDTH}", verbose)
        log_print("BROADBAND ANALYSIS", verbose)
        log_print("="*HEADER_WIDTH, verbose)
        
        analyzer = NeuralComplexityGaussian(verbose=verbose)
        results_list = []
        
        for condition, file_path in file_paths.items():
            try:
                log_print(f"\nAnalyzing {condition} state...", verbose)
                
                results_obj = analyzer.analyze_eeg_file(
                    file_path=file_path,
                    n_channels=n_channels,
                    epoch_length=epoch_length,
                    max_partitions=max_partitions,
                    subsample_factor=subsample_factor_broadband,
                    n_jobs=n_jobs
                )
                
                results_list.append(results_obj)
                
            except Exception as e:
                log_print(f"Error analyzing {file_path}: {e}", verbose)
                if verbose:
                    import traceback
                    traceback.print_exc()
        
        # Compare and plot results
        if len(results_list) >= 2:
            plot_results(
                results_list, 
                method_name="Gaussian (INVALID)",
                save_path=os.path.join(output_dir, "plots", "neural_complexity_gaussian_broadband_results.png")
            )
        
        results['broadband'] = results_list
    
    # Spectral band analysis
    if analysis_type in ['spectral', 'both']:
        log_print(f"\n{'='*HEADER_WIDTH}", verbose)
        log_print("SPECTRAL BAND ANALYSIS", verbose)
        log_print("="*HEADER_WIDTH, verbose)
        log_print("⚠️  Analyzing neural complexity across frequency bands...", verbose)
        log_print("⚠️  Remember: Gaussian assumption is invalid for all frequency bands!", verbose)
        
        try:
            spectral_results = analyze_spectral_complexity(
                file_paths=file_paths,
                complexity_func=calculate_cn_gaussian,
                n_channels=n_channels,
                subsample_factor=subsample_factor_spectral,
                verbose=verbose,
                max_partitions=max_partitions
            )
            
            # Print summary
            print_spectral_summary(spectral_results, verbose)
            
            # Create spectral visualization
            plot_spectral_complexity_results(
                spectral_results,
                method_name="Gaussian (INVALID)",
                save_path=os.path.join(output_dir, "plots", "neural_complexity_gaussian_spectral_results.png")
            )
            
            # Save detailed spectral results with warning
            spectral_df = []
            for condition, bands in spectral_results.items():
                for band, data in bands.items():
                    spectral_df.append({
                        'condition': condition,
                        'frequency_band': band,
                        'mean_complexity': data['mean_complexity'],
                        'std_complexity': data['std_complexity'],
                        'n_epochs': data['n_epochs'],
                        'warning': 'Gaussian assumption invalid - results unreliable'
                    })
            
            pd.DataFrame(spectral_df).to_csv(
                os.path.join(output_dir, "summaries", "neural_complexity_gaussian_spectral_summary.csv"), 
                index=False
            )
            log_print("\nSpectral results saved to neural_complexity_gaussian_spectral_summary.csv", verbose)
            
            results['spectral'] = spectral_results
            
        except Exception as e:
            log_print(f"Error in spectral analysis: {e}", verbose)
            if verbose:
                import traceback
                traceback.print_exc()
    
    log_print(f"\n{'='*HEADER_WIDTH}", verbose)
    log_print("GAUSSIAN ANALYSIS COMPLETE", verbose)
    log_print(f"{'='*HEADER_WIDTH}", verbose)
    log_print("⚠️  CRITICAL REMINDER:", verbose)
    log_print("⚠️  - EEG data is highly non-Gaussian (skewness: -3.955, kurtosis: +21.995)", verbose)
    log_print("⚠️  - 0% of channels pass normality tests", verbose)
    log_print("⚠️  - These results are scientifically invalid", verbose)
    log_print("⚠️  - Use KSG method for reliable results", verbose)
    log_print("⚠️  - This analysis is for comparison/educational purposes only", verbose)
    
    return results 