#!/usr/bin/env python3
"""
Neural Complexity - KSG Method Implementation
============================================

Implementation of the KSG (Kraskov-Stögbauer-Grassberger) method for calculating 
Neural Complexity (CN) from EEG data. This non-parametric approach uses k-nearest 
neighbor distances to estimate entropy without assuming any specific distribution.

Based on PLAN.md: Project: Estimating Consciousness from EEG using Neural Complexity
This method is particularly suitable for EEG data which we found to be highly non-Gaussian.
"""

import numpy as np
import pandas as pd
from scipy import spatial
from scipy.special import digamma, gammaln
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
DEFAULT_N_CHANNELS = 8              # Number of EEG channels to analyze
DEFAULT_EPOCH_LENGTH = 5.0           # Length of each epoch in seconds
DEFAULT_MAX_PARTITIONS = 200         # Maximum bipartitions to compute
DEFAULT_SUBSAMPLE_FACTOR_BROADBAND = 10  # Subsampling for broadband analysis
DEFAULT_SUBSAMPLE_FACTOR_SPECTRAL = 5    # Subsampling for spectral analysis
DEFAULT_N_JOBS = -1                  # Number of parallel jobs (-1 for all cores)
DEFAULT_VERBOSE = True               # Enable verbose output
DEFAULT_K = 3                        # Number of nearest neighbors for KSG

# File paths for analysis
DEFAULT_FILE_PATHS = {
    'awake': 'ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr',
    'sedation': 'ds005620/sub-1010/eeg/sub-1010_task-sed2_acq-rest_run-1_eeg.vhdr'
}

# Output file names
OUTPUT_BROADBAND_PLOT = "neural_complexity_ksg_broadband_results.png"
OUTPUT_SPECTRAL_PLOT = "neural_complexity_ksg_spectral_results.png"
OUTPUT_SPECTRAL_CSV = "neural_complexity_ksg_spectral_summary.csv"

# Numerical constants
EPSILON_KSG = 1e-15                  # Small value to avoid numerical issues
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

class NeuralComplexityKSG:
    """
    Neural Complexity calculation using KSG entropy estimation.
    
    The KSG method estimates entropy using k-nearest neighbor distances,
    which is more robust for non-Gaussian data than parametric methods.
    """
    
    def __init__(self, k=DEFAULT_K, verbose=DEFAULT_VERBOSE):
        """
        Initialize KSG neural complexity calculator.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors for entropy estimation
        verbose : bool
            Whether to print progress messages
        """
        self.k = k
        self.verbose = verbose
        self.results = {}
        
    def ksg_entropy(self, data, k=None):
        """
        Calculate entropy using KSG k-nearest neighbor method.
        
        Based on Kraskov, Stögbauer, and Grassberger (2004) Physical Review E.
        
        Parameters:
        -----------
        data : array-like, shape (n_channels, n_samples)
            Input data matrix
        k : int, optional
            Number of nearest neighbors (uses self.k if None)
            
        Returns:
        --------
        entropy : float
            Entropy in bits
        """
        if k is None:
            k = self.k
            
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels, n_samples = data.shape
        
        if n_channels == 1:
            return self._ksg_entropy_1d(data[0], k)
        
        X = data.T
        
        valid_mask = np.all(np.isfinite(X), axis=1)
        X = X[valid_mask]
        n_samples = X.shape[0]
        
        if n_samples < k + 1:
            log_print(f"Warning: Too few samples ({n_samples}) for k={k}", self.verbose)
            return 0.0
        
        tree = spatial.cKDTree(X)
        
        distances, _ = tree.query(X, k=k+1)  # k+1 because first neighbor is the point itself
        kth_distances = distances[:, k]  # k-th neighbor distance (index k is the k-th neighbor)
        
        kth_distances = np.maximum(kth_distances, EPSILON_KSG)
        
        log_volume_d = (n_channels / 2.0) * np.log(np.pi) - gammaln(n_channels / 2.0 + 1)

        entropy_nats = -digamma(k) + digamma(n_samples) + log_volume_d + \
                       (n_channels / n_samples) * np.sum(np.log(kth_distances))

        return entropy_nats / np.log(2)
    
    def _ksg_entropy_1d(self, data, k):
        """
        KSG entropy estimation for 1D data.
        
        Parameters:
        -----------
        data : array-like, shape (n_samples,)
            1D data array
        k : int
            Number of nearest neighbors
            
        Returns:
        --------
        entropy : float
            Entropy in bits
        """
        data = data[np.isfinite(data)]
        n_samples = len(data)
        
        if n_samples < k + 1:
            return 0.0
        
        sorted_data = np.sort(data)
        
        distances = np.array([sorted_data[i+k] - sorted_data[i] for i in range(n_samples - k)])
        distances = np.maximum(distances, EPSILON_KSG)

        entropy_nats = digamma(n_samples) - digamma(k) + np.log(n_samples - k) + np.mean(np.log(distances))
        return entropy_nats / np.log(2)
    
    def integration_bipartition(self, data, partition_indices, k=None):
        """
        Calculate integration for a single bipartition using KSG method.
        
        Integration I(S_k, X\S_k) = H(S_k) + H(X\S_k) - H(X)
        
        Parameters:
        -----------
        data : array-like, shape (n_channels, n_samples)
            Full EEG data
        partition_indices : list of int
            Indices of channels in first partition
        k : int, optional
            Number of nearest neighbors
            
        Returns:
        --------
        integration : float
            Integration value for this bipartition
        """
        if k is None:
            k = self.k
            
        n_channels = data.shape[0]
        all_indices = set(range(n_channels))
        
        subset1_indices = list(partition_indices)
        subset2_indices = list(all_indices - set(partition_indices))
        
        h_subset1 = self.ksg_entropy(data[subset1_indices, :], k)
        h_subset2 = self.ksg_entropy(data[subset2_indices, :], k)
        h_total = self.ksg_entropy(data, k)
        
        return h_subset1 + h_subset2 - h_total
    
    def calculate_neural_complexity(self, data, max_partitions=DEFAULT_MAX_PARTITIONS, k=None, verbose=DEFAULT_VERBOSE):
        """
        Calculate Neural Complexity using KSG method.
        
        CN(X) = (1/(2^(n-1) - 1)) * Σ I(S_k, X\S_k)
        
        Parameters:
        -----------
        data : array-like, shape (n_channels, n_samples)
            EEG data matrix
        max_partitions : int
            Maximum number of bipartitions to use
        k : int, optional
            Number of nearest neighbors
        verbose : bool
            Whether to print progress messages for this specific calculation.
            
        Returns:
        --------
        complexity : float
            Neural complexity value
        integration_values : list
            Integration values for each bipartition
        """
        if k is None:
            k = self.k
            
        n_channels, _ = data.shape
        
        log_print(f"Calculating complexity for epoch: {n_channels} channels, k={k}", verbose and self.verbose)
        
        partitions = generate_bipartitions(n_channels, max_partitions, verbose=False)
        
        try:
            integration_values = [self.integration_bipartition(data, p, k) for p in partitions]
        except Exception as e:
            log_print(f"Warning: Failed to calculate integration: {e}", verbose and self.verbose)
            integration_values = [0.0] * len(partitions)
        
        complexity = np.mean(integration_values) if integration_values else 0.0
        return complexity, integration_values
    
    def analyze_eeg_file(self, file_path, n_channels=DEFAULT_N_CHANNELS, epoch_length=DEFAULT_EPOCH_LENGTH, 
                        max_partitions=DEFAULT_MAX_PARTITIONS, k=None, subsample_factor=DEFAULT_SUBSAMPLE_FACTOR_BROADBAND, n_jobs=DEFAULT_N_JOBS):
        """
        Complete analysis pipeline for a single EEG file using KSG method.
        
        Parameters:
        -----------
        file_path : str
            Path to EEG file
        n_channels : int
            Number of channels to use (computational limit)
        epoch_length : float
            Length of epochs in seconds
        max_partitions : int
            Maximum bipartitions to calculate
        k : int, optional
            Number of nearest neighbors
        subsample_factor : int
            Factor to subsample data for computational efficiency
        n_jobs : int
            Number of CPU cores to use for parallel epoch processing (-1 for all)
            
        Returns:
        --------
        results : dict
            Analysis results
        """
        if not MNE_AVAILABLE:
            raise ImportError("MNE-Python is required for EEG file loading")
        
        if k is None:
            k = self.k
        
        log_print(f"\n{'='*HEADER_WIDTH}\nANALYZING (KSG): {file_path}\n{'='*HEADER_WIDTH}", self.verbose)
        
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
        
        epochs_data, selected_channels = preprocess_eeg(
            raw, n_channels=n_channels, epoch_length=epoch_length,
            subsample_factor=subsample_factor, verbose=self.verbose
        )
        
        num_cores = os.cpu_count() or 1
        n_jobs_to_use = min(n_jobs, num_cores) if n_jobs != -1 else num_cores

        log_print(f"\nProcessing {len(epochs_data)} epochs using {n_jobs_to_use} parallel jobs...", self.verbose)
        start_time = time.time()

        with Parallel(n_jobs=n_jobs_to_use) as parallel:
            results_parallel = parallel(
                delayed(self.calculate_neural_complexity)(
                    epoch_data, max_partitions=max_partitions, k=k, verbose=False
                )
                for epoch_data in tqdm(epochs_data, desc=f"Analyzing {os.path.basename(file_path)}", disable=not self.verbose)
            )
        
        complexity_values, _ = zip(*results_parallel)
        
        total_time = time.time() - start_time
        log_print(f"Epoch processing completed in {total_time:.2f} seconds.", self.verbose)
        
        results = {
            'file_path': file_path,
            'condition': _extract_condition(file_path),
            'method': 'KSG',
            'k': k,
            'n_channels': n_channels,
            'selected_channels': selected_channels,
            'n_epochs': len(epochs_data),
            'epoch_length': epoch_length,
            'complexity_values': complexity_values,
            'mean_complexity': np.mean(complexity_values),
            'std_complexity': np.std(complexity_values),
            'max_partitions_used': min(max_partitions, 2**(n_channels-1) - 1)
        }
        
        log_print(f"\nRESULTS:\nMean Neural Complexity (KSG): {results['mean_complexity']:.4f} ± {results['std_complexity']:.4f}", self.verbose)
        
        return results
    
    def compare_conditions(self, results_list):
        """
        Compare neural complexity between different conditions.
        
        Parameters:
        -----------
        results_list : list of dict
            Results from analyze_eeg_file for different conditions
            
        Returns:
        --------
        comparison : dict
            Statistical comparison results
        """
        log_print(f"\n{'='*HEADER_WIDTH}")
        log_print("CONDITION COMPARISON (KSG METHOD)")
        log_print(f"{'='*HEADER_WIDTH}")
        
        comparison = {}
        
        for results in results_list:
            condition = results['condition']
            mean_complexity = results['mean_complexity']
            std_complexity = results['std_complexity']
            n_epochs = results['n_epochs']
            k = results['k']
            
            comparison[condition] = {
                'mean': mean_complexity,
                'std': std_complexity,
                'n_epochs': n_epochs,
                'k': k,
                'complexity_values': results['complexity_values']
            }
            
            log_print(f"{condition}: {mean_complexity:.4f} ± {std_complexity:.4f} (n={n_epochs}, k={k})", self.verbose)
        
        # Calculate differences if we have two conditions
        if len(results_list) == 2:
            conditions = list(comparison.keys())
            diff = comparison[conditions[0]]['mean'] - comparison[conditions[1]]['mean']
            comparison['difference'] = diff
            comparison['percent_change'] = (diff / comparison[conditions[1]]['mean']) * 100
            
            log_print(f"\nDifference ({conditions[0]} - {conditions[1]}): {diff:.4f}", self.verbose)
            log_print(f"Percent change: {comparison['percent_change']:.1f}%", self.verbose)
        
        return comparison

# Wrapper function for spectral analysis compatibility
def calculate_cn_ksg(epoch_data, k=DEFAULT_K, max_partitions=DEFAULT_MAX_PARTITIONS, verbose=False):
    """
    Calculate Neural Complexity using KSG method for a single epoch.
    
    This function provides a simple interface for the spectral analysis framework.
    
    Parameters:
    -----------
    epoch_data : array-like, shape (n_channels, n_samples)
        Single epoch of EEG data
    k : int
        Number of nearest neighbors for KSG estimation
    max_partitions : int
        Maximum number of bipartitions to use
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    complexity : float
        Neural complexity value for this epoch
    """
    analyzer = NeuralComplexityKSG(k=k, verbose=verbose)
    complexity, _ = analyzer.calculate_neural_complexity(
        epoch_data, max_partitions=max_partitions, verbose=verbose
    )
    return complexity

def run_ksg_analysis(analysis_type, file_paths, n_channels=DEFAULT_N_CHANNELS, 
                    epoch_length=DEFAULT_EPOCH_LENGTH, max_partitions=DEFAULT_MAX_PARTITIONS,
                    k=DEFAULT_K, subsample_factor_broadband=DEFAULT_SUBSAMPLE_FACTOR_BROADBAND,
                    subsample_factor_spectral=DEFAULT_SUBSAMPLE_FACTOR_SPECTRAL,
                    n_jobs=DEFAULT_N_JOBS, verbose=DEFAULT_VERBOSE, output_dir='.'):
    """
    Run KSG neural complexity analysis.
    
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
    k : int
        Number of nearest neighbors for KSG estimation
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
    log_print("NEURAL COMPLEXITY ANALYSIS - KSG METHOD", verbose)
    log_print("=" * HEADER_WIDTH, verbose)
    log_print("Non-parametric entropy estimation using k-nearest neighbors.", verbose)
    
    results = {}
    
    # Traditional broadband analysis
    if analysis_type in ['broadband', 'both']:
        log_print(f"\n{'='*HEADER_WIDTH}", verbose)
        log_print("BROADBAND ANALYSIS", verbose)
        log_print("="*HEADER_WIDTH, verbose)
        
        analyzer = NeuralComplexityKSG(k=k, verbose=verbose)
        results_list = []
        
        for condition, file_path in file_paths.items():
            try:
                log_print(f"\nAnalyzing {condition} state...", verbose)
                
                results_obj = analyzer.analyze_eeg_file(
                    file_path=file_path,
                    n_channels=n_channels,
                    epoch_length=epoch_length,
                    max_partitions=max_partitions,
                    k=k,
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
            comparison = analyzer.compare_conditions(results_list)
            
            plot_results(
                results_list,
                method_name="KSG",
                save_path=os.path.join(output_dir, "plots", "neural_complexity_ksg_broadband_results.png")
            )
        
        results['broadband'] = results_list
    
    # Spectral band analysis
    if analysis_type in ['spectral', 'both']:
        log_print(f"\n{'='*HEADER_WIDTH}", verbose)
        log_print("SPECTRAL BAND ANALYSIS", verbose)
        log_print("="*HEADER_WIDTH, verbose)
        log_print("Analyzing neural complexity across different frequency bands...", verbose)
        log_print("Focus on gamma band (30-100 Hz) - key for consciousness!", verbose)
        
        try:
            spectral_results = analyze_spectral_complexity(
                file_paths=file_paths,
                complexity_func=calculate_cn_ksg,
                n_channels=n_channels,
                subsample_factor=subsample_factor_spectral,
                verbose=verbose,
                k=k,
                max_partitions=max_partitions
            )
            
            # Print summary
            print_spectral_summary(spectral_results, verbose)
            
            # Create spectral visualization
            plot_spectral_complexity_results(
                spectral_results,
                method_name="KSG",
                save_path=os.path.join(output_dir, "plots", "neural_complexity_ksg_spectral_results.png")
            )
            
            # Save detailed spectral results
            spectral_df = []
            for condition, bands in spectral_results.items():
                for band, data in bands.items():
                    spectral_df.append({
                        'condition': condition,
                        'frequency_band': band,
                        'mean_complexity': data['mean_complexity'],
                        'std_complexity': data['std_complexity'],
                        'n_epochs': data['n_epochs']
                    })
            
            pd.DataFrame(spectral_df).to_csv(
                os.path.join(output_dir, "summaries", "neural_complexity_ksg_spectral_summary.csv"), 
                index=False
            )
            log_print("\nSpectral results saved to neural_complexity_ksg_spectral_summary.csv", verbose)
            
            results['spectral'] = spectral_results
            
        except Exception as e:
            log_print(f"Error in spectral analysis: {e}", verbose)
            if verbose:
                import traceback
                traceback.print_exc()
    
    log_print(f"\n{'='*HEADER_WIDTH}", verbose)
    log_print("KSG ANALYSIS COMPLETE", verbose)
    log_print(f"{'='*HEADER_WIDTH}", verbose)
    log_print("Key Findings:", verbose)
    log_print("- Broadband analysis provides overall neural complexity", verbose)
    log_print("- Gamma band analysis targets consciousness-specific mechanisms", verbose)
    log_print("- Spectral decomposition reveals frequency-specific complexity patterns", verbose)
    
    return results 