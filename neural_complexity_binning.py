#!/usr/bin/env python3
"""
Neural Complexity - Binning (Histogram) Method Implementation
===========================================================

Implementation of the Binning method for calculating Neural Complexity (CN) from EEG data.
This non-parametric approach discretizes continuous data into bins and uses histograms 
to estimate joint probability distributions for entropy calculation.

Based on PLAN.md: Project: Estimating Consciousness from EEG using Neural Complexity
This method avoids distribution assumptions but is sensitive to bin size selection.
"""

import numpy as np
import pandas as pd
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
DEFAULT_N_BINS = 10                  # Number of bins for histogram discretization
DEFAULT_EPOCH_LENGTH = 5.0           # Length of each epoch in seconds
DEFAULT_MAX_PARTITIONS = 200         # Maximum bipartitions to compute
DEFAULT_SUBSAMPLE_FACTOR_BROADBAND = 10  # Subsampling for broadband analysis
DEFAULT_SUBSAMPLE_FACTOR_SPECTRAL = 5    # Subsampling for spectral analysis
DEFAULT_N_JOBS = -1                  # Number of parallel jobs (-1 for all cores)
DEFAULT_VERBOSE = True               # Enable verbose output

# File paths for analysis
DEFAULT_FILE_PATHS = {
    'awake': 'ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr',
    'sedation': 'ds005620/sub-1010/eeg/sub-1010_task-sed2_acq-rest_run-1_eeg.vhdr'
}

# Output file names
OUTPUT_BROADBAND_PLOT = "neural_complexity_binning_broadband_results.png"
OUTPUT_BROADBAND_CSV = "neural_complexity_binning_broadband_summary.csv"
OUTPUT_SPECTRAL_PLOT = "neural_complexity_binning_spectral_results.png"
OUTPUT_SPECTRAL_CSV = "neural_complexity_binning_spectral_summary.csv"

# Numerical constants
EPSILON_CONSTANT_CHANNEL = 1e-10     # Small value for handling constant channels
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

class NeuralComplexityBinning:
    """
    Neural Complexity calculation using binning (histogram) entropy estimation.
    
    The binning method discretizes continuous EEG data into bins and estimates
    entropy from the resulting probability distributions.
    """
    
    def __init__(self, n_bins=DEFAULT_N_BINS, verbose=DEFAULT_VERBOSE):
        """
        Initialize binning neural complexity calculator.
        
        Parameters:
        -----------
        n_bins : int
            Number of bins for discretization
        verbose : bool
            Whether to print progress messages
        """
        self.n_bins = n_bins
        self.verbose = verbose
        self.results = {}
        
    def binning_entropy(self, data, n_bins=None):
        """
        Calculate entropy using binning (histogram) method.
        
        Parameters:
        -----------
        data : array-like, shape (n_channels, n_samples)
            Input data matrix
        n_bins : int, optional
            Number of bins for discretization (uses self.n_bins if None)
            
        Returns:
        --------
        entropy : float
            Entropy in bits
        """
        if n_bins is None:
            n_bins = self.n_bins
            
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        n_channels, n_samples = data.shape
        
        # Remove non-finite values
        valid_mask = np.all(np.isfinite(data), axis=0)
        if not np.any(valid_mask):
            log_print("Warning: No valid samples found", self.verbose)
            return 0.0
            
        data_clean = data[:, valid_mask]
        n_samples = data_clean.shape[1]
        
        if n_samples == 0:
            return 0.0
        
        if n_channels == 1:
            return self._binning_entropy_1d(data_clean[0], n_bins)
        else:
            return self._binning_entropy_nd(data_clean, n_bins)
    
    def _binning_entropy_1d(self, data, n_bins):
        """
        Calculate entropy for 1D data using binning.
        
        Parameters:
        -----------
        data : array-like, shape (n_samples,)
            1D data array
        n_bins : int
            Number of bins
            
        Returns:
        --------
        entropy : float
            Entropy in bits
        """
        if len(data) == 0:
            return 0.0
            
        # Create histogram
        counts, _ = np.histogram(data, bins=n_bins)
        
        # Convert to probabilities
        probabilities = counts / np.sum(counts)
        
        # Remove zero probabilities to avoid log(0)
        probabilities = probabilities[probabilities > 0]
        
        if len(probabilities) == 0:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _binning_entropy_nd(self, data, n_bins):
        """
        Calculate entropy for multi-dimensional data using binning.
        
        Parameters:
        -----------
        data : array-like, shape (n_channels, n_samples)
            Multi-dimensional data array
        n_bins : int
            Number of bins per dimension
            
        Returns:
        --------
        entropy : float
            Entropy in bits
        """
        n_channels, n_samples = data.shape
        
        if n_samples == 0:
            return 0.0
        
        # Determine bin edges for each channel
        bin_edges = []
        for i in range(n_channels):
            channel_data = data[i, :]
            if np.std(channel_data) == 0:  # Handle constant channels
                # Create artificial bins around the constant value
                value = channel_data[0]
                edges = np.linspace(value - EPSILON_CONSTANT_CHANNEL, value + EPSILON_CONSTANT_CHANNEL, n_bins + 1)
            else:
                edges = np.linspace(np.min(channel_data), np.max(channel_data), n_bins + 1)
            bin_edges.append(edges)
        
        # Digitize each channel
        digitized_data = np.zeros((n_channels, n_samples), dtype=int)
        for i in range(n_channels):
            digitized_data[i, :] = np.digitize(data[i, :], bin_edges[i]) - 1
            # Ensure all values are within valid range [0, n_bins-1]
            digitized_data[i, :] = np.clip(digitized_data[i, :], 0, n_bins - 1)
        
        # Create multi-dimensional histogram
        # Convert to single index for each sample
        multipliers = np.array([n_bins**i for i in range(n_channels)])
        joint_indices = np.sum(digitized_data * multipliers.reshape(-1, 1), axis=0)
        
        # Count occurrences
        unique_indices, counts = np.unique(joint_indices, return_counts=True)
        
        # Convert to probabilities
        probabilities = counts / np.sum(counts)
        
        # Remove zero probabilities
        probabilities = probabilities[probabilities > 0]
        
        if len(probabilities) == 0:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def integration_bipartition(self, data, partition_indices, n_bins=None):
        """
        Calculate integration for a single bipartition using binning method.
        
        Integration I(S_k, X\S_k) = H(S_k) + H(X\S_k) - H(X)
        
        Parameters:
        -----------
        data : array-like, shape (n_channels, n_samples)
            Full EEG data
        partition_indices : list of int
            Indices of channels in first partition
        n_bins : int, optional
            Number of bins for discretization
            
        Returns:
        --------
        integration : float
            Integration value for this bipartition
        """
        if n_bins is None:
            n_bins = self.n_bins
            
        n_channels = data.shape[0]
        all_indices = set(range(n_channels))
        
        subset1_indices = list(partition_indices)
        subset2_indices = list(all_indices - set(partition_indices))
        
        h_subset1 = self.binning_entropy(data[subset1_indices, :], n_bins)
        h_subset2 = self.binning_entropy(data[subset2_indices, :], n_bins)
        h_total = self.binning_entropy(data, n_bins)
        
        return h_subset1 + h_subset2 - h_total
    
    def calculate_neural_complexity(self, data, max_partitions=DEFAULT_MAX_PARTITIONS, n_bins=None, verbose=DEFAULT_VERBOSE):
        """
        Calculate Neural Complexity using binning method.
        
        CN(X) = (1/(2^(n-1) - 1)) * Σ I(S_k, X\S_k)
        
        Parameters:
        -----------
        data : array-like, shape (n_channels, n_samples)
            EEG data matrix
        max_partitions : int
            Maximum number of bipartitions to use
        n_bins : int, optional
            Number of bins for discretization
        verbose : bool
            Whether to print progress messages for this specific calculation.
            
        Returns:
        --------
        complexity : float
            Neural complexity value
        integration_values : list
            Integration values for each bipartition
        """
        if n_bins is None:
            n_bins = self.n_bins
            
        n_channels, _ = data.shape
        
        log_print(f"Calculating complexity for epoch: {n_channels} channels, {n_bins} bins", verbose and self.verbose)
        
        partitions = generate_bipartitions(n_channels, max_partitions, verbose=False)
        
        try:
            integration_values = [self.integration_bipartition(data, p, n_bins) for p in partitions]
        except Exception as e:
            log_print(f"Warning: Failed to calculate integration: {e}", verbose and self.verbose)
            integration_values = [0.0] * len(partitions)
        
        complexity = np.mean(integration_values) if integration_values else 0.0
        return complexity, integration_values
    
    def analyze_eeg_file(self, file_path, n_channels=DEFAULT_N_CHANNELS, epoch_length=DEFAULT_EPOCH_LENGTH, 
                        max_partitions=DEFAULT_MAX_PARTITIONS, n_bins=None, subsample_factor=DEFAULT_SUBSAMPLE_FACTOR_BROADBAND, n_jobs=DEFAULT_N_JOBS):
        """
        Complete analysis pipeline for a single EEG file using binning method.
        
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
        n_bins : int, optional
            Number of bins for discretization
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
        
        if n_bins is None:
            n_bins = self.n_bins
        
        log_print(f"\n{'='*60}\nANALYZING (BINNING): {file_path}\n{'='*60}", self.verbose)
        
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
                    epoch_data, max_partitions=max_partitions, n_bins=n_bins, verbose=False
                )
                for epoch_data in tqdm(epochs_data, desc=f"Analyzing {os.path.basename(file_path)}", disable=not self.verbose)
            )
        
        complexity_values, _ = zip(*results_parallel)
        
        total_time = time.time() - start_time
        log_print(f"Epoch processing completed in {total_time:.2f} seconds.", self.verbose)
        
        results = {
            'file_path': file_path,
            'condition': _extract_condition(file_path),
            'method': 'Binning',
            'n_bins': n_bins,
            'n_channels': n_channels,
            'selected_channels': selected_channels,
            'n_epochs': len(epochs_data),
            'epoch_length': epoch_length,
            'complexity_values': complexity_values,
            'mean_complexity': np.mean(complexity_values),
            'std_complexity': np.std(complexity_values),
            'max_partitions_used': min(max_partitions, 2**(n_channels-1) - 1)
        }
        
        log_print(f"\nRESULTS:\nMean Neural Complexity (Binning): {results['mean_complexity']:.4f} ± {results['std_complexity']:.4f}", self.verbose)
        
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
        log_print(f"\n{'='*60}")
        log_print("CONDITION COMPARISON (BINNING METHOD)")
        log_print(f"{'='*60}")
        
        comparison = {}
        
        for results in results_list:
            condition = results['condition']
            mean_complexity = results['mean_complexity']
            std_complexity = results['std_complexity']
            n_epochs = results['n_epochs']
            n_bins = results['n_bins']
            
            comparison[condition] = {
                'mean': mean_complexity,
                'std': std_complexity,
                'n_epochs': n_epochs,
                'n_bins': n_bins,
                'complexity_values': results['complexity_values']
            }
            
            log_print(f"{condition.upper()}: {mean_complexity:.4f} ± {std_complexity:.4f} "
                     f"(n={n_epochs} epochs, {n_bins} bins)")
        
        # Calculate percentage differences
        conditions = list(comparison.keys())
        if len(conditions) >= 2:
            for i in range(len(conditions)):
                for j in range(i+1, len(conditions)):
                    cond1, cond2 = conditions[i], conditions[j]
                    mean1, mean2 = comparison[cond1]['mean'], comparison[cond2]['mean']
                    if mean2 != 0:
                        pct_diff = ((mean1 - mean2) / abs(mean2)) * 100
                        log_print(f"{cond1} vs {cond2}: {pct_diff:+.1f}% difference")
        
        return comparison


# Wrapper function for spectral analysis compatibility
def calculate_cn_binning(epoch_data, n_bins=DEFAULT_N_BINS, max_partitions=DEFAULT_MAX_PARTITIONS, verbose=False):
    """
    Calculate Neural Complexity using binning method for a single epoch.
    
    This function provides a simple interface for the spectral analysis framework.
    
    Parameters:
    -----------
    epoch_data : array-like, shape (n_channels, n_samples)
        Single epoch of EEG data
    n_bins : int
        Number of bins for discretization
    max_partitions : int
        Maximum number of bipartitions to use
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    complexity : float
        Neural complexity value for this epoch
    """
    analyzer = NeuralComplexityBinning(n_bins=n_bins, verbose=verbose)
    complexity, _ = analyzer.calculate_neural_complexity(
        epoch_data, max_partitions=max_partitions, n_bins=n_bins, verbose=verbose
    )
    return complexity


def run_binning_analysis(analysis_type, file_paths, n_channels=DEFAULT_N_CHANNELS,
                        epoch_length=DEFAULT_EPOCH_LENGTH, max_partitions=DEFAULT_MAX_PARTITIONS,
                        n_bins=DEFAULT_N_BINS, subsample_factor_broadband=DEFAULT_SUBSAMPLE_FACTOR_BROADBAND,
                        subsample_factor_spectral=DEFAULT_SUBSAMPLE_FACTOR_SPECTRAL,
                        n_jobs=DEFAULT_N_JOBS, verbose=DEFAULT_VERBOSE, output_dir='.'):
    """
    Run binning neural complexity analysis.
    
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
    n_bins : int
        Number of bins for discretization
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
    log_print("NEURAL COMPLEXITY ANALYSIS - BINNING METHOD", verbose)
    log_print("=" * HEADER_WIDTH, verbose)
    log_print("Non-parametric entropy estimation using histogram discretization.", verbose)
    log_print("Note: Sensitive to bin size selection and curse of dimensionality.", verbose)
    
    results = {}
    
    # Traditional broadband analysis
    if analysis_type in ['broadband', 'both']:
        log_print(f"\n{'='*HEADER_WIDTH}", verbose)
        log_print("BROADBAND ANALYSIS", verbose)
        log_print("="*HEADER_WIDTH, verbose)
        
        analyzer = NeuralComplexityBinning(n_bins=n_bins, verbose=verbose)
        results_list = []
        
        for condition, file_path in file_paths.items():
            if not os.path.exists(file_path):
                log_print(f"File not found: {file_path}", verbose)
                continue
                
            try:
                log_print(f"\nAnalyzing {condition} state...", verbose)
                
                results_obj = analyzer.analyze_eeg_file(
                    file_path=file_path,
                    n_channels=n_channels,
                    epoch_length=epoch_length,
                    max_partitions=max_partitions,
                    n_bins=n_bins,
                    subsample_factor=subsample_factor_broadband,
                    n_jobs=n_jobs
                )
                
                results_list.append(results_obj)
                
            except Exception as e:
                log_print(f"Error analyzing {file_path}: {e}", verbose)
                if verbose:
                    import traceback
                    traceback.print_exc()
                continue
        
        # Compare and plot results
        if len(results_list) >= 2:
            log_print(f"\n{'='*HEADER_WIDTH}", verbose)
            log_print("COMPARATIVE ANALYSIS", verbose)
            log_print("="*HEADER_WIDTH, verbose)
            
            comparison = analyzer.compare_conditions(results_list)
            
            # Create visualization
            plot_results(results_list, "Binning", 
                        save_path=os.path.join(output_dir, "plots", "neural_complexity_binning_broadband_results.png"), 
                        verbose=verbose)
            
            # Save traditional results
            results_df = pd.DataFrame([
                {
                    'condition': r['condition'],
                    'method': r['method'],
                    'n_bins': r['n_bins'],
                    'mean_complexity': r['mean_complexity'],
                    'std_complexity': r['std_complexity'],
                    'n_epochs': r['n_epochs'],
                    'n_channels': r['n_channels']
                }
                for r in results_list
            ])
            
            results_df.to_csv(os.path.join(output_dir, "summaries", "neural_complexity_binning_broadband_summary.csv"), index=False)
            log_print("\nBroadband results saved to neural_complexity_binning_broadband_summary.csv", verbose)
        
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
                complexity_func=calculate_cn_binning,
                n_channels=n_channels,
                subsample_factor=subsample_factor_spectral,
                verbose=verbose,
                n_bins=n_bins,
                max_partitions=max_partitions
            )
            
            # Print summary
            print_spectral_summary(spectral_results, verbose)
            
            # Create spectral visualization
            plot_spectral_complexity_results(
                spectral_results,
                method_name="Binning",
                save_path=os.path.join(output_dir, "plots", "neural_complexity_binning_spectral_results.png")
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
                        'n_epochs': data['n_epochs'],
                        'n_bins': n_bins
                    })
            
            pd.DataFrame(spectral_df).to_csv(
                os.path.join(output_dir, "summaries", "neural_complexity_binning_spectral_summary.csv"), 
                index=False
            )
            log_print("\nSpectral results saved to neural_complexity_binning_spectral_summary.csv", verbose)
            
            results['spectral'] = spectral_results
            
        except Exception as e:
            log_print(f"Error in spectral analysis: {e}", verbose)
            if verbose:
                import traceback
                traceback.print_exc()
    
    log_print(f"\n{'='*HEADER_WIDTH}", verbose)
    log_print("BINNING ANALYSIS COMPLETE", verbose)
    log_print(f"{'='*HEADER_WIDTH}", verbose)
    log_print("Key Findings:", verbose) 
    log_print("- Broadband analysis provides overall neural complexity", verbose)
    log_print("- Gamma band analysis targets consciousness-specific mechanisms", verbose)  
    log_print("- Binning method avoids distribution assumptions", verbose)
    log_print("- Results sensitive to bin size and dimensionality", verbose)
    log_print("- Consider using KSG method for more robust results", verbose)
    
    return results 