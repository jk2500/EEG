#!/usr/bin/env python3
"""
Neural Complexity Analysis - Interactive & Argument-Based CLI
=============================================================

A unified command-line interface for running neural complexity and MIB
analyses. Supports both an interactive, guided setup and a direct,
argument-based execution for scripting.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Local imports
from eeg_analysis.config import (
    ANALYSIS_PARAMS,
    KSG_PARAMS,
    BINNING_PARAMS,
    DEFAULT_FILE_PATHS,
    DEFAULT_OUTPUT_DIR
)
from eeg_analysis.analyzers.complexity_analyzer import ComplexityAnalyzer
from eeg_analysis.analyzers.estimators import (
    KSGEstimator,
    BinningEstimator,
    GaussianEstimator
)

# --- UI and Helper Functions ---

def print_banner():
    """Displays a welcome banner for the interactive mode."""
    print("="*80)
    print("MIB ANALYSIS FOR EEG CONSCIOUSNESS RESEARCH")
    print("="*80)
    print("\nWelcome! This tool will guide you through the analysis setup.")

def get_user_choice(prompt, options):
    """Gets a validated user choice from a list of options."""
    while True:
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        try:
            choice = int(input("Enter your choice (number): ").strip()) - 1
            if 0 <= choice < len(options):
                return options[choice].split(' ')[0].lower()
            print(f"Invalid choice. Please enter a number between 1 and {len(options)}.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)

def get_params_from_user():
    """Gathers analysis parameters from the user in interactive mode."""
    params = {}
    print("\n--- Configure Analysis Parameters ---")
    params['n_channels'] = int(input(f"Number of EEG channels (default: {ANALYSIS_PARAMS['n_channels']}): ") or ANALYSIS_PARAMS['n_channels'])
    params['epoch_length'] = float(input(f"Epoch length in seconds (default: {ANALYSIS_PARAMS['epoch_length']}): ") or ANALYSIS_PARAMS['epoch_length'])
    params['k'] = int(input(f"KSG k-neighbors (default: {KSG_PARAMS['k']}): ") or KSG_PARAMS['k'])
    params['n_bins'] = int(input(f"Binning number of bins (default: {BINNING_PARAMS['n_bins']}): ") or BINNING_PARAMS['n_bins'])
    return params

# --- Main Logic ---

def run_analysis(estimator_name, analysis_type, file_paths, output_dir, params):
    """
    Initializes and runs the selected analysis.
    """
    estimator_map = {
        'ksg': KSGEstimator,
        'binning': BinningEstimator,
        'gaussian': GaussianEstimator
    }
    
    EstimatorClass = estimator_map.get(estimator_name)
    if not EstimatorClass:
        print(f"Error: Unknown estimator '{estimator_name}'", file=sys.stderr)
        sys.exit(1)
        
    estimator = EstimatorClass(**params)
    analyzer = ComplexityAnalyzer(estimator=estimator, verbose=True, **params)
    
    print(f"\n--- Running {analyzer.method_name} Analysis ---")
    analyzer.run_analysis(
        analysis_type=analysis_type,
        file_paths=file_paths,
        output_dir=output_dir
    )
    print(f"--- {analyzer.method_name} Analysis Complete ---")

def interactive_main():
    """Guides the user through an interactive session."""
    print_banner()
    
    estimator_name = get_user_choice("SELECT ESTIMATOR:", ['KSG (Recommended)', 'Binning', 'Gaussian (Invalid)'])
    analysis_type = get_user_choice("SELECT ANALYSIS TYPE:", ['Broadband', 'Spectral', 'Both (Recommended)'])
    
    file_paths = DEFAULT_FILE_PATHS
    print(f"\nUsing default dataset files:\n- Awake: {file_paths['awake']}\n- Sedation: {file_paths['sedation']}")
    
    params = get_params_from_user()
    output_dir = input(f"\nOutput directory (default: {DEFAULT_OUTPUT_DIR}): ").strip() or DEFAULT_OUTPUT_DIR
    
    print("\n--- Analysis Summary ---")
    print("Metric: MIB (Minimum Information Bipartition)")
    print(f"Estimator: {estimator_name.upper()}")
    print(f"Type: {analysis_type}")
    print(f"Parameters: {params}")
    print(f"Output Directory: {output_dir}")
    
    if input("\nProceed with analysis? (y/n): ").lower() not in ['y', 'yes']:
        print("Analysis cancelled.")
        return
        
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    run_analysis(estimator_name, analysis_type, file_paths, output_dir, params)

def argument_main():
    """Handles command-line arguments for non-interactive execution."""
    parser = argparse.ArgumentParser(
        description='Run Neural Complexity and MIB analyses.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--estimator', required=True, choices=['ksg', 'binning', 'gaussian'],
                        help='The MI estimator to use.')
    parser.add_argument('--type', default='both', choices=['broadband', 'spectral', 'both'],
                        help='The type of analysis to perform (default: both).')
    parser.add_argument('--files', nargs='+', default=list(DEFAULT_FILE_PATHS.values()),
                        help='Paths to the EEG files to analyze.')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR,
                        help=f'Directory to save results (default: {DEFAULT_OUTPUT_DIR}).')
    parser.add_argument('--channels', type=int, default=ANALYSIS_PARAMS['n_channels'])
    parser.add_argument('--epoch', type=float, default=ANALYSIS_PARAMS['epoch_length'])
    parser.add_argument('--k', type=int, default=KSG_PARAMS['k'], help='k-neighbors for KSG.')
    parser.add_argument('--bins', type=int, default=BINNING_PARAMS['n_bins'], help='Number of bins for Binning.')
    
    args = parser.parse_args()
    
    if len(args.files) != 2:
        print("Error: Please provide exactly two file paths (e.g., one for awake, one for sedation).", file=sys.stderr)
        sys.exit(1)
        
    file_paths = {'condition1': args.files[0], 'condition2': args.files[1]}
    
    params = {
        'n_channels': args.channels,
        'epoch_length': args.epoch,
        'k': args.k,
        'n_bins': args.bins
    }
    
    os.makedirs(os.path.join(args.output, 'plots'), exist_ok=True)
    run_analysis(args.estimator, args.type, file_paths, args.output, params)

def main():
    """Main entry point."""
    if len(sys.argv) == 1 or '--interactive' in sys.argv:
        interactive_main()
    else:
        argument_main()

if __name__ == "__main__":
    main()
