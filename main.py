#!/usr/bin/env python3
"""
Neural Complexity Analysis - Interactive CLI Interface
=====================================================

Interactive command-line interface for running neural complexity analysis using different methods:
- KSG (Kraskov-Stögbauer-Grassberger): Non-parametric, robust for non-Gaussian data
- Binning: Histogram-based discretization method
- Gaussian: Parametric method (WARNING: Invalid assumption for EEG data)

Based on PLAN.md: Project: Estimating Consciousness from EEG using Neural Complexity
"""

import argparse
import os
import sys
from pathlib import Path

def print_banner():
    """Display welcome banner."""
    print("="*80)
    print("🧠 NEURAL COMPLEXITY ANALYSIS FOR EEG CONSCIOUSNESS RESEARCH 🧠")
    print("="*80)
    print()
    print("This tool analyzes neural complexity in EEG data to study consciousness.")
    print("Developed based on the Neural Complexity framework for consciousness research.")
    print()

def print_method_info():
    """Display detailed information about available methods."""
    print("📊 AVAILABLE METHODS:")
    print("-" * 50)
    print()
    print("1. 🔬 KSG Method (RECOMMENDED)")
    print("   • Kraskov-Stögbauer-Grassberger entropy estimation")
    print("   • Non-parametric, robust for non-Gaussian data")
    print("   • Best choice for EEG analysis")
    print("   • Uses k-nearest neighbor distances")
    print()
    print("2. 📊 Binning Method")
    print("   • Histogram-based discretization")
    print("   • Non-parametric but sensitive to bin size")
    print("   • Alternative approach for comparison")
    print("   • May suffer from curse of dimensionality")
    print()
    print("3. ⚠️  Gaussian Method (COMPARISON ONLY)")
    print("   • Assumes multivariate Gaussian distribution")
    print("   • INVALID for EEG data (highly non-Gaussian)")
    print("   • Included only for educational/comparison purposes")
    print("   • Results are scientifically unreliable")
    print()
    print("4. 🔬📊⚠️  All Methods")
    print("   • Run all three methods for comparison")
    print("   • Comprehensive analysis")
    print("   • Best for research and method evaluation")
    print()

def print_analysis_info():
    """Display information about analysis types."""
    print("🔍 ANALYSIS TYPES:")
    print("-" * 50)
    print()
    print("1. 📡 Broadband Analysis")
    print("   • Traditional analysis across full frequency spectrum (1-40 Hz)")
    print("   • Provides overall neural complexity measure")
    print("   • Fast and straightforward")
    print()
    print("2. 🌈 Spectral Band Analysis")
    print("   • Analysis by frequency bands:")
    print("     - Delta (0.5-4 Hz): Deep sleep, unconscious states")
    print("     - Theta (4-8 Hz): Memory, meditation")
    print("     - Alpha (8-13 Hz): Relaxed awareness")
    print("     - Beta (13-30 Hz): Active thinking")
    print("     - Gamma (30-100 Hz): ⭐ Consciousness marker")
    print("   • Focus on gamma band for consciousness research")
    print("   • More detailed frequency-specific analysis")
    print()
    print("3. 🔬🌈 Both Analyses")
    print("   • Complete analysis package")
    print("   • Recommended for research")
    print("   • Provides comprehensive results")
    print()

def get_user_choice(prompt, options, descriptions=None):
    """Get user choice with input validation."""
    while True:
        print(prompt)
        if descriptions:
            for i, (option, desc) in enumerate(zip(options, descriptions), 1):
                print(f"  {i}. {option} - {desc}")
        else:
            for i, option in enumerate(options, 1):
                print(f"  {i}. {option}")
        print()
        
        try:
            choice = input("Enter your choice (number): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(options):
                return choice_idx, options[choice_idx]
            else:
                print(f"❌ Invalid choice. Please enter a number between 1 and {len(options)}.")
                print()
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
            print()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)

def get_yes_no(prompt, default=True):
    """Get yes/no input from user."""
    while True:
        default_text = "Y/n" if default else "y/N"
        response = input(f"{prompt} ({default_text}): ").strip().lower()
        
        if not response:
            return default
        elif response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("❌ Please enter 'y' for yes or 'n' for no.")

def get_integer(prompt, default, min_val=1, max_val=None):
    """Get integer input with validation."""
    while True:
        response = input(f"{prompt} (default: {default}): ").strip()
        
        if not response:
            return default
        
        try:
            value = int(response)
            if value < min_val:
                print(f"❌ Value must be at least {min_val}.")
                continue
            if max_val and value > max_val:
                print(f"❌ Value must be at most {max_val}.")
                continue
            return value
        except ValueError:
            print("❌ Please enter a valid integer.")

def get_float(prompt, default, min_val=0.1):
    """Get float input with validation."""
    while True:
        response = input(f"{prompt} (default: {default}): ").strip()
        
        if not response:
            return default
        
        try:
            value = float(response)
            if value < min_val:
                print(f"❌ Value must be at least {min_val}.")
                continue
            return value
        except ValueError:
            print("❌ Please enter a valid number.")

def get_file_paths():
    """Get EEG file paths from user."""
    print("📁 EEG FILE SELECTION:")
    print("-" * 50)
    print()
    
    use_default = get_yes_no("Use default dataset files?", default=True)
    
    if use_default:
        file_paths = {
            'awake': 'ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr',
            'sedation': 'ds005620/sub-1010/eeg/sub-1010_task-sed2_acq-rest_run-1_eeg.vhdr'
        }
        
        # Check if files exist
        missing_files = [path for path in file_paths.values() if not os.path.exists(path)]
        if missing_files:
            print("\n❌ Default files not found:")
            for path in missing_files:
                print(f"  - {path}")
            print("\nPlease provide custom file paths.")
            return get_custom_file_paths()
        else:
            print("\n✅ Using default dataset files:")
            for condition, path in file_paths.items():
                print(f"  - {condition}: {path}")
            return file_paths
    else:
        return get_custom_file_paths()

def get_custom_file_paths():
    """Get custom file paths from user."""
    print("\n📂 CUSTOM FILE PATHS:")
    print("Enter paths to your EEG files (.vhdr format)")
    print()
    
    file_paths = {}
    
    # Get first file
    while True:
        path1 = input("Path to first EEG file: ").strip()
        if os.path.exists(path1):
            condition1 = input("Label for this file (e.g., 'awake', 'baseline'): ").strip() or 'file1'
            file_paths[condition1] = path1
            break
        else:
            print(f"❌ File not found: {path1}")
    
    # Ask for second file
    if get_yes_no("\nAdd a second file for comparison?", default=True):
        while True:
            path2 = input("Path to second EEG file: ").strip()
            if os.path.exists(path2):
                condition2 = input("Label for this file (e.g., 'sedation', 'treatment'): ").strip() or 'file2'
                file_paths[condition2] = path2
                break
            else:
                print(f"❌ File not found: {path2}")
    
    print(f"\n✅ Using custom files:")
    for condition, path in file_paths.items():
        print(f"  - {condition}: {path}")
    
    return file_paths

def interactive_main():
    """Main interactive CLI function."""
    print_banner()
    
    # Method selection
    print_method_info()
    method_options = ['KSG (Recommended)', 'Binning', 'Gaussian (Invalid)', 'All Methods']
    method_map = ['ksg', 'binning', 'gaussian', 'all']
    
    method_idx, _ = get_user_choice("🔬 SELECT ANALYSIS METHOD:", method_options)
    selected_method = method_map[method_idx]
    
    print(f"\n✅ Selected method: {method_options[method_idx]}")
    
    # Analysis type selection
    print("\n" + "="*60)
    print_analysis_info()
    analysis_options = ['Broadband', 'Spectral Bands', 'Both (Recommended)']
    analysis_map = ['broadband', 'spectral', 'both']
    
    analysis_idx, _ = get_user_choice("🔍 SELECT ANALYSIS TYPE:", analysis_options)
    selected_analysis = analysis_map[analysis_idx]
    
    print(f"\n✅ Selected analysis: {analysis_options[analysis_idx]}")
    
    # File selection
    print("\n" + "="*60)
    file_paths = get_file_paths()
    
    # Parameters configuration
    print("\n" + "="*60)
    print("⚙️  ANALYSIS PARAMETERS:")
    print("-" * 50)
    print()
    
    print("🧠 EEG Processing Parameters:")
    n_channels = get_integer("Number of EEG channels to analyze", 8, min_val=2, max_val=64)
    epoch_length = get_float("Epoch length in seconds", 5.0, min_val=1.0)
    
    print("\n⚡ Performance Parameters:")
    subsample = get_integer("Subsampling factor (higher = faster but less precise)", 10, min_val=1, max_val=50)
    
    if selected_analysis in ['spectral', 'both']:
        subsample_spectral = get_integer("Spectral analysis subsampling factor", 5, min_val=1, max_val=20)
    else:
        subsample_spectral = 5
    
    max_partitions = get_integer("Maximum bipartitions to compute", 200, min_val=10, max_val=1000)
    
    # Method-specific parameters
    if selected_method in ['ksg', 'all']:
        print("\n🔬 KSG Method Parameters:")
        k_neighbors = get_integer("Number of nearest neighbors (k)", 3, min_val=1, max_val=10)
    else:
        k_neighbors = 3
    
    if selected_method in ['binning', 'all']:
        print("\n📊 Binning Method Parameters:")
        n_bins = get_integer("Number of bins for discretization", 10, min_val=5, max_val=50)
    else:
        n_bins = 10
    
    # Output configuration
    print("\n💾 Output Configuration:")
    output_dir = input("Output directory (default: results): ").strip() or "results"
    verbose = get_yes_no("Enable verbose output?", default=True)
    
    # Summary and confirmation
    print("\n" + "="*60)
    print("📋 ANALYSIS SUMMARY:")
    print("-" * 50)
    print(f"Method: {method_options[method_idx]}")
    print(f"Analysis: {analysis_options[analysis_idx]}")
    print(f"Files: {len(file_paths)} file(s)")
    for condition, path in file_paths.items():
        print(f"  - {condition}: {os.path.basename(path)}")
    print(f"Channels: {n_channels}")
    print(f"Epoch length: {epoch_length}s")
    print(f"Subsampling: {subsample}")
    if selected_analysis in ['spectral', 'both']:
        print(f"Spectral subsampling: {subsample_spectral}")
    print(f"Max partitions: {max_partitions}")
    if selected_method in ['ksg', 'all']:
        print(f"KSG k-neighbors: {k_neighbors}")
    if selected_method in ['binning', 'all']:
        print(f"Binning bins: {n_bins}")
    print(f"Output directory: {output_dir}")
    print(f"Verbose output: {'Yes' if verbose else 'No'}")
    print()
    
    if not get_yes_no("🚀 Proceed with analysis?", default=True):
        print("❌ Analysis cancelled.")
        return 1
    
    # Run analysis
    print("\n" + "="*80)
    print("🔬 STARTING ANALYSIS...")
    print("="*80)
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'summaries'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
    
    # Prepare analysis parameters
    analysis_params = {
        'file_paths': file_paths,
        'n_channels': n_channels,
        'epoch_length': epoch_length,
        'max_partitions': max_partitions,
        'subsample_factor_broadband': subsample,
        'subsample_factor_spectral': subsample_spectral,
        'n_jobs': -1,  # Use all cores
        'verbose': verbose,
        'output_dir': output_dir
    }
    
    # Method-specific parameters
    ksg_params = analysis_params.copy()
    ksg_params['k'] = k_neighbors
    
    binning_params = analysis_params.copy()
    binning_params['n_bins'] = n_bins
    
    gaussian_params = analysis_params.copy()
    
    # Run analysis based on method selection
    results = {}
    
    if selected_method in ['ksg', 'all']:
        if not verbose:
            print("\n🔬 Running KSG method...")
        try:
            from neural_complexity_ksg import run_ksg_analysis
            results['ksg'] = run_ksg_analysis(selected_analysis, **ksg_params)
        except Exception as e:
            print(f"❌ Error running KSG analysis: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    if selected_method in ['binning', 'all']:
        if not verbose:
            print("\n📊 Running Binning method...")
        try:
            from neural_complexity_binning import run_binning_analysis
            results['binning'] = run_binning_analysis(selected_analysis, **binning_params)
        except Exception as e:
            print(f"❌ Error running Binning analysis: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    if selected_method in ['gaussian', 'all']:
        if not verbose:
            print("\n⚠️  Running Gaussian method...")
            print("⚠️  WARNING: Results are scientifically invalid for EEG data!")
        try:
            from neural_complexity_gaussian import run_gaussian_analysis
            results['gaussian'] = run_gaussian_analysis(selected_analysis, **gaussian_params)
        except Exception as e:
            print(f"❌ Error running Gaussian analysis: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*80)
    
    if results:
        print("✅ Successfully completed methods:")
        for method in results:
            print(f"  • {method.upper()}")
        
        print(f"\n📁 Results saved to: {output_dir}/")
        print("   • Plots: plots/")
        print("   • Summaries: summaries/") 
        print("   • Data: data/")
    else:
        print("❌ No methods completed successfully.")
        return 1
    
    print("\n💡 RECOMMENDATIONS:")
    print("• Use KSG method results for scientific conclusions")
    print("• Focus on gamma band (30-100 Hz) for consciousness research")
    print("• Compare conditions to identify consciousness-related changes")
    print("• Binning and Gaussian methods included for comparison only")
    
    print("\n🔬 Happy researching! 🧠")
    return 0

def argument_main():
    """Original argument-based CLI function."""
    # Use exec to avoid import issues in interactive mode
    with open('main_backup.py', 'r') as f:
        backup_code = f.read()
    
    # Extract and execute the argument_main function
    import sys
    backup_globals = {
        '__name__': '__main__',
        'sys': sys,
        'os': os,
        'argparse': argparse,
        'Path': Path
    }
    exec(backup_code, backup_globals)
    return backup_globals['argument_main']()

def main():
    """Main entry point - chooses between interactive and argument modes."""
    # Check if any arguments were provided (excluding script name)
    if len(sys.argv) > 1:
        # Check for special interactive flag
        if '--interactive' in sys.argv:
            sys.argv.remove('--interactive')
            if len(sys.argv) == 1:  # Only script name left
                return interactive_main()
        
        # Use argument-based interface
        print("Using argument-based interface. For interactive mode, run: python main.py")
        print("Or use: python main.py --interactive")
        print("=" * 70)
        return argument_main()
    else:
        # No arguments provided, use interactive interface
        return interactive_main()

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user. Goodbye!")
        sys.exit(0) 