#!/usr/bin/env python3
"""
Unified EEG Analysis Runner - Interactive & Batch CLI
=====================================================

A centralized, powerful script for running batch analyses of the Minimum Information
Bipartition (MIB) on the entire ds005620 dataset. This
script provides both an interactive guided setup and direct command-line execution.

Key Features:
- Interactive CLI mode for guided analysis setup
- Unified interface for all analysis estimators (KSG, Binning, Gaussian).
- Focused exclusively on the MIB metric for clarity.
- Robust processing with periodic checkpointing and resume capability.
- Sample run mode for quick testing and validation.
- Optimized parallel processing and memory management.
"""

import os
import sys
import time
import argparse
import pickle
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Local imports
from eeg_analysis.eeg_utils import (
    create_subject_file_map,
    preprocess_eeg_by_bands,
    _extract_condition,
    EEGDataCache,
    log_print
)
from eeg_analysis.config import (
    ANALYSIS_PARAMS,
    KSG_PARAMS,
    BINNING_PARAMS,
    GAUSSIAN_PARAMS,
    DATASET_DIR,
    DEFAULT_OUTPUT_DIR
)
from eeg_analysis.analyzers.complexity_analyzer import ComplexityAnalyzer
from eeg_analysis.analyzers.estimators import (
    KSGEstimator,
    BinningEstimator,
    GaussianEstimator
)

# --- Interactive UI Functions ---

def print_banner():
    """Displays a welcome banner for the interactive mode."""
    print("="*80)
    print("UNIFIED EEG BATCH ANALYSIS - MIB")
    print("="*80)
    print("\nWelcome! This tool will guide you through batch analysis setup.")
    print("You can process entire datasets with sophisticated checkpointing and resume capabilities.")

def get_user_choice(prompt, options, allow_custom=False):
    """Gets a validated user choice from a list of options."""
    while True:
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        if allow_custom:
            print(f"  {len(options) + 1}. Custom (enter your own)")
        
        try:
            choice_input = input("Enter your choice (number): ").strip()
            choice = int(choice_input) - 1
            
            if 0 <= choice < len(options):
                return options[choice].split(' ')[0].lower()
            elif allow_custom and choice == len(options):
                return input("Enter custom value: ").strip()
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(options) + (1 if allow_custom else 0)}.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)

def get_boolean_choice(prompt, default=True):
    """Gets a yes/no choice from the user."""
    default_text = "Y/n" if default else "y/N"
    while True:
        try:
            response = input(f"{prompt} ({default_text}): ").strip().lower()
            if not response:
                return default
            if response in ['y', 'yes', 'true', '1']:
                return True
            elif response in ['n', 'no', 'false', '0']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)

def get_numeric_input(prompt, default, input_type=int, min_val=None, max_val=None):
    """Gets a validated numeric input from the user."""
    while True:
        try:
            response = input(f"{prompt} (default: {default}): ").strip()
            if not response:
                return default
            
            value = input_type(response)
            if min_val is not None and value < min_val:
                print(f"Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            sys.exit(0)

def get_dataset_directory():
    """Gets the dataset directory from user input."""
    print(f"\nDataset Directory:")
    print(f"Current default: {DATASET_DIR}")
    
    if get_boolean_choice("Use default dataset directory?", default=True):
        return DATASET_DIR
    
    while True:
        custom_dir = input("Enter dataset directory path: ").strip()
        if os.path.exists(custom_dir):
            return custom_dir
        else:
            print(f"Directory '{custom_dir}' does not exist. Please enter a valid path.")
            if not get_boolean_choice("Try again?", default=True):
                return DATASET_DIR

def get_output_directory():
    """Gets the output directory from user input."""
    print(f"\nOutput Directory:")
    print("Leave blank for auto-generated directory based on analysis settings")
    
    custom_dir = input("Enter custom output directory (or press Enter for auto): ").strip()
    return custom_dir if custom_dir else None

def interactive_main():
    """Guides the user through an interactive batch analysis session."""
    print_banner()
    
    # Get estimator choice
    estimator_name = get_user_choice(
        "SELECT MUTUAL INFORMATION ESTIMATOR:", 
        ['KSG (K-Nearest Neighbors - Recommended)', 'Binning (Histogram-based)', 'Gaussian (Parametric)']
    )
    
    # Get dataset directory
    dataset_dir = get_dataset_directory()
    
    # Get output directory
    output_dir = get_output_directory()
    
    # Get analysis parameters
    print("\n--- Configure Analysis Parameters ---")
    n_channels = get_numeric_input(
        "Number of EEG channels to analyze", 
        ANALYSIS_PARAMS['n_channels'], 
        int, min_val=1, max_val=128
    )
    
    # Sample run option
    sample_run = get_boolean_choice("\nRun sample analysis (few subjects for testing)?", default=False)
    sample_subjects = 2
    if sample_run:
        sample_subjects = get_numeric_input(
            "Number of subjects for sample run", 
            2, int, min_val=1, max_val=10
        )
    
    # Resume option
    resume = get_boolean_choice("Resume from checkpoint if available?", default=True)
    
    # Verbose output option
    verbose = get_boolean_choice("Show detailed output during processing?", default=True)
    
    # Display summary
    print("\n" + "="*60)
    print("BATCH ANALYSIS CONFIGURATION SUMMARY")
    print("="*60)
    print("Metric: MIB (Minimum Information Bipartition)")
    print(f"Estimator: {estimator_name.upper()}")
    print(f"Dataset Directory: {dataset_dir}")
    print(f"Output Directory: {output_dir or 'Auto-generated'}")
    print(f"Channels: {n_channels}")
    print(f"Sample Run: {'Yes' if sample_run else 'No'}" + (f" ({sample_subjects} subjects)" if sample_run else ""))
    print(f"Resume from Checkpoint: {'Yes' if resume else 'No'}")
    print(f"Verbose Output: {'Yes' if verbose else 'No'}")
    print("="*60)
    
    if not get_boolean_choice("\nProceed with batch analysis?", default=True):
        print("Analysis cancelled. Goodbye!")
        return
    
    print(f"\nStarting batch analysis...")
    
    try:
        run_batch_analysis(
            estimator_name=estimator_name,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            n_channels=n_channels,
            sample_run=sample_run,
            sample_subjects=sample_subjects,
            verbose=verbose,
            resume=resume
        )
        print(f"\nBatch analysis completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user. Progress has been checkpointed.")
        print("You can resume later using the --resume flag or interactive mode.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# --- Helper Functions for Batch Processing ---

def save_progress_checkpoint(results, metadata, checkpoint_path, verbose=True):
    """Save progress checkpoint to disk."""
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({'results': results, 'metadata': metadata, 'timestamp': time.time()}, f)
    log_print(f"Progress checkpoint saved: {checkpoint_path}", verbose)

def load_progress_checkpoint(checkpoint_path, verbose=True):
    """Load progress checkpoint from disk."""
    if not os.path.exists(checkpoint_path):
        return {}, {}
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        log_print(f"Resuming from checkpoint: {checkpoint_path} (last updated: {time.ctime(checkpoint_data.get('timestamp', 0))})", verbose)
        return checkpoint_data.get('results', {}), checkpoint_data.get('metadata', {})
    except Exception as e:
        log_print(f"Could not load checkpoint: {e}", verbose)
        return {}, {}

def save_subject_results(subject_id, subject_results, output_dir, method_name, verbose=True):
    """Save individual subject results immediately to JSON."""
    subject_dir = os.path.join(output_dir, 'subjects')
    os.makedirs(subject_dir, exist_ok=True)
    
    base_filename = f'{subject_id}_{method_name.lower()}_results.json'
    json_path = os.path.join(subject_dir, base_filename)
    
    serializable_results = {}
    for condition, condition_data in subject_results.items():
        serializable_results[condition] = {}
        for band, band_data in condition_data.items():
            serializable_results[condition][band] = {
                'metric_values': [float(x) for x in band_data.get('metric_values', [])],
                'mean_metric': float(band_data.get('mean_metric', 0)),
                'std_metric': float(band_data.get('std_metric', 0)),
                'n_epochs': int(band_data.get('n_epochs', 0))
            }
            
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    log_print(f"Subject {subject_id} results saved to {json_path}", verbose)

def process_single_subject(subject_id, condition_paths, analyzer, output_dir, verbose=True):
    """Process a single subject and save its results."""
    log_print(f"\nProcessing subject: {subject_id}", verbose)
    
    cache = EEGDataCache(max_cache_size=2)
    subject_results = {}
    
    for condition, file_path in condition_paths.items():
        try:
            raw = cache.get_raw_data(file_path, verbose=verbose)
            condition_name = _extract_condition(file_path)
            
            band_data, _ = preprocess_eeg_by_bands(raw, **analyzer.params)
            
            condition_results = {}
            for band_name, epochs_data in band_data.items():
                metric_values = analyzer._evaluate_epochs(
                    epochs_data,
                    progress_label=f"{subject_id} | {condition_name} | {band_name}",
                    verbose_override=verbose
                )
                metric_values = [v for v in metric_values if v is not None]
                
                if metric_values:
                    condition_results[band_name] = {
                        'metric_values': metric_values,
                        'mean_metric': np.mean(metric_values),
                        'std_metric': np.std(metric_values),
                        'n_epochs': len(metric_values)
                    }
                    log_print(f"  {condition_name} {band_name}: {np.mean(metric_values):.4f}", verbose)
            
            if condition_results:
                subject_results[condition_name] = condition_results
                
        except Exception as e:
            log_print(f"Error processing {subject_id} {condition}: {e}", verbose)
            continue
            
    cache.clear()
    
    if subject_results:
        save_subject_results(subject_id, subject_results, output_dir, analyzer.method_name, verbose)
        
    return subject_id, subject_results

# --- Main Batch Analysis Function ---

def run_batch_analysis(estimator_name, dataset_dir, output_dir, n_channels, sample_run, sample_subjects, verbose, resume):
    """Main function to orchestrate the batch analysis."""
    
    metric_name = 'mib'
    
    estimator_map = {
        'ksg': (KSGEstimator, KSG_PARAMS),
        'binning': (BinningEstimator, BINNING_PARAMS),
        'gaussian': (GaussianEstimator, GAUSSIAN_PARAMS)
    }
    
    if estimator_name not in estimator_map:
        raise ValueError(f"Unknown estimator: {estimator_name}. Choose from {list(estimator_map.keys())}")
        
    EstimatorClass, method_params = estimator_map[estimator_name]
    final_params = {**ANALYSIS_PARAMS, **method_params, 'n_channels': n_channels}
    
    estimator = EstimatorClass(**final_params)
    analyzer = ComplexityAnalyzer(estimator, **final_params)
    
    run_type = "SAMPLE" if sample_run else "FULL"
    output_dir = output_dir or os.path.join(DEFAULT_OUTPUT_DIR, f'batch_{analyzer.method_name.lower()}_{run_type.lower()}')
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint.pkl')
    
    log_print(f"\n{'='*80}", verbose)
    log_print(f"UNIFIED BATCH ANALYSIS - {analyzer.method_name} ({run_type} RUN)", verbose)
    log_print(f"{'='*80}", verbose)
    log_print(f"Dataset: {dataset_dir}, Output: {output_dir}", verbose)
    
    all_results, metadata = {}, {}
    if resume:
        all_results, metadata = load_progress_checkpoint(checkpoint_path, verbose)
    
    processed_subjects = set(all_results.keys())
    subject_paths = create_subject_file_map(dataset_dir=dataset_dir, verbose=False)
    subjects_to_process = {k: v for k, v in subject_paths.items() if k not in processed_subjects}
    
    if sample_run:
        subjects_to_process = dict(list(subjects_to_process.items())[:sample_subjects])
    
    if not subjects_to_process:
        log_print("No new subjects to process. Analysis complete.", verbose)
        return
        
    log_print(f"\nProcessing {len(subjects_to_process)} subjects...", verbose)
    start_time = time.time()
    
    progress_bar = tqdm(subjects_to_process.items(), desc="Processing Subjects", disable=not verbose)
    
    for i, (subject_id, condition_paths) in enumerate(progress_bar):
        progress_bar.set_description(f"Processing {subject_id}")
        _, subject_results = process_single_subject(subject_id, condition_paths, analyzer, output_dir, verbose=False)
        
        if subject_results:
            all_results[subject_id] = subject_results
            
        if (i + 1) % 5 == 0 or (i + 1) == len(subjects_to_process):
            metadata = {'metric': metric_name, 'estimator': estimator_name}
            save_progress_checkpoint(all_results, metadata, checkpoint_path, verbose)
            
    total_time = time.time() - start_time
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    log_print(f"\n{'='*80}", verbose)
    log_print(f"BATCH ANALYSIS COMPLETE!", verbose)
    log_print(f"Total subjects processed: {len(subjects_to_process)} in {total_time/60:.1f} minutes", verbose)
    log_print(f"Results saved to: {output_dir}", verbose)


def run_flow_cli(args):
    """
    Entry point for the ACFlow training workflow.
    """

    if not args.flow_train_data:
        raise SystemExit("ERROR: --flow-train-data must be provided when using --flow-mode train.")

    from eeg_analysis.flows import (
        ACFlow,
        ACFlowConfig,
        ACFlowTrainer,
        ChannelwiseStandardizer,
        EEGWindowDataset,
        MaskSampler,
        MaskSamplerConfig,
        TrainerConfig,
        create_dataloader,
    )

    run_id = args.flow_run_id or time.strftime("acflow_%Y%m%d_%H%M%S")
    normalizer = ChannelwiseStandardizer() if args.flow_fit_normalizer else None
    pin_memory = not args.flow_no_pin_memory

    train_dataset = EEGWindowDataset(
        args.flow_train_data,
        memmap=True,
        normalizer=normalizer,
        fit_normalizer=args.flow_fit_normalizer,
    )
    if train_dataset.feature_dim != args.flow_input_dim:
        raise SystemExit(
            f"Train data feature dimension ({train_dataset.feature_dim}) "
            f"does not match --flow-input-dim ({args.flow_input_dim})."
        )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.flow_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.flow_workers,
        pin_memory=pin_memory,
    )

    val_loader = None
    if args.flow_val_data:
        val_dataset = EEGWindowDataset(
            args.flow_val_data,
            memmap=True,
            normalizer=normalizer,
            fit_normalizer=False,
        )
        if val_dataset.feature_dim != args.flow_input_dim:
            raise SystemExit(
                f"Validation data feature dimension ({val_dataset.feature_dim}) "
                f"does not match --flow-input-dim ({args.flow_input_dim})."
            )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=args.flow_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.flow_workers,
            pin_memory=pin_memory,
        )

    flow_config = ACFlowConfig(
        input_dim=args.flow_input_dim,
        hidden_dim=args.flow_hidden_dim,
        num_blocks=args.flow_blocks,
        conditioner_depth=args.flow_conditioner_depth,
        dropout=args.flow_dropout,
        scale_clip=args.flow_scale_clip,
    )
    trainer_config = TrainerConfig(
        batch_size=args.flow_batch_size,
        max_epochs=args.flow_epochs,
        learning_rate=args.flow_lr,
        weight_decay=args.flow_weight_decay,
        gradient_clip=args.flow_grad_clip,
        use_amp=not args.flow_disable_amp,
        val_interval=max(1, args.flow_val_interval),
        early_stopping_patience=args.flow_patience,
        log_interval=args.flow_log_interval,
        checkpoint_dir=Path(args.flow_checkpoint_dir),
        run_id=run_id,
    )
    mask_config = MaskSamplerConfig(
        dim=args.flow_input_dim,
        min_condition=args.flow_min_condition,
        seed=args.flow_seed,
    )

    trainer = ACFlowTrainer(
        ACFlow(flow_config),
        trainer_config,
        mask_sampler=MaskSampler(mask_config),
    )
    checkpoint_name = args.flow_checkpoint_name or f"{run_id}.pt"

    print(f"[ACFlowTrainer] Starting run {run_id} with {args.flow_blocks} blocks and hidden dim {args.flow_hidden_dim}.")
    trainer.fit(
        train_loader,
        val_loader=val_loader,
        checkpoint_name=checkpoint_name,
        verbose=not args.flow_quiet,
    )
    print(f"[ACFlowTrainer] Completed run {run_id}. Latest metrics: {trainer.history[-1] if trainer.history else 'N/A'}")

# --- Command-Line Interface ---

def main():
    """Main entry point that handles both interactive and command-line modes."""
    # Check if we should run in interactive mode
    if len(sys.argv) == 1 or '--interactive' in sys.argv:
        interactive_main()
        return
    
    # Command-line argument parsing for non-interactive mode
    parser = argparse.ArgumentParser(
        description='Unified EEG Batch Analysis Runner - Interactive & Command-Line',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default when no arguments provided)
  python run_analysis.py
  python run_analysis.py --interactive

  # Command-line mode examples:
  # Run a sample MIB analysis using the recommended KSG estimator
  python run_analysis.py --estimator ksg --sample

  # Run a full analysis with the Binning estimator
  python run_analysis.py --estimator binning --resume

  # Custom dataset and output directories
  python run_analysis.py --estimator ksg --dataset /path/to/data --output /path/to/results
"""
    )
    
    parser.add_argument('--interactive', action='store_true',
                        help='Force interactive mode (default when no other args provided).')
    parser.add_argument('--estimator', required=False, choices=['ksg', 'binning', 'gaussian'],
                        help='The MI estimator to use.')
    parser.add_argument('--dataset', default=DATASET_DIR,
                        help=f'Path to the dataset directory (default: {DATASET_DIR})')
    parser.add_argument('--output', default=None,
                        help='Output directory (auto-generated if not specified).')
    parser.add_argument('--channels', type=int, default=ANALYSIS_PARAMS['n_channels'],
                        help=f"Number of EEG channels to analyze (default: {ANALYSIS_PARAMS['n_channels']}).")
    parser.add_argument('--sample', action='store_true',
                        help='Run a sample analysis on a few subjects for testing.')
    parser.add_argument('--sample-subjects', type=int, default=2,
                        help='Number of subjects for a sample run (default: 2).')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output and show progress bars instead.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from a checkpoint if available.')
    parser.add_argument('--flow-mode', choices=['train'], default=None,
                        help='Run the ACFlow pipeline instead of the classical estimators.')
    parser.add_argument('--flow-train-data', default=None,
                        help='Path to the training windows (.npy) for ACFlow.')
    parser.add_argument('--flow-val-data', default=None,
                        help='Optional validation windows (.npy) for ACFlow.')
    parser.add_argument('--flow-input-dim', type=int, default=64,
                        help='Dimensionality (channels) for the ACFlow model.')
    parser.add_argument('--flow-hidden-dim', type=int, default=512,
                        help='Hidden width of ACFlow conditioners.')
    parser.add_argument('--flow-blocks', type=int, default=8,
                        help='Number of coupling blocks.')
    parser.add_argument('--flow-conditioner-depth', type=int, default=3,
                        help='Number of layers inside each conditioner MLP.')
    parser.add_argument('--flow-dropout', type=float, default=0.05,
                        help='Dropout rate inside the conditioner.')
    parser.add_argument('--flow-scale-clip', type=float, default=5.0,
                        help='Maximum log-scale magnitude for affine couplings.')
    parser.add_argument('--flow-batch-size', type=int, default=512,
                        help='Training batch size for ACFlow.')
    parser.add_argument('--flow-epochs', type=int, default=50,
                        help='Number of training epochs for ACFlow.')
    parser.add_argument('--flow-lr', type=float, default=1e-3,
                        help='Learning rate for ACFlow.')
    parser.add_argument('--flow-weight-decay', type=float, default=1e-6,
                        help='Weight decay for ACFlow optimizer.')
    parser.add_argument('--flow-grad-clip', type=float, default=1.0,
                        help='Gradient clipping value for ACFlow.')
    parser.add_argument('--flow-seed', type=int, default=0,
                        help='Random seed for mask sampling.')
    parser.add_argument('--flow-min-condition', type=int, default=8,
                        help='Minimum observed set size during training.')
    parser.add_argument('--flow-val-interval', type=int, default=1,
                        help='Validate every N epochs.')
    parser.add_argument('--flow-patience', type=int, default=10,
                        help='Early stopping patience for ACFlow.')
    parser.add_argument('--flow-log-interval', type=int, default=50,
                        help='Steps between training log prints.')
    parser.add_argument('--flow-checkpoint-dir', default='artifacts/checkpoints',
                        help='Directory to store ACFlow checkpoints.')
    parser.add_argument('--flow-checkpoint-name', default=None,
                        help='Optional checkpoint filename.')
    parser.add_argument('--flow-run-id', default=None,
                        help='Custom run identifier for ACFlow.')
    parser.add_argument('--flow-disable-amp', action='store_true',
                        help='Disable mixed precision during ACFlow training.')
    parser.add_argument('--flow-fit-normalizer', action='store_true',
                        help='Fit a channel-wise normalizer on the training data before ACFlow training.')
    parser.add_argument('--flow-workers', type=int, default=0,
                        help='Number of dataloader workers for ACFlow.')
    parser.add_argument('--flow-no-pin-memory', action='store_true',
                        help='Disable dataloader pin_memory for ACFlow (useful on CPU-only machines).')
    parser.add_argument('--flow-quiet', action='store_true',
                        help='Silence ACFlow trainer logging.')
    
    args = parser.parse_args()

    if args.flow_mode:
        run_flow_cli(args)
        return
    
    if args.estimator is None:
        parser.error("--estimator is required unless --flow-mode is provided.")
    
    try:
        run_batch_analysis(
            estimator_name=args.estimator,
            dataset_dir=args.dataset,
            output_dir=args.output,
            n_channels=args.channels,
            sample_run=args.sample,
            sample_subjects=args.sample_subjects,
            verbose=not args.quiet,
            resume=args.resume
        )
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
