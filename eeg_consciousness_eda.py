#!/usr/bin/env python3
"""
EEG Consciousness Dataset - Exploratory Data Analysis
====================================================

This script performs exploratory data analysis on the ds005620 dataset containing
EEG recordings from a propofol sedation study. The goal is to understand the data
structure and characteristics before implementing neural complexity measures for
consciousness estimation.

Dataset: A repeated awakening study exploring the capacity of complexity measures 
to capture dreaming during propofol sedation
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try importing MNE for EEG analysis
try:
    import mne
    mne.set_log_level('WARNING')
    MNE_AVAILABLE = True
except ImportError:
    print("MNE-Python not available. Install with: pip install mne")
    MNE_AVAILABLE = False

def setup_plotting():
    """Configure matplotlib and seaborn for better plots"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

def load_dataset_info(data_path):
    """Load and display basic dataset information"""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    
    # Load dataset description
    desc_file = data_path / "dataset_description.json"
    if desc_file.exists():
        with open(desc_file, 'r') as f:
            dataset_desc = json.load(f)
        
        print(f"Dataset: {dataset_desc['Name']}")
        print(f"BIDS Version: {dataset_desc['BIDSVersion']}")
        print(f"License: {dataset_desc['License']}")
        print(f"DOI: {dataset_desc.get('DatasetDOI', 'N/A')}")
        print("\nAuthors:")
        for author in dataset_desc['Authors']:
            print(f"  - {author}")
    
    # Load README
    readme_file = data_path / "README.md"
    if readme_file.exists():
        with open(readme_file, 'r') as f:
            readme_content = f.read()
        print(f"\nREADME length: {len(readme_content)} characters")

def analyze_participants(data_path):
    """Analyze participant demographics and experimental conditions"""
    print("\n" + "=" * 60)
    print("PARTICIPANT ANALYSIS")
    print("=" * 60)
    
    # Load participants data
    participants_file = data_path / "participants.tsv"
    if not participants_file.exists():
        print("participants.tsv not found")
        return None
    
    df = pd.read_csv(participants_file, sep='\t')
    print(f"Total participants: {len(df)}")
    
    # Clean and analyze demographics
    print("\nDemographics:")
    print("-" * 30)
    
    # Age analysis
    age_data = df['age'].replace('N/A', np.nan)
    age_numeric = pd.to_numeric(age_data, errors='coerce')
    valid_ages = age_numeric.dropna()
    if len(valid_ages) > 0:
        print(f"Age - Mean: {valid_ages.mean():.1f}, Range: {valid_ages.min():.0f}-{valid_ages.max():.0f}")
        print(f"Age data available for: {len(valid_ages)}/{len(df)} participants")
    
    # Sex distribution
    sex_counts = df['sex'].value_counts()
    print(f"\nSex distribution:")
    for sex, count in sex_counts.items():
        if sex != 'N/A':
            print(f"  {sex}: {count}")
    
    # Experimental conditions
    print(f"\nExperimental Conditions:")
    print("-" * 30)
    
    # Awakenings
    awakening_counts = df['awakenings'].value_counts().sort_index()
    print("Awakenings distribution:")
    for awake, count in awakening_counts.items():
        print(f"  {awake} awakenings: {count} participants")
    
    # TMS conditions
    tms_counts = df['TMS'].value_counts()
    print(f"\nTMS conditions:")
    for tms, count in tms_counts.items():
        print(f"  TMS {tms}: {count} participants")
    
    if 'tms_count' in df.columns:
        tms_count_stats = df[df['TMS'] == True]['tms_count']
        if len(tms_count_stats) > 0:
            print(f"  TMS count range: {tms_count_stats.min()}-{tms_count_stats.max()}")
    
    # Exclusions
    excluded_count = df['excluded'].sum() if 'excluded' in df.columns else 0
    print(f"\nExcluded participants: {excluded_count}")
    
    # Bad data after preprocessing
    bad_data = df['bad_after_preprocessing'].apply(lambda x: x != 'False' and x != False)
    bad_count = bad_data.sum()
    print(f"Participants with bad data after preprocessing: {bad_count}")
    
    return df

def analyze_file_structure(data_path):
    """Analyze the file structure and naming conventions"""
    print("\n" + "=" * 60)
    print("FILE STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Find all subject directories
    subject_dirs = [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith('sub-')]
    subject_dirs.sort()
    
    print(f"Number of subject directories: {len(subject_dirs)}")
    
    # Analyze file types and naming patterns
    all_files = []
    task_counts = {}
    acq_counts = {}
    file_type_counts = {}
    
    for subject_dir in subject_dirs:
        eeg_dir = subject_dir / 'eeg'
        if eeg_dir.exists():
            for file_path in eeg_dir.iterdir():
                if file_path.is_file():
                    filename = file_path.name
                    all_files.append(filename)
                    
                    # Extract file extension
                    ext = file_path.suffix
                    file_type_counts[ext] = file_type_counts.get(ext, 0) + 1
                    
                    # Parse BIDS filename
                    if '_task-' in filename:
                        parts = filename.split('_')
                        for part in parts:
                            if part.startswith('task-'):
                                task = part.split('-')[1]
                                task_counts[task] = task_counts.get(task, 0) + 1
                            elif part.startswith('acq-'):
                                acq = part.split('-')[1]
                                acq_counts[acq] = acq_counts.get(acq, 0) + 1
    
    print(f"\nTotal EEG files: {len(all_files)}")
    
    print(f"\nFile types:")
    for ext, count in sorted(file_type_counts.items()):
        print(f"  {ext}: {count}")
    
    print(f"\nTask conditions:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    
    print(f"\nAcquisition types:")
    for acq, count in sorted(acq_counts.items()):
        print(f"  {acq}: {count}")
    
    return subject_dirs, task_counts, acq_counts

def analyze_eeg_metadata(data_path, subject_dirs):
    """Analyze EEG recording parameters and metadata"""
    print("\n" + "=" * 60)
    print("EEG RECORDING PARAMETERS")
    print("=" * 60)
    
    # Collect metadata from JSON files
    metadata_list = []
    channel_counts = []
    sampling_rates = []
    recording_durations = []
    
    for subject_dir in subject_dirs[:5]:  # Analyze first 5 subjects for speed
        eeg_dir = subject_dir / 'eeg'
        if eeg_dir.exists():
            json_files = list(eeg_dir.glob('*_eeg.json'))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        metadata = json.load(f)
                    metadata_list.append(metadata)
                    
                    # Extract key parameters
                    if 'EEGChannelCount' in metadata:
                        channel_counts.append(metadata['EEGChannelCount'])
                    if 'SamplingFrequency' in metadata:
                        sampling_rates.append(metadata['SamplingFrequency'])
                    if 'RecordingDuration' in metadata:
                        recording_durations.append(metadata['RecordingDuration'])
                        
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
    
    print(f"Analyzed {len(metadata_list)} EEG recordings")
    
    if channel_counts:
        print(f"\nChannel counts: {set(channel_counts)}")
    if sampling_rates:
        print(f"Sampling rates: {set(sampling_rates)} Hz")
    if recording_durations:
        durations = np.array(recording_durations)
        print(f"Recording durations: {durations.min():.0f}-{durations.max():.0f} seconds")
        print(f"Average duration: {durations.mean():.1f} seconds")
    
    # Analyze first metadata in detail
    if metadata_list:
        sample_metadata = metadata_list[0]
        print(f"\nSample recording parameters:")
        print("-" * 30)
        for key, value in sample_metadata.items():
            print(f"  {key}: {value}")
    
    return metadata_list

def analyze_channel_layout(data_path, subject_dirs):
    """Analyze EEG channel layout and electrode positions"""
    print("\n" + "=" * 60)
    print("CHANNEL LAYOUT ANALYSIS")
    print("=" * 60)
    
    # Load channel information from first subject
    first_subject = subject_dirs[0]
    eeg_dir = first_subject / 'eeg'
    
    channel_files = list(eeg_dir.glob('*_channels.tsv'))
    if not channel_files:
        print("No channel files found")
        return None
    
    # Analyze first channel file
    channel_file = channel_files[0]
    channels_df = pd.read_csv(channel_file, sep='\t')
    
    print(f"Total channels: {len(channels_df)}")
    print(f"Channel file: {channel_file.name}")
    
    # Analyze channel types
    if 'type' in channels_df.columns:
        type_counts = channels_df['type'].value_counts()
        print(f"\nChannel types:")
        for ch_type, count in type_counts.items():
            print(f"  {ch_type}: {count}")
    
    # Display channel names by brain region
    eeg_channels = channels_df[channels_df['type'] == 'EEG']['name'].tolist()
    print(f"\nEEG Channel names ({len(eeg_channels)} total):")
    print(f"Sample channels: {', '.join(eeg_channels[:10])}...")
    
    # Categorize channels by brain region
    regions = {
        'Frontal': [ch for ch in eeg_channels if ch.startswith(('F', 'AF', 'Fp'))],
        'Central': [ch for ch in eeg_channels if ch.startswith(('C', 'FC', 'CP'))],
        'Parietal': [ch for ch in eeg_channels if ch.startswith('P')],
        'Occipital': [ch for ch in eeg_channels if ch.startswith('O')],
        'Temporal': [ch for ch in eeg_channels if ch.startswith(('T', 'TP', 'FT'))],
        'Other': [ch for ch in eeg_channels if not any(ch.startswith(prefix) for prefix in ['F', 'AF', 'Fp', 'C', 'FC', 'CP', 'P', 'O', 'T', 'TP', 'FT'])]
    }
    
    print(f"\nChannels by brain region:")
    for region, channels in regions.items():
        if channels:
            print(f"  {region}: {len(channels)} channels")
    
    return channels_df, eeg_channels

def load_sample_eeg_data(data_path, subject_dirs):
    """Load and analyze sample EEG data"""
    print("\n" + "=" * 60)
    print("SAMPLE EEG DATA ANALYSIS")
    print("=" * 60)
    
    if not MNE_AVAILABLE:
        print("MNE-Python not available - skipping EEG data loading")
        return None
    
    # Find a sample EEG file (awake condition, any acquisition)
    sample_file = None
    for subject_dir in subject_dirs[:3]:
        eeg_dir = subject_dir / 'eeg'
        vhdr_files = list(eeg_dir.glob('*task-awake*.vhdr'))
        if vhdr_files:
            sample_file = vhdr_files[0]
            break
    
    if not sample_file:
        print("No suitable EEG file found for loading")
        return None
    
    try:
        print(f"Loading: {sample_file.name}")
        
        # Load raw EEG data
        raw = mne.io.read_raw_brainvision(sample_file, preload=True, verbose=False)
        
        print(f"Successfully loaded EEG data")
        print(f"  Channels: {len(raw.ch_names)}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} seconds")
        print(f"  Data shape: {raw.get_data().shape}")
        
        # Basic signal statistics
        data = raw.get_data()
        print(f"\nSignal statistics:")
        print(f"  Mean amplitude: {np.mean(data):.2e} V")
        print(f"  Std amplitude: {np.std(data):.2e} V")
        print(f"  Min amplitude: {np.min(data):.2e} V")
        print(f"  Max amplitude: {np.max(data):.2e} V")
        
        return raw
        
    except Exception as e:
        print(f"Error loading EEG data: {e}")
        return None

def create_summary_plots(participants_df, task_counts, acq_counts):
    """Create summary plots of the dataset"""
    print("\n" + "=" * 60)
    print("CREATING SUMMARY PLOTS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EEG Consciousness Dataset Overview', fontsize=16, fontweight='bold')
    
    # Plot 1: Participant demographics
    ax1 = axes[0, 0]
    sex_counts = participants_df['sex'].value_counts()
    sex_counts = sex_counts[sex_counts.index != 'N/A']  # Remove N/A
    ax1.pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%')
    ax1.set_title('Sex Distribution')
    
    # Plot 2: Awakening distribution
    ax2 = axes[0, 1]
    awakening_counts = participants_df['awakenings'].value_counts().sort_index()
    ax2.bar(awakening_counts.index, awakening_counts.values)
    ax2.set_title('Number of Awakenings per Participant')
    ax2.set_xlabel('Awakenings')
    ax2.set_ylabel('Count')
    
    # Plot 3: Task conditions
    ax3 = axes[1, 0]
    tasks = list(task_counts.keys())
    counts = list(task_counts.values())
    ax3.bar(tasks, counts)
    ax3.set_title('Task Conditions')
    ax3.set_xlabel('Task')
    ax3.set_ylabel('Number of Recordings')
    ax3.tick_params(axis='x', rotation=45)
    
    # Plot 4: Acquisition types
    ax4 = axes[1, 1]
    acqs = list(acq_counts.keys())
    acq_counts_list = list(acq_counts.values())
    ax4.bar(acqs, acq_counts_list)
    ax4.set_title('Acquisition Types')
    ax4.set_xlabel('Acquisition')
    ax4.set_ylabel('Number of Recordings')
    
    plt.tight_layout()
    plt.savefig('eeg_dataset_overview.png', dpi=300, bbox_inches='tight')
    print("Saved: eeg_dataset_overview.png")
    plt.show()

def consciousness_analysis_preview():
    """Preview the neural complexity analysis approach"""
    print("\n" + "=" * 60)
    print("NEURAL COMPLEXITY ANALYSIS PREVIEW")
    print("=" * 60)
    
    print("Based on the PLAN.md, the next steps for consciousness estimation:")
    print("\n1. Data Preprocessing Pipeline:")
    print("   - Band-pass filtering (1-40 Hz)")
    print("   - Epoching (2-5 second windows)")  
    print("   - Artifact rejection (ICA)")
    print("   - Downsampling if needed for computational efficiency")
    
    print("\n2. Neural Complexity Calculation Methods:")
    print("   - Gaussian Method: Fast, assumes normal distribution")
    print("   - Binning Method: Non-parametric, histogram-based")
    print("   - KSG Method: k-NN based, most accurate but slowest")
    
    print("\n3. Consciousness State Comparison:")
    print("   - Awake (EC/EO) vs Sedation conditions")
    print("   - Different levels of sedation depth")
    print("   - Statistical analysis of complexity differences")
    
    print(f"\n4. Computational Considerations:")
    print(f"   - 65 EEG channels = 2^64 possible bipartitions")
    print(f"   - Need to reduce channel count or use approximations")
    print(f"   - Consider hierarchical or random sampling approaches")
    
    # Show sample complexity calculation framework
    n_channels = 10  # Example with reduced channels
    n_bipartitions = 2**(n_channels-1) - 1
    print(f"\nExample: With {n_channels} channels:")
    print(f"   Number of bipartitions: {n_bipartitions}")
    print(f"   Computational complexity: O(n × 2^n)")

def main():
    """Main analysis function"""
    setup_plotting()
    
    # Define data path
    data_path = Path("ds005620")
    
    if not data_path.exists():
        print(f"Dataset directory {data_path} not found!")
        print("Please ensure the dataset is in the correct location.")
        return
    
    print("EEG CONSCIOUSNESS DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    print("Analyzing dataset for Neural Complexity estimation...")
    
    # Run analysis steps
    load_dataset_info(data_path)
    participants_df = analyze_participants(data_path)
    subject_dirs, task_counts, acq_counts = analyze_file_structure(data_path)
    analyze_eeg_metadata(data_path, subject_dirs)
    channels_df, eeg_channels = analyze_channel_layout(data_path, subject_dirs)
    
    # Try to load sample EEG data
    sample_raw = load_sample_eeg_data(data_path, subject_dirs)
    
    # Create summary visualizations
    if participants_df is not None:
        create_summary_plots(participants_df, task_counts, acq_counts)
    
    # Preview consciousness analysis approach
    consciousness_analysis_preview()
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print("Next steps:")
    print("1. Implement preprocessing pipeline")
    print("2. Develop neural complexity calculation methods")
    print("3. Apply to different consciousness states")
    print("4. Statistical analysis and validation")
    
    print(f"\nKey findings:")
    print(f"- {len(subject_dirs)} subjects with EEG data")
    print(f"- Multiple consciousness states: awake, sedation")  
    print(f"- High-density EEG: ~65 channels at 5000 Hz")
    print(f"- Suitable for neural complexity analysis")

if __name__ == "__main__":
    main() 