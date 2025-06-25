# Neural Complexity CLI Interface

This project now has an **interactive command-line interface** for running neural complexity analysis with different methods, designed to be user-friendly for researchers.

## Quick Start

### Interactive Mode (Recommended)
```bash
# Launch interactive interface with guided setup
python main.py
```

The interactive interface will guide you through:
- 🔬 Method selection with detailed explanations
- 🔍 Analysis type selection (broadband, spectral, or both)  
- 📁 File selection (default dataset or custom files)
- ⚙️ Parameter configuration with smart defaults
- 📋 Summary and confirmation before running

### Argument Mode (Advanced Users)
```bash
# Run with command-line arguments (for scripting/automation)
python main.py --method ksg --analysis broadband --channels 8

# Force interactive mode even with arguments
python main.py --interactive

# Run all methods with verbose output
python main.py --method all --verbose

# Analyze specific files with 12 channels
python main.py --method ksg --files path/to/awake.vhdr path/to/sedation.vhdr --channels 12

# Run only spectral analysis with binning method
python main.py --method binning --analysis spectral --n-bins 15
```

## Available Methods

- **KSG** (default): Kraskov-Stögbauer-Grassberger method - non-parametric, robust for non-Gaussian data
- **Binning**: Histogram-based discretization method
- **Gaussian**: Parametric method (WARNING: Invalid assumption for EEG data)
- **All**: Run all three methods for comparison

## Analysis Types

- **broadband**: Traditional analysis across full spectrum
- **spectral**: Analysis by frequency bands (delta, theta, alpha, beta, gamma)
- **both** (default): Run both broadband and spectral analysis

## Key Parameters

- `--channels`: Number of EEG channels to analyze (default: 8)
- `--epoch-length`: Length of epochs in seconds (default: 5.0)
- `--subsample`: Subsampling factor for computational efficiency (default: 10)
- `--k-neighbors`: Number of nearest neighbors for KSG method (default: 3)
- `--n-bins`: Number of bins for binning method (default: 10)
- `--jobs`: Number of parallel jobs (default: -1 for all cores)

## Examples

### Basic Usage
```bash
# Quick analysis with KSG method
python main.py --method ksg --analysis broadband --verbose

# Compare all methods
python main.py --method all --analysis both
```

### Advanced Usage
```bash
# Custom analysis with specific parameters
python main.py \
  --method ksg \
  --analysis spectral \
  --channels 16 \
  --epoch-length 10.0 \
  --k-neighbors 5 \
  --subsample 5 \
  --output-dir results/ \
  --verbose

# Analyze custom files
python main.py \
  --method binning \
  --files data/subject1_awake.vhdr data/subject1_sedation.vhdr \
  --n-bins 20 \
  --channels 20
```

## File Structure

After running the analysis, results are saved to the organized output directory:

```
results/
├── plots/
│   ├── neural_complexity_{method}_broadband_results.png
│   └── neural_complexity_{method}_spectral_results.png
├── summaries/
│   ├── neural_complexity_{method}_broadband_summary.csv
│   └── neural_complexity_{method}_spectral_summary.csv
└── data/
    └── (additional data files as needed)
```

## Method Recommendations

- **Use KSG method** for scientifically valid results (default choice)
- **Binning method** provides alternative non-parametric approach but sensitive to bin size
- **Gaussian method** included only for comparison - results are scientifically invalid for EEG data

## Interactive Features

The interactive interface provides:

### 🎯 **Guided Method Selection**
- Detailed explanations of each method
- Clear recommendations (KSG is recommended)
- Warnings about invalid methods (Gaussian)

### 📊 **Analysis Type Guidance** 
- Explanation of broadband vs spectral analysis
- Information about EEG frequency bands
- Recommendations for consciousness research

### 📁 **Smart File Handling**
- Auto-detection of default dataset files
- File existence validation
- Support for custom file paths with labels

### ⚙️ **Parameter Configuration**
- Smart defaults for all parameters
- Input validation with helpful error messages
- Parameter explanations and recommendations

### 📋 **Pre-Run Summary**
- Complete analysis configuration review
- Confirmation before starting analysis
- Clear progress indicators during analysis

## Help

```bash
# Interactive mode (no arguments needed)
python main.py

# Argument mode help
python main.py --help
```

For detailed documentation on the underlying methods, see the individual method files:
- `neural_complexity_ksg.py` - KSG implementation
- `neural_complexity_binning.py` - Binning implementation  
- `neural_complexity_gaussian.py` - Gaussian implementation
- `eeg_utils.py` - Common utilities 