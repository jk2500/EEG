# Neural Complexity Analysis for EEG Consciousness Research

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive toolkit for analyzing neural complexity in EEG data to study consciousness using multiple entropy estimation methods.

## 🧠 Overview

This project implements neural complexity analysis for EEG consciousness research, based on the framework that consciousness can be quantified through the complexity of neural activity patterns. The toolkit provides three different entropy estimation methods with an intuitive interactive interface.

### Key Features

- **🎯 Interactive CLI**: Guided setup with detailed explanations
- **📊 Multiple Methods**: KSG, Binning, and Gaussian entropy estimation
- **🔍 Dual Analysis**: Broadband and spectral band analysis
- **📁 Organized Output**: Structured results in plots/, summaries/, data/
- **🔬 Scientific Reproducibility**: Integration with OpenNeuro dataset via submodules
- **⚡ High Performance**: Parallel processing and optimized algorithms

## 🚀 Quick Start

### Interactive Mode (Recommended)
```bash
python main.py
```
The interactive interface guides you through method selection, parameter configuration, and analysis setup.

### Command Line Mode
```bash
# Basic analysis with KSG method
python main.py --method ksg --analysis both --verbose

# Compare all methods
python main.py --method all --analysis spectral --channels 12
```

## 📋 Requirements

- Python 3.8+
- Required packages: `numpy`, `scipy`, `matplotlib`, `mne`, `pandas`, `scikit-learn`
- Optional: `datalad` for dataset management

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/EEG.git
cd EEG
```

2. Create conda environment:
```bash
conda env create -f environment.yml
conda activate eeg_consciousness
```

3. Initialize the dataset submodule:
```bash
git submodule init
git submodule update
```

## 🔬 Methods

### 1. KSG Method (Recommended)
- **Kraskov-Stögbauer-Grassberger entropy estimation**
- Non-parametric, robust for non-Gaussian data
- Based on k-nearest neighbor distances
- Best choice for EEG analysis

### 2. Binning Method
- Histogram-based discretization
- Non-parametric but sensitive to bin size
- Alternative approach for comparison
- May suffer from curse of dimensionality

### 3. Gaussian Method (Comparison Only)
- Assumes multivariate Gaussian distribution
- ⚠️ **INVALID for EEG data** (highly non-Gaussian)
- Included only for educational/comparison purposes

## 📊 Analysis Types

### Broadband Analysis
- Traditional analysis across full frequency spectrum (1-40 Hz)
- Provides overall neural complexity measure
- Fast and straightforward

### Spectral Band Analysis
- Analysis by frequency bands:
  - **Delta (0.5-4 Hz)**: Deep sleep, unconscious states
  - **Theta (4-8 Hz)**: Memory, meditation
  - **Alpha (8-13 Hz)**: Relaxed awareness
  - **Beta (13-30 Hz)**: Active thinking
  - **Gamma (30-100 Hz)**: ⭐ **Consciousness marker**
- Focus on gamma band for consciousness research

## 📁 Dataset

This project uses the **OpenNeuro dataset ds005620** integrated as a git submodule:
- **Source**: https://github.com/OpenNeuroDatasets/ds005620
- **Description**: EEG data from awake and sedated states
- **Management**: Datalad for efficient data handling
- **Reproducibility**: Ensures consistent dataset versions

## 📈 Output Structure

```
results/
├── plots/
│   ├── neural_complexity_{method}_broadband_results.png
│   └── neural_complexity_{method}_spectral_results.png
├── summaries/
│   ├── neural_complexity_{method}_broadband_summary.csv
│   └── neural_complexity_{method}_spectral_summary.csv
└── data/
    └── (additional processed data)
```

## 🎯 Usage Examples

### Interactive Analysis
```bash
# Launch interactive mode
python main.py

# Follow the guided setup:
# 1. Select method (KSG recommended)
# 2. Choose analysis type (both recommended)
# 3. Configure parameters
# 4. Review and confirm
```

### Command Line Analysis
```bash
# Quick KSG analysis
python main.py --method ksg --analysis broadband --channels 8

# Comprehensive comparison
python main.py --method all --analysis both --verbose

# Custom files analysis
python main.py --method ksg --files custom_awake.vhdr custom_sedation.vhdr --channels 16

# Spectral analysis with custom parameters
python main.py --method ksg --analysis spectral --k-neighbors 5 --subsample 5
```

## 🔍 Scientific Background

Neural complexity analysis is based on the theory that consciousness emerges from the integrated information processing of neural networks. This toolkit implements:

1. **Entropy Estimation**: Quantifies the complexity of neural activity patterns
2. **Bipartition Analysis**: Measures information integration across brain regions
3. **Spectral Decomposition**: Analyzes frequency-specific complexity
4. **Consciousness Markers**: Focus on gamma band activity as consciousness indicator

## 📚 Documentation

- **[CLI_README.md](CLI_README.md)**: Detailed CLI usage guide
- **[docs/PLAN.md](docs/PLAN.md)**: Project planning and methodology
- **[docs/KSG_Analysis_Summary.md](docs/KSG_Analysis_Summary.md)**: KSG method analysis
- **[docs/EEG_Gaussianity_Analysis_Summary.md](docs/EEG_Gaussianity_Analysis_Summary.md)**: Gaussianity analysis results

## 🧪 Research Applications

This toolkit is designed for:
- **Consciousness Research**: Quantifying levels of consciousness
- **Anesthesia Studies**: Monitoring sedation effects
- **Clinical Applications**: Assessing neurological conditions
- **Method Comparison**: Evaluating entropy estimation techniques

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenNeuro**: For providing the EEG dataset
- **Datalad**: For efficient data management
- **MNE-Python**: For EEG processing capabilities
- **Scientific Community**: For neural complexity research foundations

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the CLI help: `python main.py --help`

---

**⭐ If this project helps your research, please consider giving it a star!** 