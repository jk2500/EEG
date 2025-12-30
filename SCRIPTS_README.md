# Scripts Documentation

This document catalogs all scripts in the project with their functionality, inputs, outputs, and CLI options.

---

## Installation

Install the package in development mode for proper imports:

```bash
pip install -e .
```

This makes `eeg_analysis` available as a proper package without needing `sys.path` hacks.

To install with all optional dependencies:

```bash
pip install -e ".[all]"
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Directory Structure

```
scripts/                           # MIB computation scripts
├── utils/
│   ├── mib_core.py               # Shared MIB computation functions
│   └── metadata.py               # Path parsing utilities
├── mib_analysis.py               # Unified MIB analysis (all datasets)
└── analyze_mib_results.py        # Summarize MIB JSON outputs

analysis/scripts/                  # Post-hoc analysis and visualization
├── utils/
│   ├── stats.py                  # Statistical functions (mean_ci, cohen_d, etc.)
│   ├── config.py                 # Constants (BANDS, CONDITIONS, etc.)
│   ├── data_loading.py           # JSON loading utilities
│   └── visualization.py          # Plotting helpers
├── eda_sub1010.py                # Raw EEG EDA for any subject (configurable)
├── eda_sub1010_results.py        # MIB results EDA for any subject
├── eda_results_all_subjects.py   # MIB results EDA for all subjects
├── analyze_hyperparams.py        # Test channel count and epoch length effects
├── analyze_bin_effect.py         # Test histogram bin count effect on MIB
└── plot_sedation_fourway.py      # Four-way sedation comparison (all plot types)
```

---

## MIB Computation Scripts (`scripts/`)

### `mib_analysis.py` (Recommended)
**Purpose**: Unified MIB analysis script supporting multiple dataset formats.

**Supported Datasets**:
- `ds005620`: BrainVision format (.vhdr) - propofol sedation study
- `sedation`: EEGLAB format (.set/.fdt) - Sedation-RestingState dataset

**Modes**:
- `dataset`: Sweep all subjects/conditions
- `single`: Analyze a single file

**Features**:
- Spectral band filtering (delta, theta, alpha, beta, gamma, broadband)
- Random channel subsampling with configurable repeats
- Fixed-channel epoch stability baseline

**CLI Examples**:
```bash
# DS005620 dataset
python scripts/mib_analysis.py ds005620 dataset --subjects sub-1010 --mode spectral
python scripts/mib_analysis.py ds005620 single --file path/to/file.vhdr

# Sedation-RestingState dataset
python scripts/mib_analysis.py sedation dataset --conditions baseline deep_sedation
python scripts/mib_analysis.py sedation single --file path/to/file.set
```

**Outputs**: JSON files with per-band MIB statistics

---

### `analyze_mib_results.py`
**Purpose**: Summarize and analyze MIB JSON outputs.

**Features**:
- Per-condition stability via CV
- Within-subject ordering check (awake > sedation)
- Cross-subject difference analysis

**CLI**: `python scripts/analyze_mib_results.py --root results/ds005620 --mode spectral`

**Outputs**: Console summary tables

---

## Analysis Scripts (`analysis/scripts/`)

### `eda_sub1010.py`
**Purpose**: Raw EEG exploratory analysis for any subject.

**CLI**:
```bash
python analysis/scripts/eda_sub1010.py                           # Default: sub-1010
python analysis/scripts/eda_sub1010.py --subject sub-1019
python analysis/scripts/eda_sub1010.py --subject sub-1010 --dataset /path/to/ds005620
```

**Outputs** (in `analysis/outputs/ds005620/{subject}/`):
- `run_summary.csv` - Per-run statistics
- `channel_stats.csv` - Per-channel amplitude stats
- `bandpower.csv` - Spectral band power
- `psd_by_condition.png` - PSD comparison plot
- `relative_bandpower.png` - Relative power bar chart

---

### `eda_sub1010_results.py`
**Purpose**: Detailed MIB results EDA for any subject.

**CLI**: `python analysis/scripts/eda_sub1010_results.py --subject sub-1010 --epoch 5`

**Outputs** (in `analysis/outputs/ds005620/{subject}_results_epoch{N}/`):
- `results_summary.csv` - Per-band summary statistics
- `repeat_means.csv` - All repeat-level means
- `epoch_distribution.csv` - Epoch-level distributions
- `epoch_stability.csv` - Fixed-channel stability
- `channel_selection_counts.csv` - Channel selection frequencies
- `selection_jaccard.csv` - Selection set similarity
- `condition_comparisons.csv` - Statistical tests between conditions
- Multiple PNG visualizations

---

### `eda_results_all_subjects.py`
**Purpose**: EDA across all subjects for a given epoch length.

**CLI**: `python analysis/scripts/eda_results_all_subjects.py --epoch 5`

**Outputs** (in `analysis/outputs/ds005620/all_subjects_results_epoch{N}/`):
- `band_means.csv` - Per-subject, per-condition, per-band means
- `composites.csv` - Wide format with composite metrics
- `awake_vs_sedation.csv` - Condition comparison statistics
- Multiple PNG visualizations

---

### `analyze_hyperparams.py`
**Purpose**: Test how channel count and epoch length affect MIB.

**CLI**:
```bash
python analysis/scripts/analyze_hyperparams.py                    # Default: sub-1067
python analysis/scripts/analyze_hyperparams.py --subject sub-1010
python analysis/scripts/analyze_hyperparams.py --test channels    # Only channel test
python analysis/scripts/analyze_hyperparams.py --test epochs      # Only epoch test
```

**Tests**:
1. Channel count effect (4-24 channels) on broadband MIB
2. Epoch length effect (2-20 seconds) on broadband MIB

**Outputs**:
- `channel_count_effect.png`
- `epoch_length_effect.png`

---

### `analyze_bin_effect.py`
**Purpose**: Analyze histogram bin count effect on MIB with Gaussian baseline comparison.

**CLI**:
```bash
python analysis/scripts/analyze_bin_effect.py                     # Default: sub-1067
python analysis/scripts/analyze_bin_effect.py --subject sub-1010
python analysis/scripts/analyze_bin_effect.py --gaussianity       # Only Gaussianity test
python analysis/scripts/analyze_bin_effect.py --n-channels 12
```

**Features**:
- Tests MIB across bin counts (5-500)
- Computes analytical Gaussian MIB baseline
- Gaussianity tests (Shapiro-Wilk, Jarque-Bera, etc.)

**Outputs**:
- `bin_count_effect_multi.png`
- `gaussianity_comparison.png`

---

### `plot_sedation_fourway.py`
**Purpose**: Four-way sedation comparison (baseline, light, deep, recovery) across bands.

**CLI**:
```bash
python analysis/scripts/plot_sedation_fourway.py                  # All plots
python analysis/scripts/plot_sedation_fourway.py --plot trajectories
python analysis/scripts/plot_sedation_fourway.py --plot heatmap ranges
python analysis/scripts/plot_sedation_fourway.py --root /path/to/results
```

**Plot Types**:
- `trajectories`: Subject trajectories per band
- `heatmap`: Mean values heatmap by band/condition
- `ranges`: Mean with min-max ranges

**Outputs**:
- `fourway_band_means_raw.csv` - Raw data
- `fourway_band_summary.csv` - Summary statistics
- `fourway_band_pairwise.csv` - Pairwise comparisons
- `fourway_all_bands.png` - Subject trajectories
- `fourway_band_heatmap.png` - Mean values heatmap
- `fourway_band_mean_range.png` - Mean with ranges

---

## Utility Modules

### `scripts/utils/mib_core.py`
Shared MIB computation functions:
- `build_mib_analyzer()` - Create ComplexityAnalyzer with BinningEstimator
- `compute_single_random_channel_mib()` - Single random sample MIB
- `compute_random_channel_stats()` - Multi-sample statistics
- `compute_epoch_stability()` - Fixed-channel baseline
- `aggregate_repeat_results()` - Summarize across repeats
- `select_channel_indices()` - Channel selection with preferences

### `scripts/utils/metadata.py`
Path parsing utilities:
- `infer_subject_from_path()` - Extract subject ID
- `infer_condition_from_path()` - Extract condition name
- `parse_brainvision_filename()` - Parse BIDS-like filenames

### `analysis/scripts/utils/stats.py`
Statistical functions:
- `mean_ci()` - 95% confidence interval
- `summarize_diff()` - Difference statistics with CI
- `cohen_d()` - Effect size
- `fdr_bh()` - Benjamini-Hochberg FDR correction
- `jaccard_mean()` - Mean Jaccard similarity

### `analysis/scripts/utils/config.py`
Constants:
- `BANDS` - Frequency band list
- `COMPOSITE_BANDS` - Band combinations (high_abc, low_dt, etc.)
- `DS005620_CONDITIONS` - DS005620 condition names
- `SEDATION_CONDITIONS` - Sedation dataset condition names
- `SEDATION_CONDITION_PAIRS` - Pairwise comparisons

### `analysis/scripts/utils/data_loading.py`
Data loading:
- `load_mib_results()` - Load MIB JSON files to DataFrame
- `extract_band_records()` - Extract band data from single JSON
- `aggregate_subject_results()` - Pivot to wide format with composites

### `analysis/scripts/utils/visualization.py`
Plotting helpers:
- `BAND_COLORS`, `CONDITION_COLORS`, `CONDITION_MARKERS`
- `plot_paired_trajectories()` - Subject trajectory plots
- `plot_condition_bars()` - Bar plots with CI
- `setup_faceted_subplots()` - Multi-panel figure setup

---

## Refactoring History

### Completed Consolidations

1. **Unified MIB scripts**: Created `mib_analysis.py` with dataset plugins (ds005620, sedation)

2. **All scripts CLI-configurable**: Removed hardcoded paths
   - `eda_sub1010.py`: Now accepts `--subject` and `--dataset` args
   - `analyze_hyperparams.py`: Now accepts `--subject`, `--dataset`, `--test` args
   - `analyze_bin_effect.py`: Now accepts `--subject`, `--dataset`, `--gaussianity` args
   - `plot_sedation_fourway.py`: Now accepts `--root`, `--output`, `--plot` args

3. **Merged duplicate scripts**:
   - `eda_results_all_subjects.py` + `eda_results_all_subjects_epoch10.py` → single script with `--epoch` arg
   - `plot_sedation_fourway_all_bands.py` + `plot_sedation_fourway_band_ranges.py` → `plot_sedation_fourway.py`

4. **Centralized utilities**:
   - `analysis/scripts/utils/` - stats, config, data_loading, visualization
   - `scripts/utils/` - mib_core, metadata

### Code Quality Improvements (Latest)

5. **Type hints added** throughout the codebase:
   - All `src/eeg_analysis/` modules now have proper type annotations
   - Uses `from __future__ import annotations` for modern syntax
   - TYPE_CHECKING guards for runtime-optional imports

6. **Exception handling improved**:
   - Replaced bare `except Exception:` with specific `except (ValueError, RuntimeError):`
   - Added comments explaining fallback behavior

7. **sys.path manipulation** made conditional:
   - Scripts now try importing normally first
   - Only fall back to sys.path manipulation if package not installed
   - Cleaner with `pip install -e .`

8. **Centralized `format_epoch_path()`**:
   - Moved to `analysis/scripts/utils/config.py`
   - Removed duplicates from `eda_sub1010_results.py` and `eda_results_all_subjects.py`

9. **Test coverage expanded**:
   - Added `test_stats.py` (17 tests for statistical utilities)
   - Added `test_config.py` (10 tests for config constants)
   - Added `test_metadata.py` (12 tests for path parsing)
   - Fixed existing tests for refactored modules
   - Total: 51 passing tests

10. **Legacy code removed**:
    - Removed `random_channels_mib.py` and `sedation_mib_analysis.py` (use `mib_analysis.py` instead)
    - Removed `eeg_utils.py` backward compatibility shim
    - Updated all imports to use modular structure (`eeg_analysis.core`, `eeg_analysis.utils`, etc.)

11. **DS005620 loader module** created:
    - Added `src/eeg_analysis/loaders/ds005620.py` with epoch loading and preprocessing
    - Parallel structure to sedation loader for consistency
