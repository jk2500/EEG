# Methodology: Random-Channel MIB EEG Analysis

## Overview
This project estimates Minimum Information Bipartition (MIB) values from EEG recordings using repeated random channel subsets. The primary workflow is implemented in `scripts/random_channels_mib.py` and uses a binning-based mutual information estimator. Results are generated for broadband data and/or for predefined spectral bands.

## Theoretical background (methodology)
The pipeline is grounded in information-theoretic measures of integration. For a multichannel system, mutual information between two channel subsets captures how much the activity of one subset reduces uncertainty about the other. The Minimum Information Bipartition (MIB) is defined as the smallest mutual information across all non-trivial bipartitions, providing a lower bound on how integrated the system is across all possible splits.

EEG signals are continuous, so entropy and mutual information are estimated by discretizing amplitudes into bins. This binning approach is simple and deterministic, making it appropriate for large sweeps where consistent estimation across conditions is more important than absolute MI magnitude. To compare conditions fairly, the same estimator settings, resampling rate, and epoching are applied to every recording.

MIB is computed per epoch under a stationarity assumption within each window. The distribution of MIB values across epochs reflects temporal variability, while repeated random channel subsets reflect sensitivity to spatial sampling. Using both fixed-channel and random-channel views separates within-recording temporal variability from cross-channel selection effects.

Spectral analyses apply the same MIB computation to band-limited signals. Bandpass filtering isolates canonical rhythms, allowing the method to test whether integration differs by frequency range while keeping the information-theoretic metric unchanged.

## Design choices and justifications
- **Random channel sampling**: EEG montages vary and channel selection can bias integration estimates. Random draws quantify sensitivity to channel choice and provide stability statistics across subsets.
- **Fixed-channel stability**: A deterministic channel set isolates epoch-to-epoch variability within a single montage, separating temporal effects from channel sampling effects.
- **`n_channels = 8` default**: This balances spatial coverage with computational feasibility. For 8 channels the full bipartition set is 127, which matches the default partition cap and keeps per-epoch costs bounded.
- **Bipartition cap (`max_partitions = 127`)**: Limits combinatorial growth for larger channel counts while matching the full set for the default 8-channel case.
- **Binning estimator**: Chosen for deterministic, scalable MI estimation that is stable under large batch runs. It avoids heavy hyperparameter tuning and is efficient for repeated epochs.
- **Number of bins (`n_bins = 10`)**: A compromise between resolution and sample sparsity within each epoch; fewer bins reduce estimator variance for limited samples.
- **Epoch length (default 5 s)**: Long enough to provide stable entropy estimates while short enough to reduce nonstationarity within a window.
- **Non-overlapping fixed-length epochs**: Simplifies interpretation and avoids dependencies introduced by sliding windows.
- **Resampling to 500 Hz**: Standardizes sampling across recordings and ensures consistent spectral bounds while reducing computational load relative to native rates.
- **Broadband filter 1 to 40 Hz**: Removes slow drifts and high-frequency noise while preserving canonical EEG rhythms relevant to resting-state comparisons.
- **Spectral band definitions (delta/theta/alpha/beta/gamma)**: Use standard, widely accepted EEG bands to maximize interpretability and comparability.
- **Band-specific subsampling factors**: Reduce computational cost while preserving band-limited content, with more aggressive subsampling in lower-frequency bands where temporal resolution is less critical.
- **A2 reference with average fallback**: Aligns with the acquisition reference when available, and uses the standard average reference when it is not.
- **Auxiliary channel exclusion (VEOG/HEOG/EMG)**: Prevents non-cortical artifacts from influencing information estimates.
- **Use all EEG channels for preprocessing, then subsample for analysis**: Avoids bias in filtering/normalization while still probing channel-count sensitivity at the analysis stage.
- **Seeded randomness (`rng_seed`, `partition_seed`)**: Ensures reproducibility of random channel draws and partition subsampling.
- **Condition inclusion rule (>=2 conditions; prefer `sedation_1`)**: Enables within-subject comparisons and avoids duplicate sedation recordings when both are present.
- **Stability metric (CV of per-repeat means)**: A scale-free measure of variability that allows comparison across bands and conditions.

## Standard practices (not further justified)
- BIDS-like data organization and BrainVision file ingestion.
- MNE-Python preprocessing utilities and default filter design (FIR with `firwin`).
- Per-epoch z-score normalization by channel.
- Parallel epoch evaluation with joblib to reduce runtime.

## Data source and organization
- Data are read from BrainVision EEG files (`.vhdr/.eeg/.vmrk`) organized in a BIDS-like layout (default root: `ds005620`, symlinked to `datasets/ds005620`).
- Conditions are inferred from filenames and mapped to:
  - `awake_eyes_open` (task-awake, acq-EO)
  - `awake_eyes_closed` (task-awake, acq-EC)
  - `sedation_1` (task-sed, acq-rest)
  - `sedation_2` (task-sed2, acq-rest)
- For dataset sweeps, a subject is included only if at least two conditions are present. If both sedation files exist, `sedation_1` is preferred and `sedation_2` is dropped for consistency.

## Preprocessing (MNE-based)
All preprocessing is performed with MNE-Python using a consistent pipeline:
1. **Channel filtering and selection**
   - Exclude auxiliary channels by name: `VEOG`, `HEOG`, `EMG`.
   - Use EEG channels only.
2. **Resampling**
   - Data are resampled to a target sampling frequency of 500 Hz (`target_sfreq`).
3. **Re-referencing**
   - If channel `A2` is present, the data are re-referenced to A2.
   - Otherwise, average reference is applied.
4. **Bandpass filtering**
   - Broadband analysis uses a 1 to 40 Hz bandpass filter.
   - Spectral analysis uses band-specific filters (see "Spectral bands").
5. **Subsampling**
   - Broadband: downsample by factor 10.
   - Spectral: downsample by factor 2 by default; per-band overrides:
     - delta: 10, theta: 10, alpha: 8, beta: 4, gamma: 2.
6. **Epoching**
   - Fixed-length, non-overlapping epochs are generated (default 5.0 s).
7. **Normalization**
   - Each epoch is z-scored per channel (subtract mean, divide by standard deviation).

## Channel sampling strategy
Two complementary stability views are produced:
- **Fixed-channel epoch stability**
  - A deterministic channel set is used.
  - Preferred channels come from `channels_list` in `src/eeg_analysis/config.py`
    (default: `Fp1, Fp2, F3, F4, P3, P4, T7, T8`).
  - If any preferred channels are missing, the pipeline falls back to the first
    available EEG channels.
- **Random-channel stability**
  - For each repeat, `n_channels` (default 8) are randomly sampled without
    replacement from all available EEG channels.
  - A fixed RNG seed (default 42) ensures reproducible random draws.

## Spectral bands
Spectral analysis runs each band independently using the following ranges:
- delta: 0.5 to 4 Hz
- theta: 4 to 8 Hz
- alpha: 8 to 13 Hz
- beta: 13 to 30 Hz
- gamma: 30 to 100 Hz
- broadband: 1 to 100 Hz (included for comparison)

## MIB computation
For each epoch, the minimum integration across all non-trivial bipartitions is
computed. For a bipartition of channels into subsets A and B, integration is:

```
I(A,B) = H(A) + H(B) - H(A,B)
```

Where `H` denotes entropy. The MIB for an epoch is:

```
MIB = min_{A,B} I(A,B)
```

### Entropy estimation (binning)
Entropy is estimated using uniform histogram binning:
- Number of bins: `n_bins = 10`.
- For each channel subset, data are discretized into bins and joint entropy is
  computed from the empirical distribution.
- Non-finite samples are removed before entropy calculation.

### Bipartition sampling
All non-trivial bipartitions are generated deterministically. To control
computational cost, the total number of bipartitions can be capped:
- `max_partitions` default: 127.
- If the total number of bipartitions exceeds this cap, a random subset of
  partitions is selected without replacement using a fixed seed
  (`partition_seed = 42`).

### Parallel execution
Epoch-level MIB computations are parallelized with joblib. The default is
`n_jobs = -1` (use all available cores).

## Random-channel aggregation and stability metrics
For each random channel subset (repeat), the pipeline computes MIB values across
epochs and records:
- `mean_mib`: mean across epochs.
- `std_mib`: standard deviation across epochs.

Across all repeats, the following summary statistics are computed:
- Mean of per-repeat means.
- Standard deviation of per-repeat means.
- Coefficient of variation (CV) of per-repeat means.
- Mean of per-repeat standard deviations.
- Min, max, and range of per-repeat means.

## Output structure and downstream summaries
Results are saved as JSON files, including metadata and per-repeat payloads:
- Broadband: `mib_random_channels_broadband_binning_<timestamp>.json`
- Spectral: `mib_random_channels_spectral_binning_<timestamp>.json`

For dataset sweeps, results are organized as:

```
results/ds005620/mib_random_channels/<mode>/binning/epoch-<len>s/<subject>/<condition>/
```

The optional summary script `scripts/analyze_mib_results.py` computes:
- Stability via CV thresholds.
- Within-subject ordering checks (expected: eyes-open > eyes-closed > sedation_1).
- Cross-subject mean differences for two-subject comparisons.

## Reproducibility
Key parameters for reproducible analysis are fixed by default:
- Random channel sampling seed: `rng_seed = 42`.
- Bipartition sampling seed: `partition_seed = 42`.
- All preprocessing parameters are centralized in `src/eeg_analysis/config.py`.
