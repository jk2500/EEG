# EEG Random-Channel MIB Runner

Compute Minimum Information Bipartition (MIB) statistics from EEG recordings by repeatedly sampling channel subsets (broadband and/or spectral bands). The single public entrypoint is `scripts/random_channels_mib.py`.

## Setup
- Use a recent Python 3 environment and install dependencies: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Place BrainVision `.vhdr/.vmrk/.eeg` files in a BIDS-like layout. By default the code expects `ds005620/sub-XXXX/eeg/*.vhdr` (symlinked to `datasets/ds005620`), but any path can be provided via CLI flags.

## Quickstarts
- Single file:  
  `python scripts/random_channels_mib.py --vhdr ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr --mode both --estimators ksg --n-channels 8 --repeats 20 --epoch-lengths 5 --output results/ds005620/mib_random_channels`
- Dataset sweep (uses `create_subject_file_map` to pick supported conditions):  
  `python scripts/random_channels_mib.py --dataset ds005620 --subjects sub-1010 sub-1011 --mode broadband --estimators ksg binning --epoch-lengths 5 10 --n-channels 8 --repeats 30`

## Key CLI flags
- `--vhdr` run a single file; otherwise `--dataset` + optional `--subjects` sweeps matching subjects.
- `--two-sample SUBJ_A SUBJ_B` convenience sweep for two specified subjects restricted to eyes-closed, eyes-open, and sedation_1 conditions (uses `--dataset` paths).
- `--mode` one of `broadband`, `spectral`, `both`.
- `--estimators` choose among `ksg`, `binning`, `gaussian`.
- `--epoch-lengths` one or more epoch durations (seconds); `--n-channels` channels per random draw; `--repeats` number of random subsets.
- `--jobs` number of parallel workers for epoch evaluation (`-1` = all cores).
- `--bands` limit spectral runs to specific bands (e.g., `--bands beta gamma`); defaults to all bands in config.
- `--fixed-channels` pins the epoch-stability check to a provided channel list; defaults to `channels_list` in the config.
- `--quiet` suppresses verbose logging.

## Outputs
- Broadband runs emit `mib_random_channels_broadband_<estimator>_<timestamp>.json`; spectral runs emit `mib_random_channels_spectral_<estimator>_<timestamp>.json`. Both capture overall stats plus per-repeat payloads; broadband also records fixed-channel epoch stability.
- Dataset sweeps nest outputs under `--output/<mode>/<estimator>/epoch-<len>s/<subject>/<condition>/`.
- Plots are produced only when using `ComplexityAnalyzer.run_analysis` (see `src/eeg_analysis/analyzers/complexity_analyzer.py`).
- Summaries: `scripts/analyze_mib_results.py` reads the JSON outputs and reports per-condition stability plus cross-subject differences. Example: `python scripts/analyze_mib_results.py --root results/ds005620/mib_random_channels --mode broadband --estimator ksg`.

## Configuration
- Default analysis parameters (channel list, epoch length, subsampling, reference, band definitions, default paths) live in `src/eeg_analysis/config.py`. Override via CLI arguments where exposed.
- Preprocessing and helpers are in `src/eeg_analysis/eeg_utils.py`; estimators in `src/eeg_analysis/analyzers/estimators.py`.

## Project layout
- `datasets/` raw datasets (`ds005620/`, `sedation_resting_state/`)
- `ds005620` and `data` are symlinks to the dataset folders above (backward compatible paths)
- `results/ds005620/` and `results/sedation_resting_state/` store JSON outputs per dataset
- `analysis/scripts/` analysis + plotting scripts; `analysis/outputs/` generated tables/plots; `analysis/reports/` writeups
- `scripts/` main CLI entrypoints
- `archives/` large downloaded artifacts (zips, PDFs)

## Utilities and tests
- Helper scripts: `compare_epoch_lengths.py` (epoch-length stability probe) and `check_file_durations.py` (quick metadata inspection).
- Run the unit tests with `pytest`.
