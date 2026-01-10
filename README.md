# EEG Minimum Information Bipartition (MIB) Pipeline

Compute an IIT-motivated integration proxy (Minimum Information Bipartition; MIB) on EEG by repeatedly sampling random channel subsets and evaluating MIB across epochs and spectral bands. The repo also contains an end-to-end pipeline to regenerate the paper figures/tables from raw dataset outputs.

## What’s in here
- `scripts/mib_analysis.py`: main CLI for computing per-file / per-dataset MIB JSON outputs (ds005620 + Sedation-RestingState).
- `analysis/scripts/`: post-hoc analysis + plotting; writes to `analysis/outputs/`.
- `paper/paper.tex`: LaTeX source; figures/tables are staged into `paper/` by the pipeline.
- `run_optimal_analysis.sh`: one-command pipeline to regenerate analysis outputs and stage paper assets.

## Setup
Python: 3.10–3.12 recommended (some deps may not support 3.13 yet).

1) Create a venv and install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

2) Data
- `datasets/ds005620/` is a git submodule (and `ds005620/` is a convenience symlink):
```bash
git submodule update --init --recursive
```
- Sedation-RestingState is expected at `datasets/sedation_resting_state/` (the repo contains a convenience symlink `Sedation-RestingState -> datasets/sedation_resting_state`). The `.set/.fdt` files are not committed.

## Run the full paper pipeline
```bash
bash run_optimal_analysis.sh
```
This will:
- compute MIB JSONs into `results/…`
- generate plots/tables into `analysis/outputs/…`
- export the required figures/tables/source CSVs into `paper/` (and, by default, overwrite previous staged assets)

Useful overrides:
- `FORCE=1` recompute even if JSONs exist
- `COMPILE_PDF=1` run `pdflatex` twice in `paper/`
- `CLEAN_PAPER=0` keep existing `paper/{figures,tables,source_data}` instead of wiping first

Example:
```bash
FORCE=1 COMPILE_PDF=1 bash run_optimal_analysis.sh
```

## Run analysis directly (without the full pipeline)
Dataset sweep (ds005620, spectral, 5s epochs):
```bash
python scripts/mib_analysis.py ds005620 dataset --mode spectral --epoch-length 5 --n-channels 16 --repeats 50 --jobs -1
```

Single file:
```bash
python scripts/mib_analysis.py ds005620 single --file ds005620/sub-1010/eeg/sub-1010_task-awake_acq-EO_eeg.vhdr --mode spectral
```

## Important numerical constraint (binning estimator)
The binning estimator encodes joint bin-states using an int64 base-`B` index (see `src/eeg_analysis/analyzers/estimators.py`). This requires:
- `N_BINS^N_CHANNELS < 2^63`

Defaults are chosen to satisfy this (`N_BINS=10`, `N_CHANNELS=16`). The pipeline includes a guard that exits early if you set an invalid combination.

## Tests
```bash
pytest
```

## Extra docs (optional)
- `METHODOLOGY.md`: methodology notes for the paper writeup.
- `SCRIPTS_README.md`: catalog of scripts, CLIs, and outputs.
