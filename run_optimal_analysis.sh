#!/usr/bin/env bash
# =============================================================================
# Optimal MIB Analysis Pipeline
# =============================================================================
# Parameters (based on hyperparameter analysis for minimum CV with coverage):
#   - n_bins: 10 (fits int64 joint-state encoding for k=16)
#   - n_channels: 16 (minimum CV)
#   - estimator: binning (signal is non-Gaussian)
#   - mode: spectral (all bands)
#
# This script is designed to regenerate all figures/tables used by
# `paper/paper.tex` (and stage them into `paper/` so it can be zipped).
# =============================================================================

set -euo pipefail

# -----------------------------
# Configuration (overridable)
# -----------------------------
N_BINS="${N_BINS:-10}"
HIGH_BINS="${HIGH_BINS:-200}"
N_CHANNELS="${N_CHANNELS:-16}"
REPEATS="${REPEATS:-50}"
JOBS="${JOBS:--1}"

# Run both epoch lengths for ds005620 (primary + sensitivity).
EPOCHS_DS005620="${EPOCHS_DS005620:-"5 10"}"

# Robustness diagnostics use a representative subject and smaller k.
DIAG_SUBJECT="${DIAG_SUBJECT:-sub-1067}"
DIAG_CHANNELS="${DIAG_CHANNELS:-8}"

# Optional: limit dataset sweeps (space-separated lists).
DS005620_SUBJECTS="${DS005620_SUBJECTS:-}"
DS005620_CONDITIONS="${DS005620_CONDITIONS:-}"
SEDATION_SUBJECTS="${SEDATION_SUBJECTS:-}"
SEDATION_CONDITIONS="${SEDATION_CONDITIONS:-}"

# Set FORCE=1 to recompute even if outputs exist.
FORCE="${FORCE:-0}"

# Optional: set COMPILE_PDF=1 to run pdflatex at the end.
COMPILE_PDF="${COMPILE_PDF:-0}"

# Optional: set CLEAN_PAPER=0 to keep any existing assets in paper/{figures,tables,source_data}.
# Default is to clean so the paper/ folder always reflects the latest run outputs.
CLEAN_PAPER="${CLEAN_PAPER:-1}"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Results directories (match analysis script defaults)
DS005620_OUTPUT_E5="results/ds005620/mib_analysis_optimal"
DS005620_OUTPUT_E10="results/ds005620/mib_analysis_optimal_epoch10"
SEDATION_OUTPUT="results/sedation_resting_state/mib_sedation"

# Analysis outputs that are later exported into paper/
ANALYSIS_ROOT="analysis/outputs"
ROBUSTNESS_OUT="$ANALYSIS_ROOT/robustness"
DS005620_ANALYSIS_E5="$ANALYSIS_ROOT/ds005620/all_subjects_results"
DS005620_ANALYSIS_E10="$ANALYSIS_ROOT/ds005620/all_subjects_results_epoch10"
SUB1010_ANALYSIS="$ANALYSIS_ROOT/ds005620/sub1010_results"
SEDATION_ANALYSIS="$ANALYSIS_ROOT/sedation_resting_state"

echo "============================================================"
echo "  Optimal MIB Analysis Pipeline"
echo "============================================================"
echo "Parameters:"
echo "  - n_bins:       $N_BINS"
echo "  - epochs (ds005620): ${EPOCHS_DS005620}"
echo "  - n_channels:   $N_CHANNELS"
echo "  - repeats:      $REPEATS"
echo "  - mode:         spectral (all bands)"
echo "Diagnostics:"
echo "  - subject:      $DIAG_SUBJECT"
echo "  - diag k:       $DIAG_CHANNELS"
echo "============================================================"

# Ensure non-interactive plotting + writable temp dirs (joblib/numba/matplotlib).
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MNE_USE_NUMBA="${MNE_USE_NUMBA:-false}"
export NUMBA_DISABLE_CACHING="${NUMBA_DISABLE_CACHING:-1}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$SCRIPT_DIR/.numba_cache}"
export TMPDIR="${TMPDIR:-$SCRIPT_DIR/.tmp}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-$TMPDIR/joblib}"
mkdir -p "$NUMBA_CACHE_DIR" "$JOBLIB_TEMP_FOLDER" "$ROBUSTNESS_OUT"

has_any_json() {
    local search_root="$1"
    local pattern="$2"
    find "$search_root" -type f -name "$pattern" -print -quit 2>/dev/null | grep -q .
}

# Step 1: Activate virtual environment if present
if [[ -d ".venv" ]]; then
    echo "[1/8] Activating virtual environment..."
    source .venv/bin/activate
elif [[ -d "venv" ]]; then
    echo "[1/8] Activating virtual environment..."
    source venv/bin/activate
else
    echo "[1/8] No virtual environment found, using system Python..."
fi

# Ensure package is installed
pip install -e . --quiet 2>/dev/null || true

# Guardrail: joint-state encoding in `BinningEstimator` uses int64 indices with a base-B multiplier.
# For k channels this requires B^k < 2^63 to avoid overflow/collisions.
python - "$N_BINS" "$N_CHANNELS" <<'PY'
import sys

n_bins = int(sys.argv[1])
n_channels = int(sys.argv[2])

if n_bins < 2:
    raise SystemExit(f"N_BINS must be >= 2 (got {n_bins}).")
if n_channels < 1:
    raise SystemExit(f"N_CHANNELS must be >= 1 (got {n_channels}).")

limit = (2**63) - 1

def max_bins_for_channels(k: int) -> int:
    if k <= 1:
        return limit
    # Start with a float approximation, then correct via integer checks.
    b = int(limit ** (1 / k))
    while pow(b + 1, k) <= limit:
        b += 1
    while pow(b, k) > limit:
        b -= 1
    return b

max_bins = max_bins_for_channels(n_channels)
if n_channels >= 2 and pow(n_bins, n_channels) > limit:
    raise SystemExit(
        "Invalid (N_BINS, N_CHANNELS) for int64 joint-state encoding: "
        f"{n_bins}^{n_channels} exceeds 2^63-1. "
        f"For N_CHANNELS={n_channels}, use N_BINS <= {max_bins} (e.g. 10)."
    )
PY

# Step 2: Run MIB analysis on ds005620 (epoch 5s and 10s)
echo ""
echo "[2/8] Running MIB analysis on ds005620 (primary dataset)..."
if [[ ! -d "ds005620" ]]; then
    echo "[2/8] ERROR: ds005620 dataset directory not found (expected ./ds005620 or ./datasets/ds005620)."
    exit 1
fi

for epoch in $EPOCHS_DS005620; do
    if [[ "$epoch" == "10" ]]; then
        out_dir="$DS005620_OUTPUT_E10"
    else
        out_dir="$DS005620_OUTPUT_E5"
    fi

    root_check="$out_dir/ds005620/spectral/binning"
    if [[ "$FORCE" == "1" ]] || ! has_any_json "$root_check" "mib_spectral_*.json"; then
        echo ""
        echo "  - epoch=${epoch}s | output=$out_dir"
        cmd=(python scripts/mib_analysis.py ds005620 dataset \
            --mode spectral \
            --n-channels "$N_CHANNELS" \
            --n-bins "$N_BINS" \
            --epoch-length "$epoch" \
            --repeats "$REPEATS" \
            --jobs "$JOBS" \
            --output "$out_dir")
        if [[ -n "$DS005620_SUBJECTS" ]]; then
            cmd+=(--subjects $DS005620_SUBJECTS)
        fi
        if [[ -n "$DS005620_CONDITIONS" ]]; then
            cmd+=(--conditions $DS005620_CONDITIONS)
        fi
        "${cmd[@]}"
    else
        echo "  - epoch=${epoch}s | output=$out_dir (found existing JSONs; skipping; set FORCE=1 to recompute)"
    fi
done
echo "[2/8] ds005620 analysis complete."

# Step 3: Run MIB analysis on Sedation-RestingState dataset
echo ""
echo "[3/8] Running MIB analysis on Sedation-RestingState dataset..."
echo "     Output: $SEDATION_OUTPUT"
echo "     Note: epoch_length is fixed by the .set files for this dataset."
echo ""

if [[ ! -d "Sedation-RestingState" ]]; then
    echo "[3/8] ERROR: Sedation-RestingState dataset directory not found (expected ./Sedation-RestingState or ./datasets/sedation_resting_state)."
    exit 1
fi

root_check="$SEDATION_OUTPUT/sedation/spectral/binning"
if [[ "$FORCE" == "1" ]] || ! has_any_json "$root_check" "mib_spectral_*.json"; then
    cmd=(python scripts/mib_analysis.py sedation dataset \
        --mode spectral \
        --n-channels "$N_CHANNELS" \
        --n-bins "$N_BINS" \
        --repeats "$REPEATS" \
        --jobs "$JOBS" \
        --output "$SEDATION_OUTPUT")
    if [[ -n "$SEDATION_SUBJECTS" ]]; then
        cmd+=(--subjects $SEDATION_SUBJECTS)
    fi
    if [[ -n "$SEDATION_CONDITIONS" ]]; then
        cmd+=(--conditions $SEDATION_CONDITIONS)
    fi
    "${cmd[@]}"
else
    echo "[3/8] Found existing sedation JSONs; skipping; set FORCE=1 to recompute."
fi
echo "[3/8] Sedation-RestingState analysis complete."

# Step 4: Aggregate results (optional convenience CSVs)
echo ""
echo "[4/8] Aggregating results summary CSVs..."

python analysis/scripts/aggregate_mib_results.py \
    --ds005620 "$DS005620_OUTPUT_E5" \
    --sedation "$SEDATION_OUTPUT" \
    --output "$ANALYSIS_ROOT/optimal_analysis"

# Step 5: Run robustness diagnostics (Gaussianity + bin sweep + hyperparams)
echo ""
echo "[5/8] Running robustness diagnostics..."

echo "  - Gaussianity diagnostics..."
python analysis/scripts/analyze_bin_effect.py \
    --subject "$DIAG_SUBJECT" \
    --dataset "ds005620" \
    --n-channels "$DIAG_CHANNELS" \
    --output "$ROBUSTNESS_OUT" \
    --gaussianity

echo "  - Bin-count sweep..."
python analysis/scripts/analyze_bin_effect.py \
    --subject "$DIAG_SUBJECT" \
    --dataset "ds005620" \
    --n-channels "$DIAG_CHANNELS" \
    --output "$ROBUSTNESS_OUT"

echo "  - Epoch-length + channel-count sensitivity..."
python analysis/scripts/analyze_hyperparams.py \
    --subject "$DIAG_SUBJECT" \
    --dataset "ds005620" \
    --output "$ROBUSTNESS_OUT"

# Step 6: Run ds005620 EDA (epoch 5s + 10s) and sub-1010 case study
echo ""
echo "[6/8] Running ds005620 EDA..."

python analysis/scripts/eda_results_all_subjects.py \
    --epoch 5 \
    --root "$DS005620_OUTPUT_E5/ds005620/spectral/binning"

python analysis/scripts/eda_results_all_subjects.py \
    --epoch 10 \
    --root "$DS005620_OUTPUT_E10/ds005620/spectral/binning"

python analysis/scripts/eda_sub1010_results.py \
    --subject "sub-1010" \
    --epoch 5 \
    --root "$DS005620_OUTPUT_E5/ds005620/spectral/binning"

# Step 7: Run Sedation-RestingState summaries/plots
echo ""
echo "[7/8] Running Sedation-RestingState summary + plots..."

python analysis/scripts/plot_sedation_fourway.py \
    --root "$SEDATION_OUTPUT/sedation/spectral/binning" \
    --output "$SEDATION_ANALYSIS" \
    --plot all

# Step 8: Export everything into paper/ (figures + tables + source_data)
echo ""
echo "[8/8] Exporting paper assets into paper/..."

if [[ "$CLEAN_PAPER" == "1" ]]; then
    echo "  - Cleaning previous paper assets (figures/, tables/, source_data/)..."
    rm -rf "paper/figures" "paper/tables" "paper/source_data"
fi

python analysis/scripts/export_paper_assets.py \
    --analysis-outputs "$ANALYSIS_ROOT" \
    --paper-dir "paper" \
    --main-bins "$N_BINS" \
    --high-bins "$HIGH_BINS"

if [[ "$COMPILE_PDF" == "1" ]]; then
    echo ""
    echo "Compiling paper/paper.tex (pdflatex)..."
    (cd paper && pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null)
    (cd paper && pdflatex -interaction=nonstopmode -halt-on-error paper.tex >/dev/null)
    echo "Paper compiled: paper/paper.pdf"
fi

# Summary
echo ""
echo "Pipeline complete!"
echo ""
echo "============================================================"
echo "  Results Summary"
echo "============================================================"
echo "ds005620 (5s) results:     $DS005620_OUTPUT_E5"
echo "ds005620 (10s) results:    $DS005620_OUTPUT_E10"
echo "Sedation results:          $SEDATION_OUTPUT"
echo "Analysis outputs:          $ANALYSIS_ROOT"
echo "Paper folder:              paper/"
echo ""
echo "Parameters used:"
echo "  n_bins=$N_BINS, channels=$N_CHANNELS, repeats=$REPEATS, jobs=$JOBS"
echo "============================================================"
