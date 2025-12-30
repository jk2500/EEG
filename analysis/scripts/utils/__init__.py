"""Utility modules for EEG analysis scripts."""

from .stats import mean_ci, summarize_diff, cohen_d, fdr_bh, jaccard_mean
from .data_loading import load_mib_results, extract_band_records, aggregate_subject_results
from .visualization import (
    BAND_COLORS,
    CONDITION_COLORS,
    CONDITION_MARKERS,
    plot_paired_trajectories,
    plot_condition_bars,
    setup_faceted_subplots,
)
from .config import (
    BANDS,
    DS005620_CONDITIONS,
    SEDATION_CONDITIONS,
    SEDATION_CONDITION_PAIRS,
    COMPOSITE_BANDS,
    format_epoch_path,
)

__all__ = [
    # stats
    "mean_ci",
    "summarize_diff",
    "cohen_d",
    "fdr_bh",
    "jaccard_mean",
    # data_loading
    "load_mib_results",
    "extract_band_records",
    "aggregate_subject_results",
    # visualization
    "BAND_COLORS",
    "CONDITION_COLORS",
    "CONDITION_MARKERS",
    "plot_paired_trajectories",
    "plot_condition_bars",
    "setup_faceted_subplots",
    # config
    "BANDS",
    "DS005620_CONDITIONS",
    "SEDATION_CONDITIONS",
    "SEDATION_CONDITION_PAIRS",
    "COMPOSITE_BANDS",
    "format_epoch_path",
]
