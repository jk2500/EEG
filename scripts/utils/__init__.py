"""Utility modules for MIB analysis scripts."""

from .mib_core import (
    build_mib_analyzer,
    compute_single_random_channel_mib,
    aggregate_repeat_results,
    compute_random_channel_stats,
    select_channel_indices,
    compute_epoch_stability,
)
from .metadata import infer_subject_from_path, infer_condition_from_path

__all__ = [
    # mib_core
    "build_mib_analyzer",
    "compute_single_random_channel_mib",
    "aggregate_repeat_results",
    "compute_random_channel_stats",
    "select_channel_indices",
    "compute_epoch_stability",
    # metadata
    "infer_subject_from_path",
    "infer_condition_from_path",
]
