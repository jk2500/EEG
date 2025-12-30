"""Tests for scripts/utils/metadata.py."""

import sys
from pathlib import Path

import pytest

# Add scripts/utils path for imports
SCRIPTS_UTILS = Path(__file__).resolve().parents[1] / "scripts" / "utils"
if str(SCRIPTS_UTILS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_UTILS))

from metadata import (
    infer_subject_from_path,
    infer_condition_from_path,
    parse_brainvision_filename,
)


class TestInferSubjectFromPath:
    """Tests for infer_subject_from_path function."""

    def test_extracts_from_filename(self):
        path = "sub-1010_task-awake_acq-EO_eeg.vhdr"
        assert infer_subject_from_path(path) == "sub-1010"

    def test_extracts_from_directory_path(self):
        path = "/data/ds005620/sub-1019/eeg/sub-1019_task-sed_acq-rest_run-1_eeg.vhdr"
        assert infer_subject_from_path(path) == "sub-1019"

    def test_returns_none_for_invalid_path(self):
        path = "/data/random_file.vhdr"
        assert infer_subject_from_path(path) is None


class TestInferConditionFromPath:
    """Tests for infer_condition_from_path function."""

    def test_extracts_awake_eo(self):
        path = "sub-1010_task-awake_acq-EO_eeg.vhdr"
        result = infer_condition_from_path(path)
        assert "awake" in result.lower()

    def test_extracts_sedation_from_path(self):
        path = "sub-1010_task-sed_acq-rest_run-1_eeg.vhdr"
        result = infer_condition_from_path(path)
        assert "sedation" in result.lower()

    def test_extracts_from_directory_structure(self):
        path = "/data/sedation/sub-001/baseline/file.set"
        assert infer_condition_from_path(path) == "baseline"

    def test_extracts_deep_sedation(self):
        path = "/data/sedation/sub-001/deep_sedation/file.set"
        assert infer_condition_from_path(path) == "deep_sedation"


class TestParseBrainvisionFilename:
    """Tests for parse_brainvision_filename function."""

    def test_parses_simple_filename(self):
        filename = "sub-1010_task-awake_acq-EO_eeg.vhdr"
        result = parse_brainvision_filename(filename)
        assert result is not None
        assert result["subject"] == "sub-1010"
        assert result["task"] == "awake"
        assert result["acq"] == "EO"
        assert result["run"] is None

    def test_parses_filename_with_run(self):
        filename = "sub-1019_task-sed_acq-rest_run-1_eeg.vhdr"
        result = parse_brainvision_filename(filename)
        assert result is not None
        assert result["subject"] == "sub-1019"
        assert result["task"] == "sed"
        assert result["acq"] == "rest"
        assert result["run"] == "1"

    def test_returns_none_for_invalid_filename(self):
        filename = "random_file.vhdr"
        assert parse_brainvision_filename(filename) is None

    def test_condition_and_run_id_fields(self):
        filename = "sub-1010_task-awake_acq-EO_eeg.vhdr"
        result = parse_brainvision_filename(filename)
        assert result["condition"] == "awake_EO"
        assert result["run_id"] == "awake_EO"  # No run number

    def test_run_id_includes_run_number(self):
        filename = "sub-1019_task-sed_acq-rest_run-2_eeg.vhdr"
        result = parse_brainvision_filename(filename)
        assert result["run_id"] == "sed_rest_run-2"
