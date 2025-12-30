"""Tests for analysis/scripts/utils/config.py."""

import sys
from pathlib import Path

import pytest

# Add analysis scripts utils path for imports
ANALYSIS_UTILS = Path(__file__).resolve().parents[1] / "analysis" / "scripts" / "utils"
if str(ANALYSIS_UTILS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_UTILS))

from config import (
    format_epoch_path,
    BANDS,
    SEDATION_CONDITIONS,
    COMPOSITE_BANDS,
)


class TestFormatEpochPath:
    """Tests for format_epoch_path function."""

    def test_integer_epoch(self):
        assert format_epoch_path(5.0) == "5p00s"

    def test_fractional_epoch(self):
        assert format_epoch_path(2.5) == "2p50s"

    def test_long_decimal(self):
        assert format_epoch_path(10.123) == "10p12s"

    def test_small_epoch(self):
        assert format_epoch_path(0.5) == "0p50s"


class TestBands:
    """Tests for BANDS constant."""

    def test_contains_standard_bands(self):
        expected = {"delta", "theta", "alpha", "beta", "gamma", "broadband"}
        assert set(BANDS) == expected

    def test_is_list(self):
        assert isinstance(BANDS, list)


class TestSedationConditions:
    """Tests for SEDATION_CONDITIONS constant."""

    def test_contains_four_conditions(self):
        assert len(SEDATION_CONDITIONS) == 4

    def test_contains_expected_conditions(self):
        expected = {"baseline", "light_sedation", "deep_sedation", "recovery"}
        assert set(SEDATION_CONDITIONS) == expected


class TestCompositeBands:
    """Tests for COMPOSITE_BANDS constant."""

    def test_is_dict(self):
        assert isinstance(COMPOSITE_BANDS, dict)

    def test_all_bands_in_composite_are_valid(self):
        valid_bands = set(BANDS)
        for name, band_list in COMPOSITE_BANDS.items():
            for band in band_list:
                assert band in valid_bands, f"Invalid band '{band}' in composite '{name}'"
