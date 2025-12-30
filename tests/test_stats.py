"""Tests for analysis/scripts/utils/stats.py functions."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add analysis scripts utils path for imports
ANALYSIS_UTILS = Path(__file__).resolve().parents[1] / "analysis" / "scripts" / "utils"
if str(ANALYSIS_UTILS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_UTILS))

from stats import mean_ci, summarize_diff, cohen_d, fdr_bh, jaccard_mean


class TestMeanCI:
    """Tests for mean_ci function."""

    def test_returns_tuple_of_two_floats(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mean_ci(arr)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_ci_contains_mean(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        low, high = mean_ci(arr)
        assert low < arr.mean() < high

    def test_single_value_returns_nan(self):
        arr = np.array([1.0])
        low, high = mean_ci(arr)
        assert np.isnan(low)
        assert np.isnan(high)

    def test_constant_array_returns_zero_width_ci(self):
        arr = np.array([5.0, 5.0, 5.0, 5.0])
        low, high = mean_ci(arr)
        assert low == high == 5.0


class TestSummarizeDiff:
    """Tests for summarize_diff function."""

    def test_returns_expected_keys(self):
        diff = np.array([0.1, 0.2, -0.1, 0.3])
        result = summarize_diff(diff)
        expected_keys = {
            "n", "mean_diff", "median_diff", "std_diff", "pos_frac",
            "t_stat", "t_p", "cohen_d", "wilcoxon_stat", "wilcoxon_p",
            "ci_low", "ci_high"
        }
        assert set(result.keys()) == expected_keys

    def test_n_matches_array_size(self):
        diff = np.array([0.1, 0.2, 0.3])
        result = summarize_diff(diff)
        assert result["n"] == 3

    def test_pos_frac_calculation(self):
        diff = np.array([1.0, 2.0, -1.0, 3.0])  # 3 out of 4 positive
        result = summarize_diff(diff)
        assert result["pos_frac"] == 0.75


class TestCohenD:
    """Tests for cohen_d function."""

    def test_zero_effect_size(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert cohen_d(a, b) == pytest.approx(0.0)

    def test_large_effect_size(self):
        a = np.array([10.0, 11.0, 12.0, 10.5, 11.5])
        b = np.array([1.0, 2.0, 3.0, 1.5, 2.5])
        d = cohen_d(a, b)
        assert d > 0.8  # Large effect size

    def test_insufficient_samples_returns_nan(self):
        a = np.array([1.0])
        b = np.array([2.0, 3.0])
        assert np.isnan(cohen_d(a, b))


class TestFdrBH:
    """Tests for fdr_bh function."""

    def test_preserves_order_of_significance(self):
        pvalues = np.array([0.01, 0.05, 0.10])
        adjusted = fdr_bh(pvalues)
        assert adjusted[0] < adjusted[1] < adjusted[2]

    def test_adjusted_values_between_0_and_1(self):
        pvalues = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        adjusted = fdr_bh(pvalues)
        assert np.all(adjusted >= 0.0)
        assert np.all(adjusted <= 1.0)

    def test_single_pvalue(self):
        pvalues = np.array([0.05])
        adjusted = fdr_bh(pvalues)
        assert adjusted[0] == pytest.approx(0.05)


class TestJaccardMean:
    """Tests for jaccard_mean function."""

    def test_identical_sets_return_one(self):
        sets = [{"a", "b", "c"}, {"a", "b", "c"}]
        assert jaccard_mean(sets) == pytest.approx(1.0)

    def test_disjoint_sets_return_zero(self):
        sets = [{"a", "b"}, {"c", "d"}]
        assert jaccard_mean(sets) == pytest.approx(0.0)

    def test_partial_overlap(self):
        sets = [{"a", "b", "c"}, {"b", "c", "d"}]  # overlap: {b, c}, union: {a, b, c, d}
        # Jaccard = 2/4 = 0.5
        assert jaccard_mean(sets) == pytest.approx(0.5)

    def test_single_set_returns_nan(self):
        sets = [{"a", "b"}]
        assert np.isnan(jaccard_mean(sets))
