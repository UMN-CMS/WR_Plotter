"""Tests for wrplotter.histogram_utils."""
import numpy as np
import pytest
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

from wrplotter.histogram_utils import rebin_histogram, extract_hist_data


class TestRebinInteger:
    """Integer-spec rebinning (merge every N bins)."""

    def test_rebin_by_2_halves_bin_count(self, uniform_hist_10bins):
        result = rebin_histogram(uniform_hist_10bins, 2)
        assert len(result.axes[0].edges) - 1 == 5

    def test_rebin_by_2_preserves_total(self, uniform_hist_10bins):
        original_total = uniform_hist_10bins.values().sum()
        result = rebin_histogram(uniform_hist_10bins, 2)
        np.testing.assert_allclose(result.values().sum(), original_total, rtol=1e-10)

    def test_rebin_by_2_merges_correct_pairs(self, uniform_hist_10bins):
        result = rebin_histogram(uniform_hist_10bins, 2)
        # Original: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Pairs: [30, 70, 110, 150, 190]
        np.testing.assert_allclose(result.values(), [30, 70, 110, 150, 190])

    def test_rebin_by_2_sums_variances(self, uniform_hist_10bins):
        result = rebin_histogram(uniform_hist_10bins, 2)
        np.testing.assert_allclose(result.variances(), [30, 70, 110, 150, 190])

    def test_rebin_by_1_is_identity(self, uniform_hist_10bins):
        result = rebin_histogram(uniform_hist_10bins, 1)
        np.testing.assert_allclose(result.values(), uniform_hist_10bins.values())

    def test_preserves_axis_name(self, uniform_hist_10bins):
        result = rebin_histogram(uniform_hist_10bins, 2)
        assert result.axes[0].name == "x"

    def test_spec_exceeds_nbins_raises(self, uniform_hist_10bins):
        with pytest.raises(ValueError):
            rebin_histogram(uniform_hist_10bins, 11)

    def test_spec_zero_raises(self, uniform_hist_10bins):
        with pytest.raises(ValueError):
            rebin_histogram(uniform_hist_10bins, 0)

    def test_rebin_by_3_remainder(self, uniform_hist_10bins):
        result = rebin_histogram(uniform_hist_10bins, 3)
        # edges[::3] from [0,10,...,100] = [0,30,60,90], plus [100]
        # 4 bins: [0-30), [30-60), [60-90), [90-100)
        assert len(result.axes[0].edges) - 1 == 4


class TestRebinVariableEdges:
    """List-spec rebinning (variable-width with density normalization)."""

    def test_output_bin_count(self, uniform_hist_10bins, variable_bin_edges):
        result = rebin_histogram(uniform_hist_10bins, variable_bin_edges)
        assert len(result.axes[0].edges) - 1 == 3

    def test_density_normalization(self, uniform_hist_10bins, variable_bin_edges):
        result = rebin_histogram(uniform_hist_10bins, variable_bin_edges)
        vals = result.values()
        # [0,20): bins with values [10,20] -> sum=30, width=20 -> 1.5
        np.testing.assert_allclose(vals[0], 30.0 / 20.0)
        # [20,50): bins with values [30,40,50] -> sum=120, width=30 -> 4.0
        np.testing.assert_allclose(vals[1], 120.0 / 30.0)

    def test_variance_scales_by_width_squared(self, uniform_hist_10bins, variable_bin_edges):
        result = rebin_histogram(uniform_hist_10bins, variable_bin_edges)
        var = result.variances()
        # [0,20): var sum=30, /20^2 = 0.075
        np.testing.assert_allclose(var[0], 30.0 / (20.0**2))

    def test_edges_out_of_range_raises(self, uniform_hist_10bins):
        with pytest.raises(ValueError, match="out of range"):
            rebin_histogram(uniform_hist_10bins, [-10, 50, 100])
        with pytest.raises(ValueError, match="out of range"):
            rebin_histogram(uniform_hist_10bins, [0, 50, 200])

    def test_single_edge_raises(self, uniform_hist_10bins):
        with pytest.raises(ValueError):
            rebin_histogram(uniform_hist_10bins, [50])

    def test_weight_storage_output(self, uniform_hist_10bins, variable_bin_edges):
        result = rebin_histogram(uniform_hist_10bins, variable_bin_edges)
        assert result.variances() is not None


class TestRebinOverflow:
    """Overflow handling."""

    def test_overflow_added_to_last_bin(self, hist_with_overflow):
        result = rebin_histogram(hist_with_overflow, 1)
        # Last regular bin=50, overflow=100 -> 150
        assert result.values()[-1] == 50 + 100

    def test_overflow_variance_added(self, hist_with_overflow):
        result = rebin_histogram(hist_with_overflow, 1)
        assert result.variances()[-1] == 50 + 100


class TestExtractHistData:

    def test_returns_four_arrays(self, uniform_hist_10bins):
        result = extract_hist_data(uniform_hist_10bins)
        assert len(result) == 4

    def test_edges_shape(self, uniform_hist_10bins):
        edges, _, _, _ = extract_hist_data(uniform_hist_10bins)
        assert len(edges) == 11  # 10 bins -> 11 edges

    def test_centers_are_midpoints(self, uniform_hist_10bins):
        edges, centers, _, _ = extract_hist_data(uniform_hist_10bins)
        expected = 0.5 * (edges[:-1] + edges[1:])
        np.testing.assert_allclose(centers, expected)

    def test_values_match(self, uniform_hist_10bins):
        _, _, vals, _ = extract_hist_data(uniform_hist_10bins)
        np.testing.assert_allclose(vals, uniform_hist_10bins.values())

    def test_errors_are_sqrt_variance(self, uniform_hist_10bins):
        _, _, _, errs = extract_hist_data(uniform_hist_10bins)
        np.testing.assert_allclose(errs, np.sqrt(uniform_hist_10bins.variances()))

    def test_non_weight_storage_fallback(self):
        h = Hist(Regular(5, 0, 50, name="x"))
        h.fill([5, 15, 25, 35, 45])
        _, _, vals, errs = extract_hist_data(h)
        np.testing.assert_allclose(errs, np.sqrt(vals))
