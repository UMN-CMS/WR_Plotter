"""Tests for wrplotter.histogram_utils."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

from wrplotter.histogram_utils import rebin_histogram, load_histogram


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



class TestLoadHistogram:
    """Tests for load_histogram() via mocked _open_root cache."""

    def test_delegates_to_open_root(self, uniform_hist_10bins):
        """load_histogram calls _open_root and converts the result via .to_hist()."""
        mock_file = MagicMock()
        mock_file.__getitem__.return_value.to_hist.return_value = uniform_hist_10bins

        with patch("wrplotter.histogram_cache._open_root", return_value=mock_file) as mock_open:
            result = load_histogram(Path("/fake/path.root"), "some/key")

        mock_open.assert_called_once_with("/fake/path.root")
        mock_file.__getitem__.assert_called_once_with("some/key")
        assert result is uniform_hist_10bins

    def test_accepts_string_path(self, uniform_hist_10bins):
        """load_histogram also accepts a plain string path."""
        mock_file = MagicMock()
        mock_file.__getitem__.return_value.to_hist.return_value = uniform_hist_10bins

        with patch("wrplotter.histogram_cache._open_root", return_value=mock_file):
            result = load_histogram("/fake/path.root", "key")

        assert result is uniform_hist_10bins


class TestOverflowHint:
    """Folded overflow is flagged by an attribute, never by a fake overflow count."""

    def test_rebin_flags_folded_overflow_without_fake_content(self, hist_with_overflow, uniform_hist_10bins):
        from wrplotter.histogram_utils import has_overflow_hint
        folded = rebin_histogram(hist_with_overflow, 1)
        assert has_overflow_hint(folded)
        assert folded.values(flow=True).sum() == folded.values().sum() == 250
        assert not has_overflow_hint(rebin_histogram(uniform_hist_10bins, 1))

    def test_add_hists_keeps_the_hint_of_either_operand(self, hist_with_overflow):
        from wrplotter.histogram_utils import add_hists, has_overflow_hint, sum_hists
        flagged = rebin_histogram(hist_with_overflow, 1)
        plain = flagged.copy()
        plain.overflow_hint = False
        assert has_overflow_hint(add_hists(plain, flagged))
        assert has_overflow_hint(add_hists(flagged, plain))
        assert add_hists(None, plain) is plain and add_hists(plain, None) is plain
        assert has_overflow_hint(sum_hists([plain, plain, flagged]))
        assert sum_hists([]) is None

    def test_for_drawing_marks_only_a_copy(self, hist_with_overflow):
        from wrplotter.histogram_utils import for_drawing
        folded = rebin_histogram(hist_with_overflow, 1)
        drawn = for_drawing(folded)
        assert drawn is not folded
        assert drawn.view(flow=True)["value"][-1] > 0
        assert folded.view(flow=True)["value"][-1] == 0
        folded.overflow_hint = False
        assert for_drawing(folded) is folded
        assert for_drawing(None) is None
