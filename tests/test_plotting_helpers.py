"""Tests for wrplotter.plotting_helpers."""
import numpy as np
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

from wrplotter.plotting_helpers import (
    custom_log_formatter,
    _format_bin_width,
    _safe_sum_hists,
)


class TestCustomLogFormatter:

    def test_one(self):
        assert custom_log_formatter(1.0, None) == "1"

    def test_ten(self):
        assert custom_log_formatter(10.0, None) == "10"

    def test_hundred(self):
        assert custom_log_formatter(100.0, None) == r"$10^{2}$"

    def test_thousand(self):
        assert custom_log_formatter(1000.0, None) == r"$10^{3}$"

    def test_tenth(self):
        assert custom_log_formatter(0.1, None) == r"$10^{-1}$"

    def test_non_power_of_ten(self):
        assert custom_log_formatter(50.0, None) == ""

    def test_zero(self):
        assert custom_log_formatter(0.0, None) == ""

    def test_negative(self):
        assert custom_log_formatter(-1.0, None) == ""


class TestFormatBinWidth:

    def test_integer_width(self):
        assert _format_bin_width(10.0) == "10"

    def test_non_integer_width(self):
        assert _format_bin_width(2.5) == "2.5"

    def test_one(self):
        assert _format_bin_width(1.0) == "1"


class TestSafeSumHists:

    def test_empty_returns_none(self):
        assert _safe_sum_hists([]) is None

    def test_single_hist(self, uniform_hist_10bins):
        result = _safe_sum_hists([uniform_hist_10bins])
        np.testing.assert_allclose(result.values(), uniform_hist_10bins.values())

    def test_two_hists_summed(self):
        h1 = Hist(Regular(5, 0, 50, name="x"), storage=Weight())
        h2 = Hist(Regular(5, 0, 50, name="x"), storage=Weight())
        h1.view(flow=False)["value"][:] = 1.0
        h2.view(flow=False)["value"][:] = 2.0
        h1.view(flow=False)["variance"][:] = 1.0
        h2.view(flow=False)["variance"][:] = 2.0
        result = _safe_sum_hists([h1, h2])
        np.testing.assert_allclose(result.values(), 3.0)
        np.testing.assert_allclose(result.variances(), 3.0)
