"""Tests for wrplotter.plotting_helpers."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

from wrplotter.plotting_helpers import (
    custom_log_formatter,
    _format_bin_width,
    _safe_sum_hists,
    plot_stack,
)
from wrplotter.regions import Region
from wrplotter.variables import Variable


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


def _make_weighted_hist(values, nbins=10, lo=0, hi=1000):
    """Helper: create a 1D Weight() histogram with given bin values."""
    h = Hist(Regular(nbins, lo, hi, name="x"), storage=Weight())
    h.view(flow=False)["value"][:] = values
    h.view(flow=False)["variance"][:] = values
    return h


class TestPlotStack:

    def test_smoke_mc_and_data(self):
        """plot_stack returns a Figure when given MC stack + data."""
        region = Region("wr_mumu_resolved_dy_cr", "muon", True, "test region")
        variable = Variable("mass_fourobject", r"$m_{lljj}$", "GeV", 10, 0, 1000)

        mc1 = _make_weighted_hist(np.arange(1, 11, dtype=float) * 10)
        mc2 = _make_weighted_hist(np.arange(1, 11, dtype=float) * 5)
        data = _make_weighted_hist(np.arange(1, 11, dtype=float) * 15)

        fig = plot_stack(
            region, variable,
            stack_list=[mc1, mc2],
            stack_colors=["#5790fc", "#f89c20"],
            stack_labels=["Z+jets", r"$t\bar{t}$+tW"],
            data_hist=[data],
            xlim=(0, 1000), ylim=(1, 1e6), lumi=59.7,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_smoke_mc_only_blinded(self):
        """plot_stack works with show_data=False (blinded SR)."""
        region = Region("wr_mumu_resolved_sr", "muon", False, "SR test")
        variable = Variable("mass_fourobject", r"$m_{lljj}$", "GeV", 10, 0, 1000)

        mc = _make_weighted_hist(np.arange(1, 11, dtype=float) * 10)

        fig = plot_stack(
            region, variable,
            stack_list=[mc],
            stack_colors=["#5790fc"],
            stack_labels=["Z+jets"],
            data_hist=[],
            xlim=(0, 1000), ylim=(1, 1e6), lumi=59.7,
            show_data=False,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_smoke_with_signal_overlay(self):
        """plot_stack works with signal histograms overlaid."""
        region = Region("wr_mumu_resolved_dy_cr", "muon", True, "CR test")
        variable = Variable("mass_fourobject", r"$m_{lljj}$", "GeV", 10, 0, 1000)

        mc = _make_weighted_hist(np.arange(1, 11, dtype=float) * 10)
        sig = _make_weighted_hist(np.ones(10) * 5)

        fig = plot_stack(
            region, variable,
            stack_list=[mc],
            stack_colors=["#5790fc"],
            stack_labels=["Z+jets"],
            data_hist=[],
            xlim=(0, 1000), ylim=(1, 1e6), lumi=59.7,
            signal_hists={"signal_WR4000_N2100": sig},
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_smoke_with_syst_hists(self):
        """plot_stack renders successfully when systematic variations are provided."""
        region = Region("wr_mumu_resolved_dy_cr", "muon", True, "CR test")
        variable = Variable("mass_fourobject", r"$m_{lljj}$", "GeV", 10, 0, 1000)

        mc   = _make_weighted_hist(np.arange(1, 11, dtype=float) * 10)
        data = _make_weighted_hist(np.arange(1, 11, dtype=float) * 12)
        up   = _make_weighted_hist(np.arange(1, 11, dtype=float) * 11)
        down = _make_weighted_hist(np.arange(1, 11, dtype=float) * 9)

        syst_hists = {
            region.name: {
                variable.name: {
                    "jes": {"up": {"DYJets": up}, "down": {"DYJets": down}}
                }
            }
        }

        fig = plot_stack(
            region, variable,
            stack_list=[mc], stack_colors=["#5790fc"], stack_labels=["Z+jets"],
            data_hist=[data], xlim=(0, 1000), ylim=(1, 1e6), lumi=59.7,
            syst_hists=syst_hists,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_syst_envelope_label_changes(self):
        """Legend label says 'stat. + syst.' when systematics are non-zero."""
        region = Region("wr_mumu_resolved_dy_cr", "muon", True, "CR test")
        variable = Variable("mass_fourobject", r"$m_{lljj}$", "GeV", 10, 0, 1000)

        mc   = _make_weighted_hist(np.full(10, 100.0))
        up   = _make_weighted_hist(np.full(10, 120.0))   # +20%
        down = _make_weighted_hist(np.full(10, 80.0))    # -20%

        # Stat-only (no syst_hists)
        fig_stat = plot_stack(
            region, variable,
            stack_list=[mc], stack_colors=["blue"], stack_labels=["MC"],
            data_hist=[], xlim=(0, 1000), ylim=(1, 1e6), lumi=59.7,
            show_data=False,
        )

        # Stat+syst
        syst_hists = {
            region.name: {
                variable.name: {"jes": {"up": {"DY": up}, "down": {"DY": down}}}
            }
        }
        fig_syst = plot_stack(
            region, variable,
            stack_list=[mc], stack_colors=["blue"], stack_labels=["MC"],
            data_hist=[], xlim=(0, 1000), ylim=(1, 1e6), lumi=59.7,
            show_data=False, syst_hists=syst_hists,
        )

        labels_stat = [t.get_text() for t in fig_stat.axes[0].get_legend().get_texts()]
        labels_syst = [t.get_text() for t in fig_syst.axes[0].get_legend().get_texts()]
        assert any("stat. unc." in l for l in labels_stat)
        assert any("syst." in l for l in labels_syst)
        plt.close(fig_stat)
        plt.close(fig_syst)
