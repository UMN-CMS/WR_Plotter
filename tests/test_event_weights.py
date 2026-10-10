"""Tests for wrplotter.event_weights (reading and plotting the analyzer's
event_weights/<region>/... histograms)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import uproot
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

from wrplotter.event_weights import (
    CORRECTIONS, component_label, component_names, component_stats, format_summary,
    has_event_weights, plot_component, plot_weight_distributions, read_components,
    read_distribution,
)

REGION = "wr_ee_resolved_sr"


def _log_hist(values, bins=(200, -8, 2)):
    h = Hist(Regular(*bins, name="log10_abs_weight"), storage=Weight())
    h.fill(log10_abs_weight=np.asarray(values, dtype=float))
    return h


def _factor_hist(values):
    h = Hist(Regular(600, 0.0, 3.0, name="factor"), storage=Weight())
    h.fill(factor=np.asarray(values, dtype=float))
    return h


def _write(path, *, weights=(1.0, 2.0, -0.5), components=True):
    """A minimal MC file: the region's nominal mass_dilepton plus event_weights/."""
    w = np.asarray(weights, dtype=float)
    nominal = Hist(Regular(10, 0, 1000, name="mass_dilepton"), storage=Weight())
    nominal.fill(mass_dilepton=np.full(len(w), 500.0), weight=w)
    with uproot.recreate(path) as f:
        f[f"{REGION}/mass_dilepton_{REGION}"] = nominal
        f[f"event_weights/{REGION}/positive"] = _log_hist(np.log10(w[w > 0]))
        f[f"event_weights/{REGION}/negative"] = _log_hist(np.log10(-w[w < 0]))
        if components:
            comp = f"event_weights/{REGION}/components"
            f[f"{comp}/genweight_positive"] = _log_hist([0.0, 0.0], bins=(400, -12, 8))
            f[f"{comp}/genweight_negative"] = _log_hist([0.0], bins=(400, -12, 8))
            f[f"{comp}/normalization"] = _log_hist([-3.0] * 3, bins=(400, -12, 8))
            f[f"{comp}/pileup"] = _factor_hist([0.9, 1.1, 1.0])
            f[f"{comp}/MuonIdSf"] = _factor_hist([0.97, 0.97, 0.97])
            f[f"{comp}/{CORRECTIONS}"] = _factor_hist([0.873, 1.067, 0.97])
    return path


@pytest.fixture
def mc_file(tmp_path):
    return _write(tmp_path / "WRAnalyzer_DYJets.root")


class TestTotals:

    def test_has_event_weights(self, mc_file, tmp_path):
        assert has_event_weights(mc_file)
        empty = tmp_path / "empty.root"
        with uproot.recreate(empty) as f:
            f["x"] = _factor_hist([1.0])
        assert not has_event_weights(empty)
        assert not has_event_weights(tmp_path / "missing.root")

    def test_read_distribution(self, mc_file):
        dist = read_distribution([mc_file], REGION)
        assert dist.n_events == 3
        assert dist.negative_fraction == pytest.approx(1 / 3)
        # N_eff = (sum w)^2 / sum w^2 from the nominal region histogram.
        assert dist.n_eff == pytest.approx(2.5 ** 2 / (1 + 4 + 0.25))

    def test_read_distribution_sums_files(self, tmp_path):
        a = _write(tmp_path / "a.root")
        b = _write(tmp_path / "b.root", weights=(4.0,))
        dist = read_distribution([a, b], REGION)
        assert dist.n_events == 4
        assert dist.sumw == pytest.approx(6.5)

    def test_plot_total(self, mc_file):
        fig = plot_weight_distributions([("DY", "#5790fc", read_distribution([mc_file], REGION))],
                                        text="test", lumi=100.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestComponents:

    def test_read_components(self, mc_file):
        comps = read_components([mc_file], REGION)
        assert set(comps) == {"genweight_positive", "genweight_negative", "normalization",
                              "pileup", "MuonIdSf", CORRECTIONS}

    def test_read_components_sums_files(self, tmp_path):
        a = _write(tmp_path / "a.root")
        b = _write(tmp_path / "b.root")
        assert read_components([a, b], REGION)["pileup"].values(flow=True).sum() == 6

    def test_legacy_file_has_no_components(self, tmp_path):
        legacy = _write(tmp_path / "legacy.root", components=False)
        assert read_components([legacy], REGION) == {}
        assert read_distribution([legacy], REGION).n_events == 3

    def test_component_names_order(self):
        keys = ["genweight_positive", "genweight_negative", CORRECTIONS, "normalization", "pileup"]
        assert component_names(keys) == ["genweight", "normalization", "pileup", CORRECTIONS]

    @pytest.mark.parametrize("name, label", [
        ("MuonRecoSf", "Muon reco SF"),
        ("LooseMuonIdSf", "Loose muon ID SF"),
        ("ElectronTrigSf", "Electron trigger SF"),
        ("pileup", "Pileup weight"),
        ("some_new_factor", "some new factor"),
    ])
    def test_component_label(self, name, label):
        assert component_label(name) == label

    def test_stats(self, mc_file):
        comps = read_components([mc_file], REGION)
        pileup = component_stats(comps, "pileup")
        assert pileup.n == 3
        assert pileup.mean == pytest.approx(1.0, abs=0.005)
        assert pileup.outside == 0
        gen = component_stats(comps, "genweight")
        assert gen.n == 3
        assert gen.negative == pytest.approx(1 / 3)
        assert component_stats(comps, "dy_zpt") is None

    def test_stats_count_flow(self):
        stats = component_stats({"pileup": _factor_hist([1.0, 5.0])}, "pileup")
        assert stats.n == 2
        assert stats.outside == pytest.approx(0.5)
        assert stats.mean == pytest.approx(1.0, abs=0.005)

    @pytest.mark.parametrize("name", ["genweight", "normalization", "pileup", CORRECTIONS])
    def test_plot_component(self, mc_file, name):
        comps = read_components([mc_file], REGION)
        fig = plot_component(name, [("DY", "#5790fc", comps)], text="test", lumi=100.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_missing_component_is_none(self, mc_file):
        comps = read_components([mc_file], REGION)
        assert plot_component("dy_zpt", [("DY", "#5790fc", comps)], text="t", lumi=1.0) is None

    def test_summary(self, mc_file):
        text = format_summary(REGION, [("DY", read_distribution([mc_file], REGION))],
                              [("DY", read_components([mc_file], REGION))])
        assert REGION in text
        for name in ("[genweight]", "[normalization]", "[pileup]", "[MuonIdSf]", f"[{CORRECTIONS}]"):
            assert name in text
        assert text.index("[pileup]") < text.index(f"[{CORRECTIONS}]")
