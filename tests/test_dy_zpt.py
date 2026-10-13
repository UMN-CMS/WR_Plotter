"""Tests for wrplotter.dy_zpt: reading a Z-pT payload, evaluating it as the analyzer does, drawing it."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

from wrplotter.dy_zpt import (
    ew_at, load_zpt_payload, plot_applied_ratio, plot_ew_correction, plot_qcd_ratio, plot_spectra,
    plot_weight, qcd_at, weight_at, weight_edges,
)

QCD_EDGES = [0, 10, 50, 100, 500, 1000, 8000]
EW_EDGES = [30, 100, 1000, 6500]


def _payload(*, spectra=True, stat=True):
    n = len(QCD_EDGES) - 1
    central = [0.7, 0.9, 1.0, 1.1, 1.2, 1.25]
    qcd = {
        "edges": QCD_EDGES, "central": central, "pdf_err": [0.01] * n,
        "scale_up": [c * 1.1 for c in central], "scale_down": [c * 0.95 for c in central],
        "alphas_up": [c * 1.02 for c in central], "alphas_down": [c * 0.98 for c in central],
        "pt_max": 7999.0,
    }
    if stat:
        qcd["stat_err"] = [0.001] * n
    if spectra:
        qcd["lo_pb_per_bin"] = [100.0, 300.0, 100.0, 20.0, 1.0, 0.01]
        qcd["nlo_pb_per_bin"] = [c * v for c, v in zip(central, qcd["lo_pb_per_bin"])]
    ew = {"edges": EW_EDGES, "central": [1.0, 0.95, 0.8], "e1_err": [0.0, 0.01, 0.03],
          "e2_err": [0.001, 0.002, 0.02], "e3_err": [0.0, 0.001, 0.03], "pt_min": 31.0, "pt_max": 6499.0}
    prov = {"era": "TestEra", "integrated": {"nlo_over_lo": 0.999},
            "ew_note": "EW correction copied unchanged from dy_zpt_reweight_run2legacy_v4_2018.json"}
    return {"provenance": prov, "qcd": qcd, "ew": ew}


@pytest.fixture
def payload(tmp_path):
    path = tmp_path / "zpt.json"
    path.write_text(json.dumps(_payload()))
    return load_zpt_payload(path)


class TestLoad:

    def test_fields(self, payload):
        assert payload.era == "TestEra"
        np.testing.assert_allclose(payload.qcd_edges, QCD_EDGES)
        np.testing.assert_allclose(payload.qcd["central"], [0.7, 0.9, 1.0, 1.1, 1.2, 1.25])
        assert payload.ew_err.shape == (3, 3)
        assert payload.lo_pb is not None and payload.nlo_pb is not None

    def test_run2_payload_without_stat_and_spectra(self, tmp_path):
        raw = _payload(spectra=False, stat=False)
        raw["provenance"] = {"year": 2018}
        path = tmp_path / "run2.json"
        path.write_text(json.dumps(raw))
        p = load_zpt_payload(path)
        assert p.era == "2018"
        assert p.lo_pb is None
        np.testing.assert_array_equal(p.qcd["stat_err"], 0.0)

    def test_wrong_length_refused(self, tmp_path):
        raw = _payload()
        raw["qcd"]["pdf_err"] = [0.01]
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="pdf_err"):
            load_zpt_payload(path)


class TestEvaluation:

    def test_qcd_bins_and_clip(self, payload):
        q = qcd_at(payload, np.array([0.0, 9.99, 10.0, 999.0, 7999.0, 1e5]))
        np.testing.assert_allclose(q["central"], [0.7, 0.7, 0.9, 1.2, 1.25, 1.25])

    def test_ew_rule(self, payload):
        # below 30 GeV the factor at pt_min (first bin); from 6500 GeV on the factor at pt_max (last bin)
        e, err = ew_at(payload, np.array([0.0, 29.9, 30.0, 100.0, 6499.0, 6500.0, 1e5]))
        np.testing.assert_allclose(e, [1.0, 1.0, 1.0, 0.95, 0.8, 0.8, 0.8])
        assert err.shape == (3, 7)

    def test_weight_is_the_product_with_a_band_around_it(self, payload):
        w = weight_at(payload, np.array([5.0, 200.0, 3000.0]))
        np.testing.assert_allclose(w["nominal"], [0.7 * 1.0, 1.1 * 0.95, 1.25 * 0.8])
        assert np.all(w["up"] > w["nominal"]) and np.all(w["down"] < w["nominal"])
        # scale is the largest variation: the band is at least that wide
        assert w["up"][0] >= 0.7 * 1.1 and w["down"][0] <= 0.7 * 0.95

    def test_matches_the_analyzer(self, tmp_path, payload):
        """The plotter evaluates a payload exactly as wrcoffea.dy_reweight does."""
        dy_reweight = pytest.importorskip("wrcoffea.dy_reweight")
        path = tmp_path / "zpt.json"
        pt = np.array([0.0, 25.0, 30.0, 75.0, 999.0, 6499.0, 6500.0, 7999.0, 9000.0])
        theirs = dy_reweight.evaluate(pt, np.ones(len(pt), dtype=bool), dy_reweight.load_payload(str(path)))
        np.testing.assert_allclose(weight_at(payload, pt)["nominal"], theirs["nominal"])

    def test_weight_edges_merge_both_payloads(self, payload):
        edges = weight_edges(payload)
        assert edges[0] == 0 and edges[-1] == 8000
        assert set(QCD_EDGES) <= set(edges) and set(EW_EDGES) <= set(edges)
        assert np.all(np.diff(edges) > 0)
        assert weight_edges(payload, 1000)[-1] == 1000


class TestPlots:

    def test_each_plot_returns_a_figure(self, payload):
        for fig in (plot_qcd_ratio(payload, reference=payload, text="t"),
                    plot_qcd_ratio(payload, xlim=(1, 8000), logx=True),
                    plot_ew_correction(payload, text="t"),
                    plot_weight(payload, text="t"),
                    plot_spectra(payload, text="t")):
            assert isinstance(fig, plt.Figure)
            texts = {t.get_text() for t in fig.axes[0].texts}
            assert "CMS" in texts and "Simulation Work in progress" in texts
            plt.close(fig)

    def test_spectra_need_the_spectra(self, tmp_path):
        path = tmp_path / "nospec.json"
        path.write_text(json.dumps(_payload(spectra=False)))
        with pytest.raises(ValueError, match="spectra"):
            plot_spectra(load_zpt_payload(path))

    def test_applied_ratio(self, payload):
        before = Hist(Regular(20, 0, 1000, name="x"), storage=Weight())
        after = before.copy()
        before.view().value[...] = 100.0
        after.view().value[...] = 110.0
        fig = plot_applied_ratio(payload, before, after, xlabel="pT [GeV]", text="t")
        line = [l for l in fig.axes[0].lines if l.get_label().startswith("DY reweighted")][0]
        np.testing.assert_allclose(line.get_ydata(), 1.1)
        plt.close(fig)
