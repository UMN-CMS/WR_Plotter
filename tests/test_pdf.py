"""PDF / alpha_s band from the analyzer's PDF members (wrplotter/pdf.py)."""
import json

import hist
import numpy as np
import pytest
import uproot

from wrplotter import histogram_cache, inputs
from wrplotter.histogram_utils import rebin_histogram
from wrplotter.pdf import member_key, pdf_variations
from wrplotter.systematics import clear_cache, sample_variations

REGION, VAR = "wr_ee_resolved_sr", "mass_fourobject"
LAYOUT = {"name": "NNPDF31_nnlo_as_0118_mc_hessian_pdfas", "error_type": "symmhessian", "n_members": 103,
          "central": 0, "eigenvectors": [1, 100], "alphas_members": {"down": 101, "up": 102},
          "alphas_values": [0.116, 0.12], "alphas_scale": 0.75}
EDGES = [0.0, 1000.0, 2000.0, 8000.0]


@pytest.fixture(autouse=True)
def _clear():
    histogram_cache.clear_cache()
    inputs.clear_manifest_cache()
    clear_cache()
    yield
    histogram_cache.clear_cache()
    inputs.clear_manifest_cache()
    clear_cache()


def _members(seed):
    """(103, 800) members: yield-preserving migrations between the 0-1 and 1-2 TeV halves."""
    rng = np.random.default_rng(seed)
    h0 = np.zeros(800)
    h0[50:200] = rng.uniform(1, 2, 150)
    m = np.tile(h0, (103, 1))
    for i in range(1, 103):
        eps = rng.normal(0, 0.02)
        m[i, 50:100] *= 1 + eps
        m[i, 100:150] *= 1 - eps * m[0, 50:100].sum() / m[0, 100:150].sum()
    return m


def _write(path, members, with_nominal=True):
    nom = hist.Hist(hist.axis.Regular(800, 0, 8000, name=VAR), storage=hist.storage.Weight())
    nom.view()["value"] = members[0]
    nom.view()["variance"] = members[0]
    th2 = hist.Hist(hist.axis.Regular(800, 0, 8000, name=VAR),
                    hist.axis.Regular(103, -0.5, 102.5, name="member", flow=False),
                    storage=hist.storage.Double())
    th2.view()[...] = members.T
    with uproot.recreate(path) as f:
        if with_nominal:
            f[f"{REGION}/{VAR}_{REGION}"] = nom
        f[member_key(REGION, VAR, 325300)] = th2
    path.with_name(path.stem + "_systematics.json").write_text(json.dumps({"pdf_sets": {"325300": LAYOUT}}))
    return nom


def _rebin(h):
    return rebin_histogram(h, EDGES, density=False)


def test_band_is_the_quadrature_of_the_rebinned_members(tmp_path):
    m = _members(1)
    path = tmp_path / "WRAnalyzer_DYJets.root"
    nom = _write(path, m)
    out = pdf_variations([path], REGION, VAR, _rebin(nom), _rebin)
    coarse = np.stack([m[:, :100].sum(axis=1), m[:, 100:200].sum(axis=1), m[:, 200:].sum(axis=1)], axis=1)
    delta = np.sqrt(((coarse[1:101] - coarse[0]) ** 2).sum(axis=0))
    np.testing.assert_allclose(out["pdf"]["up"].values() - coarse[0], delta)
    np.testing.assert_allclose(coarse[0] - out["pdf"]["down"].values(), delta)
    alphas = 0.75 * (coarse[102] - coarse[101]) / 2
    np.testing.assert_allclose(out["alphas"]["up"].values() - coarse[0], alphas)
    # the migrations stay inside 0-2 TeV: a bin holding both halves has (almost) no PDF shift
    merged = pdf_variations([path], REGION, VAR, rebin_histogram(nom, [0.0, 2000.0, 8000.0], density=False),
                            lambda h: rebin_histogram(h, [0.0, 2000.0, 8000.0], density=False))
    assert merged["pdf"]["up"].values()[0] - m[0].sum() < 1e-9 * m[0].sum()


def test_members_are_summed_over_sub_era_files_before_the_quadrature(tmp_path):
    m1, m2 = _members(2), _members(3)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    p1, p2 = tmp_path / "a" / "WRAnalyzer_DYJets.root", tmp_path / "b" / "WRAnalyzer_DYJets.root"
    n1, n2 = _write(p1, m1), _write(p2, m2)
    out = pdf_variations([p1, p2], REGION, VAR, _rebin(n1 + n2), _rebin)
    both = pdf_variations([p1], REGION, VAR, _rebin(n1), _rebin)
    coarse = lambda m: np.stack([m[:, :100].sum(1), m[:, 100:200].sum(1), m[:, 200:].sum(1)], axis=1)  # noqa: E731
    s = coarse(m1) + coarse(m2)
    np.testing.assert_allclose(out["pdf"]["up"].values() - s[0], np.sqrt(((s[1:101] - s[0]) ** 2).sum(0)))
    assert set(both) == {"pdf", "alphas"}


def test_sample_variations_returns_the_pdf_band_for_a_pdf_only_file(tmp_path):
    m = _members(4)
    path = tmp_path / "WRAnalyzer_DYJets.root"
    nom = _write(path, m)
    out = sample_variations({path: _rebin(nom)}, REGION, VAR, ["pdf", "alphas", "jes"],
                            load=lambda f, k: None, rebin=_rebin)
    assert set(out) == {"pdf", "alphas"}
    # without the rebin callable (or members) there is no PDF band
    assert sample_variations({path: _rebin(nom)}, REGION, VAR, ["pdf"], load=lambda f, k: None) == {}
    assert pdf_variations([path], REGION, "mass_dilepton", _rebin(nom), _rebin) == {}
