"""Tests for bin/emu_transfer_factor.py: fit-range semantics, edge validation, legacy loading."""
import runpy
from pathlib import Path

import numpy as np
import pytest
import uproot
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight


@pytest.fixture(scope="module")
def tf():
    script = Path(__file__).resolve().parent.parent / "bin" / "emu_transfer_factor.py"
    return runpy.run_path(str(script))


def _flat_ratio(edges, value=0.5, err=0.05):
    n = len(edges) - 1
    return np.full(n, value), np.full(n, err)


def test_fit_uses_only_bins_fully_inside_range(tf):
    edges = np.arange(0, 1001, 100, dtype=float)  # 10 bins, 800 is an edge
    ratio, err = _flat_ratio(edges)
    fv, ferr, chi2, ndf = tf["chi2_fit_constant"](ratio, err, edges, fit_range=(0, 800))
    assert ndf == 8 - 1
    fv, ferr, chi2, ndf = tf["chi2_fit_constant"](ratio, err, edges, fit_range=(800, 1000))
    assert ndf == 2 - 1
    assert fv == pytest.approx(0.5)


def test_bin_straddling_boundary_belongs_to_neither_range(tf):
    edges = np.array([0., 200., 400., 600., 750., 850., 1000., 1200.])
    ratio, err = _flat_ratio(edges)
    _, _, _, ndf_low = tf["chi2_fit_constant"](ratio, err, edges, fit_range=(0, 800))
    _, _, _, ndf_high = tf["chi2_fit_constant"](ratio, err, edges, fit_range=(800, 1200))
    assert ndf_low == 4 - 1   # bins [0,200],[200,400],[400,600],[600,750]
    assert ndf_high == 2 - 1  # bins [850,1000],[1000,1200]
    with pytest.raises(ValueError, match="must be a bin edge"):
        tf["study_fit_ranges"](edges)


def test_study_fit_ranges_split_at_800(tf):
    edges = np.array([0., 400., 800., 1600., 8000.])
    ranges, labels = tf["study_fit_ranges"](edges)
    assert ranges == [(0.0, 800.0), (800.0, 8000.0)]
    assert labels == ["low-mass", "high-mass"]


def _mass_hist(fill_values):
    h = Hist(Regular(800, 0, 8000, name="m"), storage=Weight())
    h.fill(fill_values, weight=np.ones(len(fill_values)))
    return h


def test_load_study_hist_prefers_nomass_and_falls_back_to_legacy(tf, tmp_path):
    region, var = "wr_ee_resolved_sr", "mass_fourobject"
    legacy = tmp_path / "legacy.root"
    with uproot.recreate(legacy) as f:
        f[f"{region}_tf_low/{var}_{region}_tf_low"] = _mass_hist([300.0, 700.0])
        f[f"{region}/{var}_{region}"] = _mass_hist([900.0, 2500.0, 2500.0])
    h = tf["load_study_hist"](legacy, region, var)
    assert h is not None
    assert h.values().sum() == 5.0
    i800 = int(np.searchsorted(h.axes[0].edges, 800.0))
    assert h.values()[:i800].sum() == 2.0 and h.values()[i800:].sum() == 3.0

    modern = tmp_path / "modern.root"
    with uproot.recreate(modern) as f:
        f[f"{region}_nomass/{var}_{region}_nomass"] = _mass_hist([100.0])
        f[f"{region}_tf_low/{var}_{region}_tf_low"] = _mass_hist([300.0, 700.0])
        f[f"{region}/{var}_{region}"] = _mass_hist([900.0])
    h = tf["load_study_hist"](modern, region, var)
    assert h.values().sum() == 1.0  # _nomass wins over legacy pieces

    assert tf["load_study_hist"](legacy, "wr_mumu_resolved_sr", var) is None

    older = tmp_path / "older.root"
    with uproot.recreate(older) as f:
        f[f"{region}_tf/{var}_{region}_tf"] = _mass_hist([300.0, 900.0])
        f[f"{region}/{var}_{region}"] = _mass_hist([900.0])
    h = tf["load_study_hist"](older, region, var)
    assert h.values().sum() == 2.0  # the older mass-inclusive _tf region is used as-is


def test_require_edge_accepts_present_edge(tf):
    edges = tf["require_edge"]([0, 400, 800, 1200], 800)
    assert 800 in edges
