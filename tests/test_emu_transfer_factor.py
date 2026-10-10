"""Tests for bin/emu_transfer_factor.py: fit-range semantics, edge validation,
legacy loading, and reading the eras of a combination."""
import argparse
import runpy
from pathlib import Path

import numpy as np
import pytest
import uproot
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

from wrplotter.inputs import make_inputs, root_file

DIR = "tf_test"
RESOLVED = ["wr_ee_resolved_sr", "wr_mumu_resolved_sr", "wr_resolved_flavor_cr"]
VAR = "mass_fourobject"


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


def test_require_edge_accepts_present_edge(tf):
    edges = tf["require_edge"]([0, 400, 800, 1200], 800)
    assert 800 in edges


# ── Loading ─────────────────────────────────────────────────────────────────────

def _mass_hist(fill_values):
    h = Hist(Regular(800, 0, 8000, name="m"), storage=Weight())
    h.fill(fill_values, weight=np.ones(len(fill_values)))
    return h


def _key(region, var=VAR):
    return f"{region}/{var}_{region}"


def _write(directory, contents, sample="tt_tW"):
    """contents: {hist key: histogram}."""
    directory.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(root_file(directory, sample)) as f:
        for key, h in contents.items():
            f[key] = h


def test_load_study_hist_prefers_nomass_and_falls_back_to_legacy(tf, tmp_path):
    region = "wr_ee_resolved_sr"

    def load_from(name):
        return tf["file_loader"](tmp_path / name, "tt_tW", 1)

    _write(tmp_path / "legacy", {
        _key(f"{region}_tf_low"): _mass_hist([300.0, 700.0]),
        _key(region): _mass_hist([900.0, 2500.0, 2500.0]),
    })
    h = tf["load_study_hist"](load_from("legacy"), region, VAR)
    assert h is not None
    assert h.values().sum() == 5.0
    i800 = int(np.searchsorted(h.axes[0].edges, 800.0))
    assert h.values()[:i800].sum() == 2.0 and h.values()[i800:].sum() == 3.0

    _write(tmp_path / "modern", {
        _key(f"{region}_nomass"): _mass_hist([100.0]),
        _key(f"{region}_tf_low"): _mass_hist([300.0, 700.0]),
        _key(region): _mass_hist([900.0]),
    })
    h = tf["load_study_hist"](load_from("modern"), region, VAR)
    assert h.values().sum() == 1.0  # _nomass wins over legacy pieces

    assert tf["load_study_hist"](load_from("legacy"), "wr_mumu_resolved_sr", VAR) is None

    _write(tmp_path / "older", {
        _key(f"{region}_tf"): _mass_hist([300.0, 900.0]),
        _key(region): _mass_hist([900.0]),
    })
    h = tf["load_study_hist"](load_from("older"), region, VAR)
    assert h.values().sum() == 2.0  # the older mass-inclusive _tf region is used as-is


def _era_dir(tmp_path, era, year="2022", sub=DIR):
    return tmp_path / "rootfiles" / "Run3" / year / era / sub


def _flat(value, nbins=16, hi=1600.0, start=800.0):
    """Histogram with *value* (and variance *value*) in every bin from *start* on."""
    h = Hist(Regular(nbins, 0, hi, name="m"), storage=Weight())
    centers = h.axes[0].centers
    h.view().value[...] = np.where(centers > start, value, 0.0)
    h.view().variance[...] = np.where(centers > start, value, 0.0)
    return h


@pytest.fixture
def year_2022(tmp_path):
    """2022 = Run3Summer22 + Run3Summer22EE, tt_tW only (no DYJets file).

    Run3Summer22:   ee = 1, mumu = 4, emu = 2 per bin  -> ratios 0.5, 2.0
    Run3Summer22EE: ee = 3, mumu = 2, emu = 2 per bin  -> ratios 1.5, 1.0
    2022 (summed):  ee = 4, mumu = 6, emu = 4 per bin  -> ratios 1.0, 1.5
    """
    for era, (ee, mumu, emu) in {"Run3Summer22": (1.0, 4.0, 2.0),
                                 "Run3Summer22EE": (3.0, 2.0, 2.0)}.items():
        _write(_era_dir(tmp_path, era), {_key(r): _flat(v) for r, v in zip(RESOLVED, (ee, mumu, emu))})
    return tmp_path


def test_combined_era_reads_every_era_file(tf, year_2022):
    (inputs,) = make_inputs("2022", DIR, None, working_dir=year_2022, with_dy=False)
    assert inputs.files_for("tt_tW") == [root_file(_era_dir(year_2022, e), "tt_tW")
                                         for e in ("Run3Summer22", "Run3Summer22EE")]
    hists = tf["load_hists"](inputs, "tt_tW", RESOLVED, VAR, 1)
    above = slice(8, None)  # bins above 800 GeV
    np.testing.assert_array_equal(hists["wr_ee_resolved_sr"].values()[above], 4.0)
    np.testing.assert_array_equal(hists["wr_mumu_resolved_sr"].values()[above], 6.0)
    np.testing.assert_array_equal(hists["wr_resolved_flavor_cr"].values()[above], 4.0)
    np.testing.assert_array_equal(hists["wr_resolved_flavor_cr"].variances()[above], 4.0)


def test_combined_era_fit_and_table(tf, year_2022, tmp_path):
    (inputs,) = make_inputs("2022", DIR, None, working_dir=year_2022, with_dy=False)
    args = argparse.Namespace(tf_study=False, variable_bins=None, rebin=1, output=None)
    tables = tf["process_topology"](args, inputs, "tt_tW", False,
                                    tmp_path / "plots", tmp_path / "tables")
    assert [p.name for p in tables] == ["emu_transfer_factor_resolved.tex"]
    row = next(line for line in tables[0].read_text().splitlines() if line.strip().startswith("Resolved"))
    # The fit sees the summed eras: 4/4 and 6/4, not either era alone.
    assert "$1.000 \\pm" in row and "$1.500 \\pm" in row
    assert (tmp_path / "plots" / "emu_transfer_factor_resolved_2022.pdf").is_file()


def test_combination_skipped_when_an_era_file_lacks_a_region(tf, tmp_path):
    full = {_key(r): _flat(1.0) for r in RESOLVED}
    _write(_era_dir(tmp_path, "Run3Summer22"), full)
    _write(_era_dir(tmp_path, "Run3Summer22EE"),
           {k: h for k, h in full.items() if "mumu" not in k})
    (inputs,) = make_inputs("2022", DIR, None, working_dir=tmp_path, with_dy=False)
    assert tf["load_hists"](inputs, "tt_tW", RESOLVED, VAR, 1) is None


def test_study_legacy_form_chosen_per_era_file(tf, tmp_path):
    region = "wr_ee_resolved_sr"
    _write(_era_dir(tmp_path, "Run3Summer22"), {_key(f"{region}_nomass"): _mass_hist([300.0, 900.0])})
    _write(_era_dir(tmp_path, "Run3Summer22EE"), {
        _key(f"{region}_tf_low"): _mass_hist([500.0]),
        _key(region): _mass_hist([1500.0, 2500.0]),
    })
    (inputs,) = make_inputs("2022", DIR, None, working_dir=tmp_path, with_dy=False)
    hists = tf["load_hists"](inputs, "tt_tW", [region], VAR, 1, study=True)
    assert hists[region].values().sum() == 5.0


# ── Sample choice ───────────────────────────────────────────────────────────────

def test_input_sample(tf, tmp_path):
    for era in ("Run3Summer22", "Run3Summer22EE"):
        _write(_era_dir(tmp_path, era), {}, sample="tt_tW")
    _write(_era_dir(tmp_path, "Run3Summer22"), {}, sample="TTbar")
    (inputs,) = make_inputs("2022", DIR, None, working_dir=tmp_path, with_dy=False)
    pick = tf["input_sample"]
    assert pick(inputs) == "tt_tW"
    assert pick(inputs, "WRAnalyzer_TTbar.root", tf_study=True) == "TTbar"
    # TTbar is preferred in study mode only when every era has it.
    assert pick(inputs, tf_study=True) == "tt_tW"
    _write(_era_dir(tmp_path, "Run3Summer22EE"), {}, sample="TTbar")
    assert pick(inputs, tf_study=True) == "TTbar"
    assert pick(inputs, tf_study=False) == "tt_tW"


@pytest.mark.parametrize("value", ["foo.root", "WRAnalyzer_TTbar", "sub/WRAnalyzer_TTbar.root"])
def test_input_must_be_an_analyzer_file_name(tf, value):
    with pytest.raises(SystemExit):
        tf["parse_args"](["--era", "RunIII2024Summer24", "--input", value])
    args = tf["parse_args"](["--era", "Run3", "--input", "WRAnalyzer_TTbar.root"])
    assert args.input_file == "WRAnalyzer_TTbar.root"
