"""Tests for bin/make_signal_cutflows.py: mass points found from the files, the
column layout of each mode, one cut sequence per table, era sums and output."""
import functools
import runpy
import sys
from pathlib import Path

import hist
import numpy as np
import pytest
import uproot

from wrplotter.cutflow import cutflow_key, latex_count
from wrplotter.histogram_cache import clear_cache
from wrplotter.inputs import make_inputs, mass_point, root_file

ERA = "RunIII2024Summer24"
DIR = "signal_cutflow_test"
CUTS = {"ee": ["no_cuts", "two_pteta_electrons", "e_trigger", "mll_gt400"],
        "mumu": ["no_cuts", "two_pteta_muons", "mu_trigger", "mll_gt400"]}
# In mass order: WR10000 comes last (numeric, not lexical, sorting).
POINTS = ["WR2000_N100", "WR2000_N1100", "WR4000_N100", "WR4000_N2100", "WR10000_N100"]

# Tables of each mode for POINTS: {table: column mass points}.
LAYOUTS = {
    "fixed-wr": {"WR2000": ["WR2000_N100", "WR2000_N1100"], "WR4000": ["WR4000_N100", "WR4000_N2100"],
                 "WR10000": ["WR10000_N100"]},
    "fixed-n": {"N100": ["WR2000_N100", "WR4000_N100", "WR10000_N100"], "N1100": ["WR2000_N1100"],
                "N2100": ["WR4000_N2100"]},
    "wr-compare": {"wr_compare": ["WR2000_N1100", "WR4000_N2100", "WR10000_N100"]},
}
SWAPPED = ["no_cuts", "e_trigger", "two_pteta_electrons", "mll_gt400"]  # same length, other order
HEADERS = {"fixed-wr": lambda wr, n: str(n), "fixed-n": lambda wr, n: str(wr),
           "wr-compare": lambda wr, n: f"({wr},{n})"}


@pytest.fixture(scope="module")
def script():
    return runpy.run_path(str(Path(__file__).resolve().parent.parent / "bin" / "make_signal_cutflows.py"))


def _era_dir(base, era=ERA, year="2024"):
    return base / "rootfiles" / "Run3" / year / era / DIR


def _write(directory, point, cuts=CUTS):
    """Resolved cutflows of one signal point, {channel: cuts}; rows WR, WR/2, ..., N (x2 for mumu)."""
    wr, n = mass_point(point)
    directory.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(root_file(directory, f"signal_{point}")) as f:
        for channel, scale in (("ee", 1.0), ("mumu", 2.0)):
            names = cuts[channel]
            h = hist.Hist(hist.axis.StrCategory(names, name="cut"), storage=hist.storage.Weight())
            h.view().value[...] = scale * np.array([wr / 2 ** i for i in range(len(names) - 1)] + [n])
            h.view().variance[...] = h.view().value
            for kind in ("cumulative", "onecut"):
                f[cutflow_key("resolved", channel, kind)] = h


@pytest.fixture
def signal_set(tmp_path):
    """A signal-only analysis set (no DYJets file), plus files that are not mass points."""
    base = _era_dir(tmp_path)
    for point in POINTS:
        _write(base, point)
    root_file(base, "signal_private_test").write_bytes(b"")
    (base / "WRAnalyzer_signal_WR2000_N100_systematics.json").write_text("{}")
    return tmp_path


@pytest.fixture
def inputs(signal_set):
    (inp,) = make_inputs(ERA, DIR, with_dy=False, working_dir=signal_set)
    return inp


def _build(script, inputs, mode, channels=("ee",), points=POINTS):
    return script["build_tables"](inputs, points, mode=mode, topology="resolved", kind="cumulative",
                                  channels=channels)


def test_mass_points_from_the_files_in_mass_order(inputs):
    assert inputs.signal_points() == POINTS
    assert mass_point("signal_WR2000_N1100") == (2000, 1100)
    assert mass_point("WR2000_N1100_old") is None


def test_signal_only_set_needs_no_dy_file(signal_set, inputs):
    with pytest.raises(FileNotFoundError, match="DYJets"):
        make_inputs(ERA, DIR, working_dir=signal_set)
    with pytest.raises(ValueError, match="No Drell-Yan sample"):
        inputs.files_for("DYJets")


@pytest.mark.parametrize("mode", list(LAYOUTS))
def test_column_layout(script, inputs, mode):
    tables = _build(script, inputs, mode)
    assert [t.name for t in tables] == [f"signal_{key}_resolved_ee" for key in LAYOUTS[mode]]
    for table, points in zip(tables, LAYOUTS[mode].values()):
        ((channel, columns),) = table.groups
        masses = [mass_point(p) for p in points]
        assert channel == "ee"
        assert [header for header, _ in columns] == [HEADERS[mode](*m) for m in masses]
        # Each column holds its own point's file: first row = WR, last row = N.
        assert [(cf.values[0], cf.values[-1]) for _, cf in columns] == masses
        assert table.row_labels(latex=False) == CUTS["ee"]


def test_layout_without_points(script):
    assert all(script["layout"]([], mode) == [] for mode in script["MODES"])


def test_channel_compare_puts_ee_and_mumu_side_by_side(script, inputs):
    tables = _build(script, inputs, "channel-compare", channels=("ee", "mumu"))
    assert [t.name for t in tables] == [f"signal_{key}_resolved_ee_vs_mumu" for key in LAYOUTS["fixed-wr"]]
    table = tables[0]
    assert table.channels == ["ee", "mumu"]
    assert [h for _, cols in table.groups for h, _ in cols] == ["100", "1100", "100", "1100"]
    np.testing.assert_array_equal(table.groups[1][1][0][1].values, 2 * table.groups[0][1][0][1].values)
    # Rows are matched by position; cuts that differ between the channels show both.
    assert table.row_labels(latex=False) == [
        "no_cuts", "two_pteta_electrons / two_pteta_muons", "e_trigger / mu_trigger", "mll_gt400"]
    latex = script["format_latex"](table, None, era=ERA, lumi=109.7, year="2024")
    assert r"\begin{tabular}{l | c c | c c}" in latex
    assert r"& \multicolumn{2}{c}{ee} & \multicolumn{2}{|c}{$\mu\mu$} \\" in latex
    assert "Electron trigger / Muon trigger" in latex
    assert "dielectron and dimuon channels" in latex


def test_points_with_different_cut_sequences_raise(script, signal_set, inputs):
    _write(_era_dir(signal_set), "WR4000_N2100", cuts={**CUTS, "ee": SWAPPED})
    with pytest.raises(ValueError, match="do not share one cutflow_resolved/ee/cumulative cut sequence") as exc:
        _build(script, inputs, "fixed-wr")
    assert f"WR4000_N2100: {', '.join(SWAPPED)}" in str(exc.value)
    assert f"WR2000_N100, WR2000_N1100, WR4000_N100, WR10000_N100: {', '.join(CUTS['ee'])}" in str(exc.value)
    assert len(_build(script, inputs, "fixed-wr", channels=("mumu",))) == 3  # mumu is consistent


def test_channel_compare_needs_sequences_of_one_length(script, signal_set, inputs):
    for point in POINTS:
        _write(_era_dir(signal_set), point, cuts={**CUTS, "mumu": CUTS["mumu"][:3]})
    with pytest.raises(ValueError, match="different lengths"):
        _build(script, inputs, "channel-compare", channels=("ee", "mumu"))


def test_year_sums_its_eras(script, tmp_path):
    for era in ("Run3Summer22", "Run3Summer22EE"):
        _write(_era_dir(tmp_path, era, "2022"), "WR2000_N100")
    _write(_era_dir(tmp_path, "Run3Summer22EE", "2022"), "WR4000_N100")  # in one era only
    (inp,) = make_inputs("2022", DIR, with_dy=False, working_dir=tmp_path)
    assert inp.signal_points() == ["WR2000_N100", "WR4000_N100"]
    (table,) = _build(script, inp, "fixed-wr", points=["WR2000_N100"])
    ((_, [(_, cf)]),) = table.groups
    np.testing.assert_array_equal(cf.values, 2 * np.array([2000, 1000, 500, 100]))

    _write(_era_dir(tmp_path, "Run3Summer22EE", "2022"), "WR2000_N100", cuts={**CUTS, "ee": SWAPPED})
    clear_cache()  # drop the open handle of the rewritten file
    with pytest.raises(ValueError, match="Run3Summer22EE.*the files before it"):
        _build(script, inp, "fixed-wr", points=["WR2000_N100"])


def test_text_and_latex_formats(script, inputs):
    table = _build(script, inputs, "fixed-wr")[0]
    text = script["format_text"](table, None, ERA).splitlines()
    assert text[0] == "2024 resolved ee, WR = 2000 GeV: yields; columns: N mass [GeV]"   # the era's plot label
    assert text[1].split() == ["Cut", "100", "1100"]
    running = script["format_text"](table, "running", ERA).splitlines()
    assert running[-2].split() == ["e_trigger", "25.0", "25.0"]
    latex = script["format_latex"](table, "relative", era=ERA, lumi=109.7, year="2024")
    assert r"\label{tab:signal_WR2000_resolved_ee_relative_eff_2024}" in latex
    assert "efficiencies (\\%) relative to the previous row" in latex
    assert r"Electron trigger & 50.0 & 50.0 \\" in latex
    # Small signal yields keep their digits.
    assert latex_count(0.004) == r"$4.00 \times 10^{-3}$"
    assert latex_count(0.5) == "0.50"


def test_main_writes_latex_tables_and_preview(script, signal_set, monkeypatch):
    out = signal_set / "out"
    monkeypatch.setitem(script["main"].__globals__, "make_inputs",
                        functools.partial(make_inputs, working_dir=signal_set))
    monkeypatch.setattr(sys, "argv", ["make_signal_cutflows.py", "--era", ERA, "--dir", DIR, "-o", str(out),
                                      "--no-pdf", "--wr-mass", "2000", "4000"])
    script["main"]()
    names = {p.name for p in out.glob("*.tex")} - {"preview.tex"}
    # Without --channel: fixed-wr and wr-compare, ee and mumu, yields and both efficiencies.
    expected = {f"signal_{table}_resolved_{ch}_{variant}.tex"
                for table in ("WR2000", "WR4000", "wr_compare") for ch in ("ee", "mumu")
                for variant in ("cumulative", "running_eff", "relative_eff")}
    assert names == expected
    assert (out / "preview.tex").read_text().count(r"\input{") == len(expected)
    assert "WR10000" not in (out / "signal_cutflows.txt").read_text()
