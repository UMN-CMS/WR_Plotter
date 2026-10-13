"""Tests for wrplotter/signal_plots.py and bin/make_signal_plots.py: the per-point summaries (efficiency,
CR fractions with the right denominator, peak summary), the cross sections from the config, the point
helpers, and the script's tables and plots.  The synthetic signal files come from test_signal_compare."""
import csv
import functools
import json
import runpy
from pathlib import Path

import pytest
import uproot

from tests.test_signal_compare import (  # noqa: F401  (fixtures: analyzer, clear_caches)
    DIR, LUMI, POINTS, XSEC, YIELDS, _era_dir, _gauss, _write, analyzer, clear_caches,
)
from wrplotter import signal_compare as sc
from wrplotter import signal_plots as sp
from wrplotter.inputs import mass_point, root_file

ERA = "RunIII2024Summer24"
CR_YIELD = 0.5      # events in each control region of the synthetic files (both flavours)


def _add_control_regions(base: Path, point: str) -> None:
    """Add the DY and flavour CR mass histograms to a synthetic point file."""
    wr, _ = mass_point(point)
    path = root_file(_era_dir(base, ERA), f"signal_{point}")
    with uproot.update(path) as f:
        for topology, regions in sp.CONTROL_REGIONS.items():
            for pattern in regions:
                for ch in sc.CHANNELS:
                    region = pattern.format(ch=ch)
                    f[f"{region}/{sc.MASS_VARIABLES[topology]}_{region}"] = _gauss(0.5 * wr, 0.1 * wr, CR_YIELD)


@pytest.fixture
def era_set(tmp_path, analyzer):
    for point in POINTS:
        _write(tmp_path, ERA, point)
        _add_control_regions(tmp_path, point)
    return sp.load_set(ERA, DIR, working_dir=tmp_path)


@pytest.fixture(scope="module")
def script():
    return runpy.run_path(str(Path(__file__).resolve().parent.parent / "bin" / "make_signal_plots.py"))


def test_point_helpers():
    pts = ["WR4000_N100", "WR2000_N100", "WR4000_N2100", "WR4000_N3900", "WR2000_N1100"]
    assert sp.points_by_mwr(pts) == {2000: ["WR2000_N100", "WR2000_N1100"],
                                     4000: ["WR4000_N100", "WR4000_N2100", "WR4000_N3900"]}
    assert sp.spread(["a", "b", "c", "d", "e"], 3) == ["a", "c", "e"]
    assert sp.spread(["a", "b"], 3) == ["a", "b"]
    assert sp.nearest_fraction(["WR4000_N100", "WR4000_N2100", "WR4000_N3900"], 0.5) == "WR4000_N2100"
    assert sp.nearest_fraction([], 0.5) is None
    assert sp.point_label("WR4000_N2100") == r"$m_{W_R}$ = 4000, $m_N$ = 2100 GeV"


def test_summary_efficiency_and_cr_fractions(era_set, analyzer):
    s = sp.summarize_point(era_set, "WR4000_N2100", analyzer_root=analyzer)
    produced = XSEC * LUMI[ERA] * 1000
    assert s.m_wr == 4000 and s.m_n == 2100
    assert s.produced.channels["ee"] == pytest.approx(produced)
    assert s.efficiency[("resolved", "ee")].value == pytest.approx(YIELDS[("resolved", "ee")] / produced)
    assert s.efficiency[("combined", "mumu")].value == pytest.approx(
        (YIELDS[("resolved", "mumu")] + YIELDS[("boosted", "mumu")]) / produced)
    # a DY CR is the channel's region: its yield over sigma L of that flavour
    assert s.cr_fraction[("resolved", "wr_{ch}_resolved_dy_cr", "ee")].value == pytest.approx(CR_YIELD / produced)
    # a flavour CR takes both flavours of the file: over sigma L of both
    assert s.cr_fraction[("resolved", "wr_resolved_flavor_cr", "ee")].value == pytest.approx(CR_YIELD / (2 * produced))
    assert s.cr_fraction[("boosted", "wr_emu_boosted_flavor_cr", "mumu")].value == pytest.approx(CR_YIELD / (2 * produced))
    # the peak of the synthetic resolved mass sits at 0.97 m_WR
    assert s.shape[("resolved", "ee")].median.value == pytest.approx(0.97 * 4000, rel=0.01)


def test_region_fraction_without_signal_of_the_flavour(era_set, analyzer):
    produced = sc.Produced({"ee": 100.0, "mumu": 0.0}, 100.0, ())
    assert sp.region_fraction(era_set, "WR4000_N2100", "wr_mumu_resolved_dy_cr", "mass_fourobject", "mumu",
                              produced) is None
    assert sp.region_fraction(era_set, "WR4000_N2100", "no_such_region", "mass_fourobject", "ee", produced) is None


def test_cross_sections_from_the_config(era_set, analyzer, tmp_path):
    config = sc.signal_config_path(ERA, analyzer)
    entries = json.loads(config.read_text())
    for key, entry in entries.items():
        if entry.get("xsec") is not None:
            entry["xsec_source"] = {"generator": entry["xsec"] / 1.3, "factor": 1.0, "kfactor": 1.3}
    config.write_text(json.dumps(entries))
    xs = sp.cross_sections(config, era_set, POINTS)
    assert set(xs) == set(POINTS)
    assert xs["WR4000_N2100"]["ee"]["nlo"] == pytest.approx(XSEC)
    assert xs["WR4000_N2100"]["ee"]["lo"] == pytest.approx(XSEC / 1.3)
    assert xs["WR4000_N2100"]["mumu"]["k"] == pytest.approx(1.3)


def test_efficiency_rows_and_latex(era_set, analyzer):
    summaries = [sp.summarize_point(era_set, p, analyzer_root=analyzer) for p in POINTS]
    rows = sp.efficiency_rows(era_set, summaries)
    assert len(rows) == 2 * len(POINTS)
    row = next(r for r in rows if r["point"] == "WR4000_N2100" and r["channel"] == "ee")
    produced = XSEC * LUMI[ERA] * 1000
    assert float(row["eff_resolved_pct"]) == pytest.approx(100 * YIELDS[("resolved", "ee")] / produced, abs=1e-3)
    assert float(row["cr_wr_resolved_flavor_cr_pct"]) == pytest.approx(100 * CR_YIELD / (2 * produced), abs=1e-3)
    assert set(rows[0]) == set(rows[-1])
    tex = sp.efficiency_latex(era_set, [s for s in summaries if s.m_wr == 4000], 4000, "mumu")
    assert r"\begin{tabular}{r|ccc}" in tex and "2100 &" in tex


def test_main_writes_tables_and_plots(script, era_set, analyzer, tmp_path, monkeypatch):
    monkeypatch.setitem(script["main"].__globals__, "load_set", functools.partial(sp.load_set, working_dir=tmp_path))
    out = tmp_path / "out"
    rc = script["main"](["--era", ERA, "--dir", DIR, "--analyzer-root", str(analyzer), "-o", str(out),
                         "--points", "all", "--no-pdf"])
    assert rc == 0
    with open(out / "tables" / "efficiency.csv") as f:
        rows = list(csv.DictReader(f))
    assert {r["point"] for r in rows} == set(POINTS)
    plots = out / "plots"
    for name in ("efficiency/ee_WR4000.pdf", "efficiency/mumu_WR2000.pdf", "efficiency_map/combined_ee.pdf",
                 "cr_fraction/resolved_ee_WR4000.pdf", "cr_fraction/boosted_mumu_WR2000.pdf",
                 "resolution/boosted_mumu_WR4000.pdf", "xsec/ee_WR4000.pdf", "kfactor/mumu_WR2000.pdf",
                 "mass/resolved_ee_WR4000.pdf", "mass/boosted_mumu_mN_0.5.pdf",
                 "kinematics/wr_ee_resolved_sr/pt_leading_lepton.pdf", "cutflow/resolved_ee.pdf",
                 "cutflow/boosted_mumu_WR4000.pdf"):
        assert (plots / name).is_file(), name
    assert (out / "tables" / "latex" / "signal_eff_ee_WR4000.tex").is_file()
    assert (out / "tables" / "latex" / "preview.tex").is_file()


def test_main_without_signal_files(script, tmp_path, monkeypatch):
    _era_dir(tmp_path, ERA).mkdir(parents=True)
    monkeypatch.setitem(script["main"].__globals__, "load_set", functools.partial(sp.load_set, working_dir=tmp_path))
    assert script["main"](["--era", ERA, "--dir", DIR, "-o", str(tmp_path / "out")]) == 2


def test_signal_output_dirs(tmp_path, monkeypatch):
    """Every signal output of a production goes under <dir>/signal_outputs/<part>, for plots and tables alike."""
    monkeypatch.setattr(sp, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(sp, "plots_base",
                        lambda run, year, era, subdir, local=False: tmp_path / "plots" / run / year / era / (subdir or ""))
    plots, tables = sp.signal_output_dirs(ERA, "prod", "event_level")
    assert plots == tmp_path / "plots" / "Run3" / "2024" / ERA / "prod" / "signal_outputs" / "event_level"
    assert tables == tmp_path / "tables" / "Run3" / "2024" / ERA / "prod" / "signal_outputs" / "event_level"
