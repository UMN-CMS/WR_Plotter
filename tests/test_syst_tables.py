"""Tests for wrplotter.syst_tables: yield and per-bin changes per process, the total, the
envelope, sources a process lacks, and the output formats."""
import json

import numpy as np
import pytest
import uproot
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

from wrplotter import systematics as systematics_module
from wrplotter.inputs import clear_manifest_cache, make_inputs, manifest_path, root_file
from wrplotter.regions import Region
from wrplotter.sample_groups import SampleGroup
from wrplotter.stack import Binning, hist_key
from wrplotter.syst_tables import (TOTAL_KEY, build_table, csv_rows, format_csv, format_latex,
                                   format_markdown, format_text, per_bin_text, source_label)
from wrplotter.systematics import variation_key

ERA = "RunIII2024Summer24"
DIR = "syst_table_test"
VAR = "mass_fourobject"
SR = Region("wr_ee_resolved_sr", "egamma", "ee SR")
GROUPS = [
    SampleGroup("dy", "#5790fc", "DY", ["DYJets"], stack_position="top"),
    SampleGroup("ttbar", "#f89c20", "ttbar", ["tt_tW"]),
    SampleGroup("egamma", "#000000", "Data", ["EGamma"], kind="data"),
]
SYSTS = ["lumi", "muonidsf", "jes", "scale", "dyzptew1", "pdf"]


@pytest.fixture(autouse=True)
def _fresh():
    clear_manifest_cache()
    systematics_module.clear_cache()
    yield
    clear_manifest_cache()
    systematics_module.clear_cache()


def _h(values):
    h = Hist(Regular(len(values), 0, 4, name=VAR), storage=Weight())
    h.view().value[...] = values
    h.view().variance[...] = values
    return h


def _write(directory, sample, nominal, variations=None, datasets=()):
    """nominal: per-bin values; variations: {(source, direction): per-bin values}."""
    directory.mkdir(parents=True, exist_ok=True)
    path = root_file(directory, sample)
    with uproot.recreate(path) as f:
        f[hist_key(SR.name, VAR)] = _h(nominal)
        for (source, direction), values in (variations or {}).items():
            f[variation_key(source, direction, SR.name, VAR)] = _h(values)
    manifest_path(path).write_text(json.dumps({"datasets": {d: {"physics_group": sample} for d in datasets}}))


@pytest.fixture
def inputs(tmp_path):
    base = tmp_path / "rootfiles" / "Run3" / "2024" / ERA / DIR
    # DY: 10 per bin; jes +-10 %, scale envelope of mur/muf, its own Z-pT EW nuisance
    _write(base, "DYJets", [10, 10, 10, 10], {
        ("jes", "up"): [11, 11, 11, 11], ("jes", "down"): [9, 9, 9, 9],
        ("scalemur", "up"): [12, 10, 10, 10], ("scalemur", "down"): [8, 10, 10, 10],
        ("scalemuf", "up"): [10, 10, 10, 13], ("scalemuf", "down"): [10, 10, 10, 7],
        ("dyzptew1", "up"): [10, 10, 10, 12], ("dyzptew1", "down"): [10, 10, 10, 8],
    }, datasets=["DYto2L-4Jets_Bin-HT-100to400"])
    # ttbar: 20 per bin; jes +5 % / -5 %; a muon SF variation equal to the nominal (the analyzer
    # produced the source for other regions: it does not affect this ee region)
    _write(base, "tt_tW", [20, 20, 20, 20], {("jes", "up"): [21, 21, 21, 21], ("jes", "down"): [19, 19, 19, 19],
                                             ("muonidsf", "up"): [20, 20, 20, 20], ("muonidsf", "down"): [20, 20, 20, 20]})
    # a signal point with a lumi variation only
    _write(base, "signal_WR4000_N2100", [1, 2, 3, 4], {("lumi", "up"): [1.016, 2.032, 3.048, 4.064],
                                                     ("lumi", "down"): [0.984, 1.968, 2.952, 3.936]})
    _write(base, "EGamma", [30, 30, 30, 30])
    (inp,) = make_inputs(ERA, DIR, None, working_dir=tmp_path)
    return inp


def _table(inputs, signals=("signal_WR4000_N2100",)):
    return build_table(SR, VAR, inputs=inputs, groups=GROUPS, binning=Binning(rebin=1),
                       systematics=SYSTS, signals=signals)


def test_yield_changes_per_process_and_total(inputs):
    table = _table(inputs)
    by_key = {p.key: p for p in table.processes}
    assert [p.key for p in table.processes] == ["dy", "ttbar", TOTAL_KEY, "signal_WR4000_N2100"]
    # pdf: no members anywhere; muonidsf: equal to the nominal everywhere (no effect in this region)
    assert table.sources == ["lumi", "jes", "scale", "dyzptew1"]
    assert by_key["ttbar"].effective("muonidsf") is None and "muonidsf" in by_key["ttbar"].variations
    np.testing.assert_allclose(by_key["dy"].yield_change("jes"), (10.0, -10.0))
    np.testing.assert_allclose(by_key["ttbar"].yield_change("jes"), (5.0, -5.0))
    # total: DY 40 -> 44 / 36, ttbar 80 -> 84 / 76 => 120 -> 128 / 112
    np.testing.assert_allclose(by_key[TOTAL_KEY].yield_change("jes"), (100 * 8 / 120, -100 * 8 / 120))
    # the envelope of mur/muf per bin: up = [12, 10, 10, 13], down = [8, 10, 10, 7]
    np.testing.assert_allclose(by_key["dy"].yield_change("scale"), (12.5, -12.5))
    # ttbar has no scale variation: its nominal stands in, the total moves by DY alone
    assert by_key["ttbar"].yield_change("scale") is None
    np.testing.assert_allclose(by_key[TOTAL_KEY].yield_change("scale"), (100 * 5 / 120, -100 * 5 / 120))
    np.testing.assert_allclose(by_key["signal_WR4000_N2100"].yield_change("lumi"), (1.6, -1.6), rtol=1e-6)


def test_max_bin_change_and_missing(inputs):
    table = _table(inputs)
    dy = next(p for p in table.processes if p.key == "dy")
    np.testing.assert_allclose(dy.max_bin_change("scale"), 30.0)         # last bin, muf down 10 -> 7
    np.testing.assert_allclose(dy.max_bin_change("dyzptew1"), 20.0)
    assert dy.max_bin_change("lumi") is None
    assert table.missing(dy) == ["lumi"]
    tt = next(p for p in table.processes if p.key == "ttbar")
    assert table.missing(tt) == ["lumi", "scale", "dyzptew1"]


def test_rebinning_applies_before_the_changes(inputs):
    table = build_table(SR, VAR, inputs=inputs, groups=GROUPS, binning=Binning(rebin=2),
                        systematics=SYSTS)
    dy = next(p for p in table.processes if p.key == "dy")
    assert dy.nominal.axes[0].size == 2
    np.testing.assert_allclose(dy.max_bin_change("dyzptew1"), 10.0)     # [10+10 -> 10+12] = +10 %


def test_no_mc_gives_none(tmp_path):
    base = tmp_path / "rootfiles" / "Run3" / "2024" / ERA / DIR
    _write(base, "DYJets", [1, 1, 1, 1], datasets=["DYto2L-4Jets_Bin-HT-100to400"])
    (inp,) = make_inputs(ERA, DIR, None, working_dir=tmp_path)
    other = Region("wr_mumu_resolved_sr", "muon", "mumu SR")
    assert build_table(other, VAR, inputs=inp, groups=GROUPS, binning=Binning(rebin=1), systematics=SYSTS) is None


def test_formats(inputs):
    table = _table(inputs)
    text = format_text(table)
    assert source_label("jes") in text and "+10.0 / -10.0" in text and "Total bkg." in text
    assert "does not affect" in text                      # the legend for the dashes
    # scale moves DY and the total only: dashes for ttbar and the signal
    assert [p.yield_change("scale") is None for p in table.processes] == [False, True, False, True]
    scale_row = next(l for l in text.splitlines() if l.startswith(source_label("scale")))
    assert scale_row.count(" - ") + scale_row.count(" -\n") + scale_row.endswith(" -") >= 1 and "+12.5 / -12.5" in scale_row
    md = format_markdown(table)
    assert md.startswith("### wr_ee_resolved_sr / mass_fourobject") and "| JES |" in md
    tex = format_latex(table, era=ERA, lumi=109.7, year="2024")
    assert r"\topcaption" in tex and r"$+10.0\,/\,-10.0$" in tex and "(4000, 2100)" in tex
    assert "wr\\_ee\\_resolved\\_sr" in tex
    rows = csv_rows(table, ERA)
    assert len(rows) == len(table.processes) * len(table.sources)
    dy_jes = next(r for r in rows if r["process"] == "dy" and r["source"] == "jes")
    assert dy_jes["up_pct"] == "10.000" and dy_jes["nominal"] == "40.0000"
    csv_text = format_csv(rows)
    assert csv_text.splitlines()[0] == "era,region,variable,process,source,up_pct,down_pct,max_bin_pct,nominal"
    per_bin = per_bin_text(table, next(p for p in table.processes if p.key == "dy"))
    assert "0-1" in per_bin and "+20.0/-20.0" in per_bin
