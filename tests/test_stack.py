"""Tests for wrplotter.stack: data stream per region, signal-region data never
read from blinded or legacy files, and the stack order."""
import json

import numpy as np
import pytest
import uproot
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight

import wrplotter.stack as stack_module
from wrplotter import systematics as systematics_module
from wrplotter.inputs import clear_manifest_cache, make_inputs, manifest_path, root_file
from wrplotter.regions import Region
from wrplotter.sample_groups import SampleGroup
from wrplotter.stack import Binning, build_stack, hist_key, order_for_stack
from wrplotter.systematics import variation_key

ERA = "RunIII2024Summer24"
DIR = "stack_test"
VAR = "mass_fourobject"
CR = Region("wr_resolved_flavor_cr", "muon", "flavor CR")
SR = Region("wr_ee_resolved_sr", "egamma", "ee SR")
GROUPS = [
    SampleGroup("ttbar", "#f89c20", "ttbar", ["tt_tW"]),
    SampleGroup("dy", "#5790fc", "DY", ["DYJets"], stack_position="top"),
    SampleGroup("egamma", "#000000", "Data", ["EGamma"], kind="data"),
    SampleGroup("muon", "#000000", "Data", ["Muon"], kind="data"),
]


@pytest.fixture(autouse=True)
def _fresh_manifests():
    clear_manifest_cache()
    yield
    clear_manifest_cache()
    stack_module._REPORTED.clear()
    systematics_module.clear_cache()


def _h(value):
    h = Hist(Regular(4, 0, 100, name=VAR), storage=Weight())
    h.view().value[...] = value
    h.view().variance[...] = value
    return h


def _write(directory, sample, contents, *, blinding="absent", datasets=()):
    """contents: {region name: per-bin value}; blinding: "absent" (legacy) or a dict."""
    directory.mkdir(parents=True, exist_ok=True)
    path = root_file(directory, sample)
    with uproot.recreate(path) as f:
        for region, value in contents.items():
            f[hist_key(region, VAR)] = _h(value)
    manifest = {"datasets": {d: {"physics_group": sample} for d in datasets}}
    if blinding != "absent":
        manifest["blinding"] = blinding
    manifest_path(path).write_text(json.dumps(manifest))


def _inputs(tmp_path, *, data_blinding):
    base = tmp_path / "rootfiles" / "Run3" / "2024" / ERA / DIR
    regions = {CR.name: 1.0, SR.name: 2.0}
    _write(base, "DYJets", regions, datasets=["DYto2L-4Jets_Bin-HT-100to400"])
    _write(base, "tt_tW", {CR.name: 10.0, SR.name: 20.0})
    # Both data files fill every region (no overlap removal), with different contents.
    _write(base, "EGamma", {CR.name: 100.0, SR.name: 200.0}, blinding=data_blinding)
    _write(base, "Muon", {CR.name: 300.0, SR.name: 400.0}, blinding=data_blinding)
    (inputs,) = make_inputs(ERA, DIR, None, working_dir=tmp_path)
    return inputs


def _build(inputs, region):
    return build_stack(region, VAR, inputs=inputs, groups=GROUPS,
                       binning=Binning(rebin=1, density=False))


def test_cr_data_from_the_region_stream_only(tmp_path):
    stack = _build(_inputs(tmp_path, data_blinding={"applied": True}), CR)
    assert not stack.data_blinded
    np.testing.assert_array_equal(stack.data.values(), 300.0)   # Muon, not EGamma
    assert [g.key for g, _ in stack.mc] == ["dy", "ttbar"]       # flavor CR: DY at the bottom


@pytest.mark.parametrize("blinding", ["absent", {"applied": True}],
                         ids=["legacy_file", "blinded_file"])
def test_sr_data_never_read_unless_unblinded(tmp_path, blinding):
    # A legacy (pre-blinding) data file still holds SR histograms: they must not be read.
    stack = _build(_inputs(tmp_path, data_blinding=blinding), SR)
    assert stack.data is None
    assert stack.data_blinded
    np.testing.assert_array_equal(stack.total.values(), 22.0)


def test_sr_data_read_from_unblinded_file(tmp_path):
    stack = _build(_inputs(tmp_path, data_blinding={"applied": False}), SR)
    assert not stack.data_blinded
    np.testing.assert_array_equal(stack.data.values(), 200.0)   # EGamma for ee


def test_order_for_stack():
    dy, tt = GROUPS[1], GROUPS[0]
    mc = [(tt, _h(1.0)), (dy, _h(1.0))]
    assert [g.key for g, _ in order_for_stack(mc, SR)] == ["ttbar", "dy"]   # DY drawn on top
    assert [g.key for g, _ in order_for_stack(mc, CR)] == ["dy", "ttbar"]   # flavor CR: bottom


# ---- systematic variations, era combinations, samples left out ----

def _write_varied(directory, sample, nominal, jes=None):
    """CR nominal of *sample*, plus jes up/down CR variations when *jes* = (up, down) is given."""
    directory.mkdir(parents=True, exist_ok=True)
    path = root_file(directory, sample)
    with uproot.recreate(path) as f:
        f[hist_key(CR.name, VAR)] = _h(nominal)
        if jes is not None:
            for direction, value in zip(("up", "down"), jes):
                f[variation_key("jes", direction, CR.name, VAR)] = _h(value)
    datasets = ["DYto2L-4Jets_Bin-HT-100to400"] if sample == "DYJets" else []
    manifest_path(path).write_text(json.dumps(
        {"datasets": {d: {"physics_group": sample} for d in datasets}}))


def _build_syst(inputs, groups=GROUPS):
    return build_stack(CR, VAR, inputs=inputs, groups=groups,
                       binning=Binning(rebin=1, density=False), systematics=["jes"])


def test_combination_era_without_variations_adds_its_nominal(tmp_path, caplog):
    # "2022" = Run3Summer22 (with jes) + Run3Summer22EE (nominal only).  The band used to
    # count the second era's missing variations as zero: a shift as large as its yield.
    root = tmp_path / "rootfiles" / "Run3" / "2022"
    _write_varied(root / "Run3Summer22" / DIR, "DYJets", 1.0, jes=(1.5, 0.5))
    _write_varied(root / "Run3Summer22" / DIR, "tt_tW", 10.0, jes=(11.0, 9.0))
    _write_varied(root / "Run3Summer22EE" / DIR, "DYJets", 2.0)
    _write_varied(root / "Run3Summer22EE" / DIR, "tt_tW", 20.0)
    (inputs,) = make_inputs("2022", DIR, None, working_dir=tmp_path)
    with caplog.at_level("WARNING"):
        stack = _build_syst(inputs)
    np.testing.assert_array_equal(stack.total.values(), 33.0)
    np.testing.assert_array_equal(stack.syst["jes"]["up"].values(), 34.5)     # 1.5 + 11 + 2 + 20
    np.testing.assert_array_equal(stack.syst["jes"]["down"].values(), 31.5)   # 0.5 + 9 + 2 + 20
    assert "Run3Summer22EE" in caplog.text and "has no systematic variations" in caplog.text


def test_sample_without_variations_is_reported(tmp_path, caplog):
    base = tmp_path / "rootfiles" / "Run3" / "2024" / ERA / DIR
    _write_varied(base, "DYJets", 1.0)                       # nominal-only DY
    _write_varied(base, "tt_tW", 10.0, jes=(11.0, 9.0))
    (inputs,) = make_inputs(ERA, DIR, None, working_dir=tmp_path)
    with caplog.at_level("WARNING"):
        stack = _build_syst(inputs)
        _build_syst(inputs)
    np.testing.assert_array_equal(stack.syst["jes"]["up"].values(), 12.0)
    assert caplog.text.count("DYJets has no systematic variations") == 1


def test_sample_without_file_is_reported(tmp_path, caplog):
    base = tmp_path / "rootfiles" / "Run3" / "2024" / ERA / DIR
    _write_varied(base, "DYJets", 1.0)
    _write_varied(base, "tt_tW", 10.0)
    (inputs,) = make_inputs(ERA, DIR, None, working_dir=tmp_path)
    other = SampleGroup("other", "#964a8b", "Other", ["Other"])
    with caplog.at_level("WARNING"):
        stack = _build_syst(inputs, groups=GROUPS + [other])
    assert [g.key for g, _ in stack.mc] == ["dy", "ttbar"]
    assert "No Other file in" in caplog.text


def test_loader_reads_each_dy_variant_file(tmp_path):
    """Every DY variant sits in the production's --dir as WRAnalyzer_DYJets_<tag>.root next to the default
    WRAnalyzer_DYJets.root (since 2026-10-12); the loader must read the variant's own file."""
    from wrplotter.inputs import DYSample, Inputs
    base = tmp_path / "rootfiles" / "Run3" / "2024" / ERA / DIR
    _write(base, "DYJets", {CR.name: 1.0})
    _write(base, "DYJets_nlo_inc", {CR.name: 7.0})
    for tag, name, value in (("lo_ht", "WRAnalyzer_DYJets.root", 1.0), ("nlo_inc", "WRAnalyzer_DYJets_nlo_inc.root", 7.0)):
        inputs = Inputs(era=ERA, dir_name=DIR, dirs=(base,), dy=DYSample(tag, tag, "black", (base / name,)))
        h = stack_module.HistLoader(inputs, Binning(rebin=1, density=False))("DYJets", hist_key(CR.name, VAR))
        assert h.values().tolist() == [value] * 4, tag
