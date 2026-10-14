"""Tests for wrplotter/signal_compare.py and bin/compare_signal_eras.py: flavour shares, sigma x L
from the manifests and the analyzer's configs, efficiencies summed over a year's eras, quantiles and
sigma_eff, and the script's tables and plots."""
import csv
import functools
import json
import math
import runpy
import sys
from pathlib import Path

import hist
import numpy as np
import pytest
import uproot

from wrplotter import signal_compare as sc
from wrplotter.cutflow import cutflow_key
from wrplotter.inputs import clear_manifest_cache, mass_point, root_file

DIR = "signal_compare_test"
EL = "WRtoNEltoElElJJ_MWR{}_N{}_TuneCP5_13p6TeV_madgraph-pythia8"
MU = "WRtoNMutoMuMuJJ_MWR{}_N{}_TuneCP5_13p6TeV_madgraph-pythia8"
XSEC = 0.002                              # pb per flavour
WEIGHT = 0.01                             # per-event weight of the synthetic histograms
LUMI = {"Run3Summer22": 8.08, "Run3Summer22EE": 26.64, "RunIII2024Summer24": 109.70}  # data/lumi.yaml
YEARS = {"Run3Summer22": "2022", "Run3Summer22EE": "2022", "RunIII2024Summer24": "2024"}
POINTS = ["WR2000_N1100", "WR4000_N2100"]
CUTS = {"ee": ["no_cuts", "two_tight_electrons", "mll_gt400"],
        "mumu": ["no_cuts", "two_tight_muons", "mll_gt400"]}
# SR yields per (topology, channel) of one file; the mass peak sits at peak x m_WR.
YIELDS = {("resolved", "ee"): 4.0, ("resolved", "mumu"): 5.0, ("boosted", "ee"): 1.0, ("boosted", "mumu"): 0.5}
# Skimmed fileset counts per dataset and the unweighted no_cuts row per file (two datasets per file):
# skim = 1000 / 1250 = 0.8, filters = 1800 / (2 x 1000) = 0.9.
NEVTS, GENERATED, NO_CUTS_RAW = 1000, 1250, 1800.0


@pytest.fixture(autouse=True)
def clear_caches():
    yield
    sc.clear_config_cache()
    clear_manifest_cache()


@pytest.fixture(scope="module")
def script():
    return runpy.run_path(str(Path(__file__).resolve().parent.parent / "bin" / "compare_signal_eras.py"))


def _era_dir(base: Path, era: str, dir_name: str = DIR) -> Path:
    return base / "rootfiles" / "Run3" / YEARS[era] / era / dir_name


def _gauss(peak: float, width: float, total: float) -> hist.Hist:
    """Weighted Gaussian peak in the analyzer's 10 GeV mass binning (0-8000), *total* events."""
    h = hist.Hist(hist.axis.Regular(800, 0, 8000, name="x"), storage=hist.storage.Weight())
    centers = h.axes[0].centers
    shape = np.exp(-0.5 * ((centers - peak) / width) ** 2)
    values = total * shape / shape.sum()
    h.view()["value"] = values
    h.view()["variance"] = values * WEIGHT
    return h


def _cutflow(names, values) -> hist.Hist:
    h = hist.Hist(hist.axis.StrCategory(names, name="cut"), storage=hist.storage.Weight())
    h.view().value[...] = values
    h.view().variance[...] = np.asarray(values) * WEIGHT
    return h


def _factor(value: float, events: int = 100) -> hist.Hist:
    """An event_weights component histogram: *events* events with this factor value."""
    h = hist.Hist(hist.axis.Regular(600, 0.0, 3.0, name="factor"), storage=hist.storage.Weight())
    h.fill(factor=np.full(events, value))
    return h


def _write(base: Path, era: str, point: str, *, scale: float = 1.0, peak: float = 0.97,
           datasets: list[str] | None = None, manifest: bool = True, dir_name: str = DIR,
           no_cuts: float = 20.0, sf: float = 0.951) -> None:
    """One signal file: SR mass + one kinematic variable per region, cutflows (row 0, *no_cuts*,
    shared by ee and mumu), weight components (one scale factor *sf*), manifest."""
    wr, n = mass_point(point)
    directory = _era_dir(base, era, dir_name)
    directory.mkdir(parents=True, exist_ok=True)
    path = root_file(directory, f"signal_{point}")
    with uproot.recreate(path) as f:
        for (topology, channel), y in YIELDS.items():
            region = sc.sr_region(topology, channel)
            f[f"{region}/{sc.MASS_VARIABLES[topology]}_{region}"] = _gauss(peak * wr, 0.05 * wr, scale * y)
            f[f"{region}/pt_leading_lepton_{region}"] = _gauss(0.4 * wr, 0.1 * wr, scale * y)
            f[cutflow_key(topology, channel, "cumulative")] = _cutflow(
                CUTS[channel], [scale * no_cuts, scale * 2 * y, scale * y])
            f[cutflow_key(topology, channel, "cumulative", unweighted=True)] = _cutflow(
                CUTS[channel], [NO_CUTS_RAW, 100.0, 50.0])
            for name, value in (("LeptonSf", sf), ("pileup", 0.971), ("corrections", sf), ("dy_zpt", 1.0)):
                f[f"event_weights/{region}/components/{name}"] = _factor(value)   # dy_zpt: the identity
            f[f"event_weights/{region}/components/normalization"] = _factor(2.0)   # not a factor mean
    if manifest:
        names = datasets if datasets is not None else [EL.format(wr, n), MU.format(wr, n)]
        path.with_name(path.stem + "_systematics.json").write_text(
            json.dumps({"datasets": {d: {"datatype": "mc"} for d in names}}))


@pytest.fixture
def analyzer(tmp_path) -> Path:
    """Analyzer configs with XSEC per flavour for POINTS (DAS-path keys, as in the real configs)."""
    root = tmp_path / "analyzer"
    for era, year in YEARS.items():
        entries = {}
        for point in POINTS + ["WR2000_N100"]:
            wr, n = mass_point(point)
            for name in (EL.format(wr, n), MU.format(wr, n)):
                entries[f"/{name}/{era}NanoAODv12/NANOAODSIM"] = {"dataset": name, "xsec": XSEC}
        entries["WRtoNLtoLLJJ_MWR2000_MN1100"] = {"dataset": "WRtoNLtoLLJJ_MWR2000_MN1100", "xsec": 2 * XSEC}
        entries[f"/{EL.format(2000, 100)}/{era}NanoAODv12/NANOAODSIM"]["xsec"] = None   # no cross section yet
        path = root / "data" / "configs" / "Run3" / year / era / f"{era}_signal.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(entries))
        fileset = {e["dataset"]: {"files": {}, "metadata": {"sample": e["dataset"], "nevts": NEVTS,
                                                             "genEventCount": GENERATED}}
                   for e in entries.values()}
        path = sc.signal_fileset_path(era, root)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(fileset))
    return root


@pytest.fixture
def sets(tmp_path, analyzer):
    """2024 and 2022 (= Run3Summer22 + Run3Summer22EE) with POINTS; 2022 has twice 2024's yield per fb^-1."""
    for point in POINTS:
        _write(tmp_path, "RunIII2024Summer24", point)
        _write(tmp_path, "Run3Summer22", point, scale=2 * LUMI["Run3Summer22"] / LUMI["RunIII2024Summer24"],
               peak=0.95)
        _write(tmp_path, "Run3Summer22EE", point, scale=2 * LUMI["Run3Summer22EE"] / LUMI["RunIII2024Summer24"],
               peak=0.95)
    return sc.make_era_sets([("RunIII2024Summer24", DIR), ("2022", DIR)], working_dir=tmp_path)


def test_flavour_shares():
    assert sc.flavour_shares(EL.format(2000, 100)) == {"ee": 1.0, "mumu": 0.0}
    assert sc.flavour_shares("/" + MU.format(2000, 100) + "/Run3Summer22NanoAODv12/NANOAODSIM")["mumu"] == 1.0
    assert sc.flavour_shares("WRtoNLtoLLJJ_MWR2000_MN1100") == {"ee": 0.5, "mumu": 0.5}
    with pytest.raises(ValueError, match="Unknown signal process"):
        sc.flavour_shares("DYto2E_MLL-50")


def test_value_ratio():
    r = sc.Value(2.0, 0.2) / sc.Value(4.0, 0.4)
    assert r.value == pytest.approx(0.5)
    assert r.error == pytest.approx(0.5 * math.hypot(0.1, 0.1))
    assert math.isnan((sc.Value(1.0, 0.1) / sc.Value(0.0, 0.0)).value)


def test_quantiles_of_a_flat_histogram():
    edges = np.linspace(0, 100, 11)
    flat = np.array([0.0] + [10.0] * 10 + [0.0])           # underflow, 10 bins, overflow
    assert sc.histogram_quantile(edges, flat, 0.5) == pytest.approx(50)
    assert sc.histogram_quantile(edges, flat, 0.25) == pytest.approx(25)
    assert sc.sigma_eff(edges, flat) == pytest.approx(100 * sc.COVERAGE / 2)
    in_overflow = np.array([0.0] + [1.0] * 10 + [100.0])
    assert math.isnan(sc.histogram_quantile(edges, in_overflow, 0.5))
    assert math.isnan(sc.sigma_eff(edges, in_overflow))      # 68 % only with the overflow
    assert math.isnan(sc.histogram_quantile(edges, np.zeros(12), 0.5))


def test_sigma_eff_of_a_gaussian():
    h = _gauss(4000, 200, 1000.0)
    shape = sc.shape_summary(h)
    assert shape.median.value == pytest.approx(4000, abs=1)
    assert shape.sigma_eff.value == pytest.approx(200, rel=0.01)
    assert (shape.q84 - shape.q16) / 2 == pytest.approx(200, rel=0.01)
    assert shape.n_eff == pytest.approx(1000.0 / WEIGHT)
    assert shape.median.error == pytest.approx(1.2533 * shape.sigma_eff.value / math.sqrt(shape.n_eff))


def test_overflow_fraction():
    h = hist.Hist(hist.axis.Regular(10, 0, 100, name="x"), storage=hist.storage.Weight())
    h.view()["value"] = 1.0
    h.view(flow=True)[-1] = (2.5, 0.0)    # ROOT overflow bin
    assert sc.overflow_fraction(h, 100) == pytest.approx(2.5 / 12.5)
    assert sc.overflow_fraction(h, 80) == pytest.approx(4.5 / 12.5)


def test_interpolate_in_mn():
    line = [(m, sc.Value(0.1 + m / 10000, 0.01)) for m in (100, 300, 500, 700, 900)]
    assert sc.interpolate_in_mn(line, 500) is line[2][1]                       # a grid point itself
    assert sc.interpolate_in_mn(line, 400).value == pytest.approx(0.14)         # PCHIP is exact on a line
    assert sc.interpolate_in_mn(line, 400).error == pytest.approx(0.01)
    assert sc.interpolate_in_mn(line, 200) is None and sc.interpolate_in_mn(line, 800) is None  # end intervals
    assert sc.interpolate_in_mn(line[:3], 400) is None                          # too few points


def test_display_range_covers_the_peak():
    factor, (lo, hi) = sc.display_range([_gauss(4000, 200, 1.0), _gauss(4100, 200, 1.0)])
    assert lo <= 4000 - 2.85 * 200 and hi >= 4100 + 2.85 * 200    # 0.2 % / 99.8 % quantiles: 2.88 sigma
    assert 30 <= (hi - lo) / (factor * 10) <= 50
    assert lo % (factor * 10) == 0


def test_produced_events_sum_the_eras_of_a_year(sets, analyzer):
    ref, year = sets
    assert year.eras == ("Run3Summer22", "Run3Summer22EE")
    produced = sc.produced_events(year, "WR2000_N1100", analyzer_root=analyzer)
    expected = XSEC * (LUMI["Run3Summer22"] + LUMI["Run3Summer22EE"]) * 1000
    assert produced.channels == pytest.approx({"ee": expected, "mumu": expected})
    assert produced.total == pytest.approx(2 * expected)
    assert len(produced.datasets) == 4


def test_efficiency_is_yield_over_sigma_lumi_of_the_flavour(sets, analyzer):
    ref, year = sets
    for era_set, scale in ((ref, 1.0), (year, 2.0)):
        produced = sc.produced_events(era_set, "WR4000_N2100", analyzer_root=analyzer)
        per_fb = YIELDS[("resolved", "ee")] / LUMI["RunIII2024Summer24"]
        eff = sc.efficiency(era_set, "WR4000_N2100", "resolved", "ee", produced)
        assert eff.value == pytest.approx(scale * per_fb / (XSEC * 1000))
        both = sc.efficiency(era_set, "WR4000_N2100", sc.COMBINED, "mumu", produced)
        yields = YIELDS[("resolved", "mumu")] + YIELDS[("boosted", "mumu")]
        assert both.value == pytest.approx(scale * yields / LUMI["RunIII2024Summer24"] / (XSEC * 1000))
        # row 0 (shared by ee and mumu) over sigma x L of both flavours, the rest over the electrons'
        cutflow = sc.cutflow_efficiency(era_set, "WR4000_N2100", "resolved", "ee", produced)
        assert cutflow.cuts == tuple(CUTS["ee"]) and cutflow.n_shared == 1
        assert cutflow.values[0] == pytest.approx(scale * 20.0 / LUMI["RunIII2024Summer24"] / (2 * XSEC * 1000))
        assert cutflow.values[-1] == pytest.approx(eff.value)        # region yield = last row here
        factors = sc.efficiency_factors(era_set, "WR4000_N2100", "resolved", "ee", produced)
        assert factors.scale_factors == pytest.approx(1.0)
        assert factors.selection == pytest.approx(2 * YIELDS[("resolved", "ee")] / 20.0)
        assert factors.no_cuts * factors.selection * factors.scale_factors == pytest.approx(eff.value)


def test_factors_do_not_depend_on_the_flavour_mix(tmp_path, analyzer):
    """The same muon events from a file of both flavours and from a muon-only file (half the no_cuts
    row): every factor of the mumu efficiency agrees."""
    _write(tmp_path, "RunIII2024Summer24", "WR2000_N1100")
    _write(tmp_path, "RunIII2024Summer24", "WR2000_N1100", datasets=[MU.format(2000, 1100)],
           dir_name=DIR + "_mu", no_cuts=10.0)
    both, muons = sc.make_era_sets([("RunIII2024Summer24", DIR), ("RunIII2024Summer24", DIR + "_mu")],
                                   working_dir=tmp_path)
    assert muons.label.endswith(f"({DIR}_mu)")       # same era twice: labelled by directory
    factors, effs = [], []
    for era_set in (both, muons):
        produced = sc.produced_events(era_set, "WR2000_N1100", analyzer_root=analyzer)
        factors.append(sc.efficiency_factors(era_set, "WR2000_N1100", "resolved", "mumu", produced))
        effs.append(sc.efficiency(era_set, "WR2000_N1100", "resolved", "mumu", produced).value)
        if era_set is muons:
            assert sc.efficiency(era_set, "WR2000_N1100", "resolved", "ee", produced) is None
            assert sc.cutflow_efficiency(era_set, "WR2000_N1100", "resolved", "ee", produced) is None
    assert effs[1] == pytest.approx(effs[0])
    for name in ("no_cuts", "selection", "scale_factors"):
        assert getattr(factors[1], name) == pytest.approx(getattr(factors[0], name)), name


def test_no_cuts_splits_into_skim_and_filters(sets, analyzer, tmp_path):
    for era_set in sets:
        produced = sc.produced_events(era_set, "WR2000_N1100", analyzer_root=analyzer)
        assert sc.no_cuts_breakdown(era_set, "WR2000_N1100", produced, analyzer_root=analyzer) \
            == pytest.approx((NEVTS / GENERATED, NO_CUTS_RAW / (2 * NEVTS)))
    # without the skimmed fileset (e.g. an unskimmed run's checkout) there is no split
    assert sc.no_cuts_breakdown(sets[0], "WR2000_N1100", produced, analyzer_root=tmp_path / "nowhere") is None


def test_weight_factor_means_are_yield_weighted_over_eras(tmp_path):
    _write(tmp_path, "Run3Summer22", "WR2000_N1100", scale=1.0, sf=0.901)
    _write(tmp_path, "Run3Summer22EE", "WR2000_N1100", scale=3.0, sf=0.981)
    (year,) = sc.make_era_sets([("2022", DIR)], working_dir=tmp_path)
    means = sc.weight_factor_means(year, "WR2000_N1100", "resolved", "mumu")
    assert set(means) == {"LeptonSf", "pileup", "corrections"}       # dy_zpt is 1 for every event
    # bin centres 0.9025 and 0.9825, weighted by the region yields 1 : 3
    assert means["LeptonSf"] == pytest.approx((0.9025 + 3 * 0.9825) / 4)
    assert means["pileup"] == pytest.approx(0.9725)
    _write(tmp_path, "Run3Summer22EE", "WR4000_N2100")
    _write(tmp_path, "Run3Summer22", "WR4000_N2100")
    with uproot.update(root_file(_era_dir(tmp_path, "Run3Summer22"), "signal_WR4000_N2100")) as f:
        del f["event_weights"]                       # an older file without the components
    assert sc.weight_factor_means(year, "WR4000_N2100", "resolved", "mumu") == {}


def test_fileset_replaces_config_cross_sections_and_skim_counts(tmp_path, analyzer):
    """An era whose configs lack cross sections (UL18 today): the fileset the analyzer ran with gives them."""
    _write(tmp_path, "RunIII2024Summer24", "WR2000_N100")         # EL has no cross section in the config
    (era_set,) = sc.make_era_sets([("RunIII2024Summer24", DIR)], working_dir=tmp_path)
    with pytest.raises(ValueError, match="no cross section"):
        sc.produced_events(era_set, "WR2000_N100", analyzer_root=analyzer)
    fileset = tmp_path / "fileset.json"
    fileset.write_text(json.dumps({
        f"/{name}/RunIII2024Summer24NanoAODv15/NANOAODSIM": {
            "files": {}, "metadata": {"sample": name, "xsec": 3 * XSEC, "nevts": 900, "genEventCount": 1000}}
        for name in (EL.format(2000, 100), MU.format(2000, 100))}))
    filesets = {"RunIII2024Summer24": fileset}
    produced = sc.produced_events(era_set, "WR2000_N100", analyzer_root=analyzer, filesets=filesets)
    assert produced.channels["ee"] == pytest.approx(3 * XSEC * LUMI["RunIII2024Summer24"] * 1000)
    assert sc.no_cuts_breakdown(era_set, "WR2000_N100", produced, analyzer_root=analyzer, filesets=filesets) \
        == pytest.approx((0.9, NO_CUTS_RAW / 1800))


def test_flavour_mixed_dataset_counts_half_for_each_channel(tmp_path, analyzer):
    _write(tmp_path, "RunIII2024Summer24", "WR2000_N1100", datasets=["WRtoNLtoLLJJ_MWR2000_MN1100"])
    (era_set,) = sc.make_era_sets([("RunIII2024Summer24", DIR)], working_dir=tmp_path)
    produced = sc.produced_events(era_set, "WR2000_N1100", analyzer_root=analyzer)
    events = 2 * XSEC * LUMI["RunIII2024Summer24"] * 1000
    assert produced.channels == pytest.approx({"ee": events / 2, "mumu": events / 2})


def test_missing_manifest_or_cross_section_raise(tmp_path, analyzer):
    _write(tmp_path, "RunIII2024Summer24", "WR2000_N1100", manifest=False)
    _write(tmp_path, "RunIII2024Summer24", "WR2000_N100")
    (era_set,) = sc.make_era_sets([("RunIII2024Summer24", DIR)], working_dir=tmp_path)
    with pytest.raises(ValueError, match="no analyzer manifest"):
        sc.produced_events(era_set, "WR2000_N1100", analyzer_root=analyzer)
    with pytest.raises(ValueError, match="no cross section"):
        sc.produced_events(era_set, "WR2000_N100", analyzer_root=analyzer)
    with pytest.raises(FileNotFoundError, match="No signal dataset config"):
        sc.produced_events(era_set, "WR2000_N100", analyzer_root=tmp_path / "nowhere")


def test_point_missing_in_one_era_is_left_out(sets, tmp_path):
    _write(tmp_path, "Run3Summer22", "WR6000_N3100")          # not in Run3Summer22EE
    _write(tmp_path, "RunIII2024Summer24", "WR6000_N3100")
    ref, year = sets
    assert sc.set_points(year) == POINTS
    assert sc.common_points(sets) == POINTS
    assert sc.load_point_hist(year, "WR6000_N3100", "wr_ee_resolved_sr/mass_fourobject_wr_ee_resolved_sr") is None


def test_main_writes_tables_and_plots(script, sets, analyzer, tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setitem(script["main"].__globals__, "make_era_sets",
                        functools.partial(sc.make_era_sets, working_dir=tmp_path))
    monkeypatch.setattr(sys, "argv", ["compare_signal_eras.py", "--era", f"RunIII2024Summer24={DIR}",
                                      "--era", "2022", "--dir", DIR, "--analyzer-root", str(analyzer),
                                      "-o", str(out)])
    script["main"]()

    with open(out / "efficiency.csv", newline="") as f:
        table = list(csv.DictReader(f))
    # 2 sets x 2 points x 3 topologies x 2 channels
    assert len(table) == 24
    resolved_ee = [r for r in table if r["era"] == "2022" and r["topology"] == "resolved" and r["channel"] == "ee"]
    assert all(float(r["ratio_to_ref"]) == pytest.approx(2.0) for r in resolved_ee)
    assert all(float(r["median_over_mwr"]) == pytest.approx(0.95, abs=0.005) for r in resolved_ee)
    assert {(r["skim"], r["filters"]) for r in table} == {("0.8", "0.9")}
    assert {r["scale_factors"] for r in table if r["topology"] != "combined"} == {"1"}
    # 2022 / 2024: the whole factor 2 is in no_cuts; selection and scale factors agree
    assert "|   2.000   1.000  1.000  |" in (out / "summary.txt").read_text()
    summary = (out / "summary.txt").read_text()
    assert "Mean weight factors in the signal regions" in summary
    assert "wr_ee_resolved_sr (2 points)" in summary
    with open(out / "scale_factors.csv", newline="") as f:
        factors = list(csv.DictReader(f))
    assert {r["factor"] for r in factors} == {"LeptonSf", "pileup", "corrections"}
    assert len(factors) == 2 * 2 * 4 * 3          # sets x points x regions x factors
    summary = (out / "summary.txt").read_text()
    assert "2022, 13.6 TeV" in summary and "2.000" in summary
    assert "Events kept before the selection, mean over the 2 common point(s)" in summary
    # cutflow rows of the default points (data/lumi.yaml cutflow_signals that both sets have)
    assert "WR2000_N1100" in (out / "cutflow.csv").read_text()

    for name in ("efficiency/resolved_ee_WR2000.pdf", "efficiency/combined_mumu_WR4000.pdf",
                 "resolution/boosted_ee_WR4000.pdf", "mass/wr_mumu_boosted_sr/WR2000_N1100.pdf",
                 "cutflow/resolved_ee_WR4000_N2100.pdf",
                 "kinematics/wr_ee_resolved_sr/pt_leading_lepton_WR2000_N1100.pdf"):
        assert (out / name).is_file(), name


def test_normalization_check_flags_cross_sections_changed_after_the_run(script, tmp_path, analyzer, monkeypatch):
    """no_cuts / sigma L must equal skim x filters (0.8 x 0.9 here); a cross section changed after the
    analyzer ran (x1.43, like an LO -> LO x k(NLO) update) is flagged."""
    row0 = 0.72 * 2 * XSEC * LUMI["RunIII2024Summer24"] * 1000      # consistent with XSEC in the configs
    for point in POINTS:
        for dir_name in (DIR, DIR + "_b"):
            _write(tmp_path, "RunIII2024Summer24", point, no_cuts=row0, dir_name=dir_name)
    monkeypatch.setitem(script["main"].__globals__, "make_era_sets",
                        functools.partial(sc.make_era_sets, working_dir=tmp_path))
    argv = ["compare_signal_eras.py", "--era", f"RunIII2024Summer24={DIR}", "--era", f"RunIII2024Summer24={DIR}_b",
            "--analyzer-root", str(analyzer), "--no-kinematics", "-o", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", argv)
    script["main"]()
    assert "WARNING" not in (tmp_path / "out" / "summary.txt").read_text()
    with open(tmp_path / "out" / "efficiency.csv", newline="") as f:
        assert {round(float(r["norm_check"]), 3) for r in csv.DictReader(f)} == {1.0}

    changed = tmp_path / "changed_xsec.json"
    changed.write_text(json.dumps({
        name: {"files": {}, "metadata": {"sample": name, "xsec": 1.43 * XSEC, "nevts": NEVTS, "genEventCount": GENERATED}}
        for point in POINTS for name in (EL.format(*mass_point(point)), MU.format(*mass_point(point)))}))
    monkeypatch.setattr(sys, "argv", argv + ["--fileset", f"RunIII2024Summer24={changed}", "-o", str(tmp_path / "out2")])
    script["main"]()
    summary = (tmp_path / "out2" / "summary.txt").read_text()
    assert summary.count("WARNING") == 2 and "no_cuts / sigma L = 0.699" in summary


def test_labels_name_and_efficiency_tables(script, sets, analyzer, tmp_path, monkeypatch):
    monkeypatch.setitem(script["main"].__globals__, "make_era_sets",
                        functools.partial(sc.make_era_sets, working_dir=tmp_path))
    monkeypatch.setattr(sys, "argv", ["compare_signal_eras.py", "--era", f"RunIII2024Summer24={DIR}", "--era", "2022",
                                      "--dir", DIR, "--analyzer-root", str(analyzer), "--label", "2024 private",
                                      "--label", "2022 (both eras)", "--name", "my_comparison", "--no-kinematics",
                                      "--no-pdf", "--local-plots"])
    monkeypatch.setitem(script["main"].__globals__, "repo_root", lambda: tmp_path)
    monkeypatch.setitem(script["main"].__globals__, "plots_base",
                        lambda run, year, era, subdir, local: tmp_path / "plots")
    script["main"]()
    tables = (tmp_path / "tables" / "Run3" / "2024" / "RunIII2024Summer24" / DIR / "signal_outputs" / "signal_compare"
              / "my_comparison")
    text = (tables / "efficiency_tables.txt").read_text()
    assert "2024 private" in text and "2022 (both eras) / ref." in text
    # 2022 has twice the 2024 efficiency at every point (see the sets fixture)
    assert text.count("2.000") == 2 * 3 * 2          # points x topologies (incl. combined) x channels
    latex = tables / "latex"
    assert sorted(p.name for p in latex.glob("eff_*.tex"))[:2] == ["eff_boosted_ee_WR2000.tex", "eff_boosted_ee_WR4000.tex"]
    table = (latex / "eff_resolved_mumu_WR4000.tex").read_text()
    assert r"\label{tab:signal_eff_my_comparison_resolved_mumu_WR4000}" in table
    assert "2024 private / 2024 private" not in table and "2022 (both eras) / 2024 private" in table
    assert r"\input{eff_resolved_ee_WR2000.tex}" in (latex / "preview.tex").read_text()
    assert "2024 private" in (tables / "summary.txt").read_text()
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["compare_signal_eras.py", "--era", "2022", "--era", "2023", "--label", "one"])
        script["main"]()


def test_main_refuses_a_single_era(script, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compare_signal_eras.py", "--era", "2022"])
    with pytest.raises(SystemExit):
        script["main"]()


def test_efficiency_factors_use_the_last_cut_before_the_region_sf_row(monkeypatch):
    """Files made since 2026-10-14 end their cutflows with with_region_sfs (the last cut with the region's SFs);
    the selection factor ends at the last cut and the scale-factor factor is the region yield over it."""
    cf = sc.CutflowEfficiency(("no_cuts", "mll_gt400", "with_region_sfs"), np.array([0.8, 0.5, 0.45]), np.zeros(3), 1)
    monkeypatch.setattr(sc, "cutflow_efficiency", lambda *args, **kwargs: cf)
    monkeypatch.setattr(sc, "region_yield", lambda *args, **kwargs: sc.Value(45.0, 0.0))
    produced = sc.Produced({"ee": 100.0, "mumu": 0.0}, 100.0, ())
    factors = sc.efficiency_factors(None, "WR4000_N2100", "resolved", "ee", produced)
    assert factors.no_cuts == pytest.approx(0.8)
    assert factors.selection == pytest.approx(0.5 / 0.8)
    assert factors.scale_factors == pytest.approx(0.45 / 0.5)
