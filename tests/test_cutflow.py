"""Tests for wrplotter.cutflow: cut labels from the file, the data column
(stream per channel, blinded rows, pre-blinding files) and efficiencies."""
import json

import hist
import numpy as np
import pytest
import uproot

from wrplotter.cutflow import (
    build_table, cutflow_key, efficiency, format_latex, format_text, table_variant,
)
from wrplotter.inputs import clear_manifest_cache, make_inputs, manifest_path, root_file

ERA = "RunIII2024Summer24"
DIR = "cutflow_test"
RESOLVED = ["no_cuts", "two_tight_leptons", "mlljj_gt800", "mll_gt200", "mll_gt400"]
BOOSTED = ["no_cuts", "boosted_tag", "mll_gt200_boosted", "mlj_gt800_boosted"]
BLINDED = {"applied": True, "signal_regions": ["wr_ee_resolved_sr", "wr_mumu_resolved_sr"]}


@pytest.fixture(autouse=True)
def _fresh_manifests():
    clear_manifest_cache()
    yield
    clear_manifest_cache()


def _cutflow(cuts, values):
    h = hist.Hist(hist.axis.StrCategory(cuts, name="cut"), storage=hist.storage.Weight())
    h.view().value[...] = values
    h.view().variance[...] = values
    return h


def _write(directory, sample, cutflows, *, blinding="absent"):
    """cutflows: {(topology, channel): (cuts, values)}, written as cumulative and onecut."""
    directory.mkdir(parents=True, exist_ok=True)
    path = root_file(directory, sample)
    with uproot.recreate(path) as f:
        for (topology, channel), (cuts, values) in cutflows.items():
            for kind in ("cumulative", "onecut"):
                for unweighted in (False, True):
                    f[cutflow_key(topology, channel, kind, unweighted)] = _cutflow(cuts, values)
    manifest = {"datasets": {}}
    if blinding != "absent":
        manifest["blinding"] = blinding
    manifest_path(path).write_text(json.dumps(manifest))


def _mc(scale):
    values = scale * np.array([1000.0, 500.0, 100.0, 20.0, 10.0])
    boosted = scale * np.array([800.0, 80.0, 8.0, 4.0])
    return {("resolved", ch): (RESOLVED, values) for ch in ("ee", "mumu", "em")} | \
           {("boosted", ch): (BOOSTED, boosted) for ch in ("ee", "mumu", "em")}


def _data(value, *, truncate):
    """Data cutflows; ee/mumu stop before the mll > 200 cut when *truncate*."""
    out = {}
    for topology, cuts in (("resolved", RESOLVED), ("boosted", BOOSTED)):
        stop = cuts.index("mll_gt200" if topology == "resolved" else "mll_gt200_boosted")
        for ch in ("ee", "mumu", "em"):
            keep = cuts[:stop] if truncate and ch != "em" else cuts
            out[(topology, ch)] = (keep, np.full(len(keep), value))
    return out


def _inputs(tmp_path, *, data_blinding=BLINDED, truncate=True):
    base = tmp_path / "rootfiles" / "Run3" / "2024" / ERA / DIR
    _write(base, "DYJets", _mc(1.0))
    _write(base, "tt_tW", _mc(2.0))
    _write(base, "signal_WR2000_N1100", _mc(0.01))
    _write(base, "EGamma", _data(7.0, truncate=truncate), blinding=data_blinding)
    _write(base, "Muon", _data(9.0, truncate=truncate), blinding=data_blinding)
    (inputs,) = make_inputs(ERA, DIR, None, working_dir=tmp_path)
    return inputs


def test_cuts_and_columns_from_file(tmp_path):
    table = build_table(_inputs(tmp_path), "resolved", "ee", "cumulative",
                        signal_points=["WR2000_N1100"])
    assert table.cuts == tuple(RESOLVED)
    assert [c.key for c in table.backgrounds] == ["dy", "ttbar"]  # order: stack top first
    np.testing.assert_allclose(table.total, 3.0 * np.array([1000, 500, 100, 20, 10]))
    assert [c.key for c in table.signals] == ["WR2000_N1100"]


@pytest.mark.parametrize("topology, channel, value", [
    ("resolved", "ee", 7.0), ("resolved", "mumu", 9.0), ("resolved", "em", 9.0),
    ("boosted", "ee", 7.0), ("boosted", "mumu", 9.0), ("boosted", "em", 7.0),
])
def test_data_from_the_channel_trigger_stream(tmp_path, topology, channel, value):
    table = build_table(_inputs(tmp_path, truncate=False), topology, channel, "cumulative")
    np.testing.assert_array_equal(table.data, value)


def test_truncated_rows_are_blinded(tmp_path):
    inputs = _inputs(tmp_path)
    table = build_table(inputs, "resolved", "ee", "cumulative")
    np.testing.assert_array_equal(table.blinded, [False, False, False, True, True])
    np.testing.assert_array_equal(table.data[:3], 7.0)
    assert "2 row(s) blinded" in table.data_note
    text = format_text(table)
    assert text.count("blinded") >= 2
    latex = format_latex(table, era=ERA, lumi=109.7, year="2024")
    assert latex.count(r"\textit{blinded}") == 2
    assert "Data in the signal-region rows is blinded." in latex

    boosted = build_table(inputs, "boosted", "mumu", "onecut", unweighted=True)
    np.testing.assert_array_equal(boosted.blinded, [False, False, True, True])
    # em is a flavor CR: never blinded.
    assert not build_table(inputs, "resolved", "em", "cumulative").blinded.any()


def test_pre_blinding_data_gives_no_column(tmp_path):
    table = build_table(_inputs(tmp_path, data_blinding="absent", truncate=False),
                        "resolved", "ee", "cumulative")
    assert table.data is None
    assert "no analyzer blinding record" in table.data_note
    assert "Data" not in format_text(table).splitlines()[1]


def test_data_cut_unknown_to_mc_raises(tmp_path):
    inputs = _inputs(tmp_path)
    base = inputs.dirs[0]
    _write(base, "EGamma", {("resolved", "ee"): (RESOLVED[:2] + ["surprise"], [1.0, 1.0, 1.0])},
           blinding=BLINDED)
    clear_manifest_cache()
    from wrplotter.histogram_cache import clear_cache
    clear_cache()
    with pytest.raises(ValueError, match="surprise"):
        build_table(inputs, "resolved", "ee", "cumulative")


def test_efficiencies():
    values = np.array([1000.0, 500.0, 100.0, 0.0, 0.0])
    np.testing.assert_allclose(efficiency(values, "running"), [1, 0.5, 0.1, 0, 0])
    relative = efficiency(values, "relative")
    np.testing.assert_allclose(relative[:4], [1, 0.5, 0.2, 0])
    assert np.isnan(relative[4])  # previous row empty
    assert np.isnan(efficiency(np.zeros(3), "running")).all()
    with pytest.raises(ValueError):
        efficiency(values, "n-1")


@pytest.mark.parametrize("kind, unweighted, mode, name", [
    ("cumulative", False, None, "cumulative"),
    ("onecut", True, None, "onecut_unweighted"),
    ("cumulative", False, "running", "running_eff"),
    ("cumulative", False, "relative", "relative_eff"),
    ("onecut", False, "running", "single_eff"),
])
def test_table_variant(kind, unweighted, mode, name):
    assert table_variant(kind, unweighted, mode) == name


class TestShownRows:
    """Rows a table or plot leaves out, and the per-topology label of the region-SF row."""

    def test_hem_veto_only_where_an_era_applies_it(self):
        from wrplotter.cutflow import row_shown
        assert not row_shown("hem_veto", eras=("RunIII2024Summer24",), kind="cumulative", unweighted=False)
        assert row_shown("hem_veto", eras=("RunIISummer20UL18",), kind="cumulative", unweighted=False)
        assert row_shown("mll_gt400", eras=("RunIII2024Summer24",), kind="cumulative", unweighted=False)

    def test_region_sf_row_only_in_weighted_cumulative_cutflows(self):
        from wrplotter.cutflow import CUTFLOW_SF_ROW, row_shown
        assert row_shown(CUTFLOW_SF_ROW, eras=("RunIII2024Summer24",), kind="cumulative", unweighted=False)
        assert not row_shown(CUTFLOW_SF_ROW, eras=("RunIII2024Summer24",), kind="cumulative", unweighted=True)
        assert not row_shown(CUTFLOW_SF_ROW, eras=("RunIII2024Summer24",), kind="onecut", unweighted=False)

    def test_shown_rows_keeps_values_aligned(self):
        from wrplotter.cutflow import Cutflow, shown_rows
        cf = Cutflow(("no_cuts", "hem_veto", "mll_gt400"), np.array([3.0, 2.0, 1.0]), np.array([9.0, 4.0, 1.0]))
        out = shown_rows(cf, eras=("RunIII2024Summer24",), kind="cumulative", unweighted=False)
        assert out.cuts == ("no_cuts", "mll_gt400")
        assert list(out.values) == [3.0, 1.0] and list(out.variances) == [9.0, 1.0]

    def test_region_sf_row_label_per_topology(self):
        from wrplotter.cutflow import CUTFLOW_SF_ROW, cut_label
        assert "LSF" not in cut_label(CUTFLOW_SF_ROW, "table", "resolved")
        assert "LSF" in cut_label(CUTFLOW_SF_ROW, "table", "boosted")
        assert cut_label(CUTFLOW_SF_ROW, "table") == cut_label(CUTFLOW_SF_ROW, "table", "boosted")
