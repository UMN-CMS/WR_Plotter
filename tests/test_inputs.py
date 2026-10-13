"""Tests for wrplotter.inputs: DY-sample files, the mislabel guard and the
analyzer's blinding record."""
import json
import logging

import numpy as np
import pytest
import uproot

from wrplotter.histogram_cache import clear_cache
from wrplotter.inputs import (
    blinding_recorded, clear_manifest_cache, detect_dy_tag, dy_reweight_in_file,
    expected_dy_reweight, make_inputs, manifest_path, root_file, sr_data_unblinded,
)

ERA = "RunIII2024Summer24"
DIR = "20261008_test"
LO_HT = ["DYto2L-4Jets_Bin-HT-100to400_MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8"]
LO_INC = ["DYto2E-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8",
          "DYto2Mu-4Jets_Bin-MLL-50_TuneCP5_13p6TeV_madgraphMLM-pythia8"]


@pytest.fixture(autouse=True)
def _fresh_manifests():
    from wrplotter import inputs
    clear_manifest_cache()
    clear_cache()
    inputs._PARTIAL_REPORTED.clear()
    yield
    clear_manifest_cache()
    clear_cache()


def _era_dir(tmp_path, sub):
    return tmp_path / "rootfiles" / "Run3" / "2024" / ERA / sub


def _write_sample(directory, sample, *, datasets=None, physics_group=None, blinding=None,
                  manifest=True, reweight=None):
    """A ROOT file with no histograms, or with the analyzer's DY weight-factor histograms
    for *reweight* ("none", "zpt", "zpt+reshape"; "identity": dy_zpt written but all 1)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = root_file(directory, sample)
    edges = np.linspace(0, 3, 301)
    at_one = np.zeros(300)
    at_one[100] = 50.0                       # every event in the bin of 1.0
    spread = np.zeros(300)
    spread[90:130] = 5.0
    factors = {"none": {}, "identity": {"dy_zpt": at_one}, "zpt": {"dy_zpt": spread},
               "zpt+reshape": {"dy_zpt": spread, "dy_reshape": spread}}[reweight or "none"]
    with uproot.recreate(path) as f:
        f["event_weights/wr_ee_resolved_dy_cr/components/pileup"] = (spread, edges)
        for factor, values in factors.items():
            f[f"event_weights/wr_ee_resolved_dy_cr/components/{factor}"] = (values, edges)
    if manifest:
        payload = {"datasets": {d: {"physics_group": physics_group or sample} for d in datasets or []}}
        if blinding is not None:
            payload["blinding"] = blinding
        manifest_path(path).write_text(json.dumps(payload))
    return path


def _write_dy(directory, tag=None, **kwargs):
    """The DY file of *tag* as the analyzer names it (None: the default sample)."""
    return _write_sample(directory, f"DYJets_{tag}" if tag else "DYJets", physics_group="DYJets", **kwargs)


@pytest.fixture
def analysis_set(tmp_path):
    """Base dir with the default (lo_ht) DY, the lo_inc DY and tt_tW."""
    base = _era_dir(tmp_path, DIR)
    _write_dy(base, datasets=LO_HT)
    _write_dy(base, "lo_inc", datasets=LO_INC)
    _write_sample(base, "tt_tW", datasets=["TTto2L2Nu"])
    return tmp_path


class TestDYResolution:

    def test_default_sample_from_base_dir(self, analysis_set):
        (inputs,) = make_inputs(ERA, DIR, None, working_dir=analysis_set)
        assert inputs.dy.tag == "lo_ht"
        assert inputs.files_for("DYJets") == [root_file(_era_dir(analysis_set, DIR), "DYJets")]

    def test_other_sample_from_its_file(self, analysis_set):
        lo_ht, lo_inc = make_inputs(ERA, DIR, ["lo_ht", "lo_inc"], working_dir=analysis_set)
        assert lo_inc.dy.tag == "lo_inc"
        assert lo_inc.files_for("DYJets") == [root_file(_era_dir(analysis_set, DIR), "DYJets_lo_inc")]
        assert lo_inc.dirs_for("DYJets") == [_era_dir(analysis_set, DIR)]
        # Every other sample still comes from the base directory.
        assert lo_inc.files_for("tt_tW") == lo_ht.files_for("tt_tW")

    def test_explicit_directory(self, analysis_set):
        elsewhere = _era_dir(analysis_set, "elsewhere")
        _write_dy(elsewhere, "nlo_inc", datasets=["DYto2E-2Jets_Bin-MLL-50"])
        (inputs,) = make_inputs(ERA, DIR, ["nlo_inc=elsewhere"], working_dir=analysis_set)
        assert inputs.dy.files == (root_file(elsewhere, "DYJets_nlo_inc"),)

    def test_explicit_directory_old_layout(self, analysis_set):
        _write_dy(_era_dir(analysis_set, "elsewhere"), datasets=LO_INC)
        (inputs,) = make_inputs(ERA, DIR, ["lo_inc=elsewhere"], working_dir=analysis_set)
        assert inputs.dy.files == (root_file(_era_dir(analysis_set, "elsewhere"), "DYJets"),)

    @pytest.mark.parametrize("tag, old_dir", [("nlo_inc", f"{DIR}_nlo_inc_dy"),
                                               ("lo_ht_zpt", f"{DIR}_lo_ht_zpt_dy"),
                                               ("lo_ht_zpt", f"{DIR}_lo_ht_zptonly_dy"),
                                               ("lo_ht_zptonly", f"{DIR}_lo_ht_zptonly_dy"),
                                               ("lo_ht_zpt", f"{DIR}_noreshape"),
                                               ("lo_ht_zpt_reshape", f"{DIR}_lo_ht_reweighted_dy")])
    def test_old_layout_directories(self, analysis_set, tag, old_dir):
        """Sets made before 2026-10-12: each sample as WRAnalyzer_DYJets.root in its own directory."""
        datasets = ["DYto2E-2Jets_Bin-MLL-50"] if tag == "nlo_inc" else LO_HT
        _write_dy(_era_dir(analysis_set, old_dir), datasets=datasets, reweight=expected_dy_reweight(tag))
        (inputs,) = make_inputs(ERA, DIR, [tag], working_dir=analysis_set)
        assert inputs.dy.tag == tag.replace("zptonly", "zpt")
        assert inputs.files_for("DYJets") == [root_file(_era_dir(analysis_set, old_dir), "DYJets")]

    def test_file_in_dir_preferred_over_old_layout(self, analysis_set):
        _write_dy(_era_dir(analysis_set, f"{DIR}_lo_inc_dy"), datasets=LO_INC)
        (inputs,) = make_inputs(ERA, DIR, ["lo_inc"], working_dir=analysis_set)
        assert inputs.files_for("DYJets") == [root_file(_era_dir(analysis_set, DIR), "DYJets_lo_inc")]

    def test_missing_dy_file_names_the_rerun(self, analysis_set):
        with pytest.raises(FileNotFoundError, match=f"WRAnalyzer_DYJets_nlo_mll.root.*--dy nlo_mll --dir {DIR}"):
            make_inputs(ERA, DIR, ["nlo_mll"], working_dir=analysis_set)

    def test_mislabelled_file_refused(self, analysis_set):
        _write_dy(_era_dir(analysis_set, DIR), "nlo_inc", datasets=LO_INC)
        with pytest.raises(ValueError, match="holds the 'lo_inc' DY sample"):
            make_inputs(ERA, DIR, ["nlo_inc"], working_dir=analysis_set)

    @pytest.mark.parametrize("tag", ["lo_ht_zpt_reshape", "lo_ht_zpt"])
    def test_sample_sharing_datasets(self, analysis_set, tag):
        """The lo_ht datasets made with other analyzer options."""
        base = _era_dir(analysis_set, DIR)
        _write_dy(base, tag, datasets=LO_HT, reweight=expected_dy_reweight(tag))
        lo_ht, variant = make_inputs(ERA, DIR, ["lo_ht", tag], working_dir=analysis_set)
        assert variant.files_for("DYJets") == [root_file(base, f"DYJets_{tag}")]
        assert variant.files_for("tt_tW") == lo_ht.files_for("tt_tW")
        # Another sample's datasets are still refused.
        _write_dy(_era_dir(analysis_set, "elsewhere"), datasets=LO_INC)
        with pytest.raises(ValueError, match="holds the 'lo_inc' DY sample"):
            make_inputs(ERA, DIR, [f"{tag}=elsewhere"], working_dir=analysis_set)

    @pytest.mark.parametrize("tag, flag", [("lo_ht_zpt_reshape", "--dy-reweight zpt\\+reshape"),
                                           ("lo_ht_zpt", "--dy-reweight zpt")])
    def test_missing_dy_file_names_the_analyzer_options(self, analysis_set, tag, flag):
        with pytest.raises(FileNotFoundError, match=f"{flag} --dir {DIR}"):
            make_inputs(ERA, DIR, [tag], working_dir=analysis_set)

    @pytest.mark.parametrize("reweight", ["none", "identity", "zpt", "zpt+reshape"])
    def test_reweighting_read_from_file(self, tmp_path, reweight):
        path = _write_sample(tmp_path, "DYJets", datasets=LO_HT, reweight=reweight)
        assert dy_reweight_in_file(path) == ("none" if reweight == "identity" else reweight)

    def test_reweighting_unknown_without_event_weights(self, tmp_path):
        path = root_file(tmp_path, "DYJets")
        with uproot.recreate(path):
            pass
        assert dy_reweight_in_file(path) is None

    def test_expected_reweighting(self):
        assert [expected_dy_reweight(t) for t in
                ("lo_ht", "lo_ht_zpt", "lo_ht_zpt_reshape", "lo_ht_zptonly", "lo_ht_reweighted", "lo_inc")] \
            == ["none", "zpt", "zpt+reshape", "zpt", "zpt+reshape", None]

    def test_reweighted_file_refused_as_lo_ht(self, analysis_set):
        """A set made before 2026-10-12 holds the reweighted DY in --dir: asking for lo_ht
        (not reweighted) is refused and names the tag to use."""
        _write_sample(_era_dir(analysis_set, "old"), "DYJets", datasets=LO_HT, reweight="zpt+reshape")
        with pytest.raises(ValueError, match="--dy lo_ht_zpt_reshape=old"):
            make_inputs(ERA, DIR, ["lo_ht=old"], working_dir=analysis_set)
        (inputs,) = make_inputs(ERA, DIR, ["lo_ht_zpt_reshape=old"], working_dir=analysis_set)
        assert inputs.dy.tag == "lo_ht_zpt_reshape"
        assert inputs.files_for("DYJets") == [root_file(_era_dir(analysis_set, "old"), "DYJets")]

    def test_reweighting_mismatch_for_the_default_only_warns(self, tmp_path, caplog):
        _write_sample(_era_dir(tmp_path, "old"), "DYJets", datasets=LO_HT, reweight="zpt")
        with caplog.at_level(logging.WARNING):
            (inputs,) = make_inputs(ERA, "old", None, working_dir=tmp_path)
        assert inputs.dy.tag == "lo_ht"
        assert "--dy lo_ht_zpt=old" in caplog.text

    def test_variant_with_wrong_reweighting_refused(self, analysis_set):
        _write_sample(_era_dir(analysis_set, f"{DIR}_lo_ht_zptonly_dy"), "DYJets", datasets=LO_HT,
                      reweight="zpt+reshape")
        with pytest.raises(ValueError, match="stands for --dy-reweight zpt"):
            make_inputs(ERA, DIR, ["lo_ht_zptonly"], working_dir=analysis_set)

    def test_unknown_tag(self, analysis_set):
        with pytest.raises(ValueError, match="Unknown DY sample"):
            make_inputs(ERA, DIR, ["nnlo"], working_dir=analysis_set)

    def test_duplicate_tag(self, analysis_set):
        with pytest.raises(ValueError, match="requested twice"):
            make_inputs(ERA, DIR, ["lo_ht", f"lo_ht={DIR}"], working_dir=analysis_set)

    def test_missing_base_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Input directory not found"):
            make_inputs(ERA, "nope", None, working_dir=tmp_path)

    def test_detect_dy_tag(self, tmp_path):
        assert detect_dy_tag(_write_sample(tmp_path / "a", "DYJets", datasets=LO_HT)) == "lo_ht"
        assert detect_dy_tag(_write_sample(tmp_path / "b", "DYJets", datasets=LO_INC)) == "lo_inc"
        # Mixed samples, no datasets, or no manifest: undecided.
        assert detect_dy_tag(_write_sample(tmp_path / "c", "DYJets", datasets=LO_HT + LO_INC)) is None
        assert detect_dy_tag(_write_sample(tmp_path / "d", "DYJets", datasets=[])) is None
        assert detect_dy_tag(_write_sample(tmp_path / "e", "DYJets", manifest=False)) is None


class TestBlindingRecord:

    @pytest.mark.parametrize("blinding, recorded, unblinded", [
        (None, False, False),                                   # legacy: predates the record
        ({"applied": True, "signal_regions": ["wr_ee_resolved_sr"]}, True, False),
        ({"applied": False, "signal_regions": []}, True, True),  # Run 2 --unblind
    ])
    def test_record(self, tmp_path, blinding, recorded, unblinded):
        path = _write_sample(tmp_path, "EGamma", datasets=["EGamma0"], blinding=blinding)
        assert blinding_recorded(path) is recorded
        assert sr_data_unblinded(path) is unblinded

    def test_no_manifest_is_blinded(self, tmp_path):
        path = _write_sample(tmp_path, "Muon", manifest=False)
        assert not blinding_recorded(path)
        assert not sr_data_unblinded(path)


class TestCombinedEras:

    def _year_2022(self, tmp_path):
        for sub in ("Run3Summer22", "Run3Summer22EE"):
            _write_sample(tmp_path / "rootfiles" / "Run3" / "2022" / sub / DIR, "DYJets", datasets=LO_HT)
        return tmp_path / "rootfiles" / "Run3" / "2022"

    def test_files_from_every_era(self, tmp_path):
        year = self._year_2022(tmp_path)
        (inputs,) = make_inputs("2022", DIR, None, working_dir=tmp_path)
        assert inputs.files_for("DYJets") == [root_file(year / "Run3Summer22" / DIR, "DYJets"),
                                              root_file(year / "Run3Summer22EE" / DIR, "DYJets")]

    def test_sample_missing_in_some_eras_is_reported_once(self, tmp_path, caplog):
        year = self._year_2022(tmp_path)
        _write_sample(year / "Run3Summer22" / DIR, "tt_tW", datasets=["TTto2L2Nu"])
        (inputs,) = make_inputs("2022", DIR, None, working_dir=tmp_path)
        with caplog.at_level(logging.WARNING):
            assert len(inputs.files_for("tt_tW")) == 1
            inputs.files_for("tt_tW")
        assert caplog.text.count("exists in only 1 of the 2 eras of 2022") == 1
        assert "Run3Summer22EE" in caplog.text

    def test_sample_absent_everywhere_is_not_reported(self, tmp_path, caplog):
        self._year_2022(tmp_path)
        (inputs,) = make_inputs("2022", DIR, None, working_dir=tmp_path)
        with caplog.at_level(logging.WARNING):
            assert inputs.files_for("Nonprompt") == []
        assert "exists in only" not in caplog.text
