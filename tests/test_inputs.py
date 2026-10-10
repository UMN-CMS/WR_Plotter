"""Tests for wrplotter.inputs: DY-sample directories, the mislabel guard and the
analyzer's blinding record."""
import json
import logging

import pytest

from wrplotter.inputs import (
    blinding_recorded, clear_manifest_cache, detect_dy_tag, make_inputs, manifest_path,
    root_file, sr_data_unblinded,
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
    inputs._PARTIAL_REPORTED.clear()
    yield
    clear_manifest_cache()


def _era_dir(tmp_path, sub):
    return tmp_path / "rootfiles" / "Run3" / "2024" / ERA / sub


def _write_sample(directory, sample, *, datasets=None, physics_group=None, blinding=None,
                  manifest=True):
    directory.mkdir(parents=True, exist_ok=True)
    path = root_file(directory, sample)
    path.write_bytes(b"")  # existence is all make_inputs checks
    if manifest:
        payload = {"datasets": {d: {"physics_group": physics_group or sample} for d in datasets or []}}
        if blinding is not None:
            payload["blinding"] = blinding
        manifest_path(path).write_text(json.dumps(payload))
    return path


@pytest.fixture
def analysis_set(tmp_path):
    """Base dir with the default (lo_ht) DY and tt_tW, plus a lo_inc DY dir."""
    base = _era_dir(tmp_path, DIR)
    _write_sample(base, "DYJets", datasets=LO_HT)
    _write_sample(base, "tt_tW", datasets=["TTto2L2Nu"])
    _write_sample(_era_dir(tmp_path, f"{DIR}_lo_inc_dy"), "DYJets", datasets=LO_INC)
    return tmp_path


class TestDYResolution:

    def test_default_sample_from_base_dir(self, analysis_set):
        (inputs,) = make_inputs(ERA, DIR, None, working_dir=analysis_set)
        assert inputs.dy.tag == "lo_ht"
        assert inputs.files_for("DYJets") == [root_file(_era_dir(analysis_set, DIR), "DYJets")]

    def test_other_sample_from_tagged_dir(self, analysis_set):
        lo_ht, lo_inc = make_inputs(ERA, DIR, ["lo_ht", "lo_inc"], working_dir=analysis_set)
        assert lo_inc.dy.tag == "lo_inc"
        assert lo_inc.files_for("DYJets") == [
            root_file(_era_dir(analysis_set, f"{DIR}_lo_inc_dy"), "DYJets")]
        # Every other sample still comes from the base directory.
        assert lo_inc.files_for("tt_tW") == lo_ht.files_for("tt_tW")

    def test_explicit_directory(self, analysis_set):
        _write_sample(_era_dir(analysis_set, "elsewhere"), "DYJets", datasets=LO_INC)
        (inputs,) = make_inputs(ERA, DIR, ["lo_inc=elsewhere"], working_dir=analysis_set)
        assert inputs.dy.dirs == (_era_dir(analysis_set, "elsewhere"),)

    def test_missing_dy_file_names_the_rerun(self, analysis_set):
        with pytest.raises(FileNotFoundError, match="--dy nlo_mll"):
            make_inputs(ERA, DIR, ["nlo_mll"], working_dir=analysis_set)

    def test_mislabelled_directory_refused(self, analysis_set):
        with pytest.raises(ValueError, match="holds the 'lo_inc' DY sample"):
            make_inputs(ERA, DIR, [f"lo_ht={DIR}_lo_inc_dy"], working_dir=analysis_set)

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
