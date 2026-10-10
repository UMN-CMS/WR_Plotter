"""Blinding and set parsing of signal_fitting/run2_results/compare_legacy_new.py."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from wrplotter.inputs import clear_manifest_cache

SCRIPT = Path(__file__).resolve().parents[1] / "signal_fitting" / "run2_results" / "compare_legacy_new.py"


@pytest.fixture(scope="module")
def cmp():
    spec = importlib.util.spec_from_file_location("compare_legacy_new", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["compare_legacy_new"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _fresh_manifests():
    clear_manifest_cache()
    yield
    clear_manifest_cache()


def _set(cmp, tmp_path, era="RunIISummer20UL18", run="RunII"):
    return cmp.NewSet(label="x", era=era, run=run, base=tmp_path, sub="s", dy_sub="s", lumi=59.83)


def _manifest(tmp_path, sample, blinding):
    d = tmp_path / "s"
    d.mkdir(exist_ok=True)
    (d / f"WRAnalyzer_{sample}.root").write_bytes(b"")
    payload = {} if blinding is None else {"blinding": blinding}
    (d / f"WRAnalyzer_{sample}_systematics.json").write_text(json.dumps(payload))


class TestParseSet:
    def test_default_era(self, cmp):
        s = cmp.parse_set("a=dyreshape_20260907")
        assert (s.era, s.run, s.sub, s.dy_sub) == ("RunIISummer20UL18", "RunII", "dyreshape_20260907",
                                                    "dyreshape_20260907")
        assert s.base.parts[-4:] == ("rootfiles", "RunII", "2018", "RunIISummer20UL18")

    def test_era_and_dy_dir(self, cmp):
        s = cmp.parse_set("y=RunIII2024Summer24/setA:setB")
        assert (s.era, s.run, s.sub, s.dy_sub) == ("RunIII2024Summer24", "Run3", "setA", "setB")
        assert s.scale == pytest.approx(cmp.LEGACY_LUMI / s.lumi)

    def test_missing_label(self, cmp):
        with pytest.raises(ValueError):
            cmp.parse_set("dyreshape_20260907")


class TestBlinding:
    @pytest.mark.parametrize("region, sr", [("wr_ee_resolved_sr", True), ("wr_mumu_boosted_sr", True),
                                            ("wr_ee_resolved_dy_cr", False), ("wr_resolved_flavor_cr", False)])
    def test_signal_region(self, cmp, region, sr):
        assert cmp.is_signal_region(region) is sr

    @pytest.mark.parametrize("blinding", [None, {"applied": False}, {"applied": True}])
    def test_run3_data_always_blinded(self, cmp, tmp_path, blinding):
        _manifest(tmp_path, "Muon", blinding)
        s = _set(cmp, tmp_path, era="RunIII2024Summer24", run="Run3")
        assert not cmp.sr_data_allowed(s, "Muon")
        assert cmp.new_hist(s, "Muon", "wr_mumu_resolved_sr", "mass_fourobject") is None
        assert cmp.blinded(s, "Muon", "wr_mumu_resolved_sr")
        assert not cmp.blinded(s, "Muon", "wr_mumu_resolved_dy_cr")

    def test_run3_mc_not_blinded(self, cmp, tmp_path):
        s = _set(cmp, tmp_path, era="RunIII2024Summer24", run="Run3")
        assert cmp.sr_data_allowed(s, "DYJets")

    @pytest.mark.parametrize("blinding, allowed", [(None, True),              # predates the record
                                                   ({"applied": False}, True),  # --unblind
                                                   ({"applied": True}, False)])
    def test_run2_follows_the_record(self, cmp, tmp_path, blinding, allowed):
        _manifest(tmp_path, "EGamma", blinding)
        assert cmp.sr_data_allowed(_set(cmp, tmp_path), "EGamma") is allowed

    def test_run3_cutflow_rows_blinded(self, cmp, tmp_path):
        s = _set(cmp, tmp_path, era="RunIII2024Summer24", run="Run3")
        # returns before opening any file for the blinded rows
        for stage in ("mll_gt200", "mll_gt400"):
            for ch in ("ee", "mumu"):
                assert cmp._cutflow_value(s, "EGamma", ch, stage) != cmp._cutflow_value(s, "EGamma", ch, stage)
