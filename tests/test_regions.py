"""Tests for wrplotter.regions."""
import pytest
from wrplotter.regions import Region, regions_for_era, regions_by_name, expand_region_requests


class TestRegionsForEra:

    def test_nonempty(self):
        assert len(regions_for_era("2022")) > 0

    def test_all_region_instances(self):
        for r in regions_for_era("2022"):
            assert isinstance(r, Region)

    def test_both_datasets_present(self):
        datasets = {r.primary_dataset for r in regions_for_era("2022")}
        assert "muon" in datasets
        assert "egamma" in datasets

    def test_one_data_stream_per_region(self):
        regions = regions_for_era("2022")
        assert len(regions) == len({r.name for r in regions}) == 11

    @pytest.mark.parametrize("name, stream", [
        ("wr_ee_resolved_dy_cr", "egamma"), ("wr_ee_resolved_sr", "egamma"),
        ("wr_ee_boosted_dy_cr", "egamma"), ("wr_ee_boosted_sr", "egamma"),
        ("wr_emu_boosted_flavor_cr", "egamma"),
        ("wr_mumu_resolved_dy_cr", "muon"), ("wr_mumu_resolved_sr", "muon"),
        ("wr_mumu_boosted_dy_cr", "muon"), ("wr_mumu_boosted_sr", "muon"),
        ("wr_resolved_flavor_cr", "muon"), ("wr_mue_boosted_flavor_cr", "muon"),
    ])
    def test_data_stream_is_the_trigger_stream(self, name, stream):
        """The analyzer fills every region in both data files; only the region's
        trigger stream is used (the other would add a trigger-overlap subset)."""
        assert regions_by_name("2022")[name][0].primary_dataset == stream

    def test_year_beside_the_luminosity_not_in_the_frame(self):
        for r in regions_for_era("TestEra"):
            assert r.year == "TestEra"
            assert "TestEra" not in r.tlatex_alias


class TestRegionsByName:

    def test_returns_dict(self):
        assert isinstance(regions_by_name("2022"), dict)

    def test_flavor_cr_one_entry(self):
        assert len(regions_by_name("2022")["wr_resolved_flavor_cr"]) == 1

    def test_mumu_sr_one_entry(self):
        assert len(regions_by_name("2022")["wr_mumu_resolved_sr"]) == 1


class TestExpandRegionRequests:

    def test_exact_name(self):
        result = expand_region_requests("2022", ["wr_mumu_resolved_dy_cr"])
        assert len(result) == 1
        assert result[0].name == "wr_mumu_resolved_dy_cr"
        assert result[0].primary_dataset == "muon"

    def test_name_with_dataset_filter(self):
        result = expand_region_requests("2022", ["wr_resolved_flavor_cr:muon"])
        assert len(result) == 1
        assert result[0].primary_dataset == "muon"

    def test_name_without_filter_one_region(self):
        result = expand_region_requests("2022", ["wr_resolved_flavor_cr"])
        assert [r.primary_dataset for r in result] == ["muon"]

    def test_filter_on_the_other_stream_is_unknown(self):
        with pytest.raises(ValueError, match="Unknown region"):
            expand_region_requests("2022", ["wr_resolved_flavor_cr:egamma"])

    def test_duplicate_requests_plotted_once(self):
        result = expand_region_requests("2022", ["wr_ee_resolved_sr", "resolved_sr"])
        assert [r.name for r in result].count("wr_ee_resolved_sr") == 1

    def test_shorthand_resolved_dy_cr(self):
        names = {r.name for r in expand_region_requests("2022", ["resolved_dy_cr"])}
        assert "wr_ee_resolved_dy_cr" in names
        assert "wr_mumu_resolved_dy_cr" in names

    def test_shorthand_resolved_sr(self):
        names = {r.name for r in expand_region_requests("2022", ["resolved_sr"])}
        assert "wr_ee_resolved_sr" in names
        assert "wr_mumu_resolved_sr" in names

    def test_shorthand_boosted_flavor_cr(self):
        names = {r.name for r in expand_region_requests("2022", ["boosted_flavor_cr"])}
        assert "wr_emu_boosted_flavor_cr" in names
        assert "wr_mue_boosted_flavor_cr" in names

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="Unknown region"):
            expand_region_requests("2022", ["totally_fake_region"])

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown region"):
            expand_region_requests("2022", ["wr_mumu_resolved_dy_cr:photon"])

    def test_multiple_requests(self):
        result = expand_region_requests("2022", ["wr_mumu_resolved_dy_cr", "wr_ee_resolved_dy_cr"])
        assert len(result) == 2


class TestRegionProperties:

    def test_sr_is_signal_region(self):
        r = Region("wr_mumu_resolved_sr", "muon")
        assert r.is_signal_region is True

    def test_dy_cr_is_not_signal_region(self):
        r = Region("wr_ee_resolved_dy_cr", "egamma")
        assert r.is_signal_region is False

    def test_boosted_region(self):
        r = Region("wr_mumu_boosted_sr", "muon")
        assert r.is_boosted is True

    def test_resolved_region_not_boosted(self):
        r = Region("wr_ee_resolved_sr", "egamma")
        assert r.is_boosted is False

    def test_boosted_sr_is_both(self):
        r = Region("wr_mumu_boosted_sr", "muon")
        assert r.is_signal_region is True
        assert r.is_boosted is True

    def test_boosted_dy_cr_is_boosted_not_sr(self):
        r = Region("wr_ee_boosted_dy_cr", "egamma")
        assert r.is_boosted is True
        assert r.is_signal_region is False


class TestRegionVariantsAndTopology:

    @pytest.mark.parametrize("name, is_sr", [
        ("wr_ee_resolved_sr", True), ("wr_mumu_boosted_sr", True),
        ("wr_ee_resolved_sr_nomass", True), ("wr_mumu_boosted_sr_nomass", True),
        ("wr_ee_resolved_dy_cr", False), ("wr_resolved_flavor_cr", False),
        ("wr_emu_boosted_flavor_cr", False),
    ])
    def test_is_signal_region(self, name, is_sr):
        assert Region(name, "egamma").is_signal_region is is_sr

    def test_topology(self):
        assert Region("wr_ee_boosted_sr", "egamma").topology == "boosted"
        assert Region("wr_ee_resolved_sr", "egamma").topology == "resolved"
