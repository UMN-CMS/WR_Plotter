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

    def test_flavor_cr_has_both_datasets(self):
        flavors = [r for r in regions_for_era("2022") if r.name == "wr_resolved_flavor_cr"]
        assert {r.primary_dataset for r in flavors} == {"muon", "egamma"}

    def test_era_in_tlatex(self):
        for r in regions_for_era("TestEra"):
            assert "TestEra" in r.tlatex_alias


class TestRegionsByName:

    def test_returns_dict(self):
        assert isinstance(regions_by_name("2022"), dict)

    def test_flavor_cr_two_entries(self):
        assert len(regions_by_name("2022")["wr_resolved_flavor_cr"]) == 2

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

    def test_name_without_filter_all_variants(self):
        result = expand_region_requests("2022", ["wr_resolved_flavor_cr"])
        assert len(result) == 2

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
