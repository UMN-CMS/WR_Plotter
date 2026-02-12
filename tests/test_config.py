"""Tests for wrplotter.config."""
import pytest
from wrplotter.config import (
    _deep_merge,
    list_eras,
    load_lumi,
    load_plot_settings,
    load_kfactors,
    get_kfactor,
    index_plot_settings,
    get_var_cfg,
    validate_var_cfg,
)


class TestDeepMerge:

    def test_flat_override(self):
        assert _deep_merge({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 10}
        override = {"a": {"y": 99, "z": 3}}
        assert _deep_merge(base, override) == {"a": {"x": 1, "y": 99, "z": 3}, "b": 10}

    def test_list_replaced_not_concatenated(self):
        assert _deep_merge({"s": [1, 2]}, {"s": [4, 5]}) == {"s": [4, 5]}

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": {"y": 2}})
        assert base == {"a": {"x": 1}}

    def test_empty_override(self):
        assert _deep_merge({"a": 1}, {}) == {"a": 1}

    def test_empty_base(self):
        assert _deep_merge({}, {"a": 1}) == {"a": 1}

    def test_deeply_nested(self):
        assert _deep_merge(
            {"a": {"b": {"c": 1}}},
            {"a": {"b": {"d": 2}}},
        ) == {"a": {"b": {"c": 1, "d": 2}}}

    def test_override_dict_with_scalar(self):
        assert _deep_merge({"a": {"x": 1}}, {"a": 42}) == {"a": 42}


class TestGetKfactor:

    def test_era_sample_match(self):
        scales = {"RunII": {"DYJets": 1.5}}
        assert get_kfactor(scales, "RunII", "DYJets") == 1.5

    def test_era_default_fallback(self):
        scales = {"RunII": {"_default": 1.2}}
        assert get_kfactor(scales, "RunII", "Unknown") == 1.2

    def test_all_sample_fallback(self):
        scales = {"ALL": {"DYJets": 1.3}}
        assert get_kfactor(scales, "FakeEra", "DYJets") == 1.3

    def test_all_default_fallback(self):
        scales = {"ALL": {"_default": 0.8}}
        assert get_kfactor(scales, "FakeEra", "FakeSample") == 0.8

    def test_final_default(self):
        assert get_kfactor({}, "any", "any", default=2.5) == 2.5

    def test_era_beats_all(self):
        scales = {"RunII": {"DYJets": 1.1}, "ALL": {"DYJets": 9.9}}
        assert get_kfactor(scales, "RunII", "DYJets") == 1.1

    def test_era_default_beats_all_sample(self):
        scales = {"RunII": {"_default": 2.0}, "ALL": {"DYJets": 9.9}}
        assert get_kfactor(scales, "RunII", "DYJets") == 2.0

    def test_returns_float(self):
        assert isinstance(get_kfactor({"ALL": {"DY": 2}}, "ALL", "DY"), float)


class TestLoadLumi:

    def test_list_eras_nonempty(self):
        assert len(list_eras()) > 0

    def test_known_era(self):
        info = load_lumi("RunIISummer20UL18")
        assert info["run"] == "RunII"
        assert info["year"] == "2018"
        assert isinstance(info["lumi"], (int, float))

    def test_unknown_era_raises(self):
        with pytest.raises(KeyError, match="not found"):
            load_lumi("NonExistentEra")


class TestLoadPlotSettings:

    def test_load_2022(self):
        cfg = load_plot_settings("2022")
        assert "common_variables" in cfg

    def test_common_vars_has_mass_fourobject(self):
        cfg = load_plot_settings("2022")
        assert "mass_fourobject" in cfg["common_variables"]
        assert "rebin" in cfg["common_variables"]["mass_fourobject"]

    def test_region_inherits_common(self):
        cfg = load_plot_settings("2022")
        region = cfg["wr_mumu_resolved_dy_cr"]
        assert "pt_leading_jet" in region

    def test_nonexistent_era_returns_empty(self):
        assert load_plot_settings("TotallyFakeEra") == {}


class TestIndexPlotSettings:

    def test_separates_common_and_regions(self):
        ps = {
            "common_variables": {"var1": {"rebin": 2}},
            "region_a": {"var2": {"rebin": 4}},
        }
        regions, common = index_plot_settings(ps)
        assert "common_variables" not in regions
        assert "region_a" in regions
        assert "var1" in common

    def test_region_overrides_common(self):
        regions = {"reg": {"var1": {"rebin": 10}}}
        common = {"var1": {"rebin": 2}}
        assert get_var_cfg(regions, common, "reg", "var1")["rebin"] == 10

    def test_falls_back_to_common(self):
        regions = {"reg": {}}
        common = {"var1": {"rebin": 2}}
        assert get_var_cfg(regions, common, "reg", "var1")["rebin"] == 2

    def test_returns_none_for_unknown(self):
        assert get_var_cfg({"reg": {}}, {}, "reg", "unknown") is None


class TestValidateVarCfg:

    def test_valid_keys_no_warning(self, caplog):
        validate_var_cfg("pt_leading_jet", {"rebin": 2, "xlim": [0, 100], "ylim": [1, 1e6]})
        assert "Unknown plot-setting key" not in caplog.text

    def test_typo_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            validate_var_cfg("mass_fourobject", {"rebin_varaible": [0, 100, 200]}, "my_region")
        assert "rebin_varaible" in caplog.text
        assert "my_region" in caplog.text

    def test_all_valid_keys_accepted(self, caplog):
        cfg = {"rebin": 2, "rebin_variable": [0, 50, 100], "xlim": [0, 100], "ylim": [1, 1e6], "ratio_ylim": [0.5, 1.5]}
        validate_var_cfg("test_var", cfg)
        assert "Unknown plot-setting key" not in caplog.text


class TestLoadKfactors:

    def test_loads_without_error(self):
        kf = load_kfactors()
        assert isinstance(kf, dict)
