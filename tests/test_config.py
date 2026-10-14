"""Tests for wrplotter.config."""
import pytest
from wrplotter.config import (
    _deep_merge,
    list_eras,
    load_lumi,
    load_plot_settings,
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

    def test_boosted_region_does_not_fall_back_to_common(self):
        # common_variables holds resolved-topology variables (AK4 jets, 4-object mass, ...)
        common = {"mass_fourobject": {"rebin": 2}}
        assert get_var_cfg({}, common, "wr_ee_boosted_sr", "mass_fourobject") is None
        assert get_var_cfg({}, common, "wr_ee_resolved_sr", "mass_fourobject")["rebin"] == 2

    @pytest.mark.parametrize("era", list_eras())
    def test_boosted_regions_plan_only_their_own_variables(self, era):
        # The resolved-only common variables used to be planned for every boosted region:
        # 84 plots per stack-plot run that the analyzer never fills.
        regions, common = index_plot_settings(load_plot_settings(era))
        for name, block in regions.items():
            if "boosted" in name:
                for var in set(common) - set(block):
                    assert get_var_cfg(regions, common, name, var) is None, (name, var)


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



class TestCombinations:
    """Years and Run 3 combine eras; their luminosity is derived from the eras."""

    @pytest.mark.parametrize("era, sub_eras", [
        ("2022", ["Run3Summer22", "Run3Summer22EE"]),
        ("2023", ["Run3Summer23", "Run3Summer23BPix"]),
        ("2024", ["RunIII2024Summer24"]),
        ("Run3", ["Run3Summer22", "Run3Summer22EE", "Run3Summer23", "Run3Summer23BPix",
                  "RunIII2024Summer24"]),
    ])
    def test_lumi_is_the_sum_of_the_eras(self, era, sub_eras):
        info = load_lumi(era)
        assert info["sub_eras"] == sub_eras
        lumis = [load_lumi(e)["lumi"] for e in sub_eras]
        assert info["sub_lumis"] == pytest.approx(lumis)
        assert info["lumi"] == pytest.approx(sum(lumis))
        assert info["com"] == 13.6

    @pytest.mark.parametrize("entry, match", [
        ({"lumi": 99.0}, "not the sum"),
        ({"sub_lumis": [1.0, 2.5]}, "disagree"),
        ({"sub_eras": ["A", "Combo"]}, "lists the combination"),
        ({"sub_eras": ["A", "C"]}, "centre-of-mass"),
    ])
    def test_inconsistent_combination_refused(self, monkeypatch, entry, match):
        import wrplotter.config as config
        table = {
            "A": {"run": "Run3", "year": "2022", "lumi": 1.0, "com": 13.6},
            "B": {"run": "Run3", "year": "2022", "lumi": 2.0, "com": 13.6},
            "C": {"run": "RunII", "year": "2018", "lumi": 3.0, "com": 13},
            "Combo": {"run": "Run3", "year": "2022", "sub_eras": ["A", "B"]},
        }
        table["X"] = {"run": "Run3", "year": "x", "sub_eras": ["A", "B"], **entry}
        monkeypatch.setattr(config, "_lumi_table", lambda: table)
        with pytest.raises(ValueError, match=match):
            config.load_lumi("X")

    def test_combination_borrows_plot_settings(self):
        assert load_plot_settings("Run3") == load_plot_settings("RunIII2024Summer24")
        assert load_plot_settings("2024") == load_plot_settings("RunIII2024Summer24")
        # 2022 and 2023 have no file and no alias: the default settings.
        assert load_plot_settings("2022") == load_plot_settings("2023") == load_plot_settings("default")
        # The single Run 3 eras share one override file.
        assert load_plot_settings("Run3Summer22EE") == load_plot_settings("run3_subera")

    def test_era_label(self):
        from wrplotter.config import era_label
        assert era_label("Run3") == "Run 3"
        assert era_label("2022") == "2022"
        assert era_label("NotAnEra") == "NotAnEra"



class TestPlotSettingsInheritance:
    """An era file lists only what differs from the file it inherits."""

    @pytest.fixture
    def settings_dir(self, tmp_path, monkeypatch):
        import wrplotter.config as config
        (tmp_path / "plot_settings").mkdir()
        monkeypatch.setattr(config, "data_path", lambda *parts: tmp_path.joinpath(*parts))
        monkeypatch.setattr(config, "_lumi_table", lambda: {
            "EraA": {"run": "Run3", "year": "x", "lumi": 1.0},
            "EraB": {"run": "Run3", "year": "x", "lumi": 1.0, "plot_settings": "shared"},
        })
        config.load_plot_settings.cache_clear()
        yield tmp_path / "plot_settings"
        config.load_plot_settings.cache_clear()

    def _write(self, d, name, text):
        (d / f"{name}.yaml").write_text(text)

    def test_override_merges_per_key(self, settings_dir):
        import wrplotter.config as config
        self._write(settings_dir, "default", "r1:\n  v: { rebin: 2, xlim: [0, 10], ylim: [1, 100] }\n"
                                             "r2:\n  v: { rebin: 5 }\n")
        self._write(settings_dir, "shared", "inherit: default\nr1:\n  v: { ylim: [0.1, 10] }\n")
        cfg = config.load_plot_settings("EraB")
        assert cfg["r1"]["v"] == {"rebin": 2, "xlim": [0, 10], "ylim": [0.1, 10]}
        assert cfg["r2"]["v"] == {"rebin": 5}
        assert "inherit" not in cfg

    def test_era_without_file_or_alias_uses_default(self, settings_dir):
        import wrplotter.config as config
        self._write(settings_dir, "default", "r1:\n  v: { rebin: 2 }\n")
        assert config.load_plot_settings("EraA") == {"r1": {"v": {"rebin": 2}}}
        assert config.load_plot_settings("NotAnEra") == {}

    def test_own_file_wins_over_alias(self, settings_dir):
        import wrplotter.config as config
        self._write(settings_dir, "default", "r1:\n  v: { rebin: 2 }\n")
        self._write(settings_dir, "shared", "inherit: default\n")
        self._write(settings_dir, "EraB", "inherit: default\nr1:\n  v: { rebin: 7 }\n")
        assert config.load_plot_settings("EraB")["r1"]["v"]["rebin"] == 7

    def test_inherit_cycle_and_missing_parent_raise(self, settings_dir):
        import wrplotter.config as config
        self._write(settings_dir, "default", "inherit: shared\n")
        self._write(settings_dir, "shared", "inherit: default\n")
        with pytest.raises(ValueError, match="cycle"):
            config.load_plot_settings("EraB")
        self._write(settings_dir, "EraA", "inherit: nowhere\n")
        with pytest.raises(FileNotFoundError, match="nowhere"):
            config.load_plot_settings("EraA")


class TestDYComparisons:
    """data/dy_comparisons.yaml: the named comparisons and plot sets of bin/compare_dy.py."""

    def test_comparisons_name_known_samples_reference_first(self):
        from wrplotter.config import load_dy_comparisons, load_dy_samples
        cfg = load_dy_comparisons()
        known = load_dy_samples()["samples"]
        assert cfg["comparisons"]["lo"][0] == "lo_ht"
        assert cfg["comparisons"]["nlo"][0] == "nlo_inc"
        for tags in cfg["comparisons"].values():
            assert len(tags) >= 2 and all(t in known for t in tags)

    def test_core_plots_exist_in_the_plot_settings(self):
        """Every (region, variable) of the core set is configured for 2024, so none is skipped."""
        from wrplotter.config import (index_plot_settings, load_dy_comparisons, load_plot_settings,
                                      configured_variables)
        from wrplotter.regions import expand_region_requests
        from wrplotter.variables import build_variables
        region_cfgs, common_vars = index_plot_settings(load_plot_settings("RunIII2024Summer24"))
        variables = build_variables()
        for regions, names in load_dy_comparisons()["plot_sets"]["core"].items():
            for region in expand_region_requests("RunIII2024Summer24", [regions]):
                configured = {v.name for v, _ in configured_variables(region_cfgs, common_vars,
                                                                      region.name, variables)}
                assert set(names) <= configured, (region.name, set(names) - configured)

    def test_bad_comparison_refused(self, tmp_path, monkeypatch):
        import shutil
        from wrplotter import config
        shutil.copy(config.data_path("dy_samples.yaml"), tmp_path / "dy_samples.yaml")
        (tmp_path / "dy_comparisons.yaml").write_text("comparisons:\n  x: {samples: [lo_ht, nope]}\n")
        monkeypatch.setattr(config, "data_path", lambda *parts: tmp_path.joinpath(*parts))
        config.load_dy_comparisons.cache_clear()
        try:
            with pytest.raises(ValueError, match="unknown DY samples"):
                config.load_dy_comparisons()
        finally:
            config.load_dy_comparisons.cache_clear()


class TestRegionDefaults:
    """A region's `_all` plot-settings entry applies to every variable it plots."""

    def test_all_entry_merged_under_variable_keys(self):
        from wrplotter.config import get_var_cfg
        regions = {"wr_x_resolved_dy_cr": {"_all": {"ratio_ylim": [0.5, 2.5], "rebin": 7},
                                           "pt": {"rebin": 2, "xlim": [0, 10]}}}
        cfg = get_var_cfg(regions, {"eta": {"rebin": 5}}, "wr_x_resolved_dy_cr", "pt")
        assert cfg == {"ratio_ylim": [0.5, 2.5], "rebin": 2, "xlim": [0, 10]}
        common = get_var_cfg(regions, {"eta": {"rebin": 5}}, "wr_x_resolved_dy_cr", "eta")
        assert common == {"ratio_ylim": [0.5, 2.5], "rebin": 5}
        assert get_var_cfg(regions, {}, "wr_x_resolved_dy_cr", "_absent") is None

    def test_2024_ranges(self):
        from wrplotter.config import get_var_cfg, index_plot_settings, load_plot_settings
        rc, cv = index_plot_settings(load_plot_settings("RunIII2024Summer24"))
        assert get_var_cfg(rc, cv, "wr_mumu_resolved_dy_cr", "pt_leading_jet")["ratio_ylim"] == [0.5, 2.5]
        assert get_var_cfg(rc, cv, "wr_resolved_flavor_cr", "pt_leading_jet")["ratio_ylim"] == [0, 2]
        # boosted SR and flavor CRs select m_ll > 200 GeV: the axis starts there
        assert get_var_cfg(rc, cv, "wr_ee_boosted_sr", "mass_dilepton")["xlim"][0] == 200
        assert get_var_cfg(rc, cv, "wr_emu_boosted_flavor_cr", "mass_dilepton")["xlim"][0] == 200
