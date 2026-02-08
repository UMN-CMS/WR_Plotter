"""Tests for wrplotter.sample_groups."""
import pytest
from wrplotter.sample_groups import SampleGroup, _mk_group, load_sample_groups


class TestMkGroup:

    def test_basic_group(self):
        g = _mk_group("dy", {
            "color": "#5790fc",
            "tlatex_alias": "DY+Jets",
            "samples": ["DYJets_NLO"],
        }, {"run": "Run3", "year": 2024, "mc_campaign": "Summer24"})
        assert g.key == "dy"
        assert g.color == "#5790fc"
        assert g.tlatex_alias == "DY+Jets"
        assert g.samples == ["DYJets_NLO"]
        assert g.kind == "mc"
        assert g.run == "Run3"
        assert g.year == 2024

    def test_data_kind(self):
        g = _mk_group("egamma", {
            "color": "#000000",
            "samples": ["EGamma"],
            "kind": "data",
        }, {})
        assert g.kind == "data"

    def test_missing_color_raises(self):
        with pytest.raises(ValueError, match="missing required field 'color'"):
            _mk_group("bad", {"samples": ["x"]}, {})

    def test_label_fallback_to_key(self):
        g = _mk_group("mykey", {"color": "#fff", "samples": ["s"]}, {})
        assert g.tlatex_alias == "mykey"

    def test_label_from_label_field(self):
        g = _mk_group("k", {"color": "#fff", "samples": ["s"], "label": "My Label"}, {})
        assert g.tlatex_alias == "My Label"

    def test_tlatex_alias_takes_priority(self):
        g = _mk_group("k", {
            "color": "#fff", "samples": ["s"],
            "label": "Label", "tlatex_alias": "TLatex",
        }, {})
        assert g.tlatex_alias == "TLatex"


class TestLoadSampleGroups:

    def test_returns_tuple(self):
        result = load_sample_groups("RunIII2024Summer24")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_groups_is_dict(self):
        groups, order = load_sample_groups("RunIII2024Summer24")
        assert isinstance(groups, dict)
        for key, sg in groups.items():
            assert isinstance(sg, SampleGroup)

    def test_order_is_list(self):
        groups, order = load_sample_groups("RunIII2024Summer24")
        assert isinstance(order, list)
        for key in order:
            assert key in groups

    def test_has_data_group(self):
        groups, _ = load_sample_groups("RunIII2024Summer24")
        data_groups = [g for g in groups.values() if g.kind == "data"]
        assert len(data_groups) > 0
