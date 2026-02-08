"""Tests for wrplotter.cli_utils."""
from wrplotter.cli_utils import parse_multi


class TestParseMulti:

    def test_none_input(self):
        assert parse_multi(None) is None

    def test_empty_list(self):
        assert parse_multi([]) is None

    def test_single_item(self):
        assert parse_multi(["foo"]) == ["foo"]

    def test_comma_separated(self):
        assert parse_multi(["foo,bar,baz"]) == ["foo", "bar", "baz"]

    def test_multiple_append_calls(self):
        assert parse_multi(["foo", "bar"]) == ["foo", "bar"]

    def test_mixed_commas_and_appends(self):
        assert parse_multi(["foo,bar", "baz"]) == ["foo", "bar", "baz"]

    def test_deduplication(self):
        assert parse_multi(["foo,bar,foo"]) == ["foo", "bar"]

    def test_dedup_preserves_order(self):
        assert parse_multi(["c,a,b,a,c"]) == ["c", "a", "b"]

    def test_whitespace_stripping(self):
        assert parse_multi(["foo , bar , baz"]) == ["foo", "bar", "baz"]

    def test_empty_commas_ignored(self):
        assert parse_multi([",foo,,bar,"]) == ["foo", "bar"]

    def test_all_empty_returns_none(self):
        assert parse_multi([",,", "  ,  "]) is None
