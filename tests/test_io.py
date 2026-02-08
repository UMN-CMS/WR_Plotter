"""Tests for wrplotter.io."""
import os
from pathlib import Path
from unittest.mock import patch

from wrplotter.io import (
    repo_root,
    data_path,
    resolve_eos_user,
    eos_endpoint,
    eos_base,
    is_eos_path,
)


class TestRepoRoot:

    def test_returns_path(self):
        assert isinstance(repo_root(), Path)

    def test_contains_wrplotter(self):
        assert (repo_root() / "wrplotter").is_dir()

    def test_contains_data(self):
        assert (repo_root() / "data").is_dir()


class TestDataPath:

    def test_returns_path(self):
        assert isinstance(data_path("lumi.json"), Path)

    def test_under_repo_root(self):
        p = data_path("lumi.json")
        assert str(p).startswith(str(repo_root()))

    def test_lumi_json_exists(self):
        assert data_path("lumi.json").is_file()


class TestResolveEosUser:

    def test_eosuser_path_takes_priority(self):
        with patch.dict(os.environ, {"EOSUSER_PATH": "w/wijackso", "EOSUSER": "other"}, clear=False):
            assert resolve_eos_user() == "w/wijackso"

    def test_eosuser_builds_segment(self):
        env = {"EOSUSER": "wijackso"}
        with patch.dict(os.environ, env, clear=False):
            # Remove EOSUSER_PATH if set
            os.environ.pop("EOSUSER_PATH", None)
            assert resolve_eos_user() == "w/wijackso"

    def test_falls_back_to_user(self):
        env = {"USER": "testuser"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("EOSUSER_PATH", None)
            os.environ.pop("EOSUSER", None)
            assert resolve_eos_user() == "t/testuser"

    def test_strips_slashes(self):
        with patch.dict(os.environ, {"EOSUSER_PATH": "/w/wijackso/"}, clear=False):
            assert resolve_eos_user() == "w/wijackso"


class TestEosEndpoint:

    def test_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EOS_ENDPOINT", None)
            assert eos_endpoint() == "eosuser.cern.ch"

    def test_override(self):
        with patch.dict(os.environ, {"EOS_ENDPOINT": "custom.host"}, clear=False):
            assert eos_endpoint() == "custom.host"


class TestEosBase:

    def test_default_path(self):
        with patch.dict(os.environ, {"EOSUSER": "wijackso"}, clear=False):
            os.environ.pop("EOSUSER_PATH", None)
            os.environ.pop("EOS_BASE", None)
            p = eos_base("Run3", "2024", "RunIII2024Summer24")
            assert p == Path("/eos/user/w/wijackso/Run3/2024/RunIII2024Summer24")

    def test_with_subdir(self):
        with patch.dict(os.environ, {"EOSUSER": "wijackso"}, clear=False):
            os.environ.pop("EOSUSER_PATH", None)
            os.environ.pop("EOS_BASE", None)
            p = eos_base("Run3", "2024", "RunIII2024Summer24", "mydir")
            assert p == Path("/eos/user/w/wijackso/Run3/2024/RunIII2024Summer24/mydir")

    def test_eos_base_override(self):
        with patch.dict(os.environ, {"EOS_BASE": "/eos/user/x/xuser"}, clear=False):
            p = eos_base("Run3", "2024", "era1")
            assert p == Path("/eos/user/x/xuser/Run3/2024/era1")


class TestIsEosPath:

    def test_eos_path(self):
        assert is_eos_path("/eos/user/w/wijackso/foo") is True

    def test_local_path(self):
        assert is_eos_path("/home/user/plots/foo.pdf") is False

    def test_path_object(self):
        assert is_eos_path(Path("/eos/stuff")) is True
