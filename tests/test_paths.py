"""Tests for wrplotter.paths."""
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

from wrplotter.paths import (
    repo_root,
    data_path,
    resolve_eos_user,
    eos_endpoint,
    eos_base,
    is_eos_path,
    eos_mkdir_p,
    eos_upload,
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
        assert isinstance(data_path("lumi.yaml"), Path)

    def test_under_repo_root(self):
        p = data_path("lumi.yaml")
        assert str(p).startswith(str(repo_root()))

    def test_lumi_yaml_exists(self):
        assert data_path("lumi.yaml").is_file()


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


class TestEosMkdirP:

    def test_calls_xrdfs_with_correct_args(self):
        """eos_mkdir_p passes the right command to the runner."""
        mock_runner = MagicMock()
        with patch.dict(os.environ, {"EOS_ENDPOINT": "eosuser.cern.ch"}, clear=False):
            eos_mkdir_p("/eos/user/w/wijackso/plots", _runner=mock_runner)

        mock_runner.assert_called_once()
        cmd = mock_runner.call_args[0][0]
        assert cmd[0] == "xrdfs"
        assert cmd[1] == "eosuser.cern.ch"
        assert "mkdir" in cmd
        assert "/eos/user/w/wijackso/plots" in cmd

    def test_uses_custom_endpoint(self):
        mock_runner = MagicMock()
        with patch.dict(os.environ, {"EOS_ENDPOINT": "myhost.cern.ch"}, clear=False):
            eos_mkdir_p("/eos/some/path", _runner=mock_runner)

        cmd = mock_runner.call_args[0][0]
        assert cmd[1] == "myhost.cern.ch"

    def test_passes_timeout(self):
        mock_runner = MagicMock()
        eos_mkdir_p("/eos/some/path", timeout=30, _runner=mock_runner)
        assert mock_runner.call_args[1]["timeout"] == 30

    def test_timeout_raises_runtime_error(self):
        """A TimeoutExpired from xrdfs is converted to a RuntimeError with guidance."""
        def _timedout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))

        import pytest
        with pytest.raises(RuntimeError, match="timed out"):
            eos_mkdir_p("/eos/some/path", timeout=1, _runner=_timedout)

    def test_called_process_error_raises_runtime_error(self):
        """A CalledProcessError from xrdfs is converted to a RuntimeError."""
        def _failed(*args, **kwargs):
            raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

        import pytest
        with pytest.raises(RuntimeError, match="failed"):
            eos_mkdir_p("/eos/some/path", _runner=_failed)


class TestEosUpload:

    def test_calls_xrdcp_with_correct_args(self):
        """eos_upload passes the right command to the runner."""
        mock_runner = MagicMock()
        with patch.dict(os.environ, {"EOS_ENDPOINT": "eosuser.cern.ch"}, clear=False):
            eos_upload("/tmp/plot.pdf", "/eos/user/w/wijackso/plot.pdf",
                       _runner=mock_runner)

        mock_runner.assert_called_once()
        cmd = mock_runner.call_args[0][0]
        assert cmd[0] == "xrdcp"
        assert "/tmp/plot.pdf" in cmd
        assert "root://eosuser.cern.ch//eos/user/w/wijackso/plot.pdf" in cmd

    def test_passes_timeout(self):
        mock_runner = MagicMock()
        eos_upload("/tmp/f.pdf", "/eos/dest.pdf", timeout=60, _runner=mock_runner)
        assert mock_runner.call_args[1]["timeout"] == 60

    def test_timeout_raises_runtime_error(self):
        def _timedout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 0))

        import pytest
        with pytest.raises(RuntimeError, match="timed out"):
            eos_upload("/tmp/f.pdf", "/eos/dest.pdf", timeout=1, _runner=_timedout)

    def test_called_process_error_raises_runtime_error(self):
        """A CalledProcessError from xrdcp is converted to a RuntimeError."""
        def _failed(*args, **kwargs):
            raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

        import pytest
        with pytest.raises(RuntimeError, match="failed"):
            eos_upload("/tmp/f.pdf", "/eos/dest.pdf", _runner=_failed)
