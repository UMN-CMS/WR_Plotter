"""Tests for wrplotter.histogram_cache."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import hist

from wrplotter.histogram_cache import (load_and_rebin, clear_cache, _open_root, keys_in_file,
                                      read_hist)


def _make_test_hist(values=None, bins=10, lo=0, hi=100):
    """Create a simple 1D hist.Hist for testing."""
    h = hist.Hist(hist.axis.Regular(bins, lo, hi))
    if values is not None:
        h.view()[:] = values
    else:
        h.fill(np.random.uniform(lo, hi, 100))
    return h


@pytest.fixture(autouse=True)
def _clear_root_cache():
    """Clear the ROOT file cache before and after each test."""
    clear_cache()
    yield
    clear_cache()


class TestLoadAndRebin:

    def test_returns_none_when_file_missing(self, tmp_path):
        result = load_and_rebin(
            input_dirs=[tmp_path],
            sample="NonExistent",
            hist_key="region/variable_region",
            n_rebin=1,
        )
        assert result is None

    def test_loads_and_rebins(self, tmp_path):
        """Test with a real ROOT file written via uproot."""
        uproot = pytest.importorskip("uproot")

        # Create a test histogram
        test_hist = _make_test_hist(values=np.ones(10), bins=10, lo=0, hi=100)

        # Write to a ROOT file
        root_path = tmp_path / "WRAnalyzer_TestSample.root"
        with uproot.recreate(root_path) as f:
            f["region/variable_region"] = test_hist

        result = load_and_rebin(
            input_dirs=[tmp_path],
            sample="TestSample",
            hist_key="region/variable_region",
            n_rebin=2,
        )

        assert result is not None
        # rebinned by 2: 10 bins -> 5 bins
        assert len(result.axes[0]) == 5


    def test_combines_multiple_dirs(self, tmp_path):
        """Test that histograms from multiple input dirs are summed."""
        uproot = pytest.importorskip("uproot")

        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        test_hist = _make_test_hist(values=np.ones(10) * 3.0, bins=10, lo=0, hi=100)

        for d in [dir1, dir2]:
            with uproot.recreate(d / "WRAnalyzer_Sample.root") as f:
                f["region/variable_region"] = test_hist

        result = load_and_rebin(
            input_dirs=[dir1, dir2],
            sample="Sample",
            hist_key="region/variable_region",
            n_rebin=1,
        )

        assert result is not None
        np.testing.assert_allclose(result.values(), np.ones(10) * 6.0)

    def test_missing_hist_key_returns_none(self, tmp_path):
        """Test that a missing histogram key in an existing file returns None."""
        uproot = pytest.importorskip("uproot")

        test_hist = _make_test_hist(values=np.ones(10), bins=10, lo=0, hi=100)
        root_path = tmp_path / "WRAnalyzer_Sample.root"
        with uproot.recreate(root_path) as f:
            f["region/variable_region"] = test_hist

        result = load_and_rebin(
            input_dirs=[tmp_path],
            sample="Sample",
            hist_key="nonexistent/key",
            n_rebin=1,
        )
        assert result is None


class TestKeyChecks:
    """Reads go through the file's key list; missing and unreadable files are told apart."""

    @staticmethod
    def _write(path, keys):
        uproot = pytest.importorskip("uproot")
        with uproot.recreate(path) as f:
            for key in keys:
                f[key] = _make_test_hist(values=np.ones(10))
        return path

    def test_keys_in_file_lists_folders_and_histograms(self, tmp_path):
        path = self._write(tmp_path / "a.root", ["region/var_region"])
        assert {"region", "region/var_region"} <= keys_in_file(str(path))

    def test_read_hist_missing_key_or_file_is_none(self, tmp_path):
        path = self._write(tmp_path / "a.root", ["region/var_region"])
        assert read_hist(path, "region/var_region") is not None
        assert read_hist(path, "region/missing_region") is None
        assert read_hist(path, "missing_folder/var_missing_folder") is None
        assert read_hist(tmp_path / "absent.root", "region/var_region") is None

    def test_unreadable_file_raises(self, tmp_path):
        # It used to be skipped like a missing file, silently dropping an era from the sum.
        (tmp_path / "WRAnalyzer_Sample.root").write_bytes(b"not a ROOT file")
        with pytest.raises((ValueError, OSError)):
            load_and_rebin(input_dirs=[tmp_path], sample="Sample",
                           hist_key="region/var_region", n_rebin=1)

    def test_key_missing_in_one_era_is_reported_once(self, tmp_path, caplog):
        era1, era2 = tmp_path / "era1", tmp_path / "era2"
        era1.mkdir()
        era2.mkdir()
        self._write(era1 / "WRAnalyzer_Sample.root", ["region/var_region"])
        self._write(era2 / "WRAnalyzer_Sample.root", ["other/var_other"])
        kw = dict(input_dirs=[era1, era2], sample="Sample", hist_key="region/var_region", n_rebin=1)
        with caplog.at_level("WARNING"):
            h = load_and_rebin(**kw)
            load_and_rebin(**kw)
        np.testing.assert_allclose(h.values(), np.ones(10))
        assert caplog.text.count("has no region/var_region") == 1
        assert str(era2 / "WRAnalyzer_Sample.root") in caplog.text

    def test_key_missing_everywhere_is_quiet(self, tmp_path, caplog):
        self._write(tmp_path / "WRAnalyzer_Sample.root", ["other/var_other"])
        with caplog.at_level("WARNING"):
            assert load_and_rebin(input_dirs=[tmp_path], sample="Sample",
                                  hist_key="region/var_region", n_rebin=1) is None
        assert caplog.text == ""


class TestRootFileCache:

    def test_cache_reuses_file_handle(self, tmp_path):
        """Verify that reading the same file twice hits the cache."""
        uproot = pytest.importorskip("uproot")

        test_hist = _make_test_hist(values=np.ones(10), bins=10, lo=0, hi=100)
        root_path = tmp_path / "WRAnalyzer_CacheSample.root"
        with uproot.recreate(root_path) as f:
            f["region/variable_region"] = test_hist

        kw = dict(
            input_dirs=[tmp_path],
            sample="CacheSample",
            hist_key="region/variable_region",
            n_rebin=1,
        )

        clear_cache()
        load_and_rebin(**kw)
        load_and_rebin(**kw)

        info = _open_root.cache_info()
        assert info.hits >= 1, f"Expected cache hits but got: {info}"

    def test_clear_cache_resets(self, tmp_path):
        """Verify clear_cache() empties the LRU cache."""
        uproot = pytest.importorskip("uproot")

        test_hist = _make_test_hist(values=np.ones(10), bins=10, lo=0, hi=100)
        root_path = tmp_path / "WRAnalyzer_ClearSample.root"
        with uproot.recreate(root_path) as f:
            f["region/variable_region"] = test_hist

        load_and_rebin(
            input_dirs=[tmp_path],
            sample="ClearSample",
            hist_key="region/variable_region",
            n_rebin=1,
        )

        clear_cache()
        info = _open_root.cache_info()
        assert info.currsize == 0
