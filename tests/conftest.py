"""Shared fixtures for wrplotter tests."""
import numpy as np
import pytest
from hist import Hist
from hist.axis import Regular
from hist.storage import Weight


@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Clear @lru_cache on config functions after each test."""
    yield
    from wrplotter.config import (
        _lumi_table, load_plot_settings, load_sample_groups_raw, load_kfactors,
        load_region_shorthands, load_systematics,
    )
    from wrplotter.variables import build_variables
    for fn in (
        _lumi_table, load_plot_settings, load_sample_groups_raw, load_kfactors,
        load_region_shorthands, load_systematics, build_variables,
    ):
        fn.cache_clear()


@pytest.fixture
def uniform_hist_10bins():
    """1D Weight() histogram: 10 bins over [0, 100], values [10, 20, ..., 100]."""
    h = Hist(Regular(10, 0, 100, name="x"), storage=Weight())
    view = h.view(flow=False)
    view["value"] = np.array([(i + 1) * 10.0 for i in range(10)])
    view["variance"] = view["value"].copy()
    return h


@pytest.fixture
def hist_with_overflow():
    """1D Weight() histogram: 5 bins over [0, 50], overflow=100."""
    h = Hist(Regular(5, 0, 50, name="x"), storage=Weight())
    view = h.view(flow=False)
    view["value"] = [10, 20, 30, 40, 50]
    view["variance"] = [10, 20, 30, 40, 50]
    view_flow = h.view(flow=True)
    view_flow[-1] = (100.0, 100.0)
    return h


@pytest.fixture
def variable_bin_edges():
    """Variable-width bin edges for rebinning tests."""
    return [0, 20, 50, 100]
