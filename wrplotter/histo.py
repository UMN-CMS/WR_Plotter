from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Callable

import uproot

from .histogram_utils import rebin_histogram

logger = logging.getLogger(__name__)


@lru_cache(maxsize=64)
def _open_root(path: str):
    """Cache open ROOT file handles for reuse across histogram reads."""
    return uproot.open(path)


def clear_cache():
    """Release all cached ROOT file handles."""
    _open_root.cache_clear()


def load_and_rebin(
    input_dirs: Sequence[Path],
    sample: str,
    hist_key: str,
    n_rebin,
    sublumis: Sequence[float],
    era_for_scale: str,
    get_kfactor_fn: Callable[[dict, str, str, float], float],
    scales: dict,
):
    combined = None

    for indir, sublumi in zip(input_dirs, sublumis):
        fp = indir / f"WRAnalyzer_{sample}.root"
        try:
            f = _open_root(str(fp))
            raw_hist = f[hist_key].to_hist()
        except (FileNotFoundError, KeyError, OSError) as exc:
            logger.debug("Skipping %s [%s]: %s", fp, hist_key, exc)
            continue

        rebinned = rebin_histogram(raw_hist, n_rebin)

        # per-era per-sample k-factor
        era_for_scale_eff = indir.name if indir.name in scales else era_for_scale
        k = get_kfactor_fn(scales, era_for_scale_eff, sample, default=1.0)
        rebinned = rebinned * k

        combined = rebinned if combined is None else (combined + rebinned)

    return combined
