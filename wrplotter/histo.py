from __future__ import annotations
from pathlib import Path
from typing import Sequence, Callable

import uproot

from .histogram_utils import rebin_histogram


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
            with uproot.open(fp) as f:
                raw_hist = f[hist_key].to_hist()
        except (FileNotFoundError, KeyError):
            continue

        rebinned = rebin_histogram(raw_hist, n_rebin)

        # per-era per-sample k-factor
        era_for_scale_eff = indir.name if indir.name in scales else era_for_scale
        k = get_kfactor_fn(scales, era_for_scale_eff, sample, default=1.0)
        rebinned = rebinned * k

        combined = rebinned if combined is None else (combined + rebinned)

    return combined
