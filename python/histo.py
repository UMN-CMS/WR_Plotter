# python/histio.py
from __future__ import annotations
from pathlib import Path
from typing import Sequence, Callable, Any, Optional
import uproot

def load_and_rebin(
    input_dirs: Sequence[Path],
    sample: str,
    hist_key: str,
    plotter: Any,              # needs: .rebin_hist(H), .scale_hist(H), .lumi
    is_data_group: bool,
    sublumis: Sequence[float],
    era_for_scale: str,
    get_kfactor_fn: Callable[[dict, str, str, float], float],
    scales: dict,
):
    combined = None
    original_lumi = plotter.lumi

    for indir, sublumi in zip(input_dirs, sublumis):
        fp = indir / f"WRAnalyzer_{sample}.root"
        try:
            with uproot.open(fp) as f:
                raw_hist = f[hist_key].to_hist()
        except (FileNotFoundError, KeyError):
            continue


        # Rebin first
#        if "mass_fourobject" in hist_key:
#            variable_edges = [0, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000]
#            rebinned = plotter.rebin_hist(raw_hist, variable_edges)
#        elif "pt_leading_jet" in hist_key:
#            variable_edges = [0, 40, 100, 200, 400, 600, 800, 1000, 1500, 2000]
#            rebinned = plotter.rebin_hist(raw_hist, variable_edges)
#        elif "mass_dijet" in hist_key:
#            variable_edges = [0, 200, 400, 600, 800, 1000, 1250, 1500, 2000, 4000]
#            rebinned = plotter.rebin_hist(raw_hist, variable_edges)
#        else:

        rebinned = plotter.rebin_hist(raw_hist)

        if not is_data_group:
            plotter.lumi = sublumi
            rebinned = plotter.scale_hist(rebinned)

        # per-era per-sample k-factor
        era_for_scale_eff = indir.name if indir.name in scales else era_for_scale
        k = get_kfactor_fn(scales, era_for_scale_eff, sample, default=1.0)
        rebinned = rebinned * k

        combined = rebinned if combined is None else (combined + rebinned)

    plotter.lumi = original_lumi
    return combined
