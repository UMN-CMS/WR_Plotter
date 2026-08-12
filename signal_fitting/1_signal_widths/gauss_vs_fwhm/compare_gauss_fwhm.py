#!/usr/bin/env python3
"""Compare the two window-robust widths: σ_gauss^on vs σ_FWHM^on.

Reads the baseline-window rows of gauss_fit/gauss_fit_table.csv and
fwhm/fwhm_table.csv and, per (channel, topology), writes:

  sigma_over_mWR_{channel}_{topology}.{png,pdf}
      σ_gauss^on/M_WR and σ_FWHM^on/M_WR overlaid vs x = M_N/M_WR.
  ratio_fwhm_over_gauss_{channel}_{topology}.{png,pdf}
      σ_FWHM^on / σ_gauss^on vs x, colored by M_WR.

Output: 1_signal_widths/gauss_vs_fwhm/  (reads the sibling gaussian/ & fwhm/ tables)

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from shape_estimators import plot_series_vs_x, plot_scalar_vs_x_by_mwr  # noqa: E402

logger = logging.getLogger(__name__)
BASE = "[0.8,1.2]"
GAUSS = r"\sigma_{\rm gauss}^{\rm on}"
FWHM = r"\sigma_{\rm FWHM}^{\rm on}"


def centered_ylim(vals, frac=0.30, include=None):
    """y-range centered on the full data range; optionally force `include` in view."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = float(np.min(v)), float(np.max(v))
    if include is not None:
        lo, hi = min(lo, include), max(hi, include)
    m = (hi - lo) * frac
    return (lo - m, hi + m)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--base-dir", type=Path,
                   default=Path(__file__).resolve().parents[1],
                   help="Stage-1 dir holding gaussian/ and fwhm/ tables; "
                        "gauss_vs_fwhm/ outputs are written under it.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    com = load_lumi(args.era).get("com", 13.0)

    g = pd.read_csv(args.base_dir / "gaussian" / "gauss_fit_table.csv")
    f = pd.read_csv(args.base_dir / "fwhm" / "fwhm_table.csv")
    g = g[g.fit_range == BASE][["channel", "category", "mWR", "mN",
                                "sigma_gaus_over_mWR"]]
    f = f[f.fit_range == BASE][["channel", "category", "mWR", "mN",
                                "sigma_fwhm_over_mWR"]]
    df = g.merge(f, on=["channel", "category", "mWR", "mN"])
    df["x"] = df["mN"] / df["mWR"]
    df["M_WR"] = df["mWR"]
    df["ratio"] = df["sigma_fwhm_over_mWR"] / df["sigma_gaus_over_mWR"]
    out_dir = args.base_dir / "gauss_vs_fwhm"
    # One shared y-range for all four ratio panels (includes the line at 1).
    ratio_ylim = centered_ylim(df["ratio"], include=1.0)

    print("\n=== σ_FWHM^on / σ_gauss^on (median over cells) ===")
    for channel in ["ee", "mumu"]:
        for topology in ["resolved", "boosted"]:
            sub = df[(df.channel == channel) & (df.category == topology)]
            if sub.empty:
                continue
            sub = sub.sort_values("x")
            plot_series_vs_x(
                sub,
                [("sigma_gaus_over_mWR", rf"${GAUSS} / M_{{W_R}}$", "#bd1f01"),
                 ("sigma_fwhm_over_mWR", rf"${FWHM} / M_{{W_R}}$", "#3f90da")],
                out_dir / "sigma_over_mWR" / f"{channel}_{topology}.png",
                channel=channel, topology=topology, era=args.era, com=com,
                ylabel=r"$\sigma^{\rm on} / M_{W_R}$",
                legend_loc="upper left", legend_bbox=(0.02, 0.83))
            plot_scalar_vs_x_by_mwr(
                sub, "ratio",
                out_dir / "ratio_fwhm_over_gauss" / f"{channel}_{topology}.png",
                channel=channel, topology=topology, era=args.era, com=com,
                ylabel=rf"${FWHM} / {GAUSS}$", hlines=[(1.0, "")],
                ylim=ratio_ylim)
            print(f"  {channel:<5} {topology:<9}: median ratio = "
                  f"{np.nanmedian(sub['ratio']):.3f}")
    print()


if __name__ == "__main__":
    main()
