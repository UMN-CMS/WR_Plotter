#!/usr/bin/env python3
"""Diagnostic: overlay an original MC histogram with one Poisson-resampled toy.

Demonstrates what the per-point bootstrap (used for FWHM and peak-position
uncertainties in the rest of the pipeline) actually does to the histogram.

Usage:
    python signal_fitting/bootstrap_demo.py --signal WR4000_N2000 --channel ee
    python signal_fitting/bootstrap_demo.py --signal WR4000_N2000 --channel ee --seed 7
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

from measure_fwhm import (
    ONSHELL_WINDOW_LO_FRAC,
    ONSHELL_WINDOW_HI_FRAC,
    build_hist_key,
    build_region_name,
    compute_shape_params,
    format_signal_label,
    load_and_combine_signal,
    parse_masses,
    rebin_histogram,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--channel", choices=["ee", "mumu"], default="ee")
    p.add_argument("--signal", default="WR4000_N2000",
                   help="Signal tag, e.g. WR4000_N2000.")
    p.add_argument("--topology", choices=["resolved", "boosted"], default="resolved")
    p.add_argument("--rebin", type=int, default=6,
                   help="Rebin factor (default 6 -> 60 GeV bins).")
    p.add_argument("--seed", type=int, default=12345,
                   help="RNG seed for the Poisson resample.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    info = load_lumi(args.era)
    com = info.get("com", 13.0)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)

    region = build_region_name(args.channel, args.topology)
    mass_var = "mass_twoobject" if args.topology == "boosted" else "mass_fourobject"
    hist_key = build_hist_key(region, mass_var)

    edges, vals, var = load_and_combine_signal(input_dirs, hist_key, args.signal)
    if args.rebin > 1:
        edges, vals, var = rebin_histogram(edges, vals, var, args.rebin)

    # Original measurement
    sp_orig = compute_shape_params(edges, vals, args.signal, args.channel, args.topology)

    # One bootstrap toy
    rng = np.random.default_rng(args.seed)
    lam = np.maximum(vals, 0.0)
    toy = rng.poisson(lam).astype(float)
    sp_toy = compute_shape_params(edges, toy, args.signal, args.channel, args.topology)

    logger.info("Original  peak=%.0f, FWHM=%.0f", sp_orig.peak_onshell, sp_orig.fwhm_onshell)
    logger.info("Bootstrap peak=%.0f, FWHM=%.0f  (seed=%d)",
                sp_toy.peak_onshell, sp_toy.fwhm_onshell, args.seed)

    # Plotting
    wr_mass, _ = parse_masses(args.signal)
    win_lo = ONSHELL_WINDOW_LO_FRAC * wr_mass
    win_hi = ONSHELL_WINDOW_HI_FRAC * wr_mass
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_win = (centers >= win_lo) & (centers <= win_hi)

    hep.style.use("CMS")
    fig, ax = plt.subplots()

    # Original histogram in blue
    ax.stairs(vals, edges, color="#1f77b4", linewidth=1.5, label="Original MC")
    ax.fill_between(centers[in_win], 0, vals[in_win], step="mid",
                    color="#1f77b4", alpha=0.20, zorder=-1)

    # Bootstrap toy in red
    ax.stairs(toy, edges, color="#d62728", linewidth=1.5, label="Bootstrap toy")
    ax.fill_between(centers[in_win], 0, toy[in_win], step="mid",
                    color="#d62728", alpha=0.20, zorder=-1)

    ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_xlim(0, 10000)
    _, y_hi = ax.get_ylim()
    ax.set_ylim(0, y_hi * 1.30)

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=18)

    ch = {"ee": "ee", "mumu": r"$\mu\mu$"}[args.channel]
    ax.text(
        0.05, 0.96,
        f"{ch}\n{args.topology.capitalize()} SR\n{args.era}\n"
        f"{format_signal_label(args.signal)}\n"
        rf"Shaded: $[0.7\,M_{{W_R}},\ 1.3\,M_{{W_R}}]$",
        transform=ax.transAxes, fontsize=14, verticalalignment="top",
    )

    # Legend top-right (5% buffer from right; tighter to top)
    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.98),
              fontsize=14, framealpha=0.85)

    # Stat box below the legend
    ax.text(
        0.95, 0.86,
        "Peak bin center / FWHM\n"
        rf"Orig: ${sp_orig.peak_onshell:.0f}$ / ${sp_orig.fwhm_onshell:.0f}$ GeV"
        "\n"
        rf"Toy:  ${sp_toy.peak_onshell:.0f}$ / ${sp_toy.fwhm_onshell:.0f}$ GeV",
        transform=ax.transAxes, fontsize=14,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.85),
    )

    out_dir = Path(args.output_dir or
                   str(repo_root() / "signal_fitting" / "outputs"
                       / args.era / "fwhm" / "bootstrap_demo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"bootstrap_demo_{args.signal}_{args.channel}_seed{args.seed}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
