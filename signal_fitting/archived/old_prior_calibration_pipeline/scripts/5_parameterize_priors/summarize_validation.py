#!/usr/bin/env python3
"""Summarize validate_params.py output: per-cell pull bias/spread + plot.

For each (channel, mass, N): compute the median pull (bias) and half-68%
(spread) over the toys. Report whether the spreads still sit in the [0.9, 1.1]
calibration target band.

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from wrplotter.cli_utils import setup_logging
from pull_stats import gaussian_pull_fit

logger = logging.getLogger(__name__)

CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}


def _gfit_mu(x):
    return gaussian_pull_fit(x)["mu"]


def _gfit_sigma(x):
    return gaussian_pull_fit(x)["sigma"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("signal_fitting/outputs/validate_params.csv"))
    p.add_argument("--plot-dir", type=Path,
                   default=Path("signal_fitting/outputs/plots/validate_params"))
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    df = pd.read_csv(args.input)
    df = df[(df["status"] == 0) & (df["covqual"] == 3)]
    logger.info("Loaded %d good fit rows", len(df))

    # Per-cell summary table.
    per_cell = (df.groupby(["channel", "mass", "M_WR", "M_N", "n_events"])
                  .agg(mu_bias=("pull_mu", _gfit_mu),
                       mu_spread=("pull_mu", _gfit_sigma),
                       w_bias=("pull_sigma", _gfit_mu),
                       w_spread=("pull_sigma", _gfit_sigma),
                       n_toys=("pull_mu", "size"))
                  .reset_index())

    # Print compact per-N report.
    print("\n=== Per-N summary across 15 cells ===\n")
    print(f"{'channel':<6} {'N':>4} "
          f"{'med µ_bias':>11} {'med µ_spread':>13} {'spread∈[0.9,1.1]':>17} "
          f"{'med σ_bias':>11} {'med σ_spread':>13} {'spread∈[0.9,1.1]':>17}")
    for (ch, n), sub in per_cell.groupby(["channel", "n_events"]):
        mu_b = sub["mu_bias"].median()
        mu_s = sub["mu_spread"].median()
        w_b  = sub["w_bias"].median()
        w_s  = sub["w_spread"].median()
        mu_ok = ((sub["mu_spread"] >= 0.9) & (sub["mu_spread"] <= 1.1)).sum()
        w_ok  = ((sub["w_spread"]  >= 0.9) & (sub["w_spread"]  <= 1.1)).sum()
        print(f"{ch:<6} {n:>4} "
              f"{mu_b:>+11.3f} {mu_s:>+13.3f} {mu_ok:>5d}/{len(sub):d}{'':>11} "
              f"{w_b:>+11.3f} {w_s:>+13.3f} {w_ok:>5d}/{len(sub):d}")
    print()

    # Plot: per-cell spread vs M_WR, colored by M_N/M_WR, one row per N.
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    hep.style.use("CMS")
    for ch in per_cell["channel"].unique():
        for param, col_bias, col_spread, plabel in [
            ("mu", "mu_bias", "mu_spread", r"\mu"),
            ("width", "w_bias", "w_spread", r"\sigma"),
        ]:
            sub_ch = per_cell[per_cell["channel"] == ch]
            n_vals = sorted(sub_ch["n_events"].unique())
            fig, axes = plt.subplots(2, len(n_vals), figsize=(4 * len(n_vals), 7),
                                      sharex=True)
            for j, n in enumerate(n_vals):
                sub = sub_ch[sub_ch["n_events"] == n]
                x = sub["M_N"] / sub["M_WR"]
                for ax, col, target, ylabel in [
                    (axes[0, j], col_bias, 0.0,
                     rf"median pull ${plabel}$ bias"),
                    (axes[1, j], col_spread, 1.0,
                     rf"half-68% pull ${plabel}$ spread"),
                ]:
                    sc = ax.scatter(x, sub[col], c=sub["M_WR"], cmap="viridis",
                                    s=70, edgecolor="black", linewidth=0.5)
                    ax.axhline(target, color="red", linewidth=1.2, alpha=0.7)
                    if target == 1.0:
                        ax.axhspan(0.9, 1.1, color="red", alpha=0.08, zorder=-1)
                    ax.grid(alpha=0.3)
                    ax.tick_params(labelsize=10)
                    ax.set_title(f"N={n}", fontsize=11)
                    if j == 0:
                        ax.set_ylabel(ylabel, fontsize=12)
                axes[1, j].set_xlabel(r"$M_N / M_{W_R}$", fontsize=12)
            cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.01)
            cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=11)
            fig.suptitle(rf"{CH_LAB[ch]} — pull ${plabel}$ with parameterized priors",
                         fontsize=14)
            out = args.plot_dir / f"{ch}_{param}.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
