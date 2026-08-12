#!/usr/bin/env python3
"""Pure scatter plots of the 390-cell training data: windowed mean and RMS
vs x = M_N/M_WR, colored by M_WR. No interpolation curves.

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

try:
    import ROOT
except ImportError:
    sys.exit("ERROR: PyROOT unavailable. Source LCG_106 first.")
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wrplotter.cli_utils import setup_logging
from wrplotter.paths import input_dirs_for_era, repo_root
from fit_truth import load_truth_from_mc

logger = logging.getLogger(__name__)

CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--channel", default="ee")
    p.add_argument("--plot-dir", type=Path,
                   default=Path("signal_fitting/outputs/plots/truth_params"))
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def plot_one(df, channel, era, kind, y_col, label_y, plot_dir):
    x = (df["M_N"] / df["M_WR"]).to_numpy()
    y = df[y_col].to_numpy()
    M_WR = df["M_WR"].to_numpy()

    hep.style.use("CMS")
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=M_WR, cmap="viridis", s=32,
                    edgecolor="black", linewidth=0.4)
    y_lo, y_hi = ax.get_ylim()
    y_data_hi = float(y.max())
    ax.set_ylim(top=y_data_hi + 0.08 * (y_data_hi - float(y.min())))
    # Two-size convention: 18 for labels, 16 for tick numbers.
    LBL_FS, TICK_FS = 18, 16
    ax.set_xlabel(r"$x = M_N / M_{W_R}$", fontsize=LBL_FS)
    ax.set_ylabel(label_y, fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=LBL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=LBL_FS)
    ax.text(0.04, 0.96, CH_LAB[channel],
            transform=ax.transAxes, fontsize=LBL_FS,
            verticalalignment="top", horizontalalignment="left")

    fig.tight_layout()
    out = plot_dir / f"{channel}_{kind}_scatter.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    df = load_truth_from_mc(input_dirs, [args.channel], args.topology, 0.10)
    logger.info("Loaded %d MC points for channel %s", len(df), args.channel)

    plot_one(df, args.channel, args.era, "mean", "mean",
             r"$\mu_{\rm truth}$  [GeV]", args.plot_dir)
    plot_one(df, args.channel, args.era, "rms", "rms",
             r"$\sigma_{\rm truth}$  [GeV]", args.plot_dir)


if __name__ == "__main__":
    main()
