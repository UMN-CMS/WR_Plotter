#!/usr/bin/env python3
"""Explainer plots for the TGraph2D-based truth parameterization.

Two figures per channel:
  - mean(M_WR, M_N): MC points + interpolation curves at 3 M_WR slices.
  - RMS (M_WR, M_N): same.

Each plot is self-contained (formula box describes the method).

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
from fit_truth import load_params, load_truth_from_mc

logger = logging.getLogger(__name__)

CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--channel", default="ee")
    p.add_argument("--params", type=Path,
                   default=Path("signal_fitting/outputs/truth_params.root"))
    p.add_argument("--plot-dir", type=Path,
                   default=Path("signal_fitting/outputs/plots/truth_params"))
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def plot_one(df, graph, kind, channel, plot_dir, *,
             y_col, label_y, formula_lines):
    """One panel: y_col vs M_N/M_WR, colored by M_WR, with interpolation
    curves at 3 M_WR slices (2, 4, 6 TeV)."""
    x_data = (df["M_N"] / df["M_WR"]).to_numpy()
    y_data = df[y_col].to_numpy()
    M_WR = df["M_WR"].to_numpy()

    wr_slices = [2000, 4000, 6000]
    slice_colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(wr_slices)))

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(x_data, y_data, c=M_WR, cmap="viridis", s=28,
                    edgecolor="black", linewidth=0.35,
                    label=f"MC: {len(df)} mass points")
    for wr, color in zip(wr_slices, slice_colors):
        # Sample only inside the per-M_WR x range so we don't show extrapolation.
        sub = df[df["M_WR"] == wr]
        if len(sub) < 2:
            continue
        x_lo = float((sub["M_N"] / wr).min())
        x_hi = float((sub["M_N"] / wr).max())
        xs = np.linspace(x_lo, x_hi, 200)
        y_curve = np.array([float(graph.Interpolate(float(wr), float(x * wr)))
                            for x in xs])
        # Drop out-of-envelope (TGraph2D returns 0).
        mask = y_curve != 0.0
        ax.plot(xs[mask], y_curve[mask], color=color, linewidth=2.5,
                label=rf"interp @ $M_{{W_R}}={wr}$ GeV")
    ax.set_xlabel(r"$x = M_N / M_{W_R}$", fontsize=15)
    ax.set_ylabel(label_y, fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=11,
              framealpha=0.92)
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=13)

    ax.text(0.98, 0.02, "\n".join(formula_lines),
            transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom", horizontalalignment="right",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="gray", alpha=0.95))
    ax.text(0.98, 0.98, CH_LAB[channel],
            transform=ax.transAxes, fontsize=16,
            verticalalignment="top", horizontalalignment="right")

    fig.tight_layout()
    out = plot_dir / f"{channel}_{kind}_explainer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    params = load_params(args.params)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    df_all = load_truth_from_mc(input_dirs, [args.channel], args.topology, 0.10)
    logger.info("Loaded %d MC points for channel %s", len(df_all), args.channel)

    n_pts = params[args.channel]["mean"].GetN()
    plot_one(
        df_all, params[args.channel]["mean"],
        kind="mean", channel=args.channel, plot_dir=args.plot_dir,
        y_col="mean",
        label_y=r"windowed mean of $m_{\ell\ell jj}$  [GeV]",
        formula_lines=[
            r"mean$(M_{W_R}, M_N)$ via ROOT TGraph2D::Interpolate",
            f"Delaunay triangulation, {n_pts} training cells",
            "exact on training points; smooth between",
        ],
    )
    plot_one(
        df_all, params[args.channel]["rms"],
        kind="rms", channel=args.channel, plot_dir=args.plot_dir,
        y_col="rms",
        label_y=r"windowed RMS of $m_{\ell\ell jj}$  [GeV]",
        formula_lines=[
            r"RMS$(M_{W_R}, M_N)$ via ROOT TGraph2D::Interpolate",
            f"Delaunay triangulation, {n_pts} training cells",
            "exact on training points; smooth between",
        ],
    )
    params["_file"].Close()


if __name__ == "__main__":
    main()
