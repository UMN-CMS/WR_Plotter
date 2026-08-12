#!/usr/bin/env python3
"""Illustrate the TGraph2D::Interpolate procedure on the scatter plot.

Takes the (x, mean) and (x, rms) scatters, overlays a star at the query
point and circles around the three Delaunay-triangle vertices that
actually feed the prediction.

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
from fit_truth import load_truth_from_mc, load_params, predict_priors

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
    p.add_argument("--m-wr", type=float, default=4321.0,
                   help="Query M_WR [GeV]. Default 4321.")
    p.add_argument("--m-n", type=float, default=1234.0,
                   help="Query M_N [GeV]. Default 1234.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def plot_one(df, channel, kind, y_col, label_y, plot_dir, *,
             query, triangle, pred_value):
    """Scatter + query + triangle-vertex overlay."""
    x = (df["M_N"] / df["M_WR"]).to_numpy()
    y = df[y_col].to_numpy()
    M_WR = df["M_WR"].to_numpy()

    hep.style.use("CMS")
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=M_WR, cmap="viridis", s=32,
                    edgecolor="black", linewidth=0.4, alpha=0.85, zorder=2)

    # Zoom to the interpolation region so the 4 markers are easy to see.
    pts_x = [query["x"]] + [v["x"] for v in triangle]
    pts_y = [pred_value] + [v[y_col] for v in triangle]
    x_lo, x_hi = min(pts_x), max(pts_x)
    y_lo, y_hi = min(pts_y), max(pts_y)
    pad_x = max(0.04, 0.6 * (x_hi - x_lo))
    pad_y = max(0.05 * (y_hi - y_lo), 0.04 * (float(y.max()) - float(y.min())))
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y * 2.2)  # extra room on top for the legend

    # Triangle vertices: large open red circles with small letter labels.
    for v, name in zip(triangle, ["v1", "v2", "v3"]):
        ax.plot(v["x"], v[y_col], marker="o",
                markersize=18, markerfacecolor="none",
                markeredgecolor="red", markeredgewidth=2.5, zorder=4)
        ax.annotate(name, xy=(v["x"], v[y_col]), xytext=(8, 8),
                    textcoords="offset points", fontsize=13, color="red",
                    fontweight="bold", zorder=5)

    # Query point: bright red dot (same size as the data scatter dots).
    ax.scatter(query["x"], pred_value, s=32, c="#ff0000",
               edgecolor="black", linewidth=0.4, zorder=6)

    # Single legend box (top-right): Q first, then the three vertices.
    legend_lines = [
        r"$\bf{Interpolation\ at\ Q}$",
        "",
        f"Q : ($M_{{W_R}}$={int(query['M_WR'])}, $M_N$={int(query['M_N'])})",
        f"pred = {pred_value:.1f} GeV",
        "",
    ]
    for v, name in zip(triangle, ["v1", "v2", "v3"]):
        legend_lines.append(
            f"{name}: ($M_{{W_R}}$={int(v['M_WR'])}, $M_N$={int(v['M_N'])})  "
            f"w = {v['weight']:.3f}"
        )
    ax.text(0.97, 0.97, "\n".join(legend_lines),
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top", horizontalalignment="right",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.95), zorder=7)

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
    out = plot_dir / f"{channel}_{kind}_interp_example.png"
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

    params = load_params(args.params)
    mean_pred, rms_pred = predict_priors(args.channel, args.m_wr, args.m_n, params)
    logger.info("Query (%s, M_WR=%g, M_N=%g) -> mean=%.2f  RMS=%.2f",
                args.channel, args.m_wr, args.m_n, mean_pred, rms_pred)

    # Hardcode the three Delaunay vertices for (4321, 1234) — verified analytically
    # against TGraph2D::Interpolate (see explanation in chat).
    # Weights are barycentric: α=0.225 (v1), β=0.605 (v2), γ=0.170 (v3).
    vert_specs = [
        {"M_WR": 4200, "M_N": 1200, "weight": 0.225},
        {"M_WR": 4200, "M_N": 1400, "weight": 0.170},
        {"M_WR": 4400, "M_N": 1200, "weight": 0.605},
    ]
    triangle = []
    for v in vert_specs:
        row = df[(df.M_WR == v["M_WR"]) & (df.M_N == v["M_N"])]
        if len(row) == 0:
            logger.warning("Vertex (%g, %g) not in data — skipping", v["M_WR"], v["M_N"])
            continue
        v["x"]    = float(v["M_N"] / v["M_WR"])
        v["mean"] = float(row["mean"].iloc[0])
        v["rms"]  = float(row["rms"].iloc[0])
        triangle.append(v)

    query = {
        "M_WR": args.m_wr, "M_N": args.m_n,
        "x": args.m_n / args.m_wr,
        "text_offset_mean": (-40, -65), "text_offset_rms": (-40, -65), "ha": "right",
    }

    plot_one(df, args.channel, "mean", "mean",
             r"$\mu_{\rm truth}$  [GeV]", args.plot_dir,
             query=query, triangle=triangle, pred_value=mean_pred)
    plot_one(df, args.channel, "rms", "rms",
             r"$\sigma_{\rm truth}$  [GeV]", args.plot_dir,
             query=query, triangle=triangle, pred_value=rms_pred)


if __name__ == "__main__":
    main()
