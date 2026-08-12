#!/usr/bin/env python3
"""Leave-one-out cross-validation of the TGraph2D interpolator.

For each of the 390 MC training points:
  1. Build a TGraph2D from the OTHER 389 points.
  2. Interpolate at the held-out (M_WR, M_N).
  3. Compare the interpolated (mean, RMS) to the actual MC values.

Reports per-cell residuals, summary statistics, and a diagnostic plot.

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
    p.add_argument("--channels", nargs="+", default=["ee", "mumu"])
    p.add_argument("--output", type=Path,
                   default=Path("signal_fitting/outputs/loo_residuals.csv"))
    p.add_argument("--plot-dir", type=Path,
                   default=Path("signal_fitting/outputs/plots/truth_params"))
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def loo_one_channel(df_ch, channel):
    """Leave-one-out: for each cell, build TGraph2D from the rest, predict, compare."""
    M_WR  = df_ch["M_WR"].to_numpy(dtype=np.float64)
    M_N   = df_ch["M_N"].to_numpy(dtype=np.float64)
    means = df_ch["mean"].to_numpy(dtype=np.float64)
    rmses = df_ch["rms"].to_numpy(dtype=np.float64)
    N = len(df_ch)

    rows = []
    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        xs = np.ascontiguousarray(M_WR[mask])
        ys = np.ascontiguousarray(M_N[mask])
        g_mean = ROOT.TGraph2D(N - 1, xs, ys,
                                np.ascontiguousarray(means[mask]))
        g_rms  = ROOT.TGraph2D(N - 1, xs, ys,
                                np.ascontiguousarray(rmses[mask]))
        mean_pred = float(g_mean.Interpolate(float(M_WR[i]), float(M_N[i])))
        rms_pred  = float(g_rms .Interpolate(float(M_WR[i]), float(M_N[i])))
        in_hull = mean_pred != 0.0 and rms_pred != 0.0
        rows.append({
            "channel": channel,
            "M_WR": float(M_WR[i]), "M_N": float(M_N[i]),
            "x": float(M_N[i] / M_WR[i]),
            "mean_actual": float(means[i]), "mean_pred": mean_pred,
            "mean_resid": (mean_pred - float(means[i])) if in_hull else float("nan"),
            "rms_actual":  float(rmses[i]), "rms_pred":  rms_pred,
            "rms_resid":  (rms_pred - float(rmses[i]))  if in_hull else float("nan"),
            "in_hull": in_hull,
        })
        if (i + 1) % 50 == 0:
            logger.info("  %s: %d / %d done", channel, i + 1, N)
    return pd.DataFrame(rows)


def plot_residuals(df_loo, channel, plot_dir):
    inside  = df_loo[df_loo["in_hull"]]
    outside = df_loo[~df_loo["in_hull"]]
    logger.info("[%s] %d/%d cells inside convex hull; %d on the boundary",
                channel, len(inside), len(df_loo), len(outside))

    hep.style.use("CMS")
    LBL_FS, TICK_FS = 18, 16
    for kind, resid_col, actual_col, label in [
        ("mean", "mean_resid", "mean_actual", r"$\mu_{\rm truth}$"),
        ("rms",  "rms_resid",  "rms_actual",  r"$\sigma_{\rm truth}$"),
    ]:
        r = inside[resid_col].to_numpy()
        a = inside[actual_col].to_numpy()
        x = inside["x"].to_numpy()
        M_WR = inside["M_WR"].to_numpy()
        rel = r / a * 100.0  # fractional residual in %

        # Clip y-range to the 99% bulk (outliers off-scale, marked).
        def bulk_lim(vals, pad_factor=1.30, min_span=1.0):
            p1, p99 = np.percentile(vals, [1, 99])
            half = max(abs(p1), abs(p99)) * pad_factor
            return max(half, min_span)
        ylim_abs = bulk_lim(r,   min_span=1.0)
        ylim_rel = bulk_lim(rel, min_span=0.5)

        n_off_abs = int(np.sum(np.abs(r)   > ylim_abs))
        n_off_rel = int(np.sum(np.abs(rel) > ylim_rel))

        # 2-panel layout with constrained_layout (handles the colorbar cleanly).
        fig, axes = plt.subplots(
            2, 1, sharex=True, figsize=(10, 9),
            gridspec_kw={"height_ratios": [1, 1]},
            constrained_layout=True,
        )
        for ax, vals, ylabel, ylim, n_off in [
            (axes[0], r,   rf"$\Delta${label} = pred $-$ actual  [GeV]",
             ylim_abs, n_off_abs),
            (axes[1], rel, rf"$\Delta${label} / {label}  [%]",
             ylim_rel, n_off_rel),
        ]:
            sc = ax.scatter(x, vals, c=M_WR, cmap="viridis", s=32,
                            edgecolor="black", linewidth=0.4)
            ax.axhline(0, color="red", linewidth=1.5, alpha=0.7)
            ax.set_ylim(-ylim, ylim)
            ax.tick_params(labelsize=TICK_FS)
            ax.grid(alpha=0.3)
            ax.set_ylabel(ylabel, fontsize=LBL_FS - 2)
            if n_off > 0:
                ax.text(0.98, 0.04, f"{n_off} pts off-scale",
                        transform=ax.transAxes, fontsize=11, color="gray",
                        verticalalignment="bottom", horizontalalignment="right")
        axes[1].set_xlabel(r"$x = M_N / M_{W_R}$", fontsize=LBL_FS)
        cbar = fig.colorbar(sc, ax=axes, pad=0.02, fraction=0.05)
        cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=LBL_FS)
        cbar.ax.tick_params(labelsize=TICK_FS)
        hep.cms.label(loc=0, ax=axes[0], data=False,
                      label="Work in Progress", com=13, fontsize=LBL_FS)
        axes[0].text(0.04, 0.96, f"{CH_LAB[channel]}  (leave-one-out)",
                     transform=axes[0].transAxes, fontsize=LBL_FS,
                     verticalalignment="top", horizontalalignment="left")
        stats = (
            f"interior cells: {len(inside)}/{len(df_loo)}\n"
            f"median |Δ| = {np.median(np.abs(r)):.1f} GeV  "
            f"({np.median(np.abs(rel)):.2f}%)\n"
            f"max    |Δ| = {np.max(np.abs(r)):.1f} GeV  "
            f"({np.max(np.abs(rel)):.2f}%)"
        )
        axes[0].text(0.97, 0.96, stats, transform=axes[0].transAxes,
                     fontsize=12, verticalalignment="top",
                     horizontalalignment="right", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                               edgecolor="gray", alpha=0.95))

        out = plot_dir / f"{channel}_{kind}_loo.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", out)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    df = load_truth_from_mc(input_dirs, args.channels, args.topology, 0.10)
    logger.info("Loaded %d total MC points", len(df))

    all_rows = []
    for channel, sub in df.groupby("channel"):
        sub = sub.reset_index(drop=True)
        logger.info("LOO on channel %s (%d cells)", channel, len(sub))
        df_loo = loo_one_channel(sub, channel)
        all_rows.append(df_loo)
        plot_residuals(df_loo, channel, args.plot_dir)

    df_all = pd.concat(all_rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(df_all), args.output)

    # Quick summary table.
    print("\n=== Leave-one-out summary ===")
    print(f"{'channel':<6} {'inside':>6}/{'total':<5} "
          f"{'mean med |Δ|':>14} {'mean med %':>11}   "
          f"{'rms med |Δ|':>13} {'rms med %':>10}")
    for channel, sub in df_all.groupby("channel"):
        inside = sub[sub.in_hull]
        n_in, n_tot = len(inside), len(sub)
        m_abs = np.median(np.abs(inside.mean_resid))
        m_pct = np.median(np.abs(inside.mean_resid / inside.mean_actual)) * 100
        r_abs = np.median(np.abs(inside.rms_resid))
        r_pct = np.median(np.abs(inside.rms_resid  / inside.rms_actual))  * 100
        print(f"{channel:<6} {n_in:>6}/{n_tot:<5} "
              f"{m_abs:>12.2f} GeV {m_pct:>9.3f}%    "
              f"{r_abs:>11.2f} GeV {r_pct:>8.3f}%")


if __name__ == "__main__":
    main()
