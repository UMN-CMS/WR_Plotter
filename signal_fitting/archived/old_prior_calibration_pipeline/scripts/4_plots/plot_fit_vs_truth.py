#!/usr/bin/env python3
"""How well does the toy Gaussian fit recover the MC histogram truth?

Companion to the leave-one-out plots (5_parameterize_priors/leave_one_out.py),
but instead of "interpolated prediction vs MC truth" this shows
"Gaussian fit to toy data vs MC truth".

For each (channel, mass) cell at a chosen N, take the production toy fits from
results.csv and compare the per-cell median fitted Gaussian (mu_fit, sigma_fit)
to the MC windowed mean / RMS (mu_truth, width_truth — the values we compute up
front from the MC histogram). Residual = median(fit) - truth, plotted vs
x = M_N / M_WR and colored by M_WR, in the same 2-panel (absolute GeV +
fractional %) layout as the LOO plots.

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Example:
    python signal_fitting/4_plots/plot_fit_vs_truth.py \
        --input signal_fitting/outputs/results.csv --n-events 20
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
from measure_fwhm import parse_masses

logger = logging.getLogger(__name__)

CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path,
                   default=Path("signal_fitting/outputs/results.csv"),
                   help="Production results.csv from scan_full.py.")
    p.add_argument("--plot-dir", type=Path,
                   default=Path("signal_fitting/outputs/plots/fit_vs_truth"))
    p.add_argument("--n-events", type=int, default=20,
                   help="Event count to plot (operating point = 20).")
    p.add_argument("--channels", nargs="+", default=["ee", "mumu"])
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def per_cell_table(df):
    """Per-(channel, mass) median fit vs MC truth, plus per-toy scatter."""
    rows = []
    for (channel, mass), sub in df.groupby(["channel", "mass"]):
        try:
            M_WR, M_N = parse_masses(mass)
        except Exception:
            logger.warning("bad mass tag %r, skipping", mass)
            continue
        mu_truth = float(sub["mu_truth"].iloc[0])
        rms_truth = float(sub["width_truth"].iloc[0])
        rows.append({
            "channel": channel, "mass": mass,
            "M_WR": float(M_WR), "M_N": float(M_N),
            "x": float(M_N) / float(M_WR),
            "mean_truth": mu_truth, "rms_truth": rms_truth,
            "mean_fit": float(sub["mu_fit"].median()),
            "rms_fit": float(sub["width_fit"].median()),
            "mean_resid": float(sub["mu_fit"].median()) - mu_truth,
            "rms_resid": float(sub["width_fit"].median()) - rms_truth,
            # per-toy scatter of the fitted value (precision, not bias).
            "mean_fit_scatter": float(sub["mu_fit"].std()),
            "rms_fit_scatter": float(sub["width_fit"].std()),
            "n_toys": int(len(sub)),
        })
    return pd.DataFrame(rows)


def plot_one(per_cell, channel, n_events, plot_dir):
    sub = per_cell[per_cell["channel"] == channel]
    if sub.empty:
        logger.warning("no cells for channel %s", channel)
        return

    hep.style.use("CMS")
    LBL_FS, TICK_FS = 18, 16
    for kind, resid_col, truth_col, label in [
        ("mean", "mean_resid", "mean_truth", r"$\mu$"),
        ("rms",  "rms_resid",  "rms_truth",  r"$\sigma$"),
    ]:
        r = sub[resid_col].to_numpy()
        t = sub[truth_col].to_numpy()
        x = sub["x"].to_numpy()
        M_WR = sub["M_WR"].to_numpy()
        rel = r / t * 100.0

        def bulk_lim(vals, pad_factor=1.30, min_span=1.0):
            p1, p99 = np.percentile(vals, [1, 99])
            half = max(abs(p1), abs(p99)) * pad_factor
            return max(half, min_span)
        ylim_abs = bulk_lim(r,   min_span=1.0)
        ylim_rel = bulk_lim(rel, min_span=0.5)
        n_off_abs = int(np.sum(np.abs(r)   > ylim_abs))
        n_off_rel = int(np.sum(np.abs(rel) > ylim_rel))

        fig, axes = plt.subplots(
            2, 1, sharex=True, figsize=(10, 9),
            gridspec_kw={"height_ratios": [1, 1]},
            constrained_layout=True,
        )
        for ax, vals, ylabel, ylim, n_off in [
            (axes[0], r,
             rf"$\Delta${label} = fit $-$ truth  [GeV]", ylim_abs, n_off_abs),
            (axes[1], rel,
             rf"$\Delta${label} / {label}$_{{\rm truth}}$  [%]", ylim_rel, n_off_rel),
        ]:
            sc = ax.scatter(x, vals, c=M_WR, cmap="viridis", s=32,
                            edgecolor="black", linewidth=0.4, zorder=2)
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
        axes[0].text(0.04, 0.96,
                     f"{CH_LAB[channel]}  (Gaussian fit, N={n_events})",
                     transform=axes[0].transAxes, fontsize=LBL_FS,
                     verticalalignment="top", horizontalalignment="left")

        out = plot_dir / f"{channel}_{kind}_fit_vs_truth.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        logger.info("wrote %s", out)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    df = pd.read_csv(args.input)
    df = df[(df["status"] == 0) & (df["covqual"] == 3)
            & (df["n_events"] == args.n_events)
            & (df["channel"].isin(args.channels))].copy()
    logger.info("Loaded %d good fit rows at N=%d", len(df), args.n_events)
    if df.empty:
        sys.exit(f"No good rows at N={args.n_events} in {args.input}")

    per_cell = per_cell_table(df)
    logger.info("Per-channel cell counts: %s",
                per_cell.groupby("channel").size().to_dict())

    # Console summary.
    print(f"\n=== Gaussian fit vs MC truth (N={args.n_events}, "
          f"median over toys per cell) ===")
    print(f"{'channel':<6} {'cells':>5} "
          f"{'µ med|Δ|':>9} {'µ med%':>8} "
          f"{'σ med|Δ|':>9} {'σ med%':>8}")
    for ch, s in per_cell.groupby("channel"):
        print(f"{ch:<6} {len(s):>5} "
              f"{np.median(np.abs(s.mean_resid)):>7.1f}G "
              f"{np.median(np.abs(s.mean_resid / s.mean_truth))*100:>7.2f}% "
              f"{np.median(np.abs(s.rms_resid)):>7.1f}G "
              f"{np.median(np.abs(s.rms_resid / s.rms_truth))*100:>7.2f}%")
    print()

    for channel in args.channels:
        plot_one(per_cell, channel, args.n_events, args.plot_dir)


if __name__ == "__main__":
    main()
