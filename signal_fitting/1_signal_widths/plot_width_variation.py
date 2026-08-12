#!/usr/bin/env python3
"""How much do the Gaussian width (sigma) and mean (mu) vary across M_N, and is
the per-M_WR median a safe choice for setting the background-fit window?

For the background fits (Stage 4) the on-shell window is built from the Stage-1
single-Gaussian fit by aggregating mu_gaus / sigma_gaus over the M_N grid with
the MEDIAN, per M_WR, at the baseline seed window [0.7,1.3]*M_WR
(see signal_fitting/4_background_fits/bkg_fit_lib.py:load_grid_widths). This
script visualizes the spread that median summarizes.

For each (channel, topology) it draws, vs x = M_N / M_WR:
  * every per-M_N point, colored by M_WR (viridis colorbar);
  * a large ringed circle per M_WR at (median_x, median_value) -- the value the
    window logic actually uses.

Two y-quantities, each in two views:
  sigma_abs/        sigma_gaus            [GeV]        (literal "sigma on y")
  sigma_over_mWR/   sigma_gaus / M_WR     [-]          (spread legible)
  mean_abs/         mu_gaus               [GeV]
  mean_over_mWR/    mu_gaus / M_WR        [-]          (spread legible)
The "/ M_WR" views are the diagnostic ones: in absolute GeV the few-percent
within-M_WR spread of mu is dwarfed by the 2000->6000 GeV span between masses.

It also prints, per (channel, topology), the per-M_WR fractional spread
(max-min)/median -- typical and worst over M_WR -- the direct answer to
"how much does it vary / is the median safe".

This script only reads the CSV and makes matplotlib plots: NO ROOT / LCG needed.

Example:
    python signal_fitting/1_signal_widths/plot_width_variation.py -v
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import mplhep as hep
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}
CMAP = "viridis"

# Each entry: (column, key for output dir, y-axis label, normalize-by-mWR?)
QUANTITIES = [
    ("sigma_gaus", "sigma_abs", r"$\sigma_{\rm gauss}$ [GeV]", False),
    ("sigma_gaus", "sigma_over_mWR", r"$\sigma_{\rm gauss} / M_{W_R}$", True),
    ("mu_gaus", "mean_abs", r"$\mu_{\rm gauss}$ [GeV]", False),
    ("mu_gaus", "mean_over_mWR", r"$\mu_{\rm gauss} / M_{W_R}$", True),
]


def setup_logging(verbosity: int) -> None:
    level = logging.WARNING - 10 * min(verbosity, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def load_com(era: str) -> float:
    """Center-of-mass energy [TeV] for the CMS label; ROOT-free, fall back to 13."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from wrplotter.config import load_lumi
        return float(load_lumi(era).get("com", 13.0))
    except Exception:  # noqa: BLE001 - plotting must not hinge on the lumi lookup
        return 13.0


def per_mwr_summary(sub: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-M_WR median x/value and min/max value over the M_N grid."""
    rows = []
    for mwr, g in sub.groupby("mWR"):
        v = g[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        x = g.loc[np.isfinite(g[col]), "x"].to_numpy(dtype=float)
        rows.append({
            "mWR": float(mwr), "n": int(v.size),
            "med_x": float(np.median(x)),
            "med": float(np.median(v)),
            "vmin": float(np.min(v)), "vmax": float(np.max(v)),
        })
    return pd.DataFrame(rows).sort_values("mWR")


def plot_variation(sub: pd.DataFrame, col: str, out_path: Path, *,
                   channel: str, topology: str, era: str, com: float,
                   ylabel: str, normalize: bool, mwr_lim: tuple[float, float]):
    """Scatter (x, value) colored by M_WR, with a per-M_WR median circle.
    `normalize` divides the value by M_WR.
    """
    df = sub.copy()
    df = df[np.isfinite(df[col])]
    if df.empty:
        logger.warning("no finite %s for %s %s", col, channel, topology)
        return
    yname = "_yval"
    df[yname] = df[col] / df["mWR"] if normalize else df[col]

    norm = Normalize(vmin=mwr_lim[0], vmax=mwr_lim[1])
    cmap = matplotlib.colormaps[CMAP]

    hep.style.use("CMS")
    LBL_FS, TICK_FS = 18, 16
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    # all per-M_N points (faint cloud)
    sc = ax.scatter(df["x"], df[yname], c=df["mWR"], cmap=CMAP, norm=norm,
                    s=22, alpha=0.55, edgecolor="none", zorder=2)

    # per-M_WR median marker, in that mass's color
    summ = per_mwr_summary(df, yname)
    for r in summ.itertuples(index=False):
        color = cmap(norm(r.mWR))
        ax.scatter([r.med_x], [r.med], s=165, marker="o",
                   facecolor=color, edgecolor="black", linewidth=1.6, zorder=5)

    # marker legend (color carries M_WR via the colorbar, so make these neutral)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor="0.5", markeredgecolor="none",
               label=r"per-$M_N$ point"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=12,
               markerfacecolor="0.5", markeredgecolor="black", markeredgewidth=1.6,
               label=r"median over $M_N$ (window value)"),
    ]
    ax.legend(handles=handles, fontsize=12, loc="upper right", framealpha=0.9)

    ax.set_xlabel(r"$x = M_N / M_{W_R}$", fontsize=LBL_FS)
    ax.set_ylabel(ylabel, fontsize=LBL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=LBL_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=LBL_FS)
    ax.text(0.04, 0.96, f"{CH_LAB[channel]}  {topology.capitalize()} SR\n{era}",
            transform=ax.transAxes, fontsize=LBL_FS,
            verticalalignment="top", horizontalalignment="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path)


def print_spread(sub: pd.DataFrame, col: str, label: str,
                 channel: str, topology: str) -> None:
    """Per-M_WR fractional spread (max-min)/median over M_N: typical & worst.

    Scale-invariant, so it is computed on the raw (unnormalized) values.
    """
    summ = per_mwr_summary(sub, col)
    if summ.empty:
        return
    frac = (summ["vmax"] - summ["vmin"]) / summ["med"]
    typ, worst = float(np.median(frac)), float(np.max(frac))
    wmwr = float(summ.loc[frac.idxmax(), "mWR"])
    print(f"  {channel:<5} {topology:<9} {label:<6}: "
          f"per-M_WR (max-min)/median  typical={typ*100:5.1f}%  "
          f"worst={worst*100:5.1f}% (M_WR={wmwr:.0f})  "
          f"[{int(summ['n'].min())}-{int(summ['n'].max())} M_N pts/mass]")


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=Path,
                   default=here / "gaussian" / "gauss_fit_table.csv")
    p.add_argument("--out-dir", type=Path, default=here / "width_variation")
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--fit-range", default="[0.7,1.3]",
                   help="Seed window to read (matches Stage-4 BASELINE_FIT_RANGE).")
    p.add_argument("--channels", nargs="+", default=["ee", "mumu"])
    p.add_argument("--topologies", nargs="+", default=["resolved", "boosted"])
    p.add_argument("--good-only", action="store_true",
                   help="Keep only fit_status==0 & cov_status==3 (default: all, "
                        "to match the unfiltered median the window logic uses).")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    com = load_com(args.era)

    df = pd.read_csv(args.table)
    df = df[df["fit_range"] == args.fit_range].copy()
    if df.empty:
        sys.exit(f"No rows with fit_range == {args.fit_range!r} in {args.table}")
    if args.good_only:
        df = df[(df["fit_status"] == 0) & (df["cov_status"] == 3)].copy()
    df["x"] = df["mN"] / df["mWR"]

    mwr_lim = (float(df["mWR"].min()), float(df["mWR"].max()))
    logger.info("loaded %d rows (fit_range %s), M_WR in [%.0f, %.0f]",
                len(df), args.fit_range, *mwr_lim)

    for col, key, ylabel, normalize in QUANTITIES:
        for channel in args.channels:
            for topology in args.topologies:
                sub = df[(df["channel"] == channel)
                         & (df["category"] == topology)]
                if sub.empty:
                    continue
                plot_variation(
                    sub, col, args.out_dir / key / f"{channel}_{topology}.png",
                    channel=channel, topology=topology, era=args.era, com=com,
                    ylabel=ylabel, normalize=normalize, mwr_lim=mwr_lim)

    print(f"\n=== per-M_WR variation over M_N  (fit_range {args.fit_range}) ===")
    print("    (the window uses the MEDIAN over M_N, per M_WR)")
    for channel in args.channels:
        for topology in args.topologies:
            sub = df[(df["channel"] == channel) & (df["category"] == topology)]
            if sub.empty:
                continue
            print_spread(sub, "sigma_gaus", "sigma", channel, topology)
            print_spread(sub, "mu_gaus", "mean", channel, topology)
    print(f"\nplots -> {args.out_dir}\n")


if __name__ == "__main__":
    main()
