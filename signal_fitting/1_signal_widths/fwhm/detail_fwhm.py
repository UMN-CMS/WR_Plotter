#!/usr/bin/env python3
"""Per-method deep dive for the on-shell RooKeysPdf FWHM width, σ_FWHM^on.

For every signal cell (both channels, both topologies; boosted x = M_N/M_WR < 0.1
from disk) and every seed window in --windows:

  1. select events in [lo, hi]*M_WR,
  2. build a RooKeysPdf (adaptive KDE) from them,
  3. evaluate on a fine grid, find the peak maximum inside the window,
  4. walk out to the left/right half-maximum crossings x_L, x_R,
  5. σ_FWHM = (x_R - x_L) / 2.3548.

Table: outputs/width_estimators/fwhm/fwhm_table.csv with
  mWR, mN, channel, category, fit_range, peak, peak_over_mWR, fwhm,
  sigma_fwhm, sigma_fwhm_over_mWR, x_lo, x_hi

Plots (outputs/width_estimators/fwhm/), baseline window, M_WR colorbar:
  1. sigma_fwhm_over_mWR_*   σ_FWHM^on / M_WR vs x        (y-centered)
     peak_over_mWR_*         peak / M_WR vs x (line at 1) (y-centered)
  2. robustness_{channel}_{topology}.{png,pdf}
        σ_FWHM(window)/σ_FWHM([0.8,1.2]) vs x, shared y-scale, M_WR colorbar.
  3. histograms/{channel}_{topology}/{mass}.png   (one per cell)
        MC histogram + the RooKeysPdf curve + the half-max line and the
        x_L/x_R crossings (the FWHM), with the seed window drawn for reference.

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Example:
    python signal_fitting/1_signal_samples/detail_fwhm.py -v
    python signal_fitting/1_signal_samples/detail_fwhm.py --plots-only -v
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

from shape_estimators import (  # noqa: E402
    CH_LAB, FWHM_TO_GAUSS_SIGMA, collect_cells, load_master_masses,
    keys_fwhm_detail, plot_scalar_vs_x_by_mwr, plot_ratio_overlay_by_mwr,
    window_stability_report,
)

logger = logging.getLogger(__name__)

MASS_VAR_LABEL = {"resolved": r"$m_{\ell\ell jj}$", "boosted": r"$m_{\ell J}$"}
DEFAULT_WINDOWS = [(0.70, 1.30), (0.80, 1.20), (0.85, 1.15)]
BASELINE_WINDOW = (0.80, 1.20)
SIGMA_FWHM_ON = r"\sigma_{\rm FWHM}^{\rm on}"

TABLE_COLUMNS = [
    "mWR", "mN", "channel", "category", "fit_range",
    "peak", "peak_over_mWR", "fwhm",
    "sigma_fwhm", "sigma_fwhm_over_mWR", "x_lo", "x_hi",
]


def win_str(lo, hi) -> str:
    return f"[{lo:g},{hi:g}]"


def parse_windows(specs):
    return [tuple(float(v) for v in s.split(",")) for s in specs]


def centered_ylim(vals, frac=0.30):
    """y-range centered on the full data range (every point on-scale)."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:
        hi = lo + 1e-6
    m = (hi - lo) * frac
    return (lo - m, hi + m)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260624_signals")
    p.add_argument("--channels", nargs="+", default=["ee", "mumu"])
    p.add_argument("--topologies", nargs="+", default=["resolved", "boosted"])
    p.add_argument("--mass-csv", type=Path,
                   default=Path(__file__).resolve().parents[2] / "master_masses.csv")
    p.add_argument("--windows", nargs="+", default=None,
                   help='Seed windows "lo,hi". Default 0.70,1.30 0.80,1.20 0.85,1.15.')
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).resolve().parent)
    p.add_argument("--min-events", type=float, default=100.0)
    p.add_argument("--boosted-max-x", type=float, default=0.1)
    p.add_argument("--no-hist-plots", action="store_true")
    p.add_argument("--plots-only", action="store_true",
                   help="Skip KDE; rebuild summary plots (1 & 2) from the table.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-cell histogram + KDE + FWHM overlay
# ---------------------------------------------------------------------------

def plot_fwhm_overlay(cell, det, out_path, *, era, com, seed_window):
    """MC histogram + RooKeysPdf curve + half-max line and x_L/x_R crossings."""
    from matplotlib.lines import Line2D

    edges, vals = cell.edges, cell.vals
    M_WR = cell.M_WR
    s_lo, s_hi = seed_window
    seed_lo, seed_hi = s_lo * M_WR, s_hi * M_WR
    peak, x_lo, x_hi, half = det["peak"], det["x_lo"], det["x_hi"], det["half_max"]
    sigma_fwhm = det["fwhm"] / FWHM_TO_GAUSS_SIGMA

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.stairs(np.maximum(vals, 0.0), edges, color="#3f90da", linewidth=1.5)

    if det["xs"] is not None:
        ax.plot(det["xs"], det["ys"], color="#bd1f01", linewidth=2.4, zorder=4)
    if np.isfinite(x_lo) and np.isfinite(x_hi):
        # FWHM: horizontal half-max line + the two crossings.
        ax.hlines(half, x_lo, x_hi, color="#e76300", linewidth=2.4, zorder=5)
        ax.vlines([x_lo, x_hi], 0, half, color="#e76300", linestyle="--",
                  linewidth=1.8, zorder=5)
    if np.isfinite(peak):
        ax.axvline(peak, color="black", linestyle=":", linewidth=1.3, zorder=4)
    for xv in (seed_lo, seed_hi):
        ax.axvline(xv, color="0.45", linestyle=(0, (6, 3)), linewidth=2.0,
                   alpha=0.9, zorder=2)

    ax.set_xlim(0, min(float(edges[-1]), 1.4 * M_WR))
    ax.set_ylim(bottom=0)
    _, y_hi = ax.get_ylim()
    ax.set_ylim(0, y_hi * 1.50)
    ax.set_xlabel(MASS_VAR_LABEL.get(cell.topology, r"$m$") + " [GeV]")
    ax.set_ylabel("Events / bin")
    ax.grid(alpha=0.3)

    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=com, fontsize=16)
    ax.text(0.04, 0.96,
            f"{CH_LAB[cell.channel]}  {cell.topology.capitalize()} SR\n{era}\n"
            rf"$M_{{W_R}}={M_WR:.0f}$, $M_N={cell.M_N:.0f}$ GeV",
            transform=ax.transAxes, fontsize=13, verticalalignment="top")
    ax.text(0.96, 0.96,
            rf"${SIGMA_FWHM_ON}={sigma_fwhm:.0f}$ GeV"
            "\n"
            rf"FWHM $={det['fwhm']:.0f}$ GeV"
            "\n"
            rf"peak $={peak:.0f}$ GeV"
            "\n"
            rf"$[x_L, x_R]=[{x_lo:.0f}, {x_hi:.0f}]$",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85))

    handles = [
        Line2D([0], [0], color="#3f90da", linewidth=1.5, label="MC"),
        Line2D([0], [0], color="#bd1f01", linewidth=2.4, label="RooKeysPdf (KDE)"),
        Line2D([0], [0], color="#e76300", linewidth=2.4, label="FWHM (half-max)"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.3, label="peak"),
        Line2D([0], [0], color="0.45", linestyle=(0, (6, 3)), linewidth=2.0,
               label=rf"seed window $[{s_lo:g}\,M_{{W_R}},\,{s_hi:g}\,M_{{W_R}}]$"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.74),
              fontsize=10, framealpha=0.9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    windows = parse_windows(args.windows) if args.windows else DEFAULT_WINDOWS
    baseline = BASELINE_WINDOW if BASELINE_WINDOW in windows else windows[0]
    base_str = win_str(*baseline)
    com = load_lumi(args.era).get("com", 13.0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / "fwhm_table.csv"

    baseline_det: dict[tuple, tuple] = {}  # (ch,topo,mass) -> (cell, detail)

    if args.plots_only:
        df = pd.read_csv(table_path)
        logger.info("Plots-only: loaded %d rows from %s", len(df), table_path)
    else:
        input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
        masses = load_master_masses(args.mass_csv)
        cells, _ = collect_cells(
            input_dirs, args.channels, args.topologies, masses,
            min_events=args.min_events, boosted_max_x=args.boosted_max_x,
        )
        logger.info("Collected %d cells; windows %s", len(cells),
                    [win_str(*w) for w in windows])
        rows = []
        for i, cell in enumerate(cells, start=1):
            for lo, hi in windows:
                det = keys_fwhm_detail(cell.edges, cell.vals, cell.M_WR,
                                       lo_frac=lo, hi_frac=hi)
                sig = det["fwhm"] / FWHM_TO_GAUSS_SIGMA
                rows.append({
                    "mWR": cell.M_WR, "mN": cell.M_N, "channel": cell.channel,
                    "category": cell.topology, "fit_range": win_str(lo, hi),
                    "peak": det["peak"], "peak_over_mWR": det["peak"] / cell.M_WR,
                    "fwhm": det["fwhm"], "sigma_fwhm": sig,
                    "sigma_fwhm_over_mWR": sig / cell.M_WR,
                    "x_lo": det["x_lo"], "x_hi": det["x_hi"],
                })
                if (lo, hi) == baseline:
                    baseline_det[(cell.channel, cell.topology, cell.mass)] = (cell, det)
            if i % 100 == 0 or i == len(cells):
                logger.info("  KDE %d / %d cells", i, len(cells))
        df = pd.DataFrame(rows, columns=TABLE_COLUMNS)
        df.to_csv(table_path, index=False)
        logger.info("Wrote %s (%d rows)", table_path, len(df))

    # Window-robustness CSV + verdict (folded in from width_window_stability.py).
    window_stability_report(
        df, "sigma_fwhm", windows, baseline,
        args.out_dir / "window_stability.csv",
        channels=args.channels, topologies=args.topologies,
        est_name="σ_FWHM^on")

    df_base = df[df.fit_range == base_str].copy()
    df_base["x"] = df_base["mN"] / df_base["mWR"]
    df_base["M_WR"] = df_base["mWR"]

    # --- Plot 1: scalars vs x (baseline), M_WR colorbar, y-centered ---
    for channel in args.channels:
        for topology in args.topologies:
            sub = df_base[(df_base.channel == channel)
                          & (df_base.category == topology)]
            if sub.empty:
                continue
            sub = sub.sort_values("x")
            plot_scalar_vs_x_by_mwr(
                sub, "sigma_fwhm_over_mWR",
                args.out_dir / "sigma_over_mWR" / f"{channel}_{topology}.png",
                channel=channel, topology=topology, era=args.era, com=com,
                ylabel=rf"${SIGMA_FWHM_ON} / M_{{W_R}}$",
                ylim=centered_ylim(sub["sigma_fwhm_over_mWR"]))
            plot_scalar_vs_x_by_mwr(
                sub, "peak_over_mWR",
                args.out_dir / "peak_over_mWR" / f"{channel}_{topology}.png",
                channel=channel, topology=topology, era=args.era, com=com,
                ylabel=r"peak $/ M_{W_R}$",
                ylim=centered_ylim(sub["peak_over_mWR"]), hlines=[(1.0, "")])

    # --- Plot 2: robustness ratios (M_WR colorbar, shared y-scale) ---
    nonbase = [w for w in windows if w != baseline]
    pivot = df.pivot_table(index=["channel", "category", "mWR", "mN"],
                           columns="fit_range", values="sigma_fwhm").reset_index()
    pivot["x"] = pivot["mN"] / pivot["mWR"]
    pivot["M_WR"] = pivot["mWR"]
    ratio_cols = []
    for lo, hi in nonbase:
        col = f"ratio_{win_str(lo, hi)}"
        pivot[col] = pivot[win_str(lo, hi)] / pivot[base_str]
        ratio_cols.append((col, win_str(lo, hi)))
    allr = np.concatenate([pivot[c].to_numpy() for c, _ in ratio_cols])
    allr = allr[np.isfinite(allr)]
    p1, p99 = np.percentile(allr, [1, 99])
    half = max(max(abs(p1 - 1.0), abs(p99 - 1.0)) * 1.5, 0.05)
    shared_ylim = (1.0 - half, 1.0 + 1.5 * half)
    mwr_lim = (float(pivot["M_WR"].min()), float(pivot["M_WR"].max()))
    for channel in args.channels:
        for topology in args.topologies:
            sub = pivot[(pivot.channel == channel) & (pivot.category == topology)]
            if sub.empty:
                continue
            plot_ratio_overlay_by_mwr(
                sub.sort_values("x"), ratio_cols,
                args.out_dir / "robustness" / f"{channel}_{topology}.png",
                channel=channel, topology=topology, era=args.era, com=com,
                ylabel=rf"${SIGMA_FWHM_ON}$(window) / ${SIGMA_FWHM_ON}${base_str}",
                ylim=shared_ylim, mwr_lim=mwr_lim, hline=1.0, band=0.05)

    # --- Plot 3: per-cell overlays ---
    if baseline_det and not args.no_hist_plots:
        hist_dir = args.out_dir / "histograms"
        items = list(baseline_det.items())
        for j, ((ch, topo, tag), (cell, det)) in enumerate(items, start=1):
            plot_fwhm_overlay(cell, det, hist_dir / f"{ch}_{topo}" / f"{tag}.png",
                              era=args.era, com=com, seed_window=baseline)
            if j % 100 == 0 or j == len(items):
                logger.info("  overlay %d / %d", j, len(items))

    # --- console summary ---
    print(f"\n=== σ_FWHM^on — baseline {base_str} ===")
    for channel in args.channels:
        for topology in args.topologies:
            s = df_base[(df_base.channel == channel)
                        & (df_base.category == topology)]
            if s.empty:
                continue
            n_nan = int(s.sigma_fwhm.isna().sum())
            print(f"  {channel:<5} {topology:<9} ({len(s):>3} cells): "
                  f"med σ_FWHM/M_WR={np.nanmedian(s.sigma_fwhm_over_mWR):.4f}  "
                  f"med peak/M_WR={np.nanmedian(s.peak_over_mWR):.4f}  "
                  f"no-crossing={n_nan}")
    print()


if __name__ == "__main__":
    main()
