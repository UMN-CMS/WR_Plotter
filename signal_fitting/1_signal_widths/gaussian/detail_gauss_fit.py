#!/usr/bin/env python3
"""Per-method deep dive for the iterative-core single-Gaussian width, σ_gauss^on.

For every signal cell (both channels, both topologies; boosted restricted to the
highly-boosted regime x = M_N/M_WR < 0.1, discovered from disk) and every seed
window in --windows, run the iterative core fit
(shape_estimators.gaussian_core_fit, RooFit):

  1. seed the fit range at [lo, hi]*M_WR,
  2. fit a RooGaussian with mu floating,
  3. refit in [mu - 2σ, mu + 2σ],
  4. iterate until mu and sigma change by < 2%,
  5. record mu, sigma, sigma_err, χ²/ndf (weighted MC errors), fit status, cov.

Table: outputs/width_estimators/gauss_fit/gauss_fit_table.csv with columns
  mWR, mN, channel, category, fit_range, mu_gaus, mu_gaus_over_mWR,
  sigma_gaus, sigma_gaus_over_mWR, sigma_err,
  chi2_ndf, fit_status, cov_status

Plots (outputs/width_estimators/gauss_fit/), all baseline window, M_WR colorbar:
  1. per-cell scalars vs x = M_N/M_WR, one file per channel/topology:
        sigma_gauss_over_mWR_*   σ_gauss^on / M_WR   (y-centered)
        mu_gauss_over_mWR_*      μ_gauss^on / M_WR   (y-centered)
        chi2_ndf_*               χ²/ndf  (high => Gaussian stable but wrong)
  2. robustness_{channel}_{topology}.{png,pdf}
        σ_gauss(window)/σ_gauss([0.8,1.2]) vs x for the other two windows,
        marker per window, M_WR colorbar, one shared y-scale across all panels.
  3. histograms/{channel}_{topology}/{mass}.png   (one per cell)
        histogram + fitted Gaussian, with the converged fit range [μ±2σ]
        (red band) and the seed window [0.8,1.2]*M_WR (grey dashed) drawn as
        distinct, separately-labeled elements.

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Example:
    python signal_fitting/1_signal_samples/detail_gauss_fit.py -v
"""
from __future__ import annotations

import argparse
import logging
import math
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
    CH_LAB, collect_cells, load_master_masses,
    gaussian_core_fit, gaussian_chi2_ndf,
    plot_scalar_vs_x_by_mwr, plot_ratio_overlay_by_mwr,
    window_stability_report,
)

logger = logging.getLogger(__name__)

MASS_VAR_LABEL = {"resolved": r"$m_{\ell\ell jj}$", "boosted": r"$m_{\ell J}$"}
DEFAULT_WINDOWS = [(0.70, 1.30), (0.80, 1.20), (0.85, 1.15)]
BASELINE_WINDOW = (0.80, 1.20)
SIGMA_GAUSS_ON = r"\sigma_{\rm gauss}^{\rm on}"
MU_GAUSS_ON = r"\mu_{\rm gauss}^{\rm on}"


def centered_ylim(vals, frac=0.30):
    """y-range that puts the full data range vertically centered on the axis.

    Uses the actual min/max (not percentiles) so every point is on-scale, with
    a `frac` margin on each side. Suitable for the well-behaved fit quantities
    (sigma/M_WR, mu/M_WR) which have no pathological outliers.
    """
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi <= lo:
        hi = lo + 1e-6
    m = (hi - lo) * frac
    return (lo - m, hi + m)

TABLE_COLUMNS = [
    "mWR", "mN", "channel", "category", "fit_range",
    "mu_gaus", "mu_gaus_over_mWR",
    "sigma_gaus", "sigma_gaus_over_mWR",
    "sigma_err",
    "chi2_ndf", "fit_status", "cov_status",
]


def win_str(lo, hi) -> str:
    return f"[{lo:g},{hi:g}]"


def parse_windows(specs):
    return [tuple(float(v) for v in s.split(",")) for s in specs]


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
    p.add_argument("--no-hist-plots", action="store_true",
                   help="Skip the per-cell histogram+fit overlays (one per cell).")
    p.add_argument("--plots-only", action="store_true",
                   help="Skip the fits; rebuild the summary plots (1 & 2) from "
                        "the existing gauss_fit_table.csv.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-cell histogram + Gaussian-fit overlay
# ---------------------------------------------------------------------------

def plot_fit_overlay(cell, fit, out_path, *, era, com, seed_window):
    """Histogram + fitted Gaussian, with the seed window and the converged
    fit range drawn as two clearly distinct, separately-labeled elements.

      * red curve + light-red shaded band = the Gaussian and the *converged*
        fit range [mu - 2 sigma, mu + 2 sigma] (what the fit actually used);
      * grey dashed lines = the *seed* window [0.8, 1.2]*M_WR we started the
        iteration from (drawn for reference; not used in the final fit).
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    edges, vals = cell.edges, cell.vals
    mu, sigma, lo, hi = fit["mu"], fit["sigma"], fit["lo"], fit["hi"]
    M_WR = cell.M_WR
    s_lo, s_hi = seed_window
    seed_lo, seed_hi = s_lo * M_WR, s_hi * M_WR
    centers = 0.5 * (edges[:-1] + edges[1:])

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.stairs(np.maximum(vals, 0.0), edges, color="#3f90da", linewidth=1.5)

    # Converged fit range: light-red band + the Gaussian curve over it.
    ax.axvspan(lo, hi, color="#bd1f01", alpha=0.10, zorder=0)
    sel = (centers >= lo) & (centers <= hi)
    n_in = float(np.maximum(vals[sel], 0.0).sum())
    bin_w = float(edges[1] - edges[0])
    norm = 0.5 * (math.erf((hi - mu) / (sigma * math.sqrt(2)))
                  - math.erf((lo - mu) / (sigma * math.sqrt(2))))
    xs = np.linspace(lo, hi, 400)
    pdf = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    ax.plot(xs, n_in * bin_w * pdf / norm if norm > 0 else pdf * 0,
            color="#bd1f01", linewidth=2.4, zorder=4)
    ax.axvline(mu, color="black", linestyle=":", linewidth=1.3, zorder=4)

    # Seed window: grey dashed verticals (reference only).
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
            rf"${SIGMA_GAUSS_ON}={sigma:.0f}\pm{fit['sigma_err']:.0f}$ GeV"
            "\n"
            rf"$\mu={mu:.0f}$ GeV"
            "\n"
            rf"fit range $[{lo:.0f},{hi:.0f}]$ GeV"
            "\n"
            rf"$\chi^2/$ndf$={fit['chi2_ndf']:.1f}$"
            "\n"
            rf"status$={fit['fit_status']}$, cov$={fit['cov_status']}$",
            transform=ax.transAxes, fontsize=11,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85))

    handles = [
        Line2D([0], [0], color="#3f90da", linewidth=1.5, label="MC"),
        Line2D([0], [0], color="#bd1f01", linewidth=2.4, label="Gaussian core fit"),
        Patch(facecolor="#bd1f01", alpha=0.10,
              label=r"fit range $[\mu-2\sigma,\ \mu+2\sigma]$"),
        Line2D([0], [0], color="black", linestyle=":", linewidth=1.3, label=r"$\mu$"),
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
    table_path = args.out_dir / "gauss_fit_table.csv"

    # baseline fit result per cell-key, kept for the overlays.
    baseline_fit: dict[tuple, tuple] = {}

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
                fit = gaussian_core_fit(cell.edges, cell.vals, cell.M_WR, lo, hi)
                chi2_ndf, _ndf = gaussian_chi2_ndf(
                    cell.edges, cell.vals, cell.variances,
                    fit["mu"], fit["sigma"], fit["lo"], fit["hi"])
                fit["chi2_ndf"] = chi2_ndf
                rows.append({
                    "mWR": cell.M_WR, "mN": cell.M_N, "channel": cell.channel,
                    "category": cell.topology, "fit_range": win_str(lo, hi),
                    "mu_gaus": fit["mu"], "mu_gaus_over_mWR": fit["mu"] / cell.M_WR,
                    "sigma_gaus": fit["sigma"],
                    "sigma_gaus_over_mWR": fit["sigma"] / cell.M_WR,
                    "sigma_err": fit["sigma_err"],
                    "chi2_ndf": chi2_ndf,
                    "fit_status": fit["fit_status"], "cov_status": fit["cov_status"],
                })
                if (lo, hi) == baseline:
                    baseline_fit[(cell.channel, cell.topology, cell.mass)] = (cell, fit)
            if i % 100 == 0 or i == len(cells):
                logger.info("  fit %d / %d cells", i, len(cells))
        df = pd.DataFrame(rows, columns=TABLE_COLUMNS)
        df.to_csv(table_path, index=False)
        logger.info("Wrote %s (%d rows)", table_path, len(df))

    # Window-robustness CSV + verdict (folded in from width_window_stability.py).
    window_stability_report(
        df, "sigma_gaus", windows, baseline,
        args.out_dir / "window_stability.csv",
        channels=args.channels, topologies=args.topologies,
        est_name="σ_gauss^on (iter core)")

    df_base = df[df.fit_range == base_str].copy()
    df_base["x"] = df_base["mN"] / df_base["mWR"]
    df_base["M_WR"] = df_base["mWR"]  # for the shared plotter

    # --- Plot 1: per-cell scalars vs x (baseline), M_WR colorbar ---
    #   sigma_gauss^on/M_WR and mu_gauss^on/M_WR are y-centered per panel so the
    #   band sits in the middle of each plot; chi2/ndf uses ONE shared y-range
    #   across all four panels (so resolved vs boosted are directly comparable),
    #   with the chi2/ndf = 1 "good fit" reference line.
    chi2_all = df_base["chi2_ndf"].to_numpy()
    chi2_all = chi2_all[np.isfinite(chi2_all)]
    chi2_ylim = (0.0, float(np.percentile(chi2_all, 99)) * 1.1) if chi2_all.size else None

    for channel in args.channels:
        for topology in args.topologies:
            sub = df_base[(df_base.channel == channel)
                          & (df_base.category == topology)]
            if sub.empty:
                continue
            sub = sub.sort_values("x")
            plot_scalar_vs_x_by_mwr(
                sub, "sigma_gaus_over_mWR",
                args.out_dir / "sigma_over_mWR" / f"{channel}_{topology}.png",
                channel=channel, topology=topology, era=args.era, com=com,
                ylabel=rf"${SIGMA_GAUSS_ON} / M_{{W_R}}$",
                ylim=centered_ylim(sub["sigma_gaus_over_mWR"]))
            plot_scalar_vs_x_by_mwr(
                sub, "mu_gaus_over_mWR",
                args.out_dir / "mu_over_mWR" / f"{channel}_{topology}.png",
                channel=channel, topology=topology, era=args.era, com=com,
                ylabel=rf"${MU_GAUSS_ON} / M_{{W_R}}$",
                ylim=centered_ylim(sub["mu_gaus_over_mWR"]),
                hlines=[(1.0, "")])
            plot_scalar_vs_x_by_mwr(
                sub, "chi2_ndf",
                args.out_dir / "chi2_ndf" / f"{channel}_{topology}.png",
                channel=channel, topology=topology, era=args.era, com=com,
                ylabel=r"$\chi^2/\mathrm{ndf}$",
                hlines=[(1.0, "good fit")], ylim=chi2_ylim)

    # --- Plot 2: robustness ratios vs baseline (M_WR colorbar, shared y-scale) ---
    nonbase = [w for w in windows if w != baseline]
    pivot = df.pivot_table(index=["channel", "category", "mWR", "mN"],
                           columns="fit_range", values="sigma_gaus").reset_index()
    pivot["x"] = pivot["mN"] / pivot["mWR"]
    pivot["M_WR"] = pivot["mWR"]
    ratio_cols = []
    for lo, hi in nonbase:
        col = f"ratio_{win_str(lo, hi)}"
        pivot[col] = pivot[win_str(lo, hi)] / pivot[base_str]
        ratio_cols.append((col, win_str(lo, hi)))
    # One shared y-range (robust 1–99% bulk, symmetric about 1) and one shared
    # M_WR colorbar range, so all four panels are directly comparable.
    allr = np.concatenate([pivot[c].to_numpy() for c, _ in ratio_cols])
    allr = allr[np.isfinite(allr)]
    p1, p99 = np.percentile(allr, [1, 99])
    # Margin set so the worst near-bulk cells (the high-x resolved tail) stay on
    # scale; a 1/99-pct base keeps the few pathological extreme-boosted outliers
    # (ratio up to ~2) from blowing up the range — they remain annotated off-scale.
    half = max(max(abs(p1 - 1.0), abs(p99 - 1.0)) * 1.5, 0.05)
    # Extra top headroom so the upper-right legend clears the points.
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
                ylabel=rf"${SIGMA_GAUSS_ON}$(window) / ${SIGMA_GAUSS_ON}${base_str}",
                ylim=shared_ylim, mwr_lim=mwr_lim, hline=1.0, band=0.05)

    # --- Per-cell histogram + Gaussian-fit overlays (all cells) ---
    if baseline_fit and not args.no_hist_plots:
        hist_dir = args.out_dir / "histograms"
        items = list(baseline_fit.items())
        for j, ((ch, topo, tag), (cell, fit)) in enumerate(items, start=1):
            plot_fit_overlay(
                cell, fit, hist_dir / f"{ch}_{topo}" / f"{tag}.png",
                era=args.era, com=com, seed_window=baseline)
            if j % 100 == 0 or j == len(items):
                logger.info("  overlay %d / %d", j, len(items))

    # --- console summary ---
    print(f"\n=== σ_gauss^on (iter core) — baseline {base_str} ===")
    for channel in args.channels:
        for topology in args.topologies:
            s = df_base[(df_base.channel == channel)
                        & (df_base.category == topology)]
            if s.empty:
                continue
            bad = int(((s.fit_status != 0) | (s.cov_status != 3)).sum())
            print(f"  {channel:<5} {topology:<9} ({len(s):>3} cells): "
                  f"med σ/M_WR={np.nanmedian(s.sigma_gaus_over_mWR):.4f}  "
                  f"med χ²/ndf={np.nanmedian(s.chi2_ndf):.2f}  "
                  f"bad fit/cov={bad}")
    print()


if __name__ == "__main__":
    main()
