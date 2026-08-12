#!/usr/bin/env python3
"""Stage 2 (wr) -- collapse the (m_WR, m_N) width grid to a m_WR-ONLY window map.

The wr_mn parameterization needs both masses (sigma as a function of x = m_N/m_WR
and m_WR). For defining a SEARCH WINDOW we only know the W_R mass, so here we
marginalize over m_N: at each grid m_WR take the median over m_N of the Stage-1
Gaussian peak mu and width sigma, then fit simple LINEAR functions

    mu(m_WR)     = a_mu + b_mu * m_WR
    sigma(m_WR)  = a_s  + b_s  * m_WR

so any m_WR (on- or off-grid, e.g. 2341 GeV) yields a single (mu, sigma) used to
centre the window and constrain the peak.

Two sigma conventions are fit and stored:
  * median        -- median sigma over m_N (central; matches current usage)
  * conservative  -- max sigma over m_N AFTER trimming the super-compressed
                     x = m_N/m_WR > x_trim points (whose width blows up and is
                     unrepresentative); covers the bulk without that tail.

mu's spread over m_N is small (~5-9%), so a single mu(m_WR) is safe.

Outputs (per channel/topology) under --out-dir (default this folder):
  window_params.json                          fitted coefficients + metadata
  fit/{ch}_{topo}_mu.*                         mu vs m_WR, data + linear (median) fit
  fit/{ch}_{topo}_sigma.*                      sigma vs m_WR, data + linear (median) fit
  closure/{ch}_{topo}/mu_pred_vs_meas.*        fit-predicted vs measured (mu)
  closure/{ch}_{topo}/mu_residual.*            fit - measured vs m_WR (mu)
  closure/{ch}_{topo}/sigma_pred_vs_meas.*     fit-predicted vs measured (sigma)
  closure/{ch}_{topo}/sigma_residual.*         fit - measured vs m_WR (sigma)
(in-sample fit closure -- predicted = the linear fit evaluated at each m_WR.)

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                       # repo root
sys.path.insert(0, str(HERE.parents[1] / "4_background_fits"))  # bkg_fit_lib (labels)

from wrplotter.cli_utils import setup_logging                  # noqa: E402

logger = logging.getLogger("parameterize_window")

CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}
TOPO_LAB = {"resolved": "Resolved", "boosted": "Boosted"}
BASELINE_FIT_RANGE = "[0.7,1.3]"


# ---------------------------------------------------------------------------
# Aggregation + fit
# ---------------------------------------------------------------------------

def load_grid(width_csv, channel, topology, fit_range):
    """{m_WR: [(m_N, mu, sigma), ...]} for one (channel, topology)."""
    out = collections.defaultdict(list)
    with open(width_csv) as fh:
        for r in csv.DictReader(fh):
            if (r["channel"] == channel and r["category"] == topology
                    and r["fit_range"] == fit_range):
                out[float(r["mWR"])].append(
                    (float(r["mN"]), float(r["mu_gaus"]), float(r["sigma_gaus"])))
    return out


def aggregate(grid, x_trim):
    """Per-m_WR (mu_median, sigma_median, sigma_conservative). Conservative =
    max sigma after dropping x = m_N/m_WR > x_trim (the super-compressed tail)."""
    masses = np.array(sorted(grid))
    mu_med, sig_med, sig_cons = [], [], []
    for m in masses:
        mus = np.array([mu for _, mu, _ in grid[m]])
        sigs = np.array([s for _, _, s in grid[m]])
        xs = np.array([mn / m for mn, _, _ in grid[m]])
        mu_med.append(np.median(mus))
        sig_med.append(np.median(sigs))
        keep = sigs[xs <= x_trim]
        sig_cons.append(float(keep.max()) if keep.size else float(sigs.max()))
    return (masses, np.array(mu_med), np.array(sig_med), np.array(sig_cons))


def linfit(x, y):
    """Linear fit; returns (slope, intercept) as plain floats."""
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _cms(ax, com=13.6, lumi=None):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=com, fontsize=14)


def plot_fit(masses, mu_med, sig_med, fits, out, *, channel, topology):
    """Two separate figures -- mu(m_WR) and sigma(m_WR) -- each the per-m_WR
    median over m_N plus its linear fit. Written to {out}_mu.* and {out}_sigma.*.
    """
    hep.style.use("CMS")
    grid = np.linspace(masses.min(), masses.max(), 200)
    label = f"{CH_LAB[channel]}  {TOPO_LAB[topology]}"

    for med, key, ylabel, stem in [
        (mu_med, "mu", r"$\mu$  [GeV]", "mu"),
        (sig_med, "sigma_median", r"$\sigma$  [GeV]", "sigma"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(masses, med, c="#1f77b4", s=30, zorder=3,
                   label="median over $m_N$")
        b, a = fits[key]
        ax.plot(grid, a + b * grid, color="#1f77b4", lw=1.8, zorder=2,
                label=fr"fit: ${a:.0f}+{b:.4f}\,m_{{W_R}}$")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"$m_{W_R}$  [GeV]")
        ax.text(0.04, 0.96, label, transform=ax.transAxes,
                ha="left", va="top", fontsize=13)
        ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(0.04, 0.89))
        _cms(ax)
        _save(fig, out.parent / f"{out.name}_{stem}")


def plot_pred_vs_meas(meas, pred, masses, sym, unit, out, *, channel, topology):
    """In-sample predicted (linear fit) vs measured (median over m_N), m_WR on
    the colorbar, with the y=x line."""
    rms = float(np.sqrt(np.mean((pred - meas) ** 2)))
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    lo, hi = min(meas.min(), pred.min()), max(meas.max(), pred.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, zorder=1,
            label="y = x")
    sc = ax.scatter(meas, pred, c=masses, cmap="viridis", s=55,
                    edgecolor="black", linewidth=0.3, vmin=masses.min(),
                    vmax=masses.max(), zorder=3)
    ax.set_xlabel(f"measured {sym} (median over $m_N$) [{unit}]")
    ax.set_ylabel(f"fit-predicted {sym} [{unit}]")
    ax.set_title(f"{CH_LAB[channel]} {TOPO_LAB[topology]} — {sym}: "
                 f"RMS = {rms:.1f} {unit}", fontsize=14)
    ax.legend(fontsize=11, loc="upper left")
    fig.colorbar(sc, ax=ax, pad=0.02).set_label(r"$m_{W_R}$ [GeV]", fontsize=13)
    _save(fig, out)


def plot_residual(masses, resid, sym, unit, out, *, channel, topology):
    """Fit residual (predicted - measured) vs m_WR."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    ax.axhline(0, color="black", lw=0.8)
    sc = ax.scatter(masses, resid, c=masses, cmap="viridis", s=55,
                    edgecolor="black", linewidth=0.3, vmin=masses.min(),
                    vmax=masses.max())
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(f"fit $-$ measured  {sym} [{unit}]")
    ax.set_title(f"{CH_LAB[channel]} {TOPO_LAB[topology]} — {sym} residual",
                 fontsize=14)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label(r"$m_{W_R}$ [GeV]", fontsize=13)
    _save(fig, out)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width-csv", type=Path,
                   default=HERE.parents[1] / "1_signal_widths" / "gaussian"
                   / "gauss_fit_table.csv")
    p.add_argument("--fit-range", default=BASELINE_FIT_RANGE)
    p.add_argument("--x-trim", type=float, default=0.9,
                   help="drop x=m_N/m_WR above this for the conservative sigma")
    p.add_argument("--channels", nargs="+", default=["ee", "mumu"])
    p.add_argument("--topologies", nargs="+", default=["resolved", "boosted"])
    p.add_argument("--out-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    params = {}
    for channel in args.channels:
        for topology in args.topologies:
            grid = load_grid(args.width_csv, channel, topology, args.fit_range)
            if not grid:
                logger.info("%s %s: no rows, skipping", channel, topology)
                continue
            masses, mu_med, sig_med, sig_cons = aggregate(grid, args.x_trim)
            if len(masses) < 2:
                logger.info("%s %s: <2 masses, skipping", channel, topology)
                continue
            fits = {"mu": linfit(masses, mu_med),
                    "sigma_median": linfit(masses, sig_med),
                    "sigma_conservative": linfit(masses, sig_cons)}
            params.setdefault(channel, {})[topology] = {
                "mu": list(fits["mu"]),
                "sigma_median": list(fits["sigma_median"]),
                "sigma_conservative": list(fits["sigma_conservative"]),
                "x_trim": args.x_trim,
                "mwr_min": float(masses.min()), "mwr_max": float(masses.max()),
                "n_mwr": int(len(masses)), "fit_range": args.fit_range}
            tag = f"{channel}_{topology}"
            plot_fit(masses, mu_med, sig_med, fits,
                     args.out_dir / "fit" / tag, channel=channel, topology=topology)
            # in-sample predicted (fit line) vs measured -- 4 separate plots
            cdir = args.out_dir / "closure" / tag
            for sym, meas, key in [(r"$\mu$", mu_med, "mu"),
                                   (r"$\sigma$", sig_med, "sigma_median")]:
                b, a = fits[key]
                pred = a + b * masses
                stem = "mu" if key == "mu" else "sigma"
                plot_pred_vs_meas(meas, pred, masses, sym, "GeV",
                                  cdir / f"{stem}_pred_vs_meas",
                                  channel=channel, topology=topology)
                plot_residual(masses, pred - meas, sym, "GeV",
                              cdir / f"{stem}_residual",
                              channel=channel, topology=topology)
            bm, am = fits["mu"]; bs, as_ = fits["sigma_median"]
            logger.info("%s %s: mu=%.0f+%.4f*mWR, sigma=%.0f+%.4f*mWR (%d masses)",
                        channel, topology, am, bm, as_, bs, len(masses))

    if not params:
        logger.error("No parameterizations produced.")
        sys.exit(1)
    out_json = args.out_dir / "window_params.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(params, fh, indent=2)
    logger.info("Wrote %s", out_json)


if __name__ == "__main__":
    main()
