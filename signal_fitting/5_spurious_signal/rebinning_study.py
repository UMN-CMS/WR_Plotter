#!/usr/bin/env python3
"""Stage 5 -- spurious-signal rebinning study.

Does the BIN WIDTH matter for the fake (spurious) yield? This is the Stage-5
spurious-signal fit (background TF1 + fixed-shape Gaussian fit to the
background-only MC template inside the [mu +/- k*sigma] window, NO signal
injected), run over a *sweep* of bin widths. For each (channel, topology),
function, and grid mass m_WR we record the spurious yield N_sp, its pull
N_sp/sigma(N_sp), and chi2/ndf at every bin width, then plot how each varies
with the binning.

The window, recentering pivot, and fixed Gaussian shape are identical to
spurious_signal.py and do NOT depend on the bin width -- only the histogram the
fit sees is rebinned. So any change in N_sp with bin width is a genuine
binning artefact (empty-bin dropping at fine binning, shape information lost at
coarse binning), not a change in the analysis definition.

Outputs (namespaced by {channel}_{topology}):
  rebinning_study/{ch}_{topo}/yield_vs_mass_{fn}.*     N_sp vs m_WR, curve/binwidth
  rebinning_study/{ch}_{topo}/pull_vs_mass_{fn}.*      pull vs m_WR, curve/binwidth
  rebinning_study/{ch}_{topo}/yield_vs_binwidth_{fn}.* N_sp vs binwidth, curve/mass
  rebinning_table_{ch}_{topo}.csv                      everything

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                       # repo root
sys.path.insert(0, str(HERE.parents[0] / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(HERE.parents[0] / "shared"))           # sb_fit, loaders

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402

from bkg_fit_lib import (                                               # noqa: E402
    FUNCS, MASS_VAR, CH_LAB, TOPO_LAB,
    load_grid_widths, grid_widths_from_params, load_summed_background,
)
from measure_fwhm import parse_masses                                   # noqa: E402
from shape_estimators import load_master_masses                         # noqa: E402
from sb_fit import fit_splusb, pick_signal_tag                          # noqa: E402

logger = logging.getLogger("rebinning_study")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _bw_colors(bin_widths):
    cmap = matplotlib.colormaps["viridis"]
    n = max(len(bin_widths) - 1, 1)
    return {bw: cmap(i / n) for i, bw in enumerate(sorted(bin_widths))}


def _hep():
    import mplhep as hep
    hep.style.use("CMS")
    return hep


def _cms(ax, hep, lumi, com):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)


def _robust_ylim(ax, values, pad=0.25, symmetric=True):
    """Set y-limits from the 5-95 percentile of central values (ignoring the
    error bars), so a few high-mass points with huge uncertainties don't crush
    the scale. Symmetric about zero by default."""
    v = np.asarray([x for x in values if np.isfinite(x)])
    if v.size == 0:
        return
    lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    span = max(hi - lo, 1e-6)
    lo, hi = lo - pad * span, hi + pad * span
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    ax.set_ylim(lo, hi)


def plot_vs_mass(name, by_bw, key, ylabel, out, *, channel, topology, com, lumi,
                 k, bands=False):
    """`key` (nsig or pull) vs m_WR, one curve per bin width."""
    hep = _hep()
    fig, ax = plt.subplots()
    if bands:
        ax.axhspan(-0.5, 0.5, color="#f7d600", alpha=0.20, lw=0, zorder=0)
        ax.axhspan(-0.2, 0.2, color="#74c476", alpha=0.30, lw=0, zorder=0)
    ax.axhline(0.0, color="black", lw=0.8, zorder=1)
    colors = _bw_colors(list(by_bw))
    allcentral = []
    for bw in sorted(by_bw):
        pts = [p for p in by_bw[bw] if np.isfinite(p[key])]
        if not pts:
            continue
        m = [p["mWR"] for p in pts]
        y = [p[key] for p in pts]
        allcentral += y
        ye = [p.get(key + "_err") for p in pts] if key + "_err" in pts[0] else None
        ax.errorbar(m, y, yerr=ye, fmt="o-", color=colors[bw], ms=3.5, lw=1.3,
                    elinewidth=0.7, capsize=1.2, label=f"{bw:.0f}")
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(ylabel)
    if bands:
        ax.set_ylim(-1.5, 1.5)
    else:
        _robust_ylim(ax, allcentral)
    ax.text(0.03, 0.95,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            f"{name}: {FUNCS[name][1]}\n"
            fr"$m_{{c}}\pm{k:g}\sigma$ window",
            transform=ax.transAxes, va="top", fontsize=12)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="bin width\n[GeV]")
    _cms(ax, hep, lumi, com)
    _save(fig, out)


def plot_yield_vs_binwidth(name, by_mass, out, *, channel, topology, com, lumi, k):
    """Spurious yield vs bin width, one curve per m_WR -- the direct view of
    'does the bin size matter'."""
    hep = _hep()
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="black", lw=0.8, ls="--", zorder=1)
    masses = sorted(by_mass)
    cmap = matplotlib.colormaps["plasma"]
    nm = max(len(masses) - 1, 1)
    allcentral = []
    for i, mWR in enumerate(masses):
        pts = [p for p in by_mass[mWR] if np.isfinite(p["nsig"])]
        if not pts:
            continue
        pts.sort(key=lambda p: p["bin_width"])
        bw = [p["bin_width"] for p in pts]
        y = [p["nsig"] for p in pts]
        allcentral += y
        ye = [p["nsig_err"] for p in pts]
        ax.errorbar(bw, y, yerr=ye, fmt="o-", color=cmap(i / nm), ms=3.5,
                    lw=1.3, elinewidth=0.7, capsize=1.2,
                    label=fr"{mWR:.0f}")
    ax.set_xlabel("bin width [GeV]")
    ax.set_ylabel(r"spurious yield $N_{\rm sp}$ [events]")
    _robust_ylim(ax, allcentral)
    ax.text(0.03, 0.95,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            f"{name}: {FUNCS[name][1]}\n"
            fr"$m_{{c}}\pm{k:g}\sigma$ window",
            transform=ax.transAxes, va="top", fontsize=12)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title=r"$m_{W_R}$ [GeV]", ncol=2)
    _cms(ax, hep, lumi, com)
    _save(fig, out)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20260317_lo_dy", help="background dir")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--mn-frac", type=float, default=0.5,
                   help="Gaussian uses the signal point with m_N closest to this*m_WR")
    p.add_argument("--k", type=float, default=3.0,
                   help="window half-width in sigma (fit is INSIDE the window only)")
    p.add_argument("--window-source", default="param",
                   choices=["param", "measured"],
                   help="'param' (default): window mu/sigma from the Stage-2 "
                        "linear parameterization. 'measured': per-m_WR aggregate "
                        "of the Stage-1 widths.")
    p.add_argument("--sigma-kind", default="median",
                   choices=["median", "conservative"],
                   help="Which fitted sigma to use when --window-source param.")
    p.add_argument("--sigma-agg", default="median",
                   choices=["median", "mean", "max", "min"],
                   help="Aggregation over M_N when --window-source measured.")
    p.add_argument("--bin-widths", nargs="+", type=float,
                   default=[20.0, 50.0, 100.0, 150.0, 200.0],
                   help="bin widths [GeV] to sweep (rounded to multiples of 10)")
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--mass-min", type=float, default=1000.0)
    p.add_argument("--mass-max", type=float, default=6000.0)
    p.add_argument("--functions", nargs="+", default=list(FUNCS),
                   choices=list(FUNCS))
    p.add_argument("--width-csv", type=Path,
                   default=HERE.parents[0] / "1_signal_widths" / "gaussian"
                   / "gauss_fit_table.csv")
    p.add_argument("--window-params", type=Path,
                   default=HERE.parents[0] / "2_width_parameterization" / "wr"
                   / "window_params.json",
                   help="Stage-2 linear window parameterization (window_source param).")
    p.add_argument("--mass-csv", type=Path,
                   default=HERE.parents[0] / "master_masses.csv")
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    channel, topology, k = args.channel, args.topology, args.k
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    funcs = [f for f in FUNCS if f in args.functions]

    measured = load_grid_widths(args.width_csv, channel, topology, args.sigma_agg)
    masses = sorted(m for m in measured if args.mass_min <= m <= args.mass_max)
    if args.window_source == "param":
        grid = grid_widths_from_params(
            args.window_params, channel, topology, masses, args.sigma_kind)
        logger.info("Window source: Stage-2 linear fit (%s sigma) <- %s",
                    args.sigma_kind, args.window_params)
    else:
        grid = measured
        logger.info("Window source: measured per-m_WR %s over M_N", args.sigma_agg)
    sig_tags = load_master_masses(args.mass_csv, topology)
    bkg_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    region = f"wr_{channel}_{topology}_sr"

    # de-dup bin widths after rounding to the 10-GeV native granularity
    bin_widths = sorted({max(1, round(bw / 10.0)) * 10.0 for bw in args.bin_widths})
    logger.info("Rebinning study: %s %s -- %d masses, funcs %s, bin widths %s",
                channel, topology, len(masses), funcs, bin_widths)

    tag = f"{channel}_{topology}"
    rows = []
    for bw in bin_widths:
        factor = max(1, round(bw / 10.0))
        edges, values, _ = load_summed_background(
            bkg_dirs, region, MASS_VAR[topology], factor)
        centers = 0.5 * (edges[:-1] + edges[1:])
        binwidth = float(edges[1] - edges[0])
        for mWR in masses:
            m_c, sigma_win = grid[mWR]
            stag = pick_signal_tag(sig_tags, mWR, args.mn_frac)
            if stag is None:
                continue
            mWR_i, mN_i = parse_masses(stag)
            # injected-signal shape from the same m_WR linear fit as the window
            mu_sig, sigma_sig = m_c, sigma_win
            lo = max(m_c - k * sigma_win, args.fit_min)
            hi = min(m_c + k * sigma_win, args.fit_max)
            win = ((centers >= m_c - k * sigma_win)
                   & (centers <= m_c + k * sigma_win))
            b_window = float(values[win].sum())
            for name in funcs:
                res = fit_splusb(name, edges, values, lo, hi, m_c, sigma_win,
                                 mu_sig, sigma_sig, binwidth, k)
                base = {"channel": channel, "topology": topology,
                        "function": name, "bin_width": binwidth, "mWR": mWR,
                        "signal_tag": stag, "m_N": mN_i,
                        "m_c": round(m_c, 1),
                        "sigma_win": round(sigma_win, 2),
                        "fit_lo": round(lo, 1), "fit_hi": round(hi, 1),
                        "B_window": round(b_window, 3)}
                if res is None:
                    rows.append({**base, "fit_ok": False, "fit_passed": "",
                                 "N_spur": "", "N_spur_err": "", "pull": "",
                                 "spur_over_sqrtB": "", "chi2_ndf": "",
                                 "n_fit_bins": ""})
                    logger.debug("  bw=%.0f m=%.0f %-6s -> fit failed",
                                 binwidth, mWR, name)
                    continue
                nsp, nerr = res["nsig"], res["nsig_err"]
                pull = nsp / nerr if nerr > 0 else float("nan")
                rel = nsp / np.sqrt(b_window) if b_window > 0 else float("nan")
                cn = res["chi2"] / res["ndf"] if res["ndf"] > 0 else float("nan")
                rows.append({
                    **base, "fit_ok": True, "fit_passed": res["passed"],
                    "N_spur": round(nsp, 4),
                    "N_spur_err": round(nerr, 4), "pull": round(pull, 4),
                    "spur_over_sqrtB": round(rel, 4),
                    "chi2_ndf": round(cn, 3) if res["ndf"] > 0 else "",
                    "n_fit_bins": res["nbins"]})
                logger.info("  bw=%3.0f m=%.0f %-6s N_sp=%8.1f+/-%6.1f "
                            "pull=%+.3f nbins=%d",
                            binwidth, mWR, name, nsp, nerr, pull, res["nbins"])

    if not rows:
        logger.error("No results produced.")
        sys.exit(1)

    ok = [r for r in rows if r["fit_ok"] and r["fit_passed"]]   # passed-only
    for name in funcs:
        fn_rows = [r for r in ok if r["function"] == name]
        if not fn_rows:
            continue
        by_bw = {}
        by_mass = {}
        for r in fn_rows:
            pt = {"mWR": r["mWR"], "bin_width": r["bin_width"],
                  "nsig": r["N_spur"], "nsig_err": r["N_spur_err"],
                  "pull": r["pull"]}
            by_bw.setdefault(r["bin_width"], []).append(pt)
            by_mass.setdefault(r["mWR"], []).append(pt)
        outd = args.output_dir / "rebinning_study" / tag
        plot_vs_mass(name, by_bw, "nsig",
                     r"spurious yield $N_{\rm sp}$ [events]",
                     outd / f"yield_vs_mass_{name}", channel=channel,
                     topology=topology, com=com, lumi=lumi, k=k)
        plot_vs_mass(name, by_bw, "pull",
                     r"spurious signal / $\sigma_{N_{\rm sp}}$  (pull)",
                     outd / f"pull_vs_mass_{name}", channel=channel,
                     topology=topology, com=com, lumi=lumi, k=k, bands=True)
        plot_yield_vs_binwidth(name, by_mass,
                               outd / f"yield_vs_binwidth_{name}",
                               channel=channel, topology=topology, com=com,
                               lumi=lumi, k=k)

    csv_path = args.output_dir / f"rebinning_table_{tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s", csv_path)
    logger.info("Done. Outputs in %s", args.output_dir / "rebinning_study" / tag)


if __name__ == "__main__":
    main()
