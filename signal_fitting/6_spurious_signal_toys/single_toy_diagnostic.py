#!/usr/bin/env python3
"""Stage 6 -- SINGLE-toy S+B fit diagnostics (one Poisson toy per window).

Stage 6 (`spurious_signal_toys.py`) draws many Poisson toys of the
background-only MC and keeps only the *distribution* of the fake signal
(`nsp_hist`). This companion renders ONE toy per window as a full fit picture:
the toy pseudo-data, the S+B fit, the background-only component, and the
shaded +-k*sigma window -- so the low-mass window geometry is visible.

At low m_WR the nominal window [m_c-k*sigma, m_c+k*sigma] runs off the low end
of the spectrum, so the fit range is clamped at `fit_min` (default 800 GeV).
That clamp is drawn explicitly: the nominal window is shaded blue, the part
below `fit_min` is hatched red ("clamped off"), and a solid line marks
`fit_min`. This is the picture to look at when asking "how bad is the low-edge
clamp at k=5?".

The toy is the SAME draw Stage 6 uses: identical seed scheme
(`seed * 1_000_003 + mWR*1009 + func_index`) and identical ROOT TRandom3 RNG,
so `--toy-index i` reproduces the (i+1)-th toy of that Stage-6 run bit-for-bit.

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Example (run2, k=5, the low-edge masses):
  python single_toy_diagnostic.py --era RunIISummer20UL18 --dir 20260714_run2_bkgs \
      --channel ee --topology resolved --k 5 --functions expo \
      --masses 1000 1200 1400 1600 1800 \
      --output-dir run2/k5/single_toy_fits
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
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                        # repo root
sys.path.insert(0, str(HERE.parents[0] / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(HERE.parents[0] / "shared"))            # sb_fit, loaders

from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root                # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter              # noqa: E402

import ROOT                                                              # noqa: E402
from bkg_fit_lib import (                                                # noqa: E402
    FUNCS, MASS_VAR, MASS_LABEL, CH_LAB, TOPO_LAB,
    predict, coef_text, load_grid_widths, grid_widths_from_params,
    load_summed_background,
)
from sb_fit import CHECK_KEYS, CHECK_SHORT, fit_splusb, gauss_per_bin    # noqa: E402

logger = logging.getLogger("single_toy")


def draw_toy(mu_bkg, seed, toy_index):
    """Reproduce Stage 6's ROOT TRandom3 stream and return the (toy_index)-th
    per-bin Poisson draw of the background-only expectation."""
    rng = ROOT.TRandom3(seed)
    toy = None
    for _ in range(toy_index + 1):
        toy = np.array([rng.Poisson(mu) for mu in mu_bkg], dtype=float)
    return toy


def plot_single_toy(edges, toy, mu_bkg, res, *, name, m_c, sigma_win,
                    mu_sig, sigma_sig, binwidth, k, lo, hi, fit_min,
                    channel, topology, com, lumi, mWR, toy_index, view_k, out):
    centers = 0.5 * (edges[:-1] + edges[1:])
    npar = FUNCS[name][2]
    nom_lo, nom_hi = m_c - k * sigma_win, m_c + k * sigma_win
    clamped = nom_lo < fit_min - 1e-6

    vlo = max(float(edges[0]), min(nom_lo, fit_min) - 150.0)
    vhi = min(float(edges[-1]), m_c + view_k * sigma_win + 150.0)
    view = (centers >= vlo) & (centers <= vhi)
    fitsel = (centers >= lo) & (centers <= hi) & (toy > 0)

    hep.style.use("CMS")
    fig, ax = plt.subplots()

    # nominal +-k sigma window, and the part clamped away below fit_min
    ax.axvspan(nom_lo, nom_hi, color="#5790fc", alpha=0.12, zorder=0,
               label=fr"$m_c\pm{k:g}\sigma$ window")
    if clamped:
        ax.axvspan(nom_lo, fit_min, facecolor="none", edgecolor="#e42536",
                   hatch="///", linewidth=0.0, alpha=0.55, zorder=1,
                   label="clamped off (below fit min)")
        ax.axvline(fit_min, color="#e42536", lw=1.6, ls="-", zorder=2,
                   label=fr"fit min $= {fit_min:.0f}$ GeV")

    # MC background expectation (grey stairs) and the single Poisson toy (points)
    ax.stairs(mu_bkg, edges, color="#888888", linewidth=1.3, zorder=2,
              label="MC Bkg (Poisson mean)")
    ax.errorbar(centers[fitsel], toy[fitsel], yerr=np.sqrt(toy[fitsel]),
                fmt="o", color="black", ms=4, elinewidth=0.8, capsize=1.5,
                zorder=5, label="Toy data (fit bins)")
    out_of_fit = view & (toy > 0) & ~fitsel
    if out_of_fit.any():
        ax.errorbar(centers[out_of_fit], toy[out_of_fit],
                    yerr=np.sqrt(toy[out_of_fit]), fmt="o",
                    mfc="white", mec="0.5", ecolor="0.7", ms=4,
                    elinewidth=0.8, capsize=1.5, zorder=4,
                    label="Toy data (outside fit)")

    # fitted S+B and its background-only component, over the clamped fit range
    grid = np.linspace(lo, hi, 500)
    nsig = res["nsig"]
    sgrid = nsig * gauss_per_bin(grid, mu_sig, sigma_sig, binwidth)
    ax.plot(grid, predict(name, res["params"][:npar], grid, m_c) + sgrid,
            color="#e42536", lw=2.0, zorder=3, label="B+S fit (this toy)")
    ax.plot(grid, predict(name, res["params"][:npar], grid, m_c),
            color="#1f77b4", ls="--", lw=1.8, zorder=3, label="Bkg component")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_xlim(vlo, vhi)
    pos = toy[view][toy[view] > 0]
    if pos.size:
        ax.set_ylim(max(0.3, pos.min() / 3.0), pos.max() * 30.0)
    ax.set_xlabel(MASS_LABEL[topology])
    ax.set_ylabel(f"Events / {binwidth:.0f} GeV")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=16)

    ne = res["nsig_err"]
    pull = nsig / ne if ne > 0 else float("nan")
    chi2_ndf = res["chi2"] / res["ndf"] if res["ndf"] else float("nan")
    clamp_line = (fr"Fit range clamped: $[{lo:.0f},\,{hi:.0f}]$ "
                  fr"(nominal low $= {nom_lo:.0f}$)" if clamped
                  else fr"Fit range $=[{lo:.0f},\,{hi:.0f}]$ (unclamped)")
    ax.text(0.05, 0.95,
            f"{CH_LAB[channel]}  {TOPO_LAB[topology]}\n"
            fr"Search for $m_{{W_R}} = {mWR:.0f}$ GeV,  toy #{toy_index}" "\n"
            fr"$\mu=m_c={m_c:.0f}$ GeV,  $\sigma={sigma_win:.0f}$ GeV" "\n"
            fr"Window $[m_c-{k:g}\sigma,\,m_c+{k:g}\sigma]=[{nom_lo:.0f},\,{nom_hi:.0f}]$"
            "\n" + clamp_line,
            transform=ax.transAxes, fontsize=13, va="top")
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.97), fontsize=11)

    checks = res["checks"]
    if res["passed"]:
        flag = "fit: PASSED"
    else:
        flag = "fit: FAILED (" + ", ".join(
            CHECK_SHORT[c] for c in CHECK_KEYS if not checks[c]) + ")"
    nsig_str = fr"$\hat{{N}}_{{\rm sig}}={nsig:.1f}\pm{ne:.1f}$"
    stat = ("Bkg: " + FUNCS[name][1] + "\n"
            + coef_text(res["params"][:npar], res["perr"][:npar]) + "\n"
            + nsig_str + "\n" + fr"pull $={pull:+.2f}$" + "\n"
            + fr"$\chi^2/\mathrm{{ndf}}={chi2_ndf:.2f}$" + "\n" + flag)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bb.x1 - 0.02, bb.y0 - 0.03, stat, transform=ax.transAxes,
            fontsize=11, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.7", alpha=0.9))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("  m=%.0f %-6s toy#%d -> Nsig=%.1f+/-%.1f pull=%+.2f chi2/ndf=%.2f "
                "%s -> %s", mWR, name, toy_index, nsig, ne, pull, chi2_ndf,
                "clamped" if clamped else "unclamped", out.with_suffix(".png"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260714_run2_bkgs", help="background dir")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved", choices=["resolved", "boosted"])
    p.add_argument("--k", type=float, default=5.0)
    p.add_argument("--masses", nargs="+", type=float,
                   default=[1000, 1200, 1400, 1600, 1800],
                   help="grid m_WR points to draw (default: the low-edge set)")
    p.add_argument("--functions", nargs="+", default=["expo"])
    p.add_argument("--toy-index", type=int, default=0,
                   help="which toy of the Stage-6 stream to render (0 = first)")
    p.add_argument("--seed", type=int, default=12345,
                   help="Stage-6 base seed (keep default to match its toys)")
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--view-k", type=float, default=6.0,
                   help="x-range half-width in sigma about m_c (>=k to see the window)")
    p.add_argument("--sigma-kind", default="median",
                   choices=["median", "conservative"])
    p.add_argument("--sigma-agg", default="median", choices=["median", "mean"])
    p.add_argument("--window-params", type=Path,
                   default=HERE.parents[0] / "2_width_parameterization" / "wr"
                   / "window_params.json")
    p.add_argument("--width-csv", type=Path,
                   default=HERE.parents[0] / "1_signal_widths" / "gaussian"
                   / "gauss_fit_table.csv")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <script dir>/<run2|run3>/single_toy_fits")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    channel, topology, k = args.channel, args.topology, args.k
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    run_sub = {"RunII": "run2", "Run3": "run3"}[str(info["run"])]
    out_dir = args.output_dir or (HERE / run_sub / "single_toy_fits")

    funcs = [f for f in FUNCS if f in args.functions]
    masses = sorted(float(m) for m in args.masses)
    grid = grid_widths_from_params(
        args.window_params, channel, topology, masses, args.sigma_kind)

    bkg_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    region = f"wr_{channel}_{topology}_sr"
    factor = max(1, round(args.bin_width / 10.0))
    edges, values, _ = load_summed_background(
        bkg_dirs, region, MASS_VAR[topology], factor)
    binwidth = float(edges[1] - edges[0])
    mu_bkg = [float(v) if (math.isfinite(v) and v > 0.0) else 0.0 for v in values]

    tag = f"{channel}_{topology}"
    logger.info("Single-toy diagnostics: %s %s k=%g, toy#%d, masses %s",
                channel, topology, k, args.toy_index, [int(m) for m in masses])
    for mWR in masses:
        m_c, sigma_win = grid[mWR]
        lo = max(m_c - k * sigma_win, args.fit_min)
        hi = min(m_c + k * sigma_win, args.fit_max)
        for name in funcs:
            seed = args.seed * 1_000_003 + int(mWR) * 1009 + funcs.index(name)
            toy = draw_toy(mu_bkg, seed, args.toy_index)
            res = fit_splusb(name, edges, toy, lo, hi, m_c, sigma_win,
                             m_c, sigma_win, binwidth, k)
            if res is None:
                logger.warning("  m=%.0f %-6s -> fit returned None (too few bins)",
                               mWR, name)
                continue
            out = out_dir / tag / name / f"m{int(mWR)}"
            plot_single_toy(
                edges, toy, np.array(mu_bkg), res, name=name, m_c=m_c,
                sigma_win=sigma_win, mu_sig=m_c, sigma_sig=sigma_win,
                binwidth=binwidth, k=k, lo=lo, hi=hi, fit_min=args.fit_min,
                channel=channel, topology=topology, com=com, lumi=lumi,
                mWR=mWR, toy_index=args.toy_index, view_k=args.view_k, out=out)
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
