#!/usr/bin/env python3
"""Stage 6 (Asimov) -- signal-injection bias study (real signal MC injected).

This is the Asimov variant: the expected signal template is injected with NO
statistical fluctuation and fit once, giving the central recovery bias. The toy
counterpart (../toys/signal_injection_toys.py) Poisson-fluctuates the pseudo-data
over many toys to get the pull width / coverage as well.

For each (channel, topology) and grid mass m_WR, we inject the REAL signal MC
shape (the WR{m_WR}_N{m_N} sample with m_N closest to mn_frac*m_WR, default
m_N = m_WR/2) onto the MC background, then fit

    background TF1 (Stage-4 recentered)  +  fixed-shape Gaussian(mu, sigma)

INSIDE the window [mu - k*sigma, mu + k*sigma] only -- the same range as the
Stage-4 background fits (no sidebands). The Gaussian (mu, sigma from Stage-1/2)
is the analysis signal
model; the injected bump is the true MC shape, so the fit tests whether the
Gaussian recovers the injected yield -- i.e. the signal-model bias, on top of
the background bias.

Injection is Asimov (no fluctuation): the signal template is normalized to unit
area and scaled to a chosen number of events N, then added to the background.
The signal MC is NOT xsec/lumi-scaled, but that cancels in the unit-area shape,
so N is an arbitrary chosen test strength (events), not a predicted yield.

  N = 0     -> spurious signal (background-only fit; what fake signal appears)
  N = 10    -> a small realistic injection (does the Gaussian recover ~10?)
  N = 1000  -> an extreme injection (the Gaussian vs the true shape is obvious)

Outputs are split by injection level into N{N}/ directories:
  N{N}/fit_diagnostics/{ch}_{topo}/{fn}/m{mWR}.*   the S+B fit (data + B + B+S)
  N{N}/fitted_yield_vs_mass/{ch}_{topo}.*          fitted N_sig vs m_WR per fn
  N{N}/injection_table_{ch}_{topo}.csv             everything

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
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

HERE = Path(__file__).resolve().parent              # .../8_signal_injection_study/asimov
SIGDIR = HERE.parents[1]                             # signal_fitting
sys.path.insert(0, str(HERE.parents[2]))            # repo root
sys.path.insert(0, str(SIGDIR / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(SIGDIR / "shared"))         # signal loaders

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter             # noqa: E402

import ROOT                                                             # noqa: E402,F401
from bkg_fit_lib import (                                               # noqa: E402
    FUNCS, MASS_VAR, MASS_LABEL, CH_LAB, TOPO_LAB,
    predict, coef_text, fit_model, load_grid_widths, load_summed_background,
)
from measure_fwhm import (                                              # noqa: E402
    build_region_name, build_hist_key, parse_masses,
)
from shape_estimators import load_master_masses                         # noqa: E402
from sb_fit import (                                                    # noqa: E402
    fit_splusb, gauss_per_bin, load_point_widths, pick_signal_tag,
    load_signal_shape,
)

logger = logging.getLogger("signal_injection")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_fit_diagnostic(edges, data, bkg, lo, hi, name, params, perr, bkg_params,
                        m_c, sigma_win, mu_sig, sigma_sig, binwidth, k, nsig,
                        nsig_err, n_inj, sig_tag, chi2_ndf, out, *, channel,
                        topology, com, lumi, mWR):
    """S+B fit confined to the window, in the in_window_fit style and with the
    SAME x/y axis limits (x-range and y-range computed from the background) so
    the two are directly comparable. Window + background recentering use the
    median (m_c); the Gaussian uses the injected point's (mu_sig, sigma_sig).
    The Stage-4 background-only fit is overlaid."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    npar = FUNCS[name][2]
    span = hi - lo
    vlo, vhi = lo - 1.3 * span, hi + 1.3 * span      # match in_window_fit x-range
    view = (centers >= vlo) & (centers <= vhi)
    fitsel = (centers >= lo) & (centers <= hi) & (data > 0)
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    grid = np.linspace(lo, hi, 500)
    sgrid = nsig * gauss_per_bin(grid, mu_sig, sigma_sig, binwidth)
    ax.stairs(bkg, edges, color="#888888", linewidth=1.3, zorder=2,
              label="MC Bkg")
    ax.stairs(data, edges, color="#888888", linestyle="--", linewidth=1.3,
              zorder=2, label="MC Bkg + MC Sig")
    ax.plot(grid, predict(name, params[:npar], grid, m_c) + sgrid,
            color="#e42536", lw=2.0, zorder=3, label="B+S fit")
    if bkg_params is not None:
        ax.plot(grid, predict(name, bkg_params[:npar], grid, m_c),
                color="#1f77b4", ls="--", lw=1.8, zorder=3,
                label="Bkg-only fit (in-window)")
    ax.axvspan(m_c - k * sigma_win, m_c + k * sigma_win, color="#5790fc",
               alpha=0.12, zorder=0, label=fr"$m_{{c}}\pm{k:g}\sigma$")
    ax.errorbar(centers[fitsel], data[fitsel], yerr=np.sqrt(data[fitsel]),
                fmt="o", color="black", ms=4, elinewidth=0.8, capsize=1.5,
                zorder=4, label="Pseudodata (S+B)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_xlim(vlo, vhi)
    pos = bkg[view][bkg[view] > 0]                   # y-range from background (match Stage 4)
    if pos.size:
        ax.set_ylim(pos.min() / 3.0, pos.max() * 30.0)
    ax.set_xlabel(MASS_LABEL[topology])
    ax.set_ylabel(f"Events / {binwidth:.0f} GeV")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=16)
    win_lo, win_hi = m_c - k * sigma_win, m_c + k * sigma_win
    m_N = parse_masses(sig_tag)[1]
    ax.text(0.05, 0.95,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            fr"$m_{{W_R}}={mWR:.0f}$ GeV, $m_N={m_N:.0f}$ GeV" "\n"
            fr"$[m_{{c}}-{k:g}\sigma,\,m_{{c}}+{k:g}\sigma]=[{win_lo:.0f},\,{win_hi:.0f}]$"
            " (median)\n"
            fr"$\mu_{{\rm sig}}={mu_sig:.0f}$, $\sigma_{{\rm sig}}={sigma_sig:.0f}$ GeV"
            "\n" fr"$N_{{\rm inj}}={n_inj:g}$",
            transform=ax.transAxes, fontsize=14, va="top")
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=10.5)
    # stat box under the legend (Stage-4 style)
    bform = FUNCS[name][1]
    chi2s = (fr"$\chi^2/\mathrm{{ndf}}={chi2_ndf:.2f}$"
             if np.isfinite(chi2_ndf) else "")
    nsig_str = (fr"$N_{{\rm sig}}={nsig:.1f} \pm {nsig_err:.1f}$"
                if np.isfinite(nsig_err) else fr"$N_{{\rm sig}}={nsig:.1f}$")
    stat = ("Bkg: " + bform + "\n"
            "S+B: " + bform + r" $+\,N_{\rm sig}\,\mathcal{G}$" + "\n"
            + coef_text(params[:npar], perr[:npar]) + "\n"
            + nsig_str + "\n" + chi2s)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bb.x1 - 0.02, bb.y0 - 0.03, stat, transform=ax.transAxes,
            fontsize=9.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.7", alpha=0.9))
    _save(fig, out)


def plot_yield_vs_mass(per_fn, out, n_inj, *, channel, topology, com, lumi):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhline(n_inj, color="black", lw=1.0, ls="--",
               label=fr"injected $N={n_inj:g}$")
    for name in FUNCS:
        pts = per_fn.get(name)
        if not pts:
            continue
        m = [p["mWR"] for p in pts]
        y = [p["nsig"] for p in pts]
        e = [p["nsig_err"] for p in pts]
        ax.errorbar(m, y, yerr=e, fmt="o-", color=FUNCS[name][0], ms=4,
                    lw=1.2, elinewidth=0.8, capsize=1.5, label=name)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"fitted signal yield $N_{\rm sig}$ [events]")
    ax.text(0.03, 0.95, f"{CH_LAB[channel]}  {TOPO_LAB[topology]}\n"
            fr"injected $N={n_inj:g}$ (real signal MC)",
            transform=ax.transAxes, va="top", fontsize=12)
    ax.legend(fontsize=9, ncol=2, loc="best")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    _save(fig, out)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20260317_lo_dy", help="background dir")
    p.add_argument("--signal-dir", default="20260406_signals")
    p.add_argument("--signal-era", default="RunIISummer20UL18",
                   help="era for the signal MC (as in 0_signal_samples); may "
                        "differ from the background --era")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--mn-frac", type=float, default=0.5,
                   help="inject the signal point with m_N closest to this*m_WR")
    p.add_argument("--inject", type=float, nargs="+", default=[0, 10, 1000],
                   help="injected signal yields (events); a dir N{N} per value")
    p.add_argument("--k", type=float, default=3.0,
                   help="window half-width in sigma; the S+B fit is INSIDE "
                        "[mu-k*sigma, mu+k*sigma] only (matches Stage 4)")
    p.add_argument("--sigma-agg", default="median",
                   choices=["median", "mean", "max", "min"])
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--mass-min", type=float, default=2000.0)
    p.add_argument("--mass-max", type=float, default=6000.0)
    p.add_argument("--functions", nargs="+", default=list(FUNCS),
                   choices=list(FUNCS))
    p.add_argument("--width-csv", type=Path,
                   default=SIGDIR / "1_signal_widths" / "gaussian"
                   / "gauss_fit_table.csv")
    p.add_argument("--mass-csv", type=Path, default=SIGDIR / "master_masses.csv")
    p.add_argument("--no-fit-plots", action="store_true",
                   help="skip the per-(mass,function) fit diagnostics")
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
    inj_levels = [float(n) for n in args.inject]

    grid = load_grid_widths(args.width_csv, channel, topology, args.sigma_agg)
    point_widths = load_point_widths(args.width_csv, channel, topology)
    masses = sorted(m for m in grid if args.mass_min <= m <= args.mass_max)
    sig_tags = load_master_masses(args.mass_csv, topology)
    bkg_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    sig_dirs, _ = input_dirs_for_era(args.signal_era, repo_root(), args.signal_dir)
    region = f"wr_{channel}_{topology}_sr"
    hist_key = build_hist_key(build_region_name(channel, topology),
                              MASS_VAR[topology])
    factor = max(1, round(args.bin_width / 10.0))
    edges, values, variances = load_summed_background(
        bkg_dirs, region, MASS_VAR[topology], factor)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binwidth = float(edges[1] - edges[0])
    logger.info("Category %s %s -- %d masses, functions %s, inject N=%s, mn~%.2f*mWR",
                channel, topology, len(masses), funcs, inj_levels, args.mn_frac)

    tag = f"{channel}_{topology}"
    # rows[N] collects table rows; per_fn[N][fn] collects yield-vs-mass points
    rows = {n: [] for n in inj_levels}
    per_fn = {n: {f: [] for f in funcs} for n in inj_levels}

    for mWR in masses:
        m_c, sigma_win = grid[mWR]                  # median -> window + recentering
        stag = pick_signal_tag(sig_tags, mWR, args.mn_frac)
        if stag is None:
            logger.info("  m=%.0f: no %s signal point, skipping", mWR, topology)
            continue
        try:
            shape = load_signal_shape(sig_dirs, hist_key, stag, edges)
        except Exception as e:
            logger.warning("  m=%.0f %s: signal load failed (%s), skipping",
                           mWR, stag, e)
            continue
        if shape is None:
            logger.info("  m=%.0f %s: empty signal template, skipping", mWR, stag)
            continue
        # Gaussian model uses the INJECTED point's own (mu, sigma), not the median.
        mWR_i, mN_i = parse_masses(stag)
        mu_sig, sigma_sig = point_widths.get((mWR_i, mN_i), (m_c, sigma_win))
        lo = max(m_c - k * sigma_win, args.fit_min)   # window (median); fit inside only
        hi = min(m_c + k * sigma_win, args.fit_max)
        win = (centers >= m_c - k * sigma_win) & (centers <= m_c + k * sigma_win)
        b_window = float(values[win].sum())
        # Stage-4 background-only fit (no signal), in-window -- the same line as
        # in_window_fit; N-independent, drawn on every injection plot.
        bkg_fits = {n: fit_model(n, edges, values, variances, lo, hi, m_c)
                    for n in funcs}

        for n_inj in inj_levels:
            data = values + n_inj * shape
            for name in funcs:
                res = fit_splusb(name, edges, data, lo, hi, m_c, sigma_win,
                                 mu_sig, sigma_sig, binwidth, k)
                base = {"channel": channel, "topology": topology,
                        "function": name, "mWR": mWR, "signal_tag": stag,
                        "m_N": mN_i, "m_c": round(m_c, 1),
                        "sigma_win": round(sigma_win, 2),
                        "mu_sig": round(mu_sig, 1),
                        "sigma_sig": round(sigma_sig, 2), "N_inj": n_inj,
                        "fit_lo": round(lo, 1), "fit_hi": round(hi, 1),
                        "B_window": round(b_window, 3)}
                if res is None:
                    rows[n_inj].append({**base, "fit_ok": False, "N_sig": "",
                                        "N_sig_err": "", "recovered": "",
                                        "pull": "", "status": "", "cov_status": "",
                                        "chi2_ndf": "", "n_fit_bins": ""})
                    continue
                nsp, nerr = res["nsig"], res["nsig_err"]
                pull = nsp / nerr if nerr > 0 else float("nan")
                recovered = nsp / n_inj if n_inj else float("nan")
                cn = res["chi2"] / res["ndf"] if res["ndf"] > 0 else float("nan")
                rows[n_inj].append({
                    **base, "fit_ok": True, "N_sig": round(nsp, 4),
                    "N_sig_err": round(nerr, 4),
                    "recovered": round(recovered, 4) if n_inj else "",
                    "pull": round(pull, 4), "status": res["status"],
                    "cov_status": res["cov"],
                    "chi2_ndf": round(cn, 3) if res["ndf"] > 0 else "",
                    "n_fit_bins": res["nbins"]})
                per_fn[n_inj][name].append(
                    {"mWR": mWR, "nsig": nsp, "nsig_err": nerr})
                logger.info("  m=%.0f N=%-5g %-6s -> N_sig=%.1f+/-%.1f (%s)",
                            mWR, n_inj, name, nsp, nerr, stag)
                if not args.no_fit_plots:
                    bkg_res = bkg_fits[name]
                    plot_fit_diagnostic(
                        edges, data, values, lo, hi, name, res["params"],
                        res["perr"], bkg_res.params if bkg_res else None,
                        m_c, sigma_win, mu_sig, sigma_sig, binwidth, k,
                        nsp, nerr, n_inj, stag, cn,
                        args.output_dir / f"N{int(n_inj)}" / "fit_diagnostics"
                        / tag / name / f"m{int(mWR)}",
                        channel=channel, topology=topology, com=com, lumi=lumi,
                        mWR=mWR)

    for n_inj in inj_levels:
        if not rows[n_inj]:
            continue
        ndir = args.output_dir / f"N{int(n_inj)}"
        plot_yield_vs_mass(per_fn[n_inj], ndir / "fitted_yield_vs_mass" / tag,
                           n_inj, channel=channel, topology=topology,
                           com=com, lumi=lumi)
        csv_path = ndir / f"injection_table_{tag}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[n_inj][0].keys()))
            wtr.writeheader()
            wtr.writerows(rows[n_inj])
        logger.info("  N=%g: wrote %s", n_inj, csv_path)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
