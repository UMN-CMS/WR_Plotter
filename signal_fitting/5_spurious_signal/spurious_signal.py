#!/usr/bin/env python3
"""Stage 5 -- spurious-signal study.

Does the background model FAKE a signal when there is none? For each
(channel, topology), grid mass m_WR, and background function, fit

    background TF1 (Stage-4 recentered)  +  fixed-shape Gaussian(mu, sigma)

to the **background-only** MC template (NO signal injected), inside the window
[m_c - k*sigma_win, m_c + k*sigma_win] only (same range as the Stage-4
background fits). The fitted yield `N_sig` is the **spurious signal** -- the fake
signal the background mismodelling produces. Only `N_sig` and the background
coefficients float; the Gaussian shape is fixed.

The window is defined from the **median** over m_N (one window per m_WR, as the
analysis does); the Gaussian uses the on-shell point's own (mu, sigma)
(m_N closest to mn_frac*m_WR, default m_N = m_WR/2) from the Stage-1 fits.

The headline metric is the **pull** = N_spurious / sigma(N_sig): a fit passes if
the fake signal is small relative to its statistical uncertainty (a common
acceptance band is |pull| < 0.2-0.5).

Outputs (namespaced by {channel}_{topology}):
  pull_vs_mass/{ch}_{topo}.*            N_sig/sigma vs m_WR per fn (+-0.2/0.5 bands)
  spurious_yield_vs_mass/{ch}_{topo}.*  spurious N_sig +- sigma vs m_WR per fn
  fit_diagnostics/{ch}_{topo}/{fn}/m{mWR}.*  the S+B fit to background-only
  spurious_table_{ch}_{topo}.csv        everything

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
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                       # repo root
sys.path.insert(0, str(HERE.parents[0] / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(HERE.parents[0] / "shared"))           # sb_fit, loaders

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter             # noqa: E402

from bkg_fit_lib import (                                               # noqa: E402
    FUNCS, MASS_VAR, MASS_LABEL, CH_LAB, TOPO_LAB,
    predict, coef_text, fit_model, load_grid_widths, grid_widths_from_params,
    load_summed_background,
)
from measure_fwhm import parse_masses                                   # noqa: E402
from shape_estimators import load_master_masses                         # noqa: E402
from sb_fit import (                                                    # noqa: E402
    CHECK_KEYS, CHECK_SHORT, fit_splusb, gauss_per_bin, pick_signal_tag,
)

logger = logging.getLogger("spurious_signal")

COEF_NAMES = ["a", "b", "c", "d"]   # background-component coefficient labels


def _coef_cols(params=None, perr=None, npar=0):
    """coef_a..coef_d (+ _err) for the BACKGROUND component of the S+B fit, in the
    recentered basis (a = log-yield at m_c, b = slope). Always 4 pairs so every
    CSV row shares the same columns; unused ones (npar<4, or a failed fit) blank."""
    cols = {}
    for i, nm in enumerate(COEF_NAMES):
        if params is not None and i < npar:
            cols[f"coef_{nm}"] = round(float(params[i]), 6)
            cols[f"coef_{nm}_err"] = round(float(perr[i]), 6)
        else:
            cols[f"coef_{nm}"] = cols[f"coef_{nm}_err"] = ""
    return cols


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_fit_diagnostic(edges, bkg, lo, hi, name, params, perr, bkg_params,
                        m_c, sigma_win, mu_sig, sigma_sig, binwidth, k, nsig,
                        nsig_err, pull, sig_tag, chi2_ndf, out, *, channel,
                        topology, com, lumi, mWR, view_k, passed, checks):
    """The S+B fit to the BACKGROUND-ONLY template, in the in_window_fit style.
    The Stage-4 background-only fit is overlaid; the red B+S curve minus the blue
    background is the fake (spurious) signal.

    The view window is a FIXED multiple `view_k` of sigma_win about m_c -- both
    k-independent -- so the x (and hence the data-derived y) limits are identical
    across the k=2..5 runs for a given mass, making the diagnostics comparable."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    npar = FUNCS[name][2]
    vlo = max(m_c - view_k * sigma_win, float(edges[0]))    # k-independent view
    vhi = min(m_c + view_k * sigma_win, float(edges[-1]))
    view = (centers >= vlo) & (centers <= vhi)
    fitsel = (centers >= lo) & (centers <= hi) & (bkg > 0)
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    grid = np.linspace(lo, hi, 500)
    sgrid = nsig * gauss_per_bin(grid, mu_sig, sigma_sig, binwidth)
    ax.stairs(bkg, edges, color="#888888", linewidth=1.3, zorder=2,
              label="MC Bkg")
    ax.plot(grid, predict(name, params[:npar], grid, m_c) + sgrid,
            color="#e42536", lw=2.0, zorder=3, label="B+S fit")
    if bkg_params is not None:
        ax.plot(grid, predict(name, bkg_params[:npar], grid, m_c),
                color="#1f77b4", ls="--", lw=1.8, zorder=3,
                label="Bkg-only fit (in-window)")
    ax.axvspan(m_c - k * sigma_win, m_c + k * sigma_win, color="#5790fc",
               alpha=0.12, zorder=0, label=fr"$m_{{c}}\pm{k:g}\sigma$")
    ax.errorbar(centers[fitsel], bkg[fitsel], yerr=np.sqrt(bkg[fitsel]),
                fmt="o", color="black", ms=4, elinewidth=0.8, capsize=1.5,
                zorder=4, label="MC Bkg (fit bins)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_xlim(vlo, vhi)
    pos = bkg[view][bkg[view] > 0]
    if pos.size:
        ax.set_ylim(pos.min() / 3.0, pos.max() * 30.0)
    ax.set_xlabel(MASS_LABEL[topology])
    ax.set_ylabel(f"Events / {binwidth:.0f} GeV")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=16)
    win_lo, win_hi = m_c - k * sigma_win, m_c + k * sigma_win
    ax.text(0.05, 0.95,
            f"{CH_LAB[channel]}  {TOPO_LAB[topology]}\n"
            fr"Search for $m_{{W_R}} = {mWR:.0f}$ GeV" "\n"
            r"$\mu,\,\sigma$ from linear fit: "
            fr"$\mu = m_c = {m_c:.0f}$ GeV, $\sigma = {sigma_win:.0f}$ GeV" "\n"
            fr"Window $[m_c-{k:g}\sigma,\,m_c+{k:g}\sigma]=[{win_lo:.0f},\,{win_hi:.0f}]$"
            "\n"
            r"Gaussian parameters fixed to the same $(\mu,\sigma)$" "\n"
            "No signal injected",
            transform=ax.transAxes, fontsize=14, va="top")
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=13)
    bform = FUNCS[name][1]
    chi2s = (fr"$\chi^2/\mathrm{{ndf}}={chi2_ndf:.2f}$"
             if np.isfinite(chi2_ndf) else "")
    nsig_str = (fr"$\hat{{N}}_{{\rm sig}}={nsig:.1f} \pm {nsig_err:.1f}$"
                if np.isfinite(nsig_err) else fr"$\hat{{N}}_{{\rm sig}}={nsig:.1f}$")
    pull_str = fr"pull $={pull:+.2f}$" if np.isfinite(pull) else "pull = n/a"
    if passed:
        flag = "fit: PASSED"
    else:
        bad = [CHECK_SHORT[c] for c in CHECK_KEYS if not checks[c]]
        flag = "fit: FAILED (" + ", ".join(bad) + ")"
    stat = ("Bkg: " + bform + "\n"
            "S+B: " + bform + r" $+\,\hat{N}_{\rm sig}\,\mathcal{G}$" + "\n"
            + coef_text(params[:npar], perr[:npar]) + "\n"
            + nsig_str + "\n" + pull_str + "\n" + chi2s
            + ("\n" if chi2s else "") + flag)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bb.x1 - 0.02, bb.y0 - 0.03, stat, transform=ax.transAxes,
            fontsize=12, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="0.7", alpha=0.9))
    _save(fig, out)


def plot_pull_vs_mass(name, pts, out, *, channel, topology, com, lumi, k):
    """Spurious pull vs mass for a single background function."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhspan(-0.5, 0.5, color="#f7d600", alpha=0.25, lw=0, zorder=0,
               label=r"$|{\rm pull}|<0.5$")
    ax.axhspan(-0.2, 0.2, color="#74c476", alpha=0.35, lw=0, zorder=0,
               label=r"$|{\rm pull}|<0.2$")
    ax.axhline(0.0, color="black", lw=0.8, zorder=1)
    finite = [(p["mWR"], p["pull"]) for p in pts if np.isfinite(p["pull"])]
    if finite:
        m, pull = zip(*finite)
        ax.plot(m, pull, "o-", color=FUNCS[name][0], ms=4, lw=1.5, label=name)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"spurious signal / $\sigma_{\hat{N}_{\rm sig}}$  (pull)")
    ax.set_ylim(-1.5, 1.5)
    ax.text(0.03, 0.95,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            f"{name}: {FUNCS[name][1]}\n"
            fr"$m_{{c}}\pm{k:g}\sigma$ window",
            transform=ax.transAxes, va="top", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    _save(fig, out)


def plot_yield_vs_mass(name, pts, out, *, channel, topology, com, lumi, k):
    """Spurious yield vs mass for a single background function."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="black", lw=1.0, ls="--", label="no fake signal")
    m = [p["mWR"] for p in pts]
    y = [p["nsig"] for p in pts]
    e = [p["nsig_err"] for p in pts]
    ax.errorbar(m, y, yerr=e, fmt="o-", color=FUNCS[name][0], ms=4,
                lw=1.3, elinewidth=0.8, capsize=1.5, label=name)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"fitted signal yield $\hat{N}_{\rm sig}$ [events]")
    ax.text(0.03, 0.95,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            f"{name}: {FUNCS[name][1]}\n"
            fr"$m_{{c}}\pm{k:g}\sigma$ window",
            transform=ax.transAxes, va="top", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
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
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--mn-frac", type=float, default=0.5,
                   help="Gaussian uses the signal point with m_N closest to this*m_WR")
    p.add_argument("--k", type=float, default=3.0,
                   help="window half-width in sigma; the S+B fit is INSIDE "
                        "[mu-k*sigma, mu+k*sigma] only (matches Stage 4)")
    p.add_argument("--view-k", type=float, default=8.0,
                   help="diagnostic plot x-range half-width in sigma_win about "
                        "m_c; k-INDEPENDENT so axis limits match across k runs")
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
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--mass-min", type=float, default=1000.0)
    p.add_argument("--mass-max", type=float, default=6000.0)
    p.add_argument("--masses", nargs="+", type=float, default=None,
                   help="Explicit m_WR list (overrides the grid; may be OFF-GRID, "
                        "e.g. 2341). Window + signal shape come from the linear fit, "
                        "so no signal MC point is needed. Requires --window-source param.")
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
    p.add_argument("--no-fit-plots", action="store_true",
                   help="skip the per-(mass,function) fit diagnostics")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <script dir>/<run2|run3>, chosen by --era.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    channel, topology, k = args.channel, args.topology, args.k
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    if args.output_dir is None:
        args.output_dir = HERE / {"RunII": "run2", "Run3": "run3"}[str(info["run"])]
    funcs = [f for f in FUNCS if f in args.functions]

    measured = load_grid_widths(args.width_csv, channel, topology, args.sigma_agg)
    if args.masses:
        masses = sorted(float(m) for m in args.masses)
    else:
        masses = sorted(m for m in measured if args.mass_min <= m <= args.mass_max)
    if args.window_source == "param":
        grid = grid_widths_from_params(
            args.window_params, channel, topology, masses, args.sigma_kind)
        logger.info("Window source: Stage-2 linear fit (%s sigma) <- %s",
                    args.sigma_kind, args.window_params)
    else:
        if args.masses and any(m not in measured for m in masses):
            sys.exit("Off-grid --masses require --window-source param (the measured "
                     "table has no widths off the signal grid).")
        grid = measured
        logger.info("Window source: measured per-m_WR %s over M_N", args.sigma_agg)
    sig_tags = load_master_masses(args.mass_csv, topology)
    bkg_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    region = f"wr_{channel}_{topology}_sr"
    factor = max(1, round(args.bin_width / 10.0))
    edges, values, variances = load_summed_background(
        bkg_dirs, region, MASS_VAR[topology], factor)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binwidth = float(edges[1] - edges[0])
    logger.info("Spurious signal: %s %s -- %d masses, functions %s",
                channel, topology, len(masses), funcs)

    tag = f"{channel}_{topology}"
    rows, per_fn = [], {n: [] for n in funcs}
    for mWR in masses:
        m_c, sigma_win = grid[mWR]                  # linear fit -> window + recentering
        # Representative signal point (informational only; the shape no longer
        # depends on it). Off-grid masses (e.g. --masses 2341) have none.
        stag = pick_signal_tag(sig_tags, mWR, args.mn_frac)
        if stag is None and not args.masses:
            logger.info("  m=%.0f: no %s signal point, skipping", mWR, topology)
            continue
        mN_i = parse_masses(stag)[1] if stag else ""
        # Injected-signal Gaussian shape from the SAME m_WR linear parameterization
        # as the window (not the per-point measured width) -> mu_sig=m_c, sigma_sig=sigma_win.
        mu_sig, sigma_sig = m_c, sigma_win
        lo = max(m_c - k * sigma_win, args.fit_min)   # window (median); fit inside only
        hi = min(m_c + k * sigma_win, args.fit_max)
        win = (centers >= m_c - k * sigma_win) & (centers <= m_c + k * sigma_win)
        b_window = float(values[win].sum())
        bkg_fits = {n: fit_model(n, edges, values, variances, lo, hi, m_c)
                    for n in funcs}

        for name in funcs:
            # fit S+B to the BACKGROUND-ONLY template (no injection)
            res = fit_splusb(name, edges, values, lo, hi, m_c, sigma_win,
                             mu_sig, sigma_sig, binwidth, k)
            base = {"channel": channel, "topology": topology, "function": name,
                    "mWR": mWR, "signal_tag": stag or "", "m_N": mN_i,
                    "m_c": round(m_c, 1), "sigma_win": round(sigma_win, 2),
                    "mu_sig": round(mu_sig, 1), "sigma_sig": round(sigma_sig, 2),
                    "fit_lo": round(lo, 1), "fit_hi": round(hi, 1),
                    "B_window": round(b_window, 3)}
            if res is None:
                rows.append({**base, "fit_ok": False, "fit_passed": "",
                             **{c: "" for c in CHECK_KEYS}, "N_spur": "",
                             "N_spur_err": "", "pull": "", "spur_over_sqrtB": "",
                             "status": "", "cov_status": "", "chi2_ndf": "",
                             "n_fit_bins": "", **_coef_cols()})
                logger.info("  m=%.0f %-6s -> S+B fit failed", mWR, name)
                continue
            nsp, nerr = res["nsig"], res["nsig_err"]
            pull = nsp / nerr if nerr > 0 else float("nan")
            rel = nsp / np.sqrt(b_window) if b_window > 0 else float("nan")
            cn = res["chi2"] / res["ndf"] if res["ndf"] > 0 else float("nan")
            rows.append({
                **base, "fit_ok": True, "fit_passed": res["passed"],
                **{c: res["checks"][c] for c in CHECK_KEYS},
                "N_spur": round(nsp, 4),
                "N_spur_err": round(nerr, 4), "pull": round(pull, 4),
                "spur_over_sqrtB": round(rel, 4), "status": res["status"],
                "cov_status": res["cov"],
                "chi2_ndf": round(cn, 3) if res["ndf"] > 0 else "",
                "n_fit_bins": res["nbins"],
                **_coef_cols(res["params"], res["perr"], FUNCS[name][2])})
            per_fn[name].append({"mWR": mWR, "nsig": nsp, "nsig_err": nerr,
                                 "pull": pull})
            failed = [CHECK_SHORT[c] for c in CHECK_KEYS if not res["checks"][c]]
            logger.info("  m=%.0f %-6s N_spur=%.1f+/-%.1f pull=%+.3f  fit:%s",
                        mWR, name, nsp, nerr, pull,
                        "PASS" if res["passed"] else "FAIL(" + ",".join(failed) + ")")
            if not args.no_fit_plots:
                bkg_res = bkg_fits[name]
                plot_fit_diagnostic(
                    edges, values, lo, hi, name, res["params"], res["perr"],
                    bkg_res.params if bkg_res else None, m_c, sigma_win,
                    mu_sig, sigma_sig, binwidth, k, nsp, nerr, pull, stag, cn,
                    args.output_dir / "fit_diagnostics" / tag / name
                    / f"m{int(mWR)}", channel=channel, topology=topology,
                    com=com, lumi=lumi, mWR=mWR, view_k=args.view_k,
                    passed=res["passed"], checks=res["checks"])

    if not rows:
        logger.error("No results produced.")
        sys.exit(1)
    for name in funcs:
        if not per_fn[name]:
            continue
        plot_pull_vs_mass(name, per_fn[name],
                          args.output_dir / "pull_vs_mass" / tag / name,
                          channel=channel, topology=topology, com=com, lumi=lumi,
                          k=k)
        plot_yield_vs_mass(name, per_fn[name],
                           args.output_dir / "spurious_yield_vs_mass" / tag / name,
                           channel=channel, topology=topology, com=com, lumi=lumi,
                           k=k)
    csv_path = args.output_dir / f"spurious_table_{tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s", csv_path)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
