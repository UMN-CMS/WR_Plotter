#!/usr/bin/env python3
"""Stage 4 — can we fit the background INSIDE the signal window at all?

Before trusting any sideband interpolation, the precondition: is the summed MC
background describable by a smooth function within the m_c±kσ window itself? For
one category (default ee resolved) and each signal-grid m_WR:

  1. Window [m_c − kσ, m_c + kσ], where the window centre m_c and width σ come
     (default) from the Stage-2 linear m_WR parameterization
     (2_width_parameterization/wr/window_params.json); pass --window-source measured
     to use the per-m_WR median over M_N of the Stage-1 widths instead.
  2. ROOT chi-square fit (TF1 + Minuit2 Migrad) of the summed MC background to
     the bins INSIDE the window — NOT the sidebands. Empty bins are skipped.
  3. Record the fit quality: χ²/ndf, parameters ± Minuit errors, the window
     yield B_fit ± err, and the four Minuit2 fit checks:
       1. valid minimum    (IsValid(): folds in EDM, call limit, Hesse)
       2. covariance accurate (CovMatrixStatus() == 3, not merely forced)
       3. no parameters at limit (slope <=0 constraint / dexp norm fence)
       4. monotonic (fitted background falls across the window, no local rise)
     A fit "passes" only if all four hold. Raw status/cov_status/edm/ncalls
     are still tabulated as diagnostics.

Candidate functions (recentered at the window centre m_c): expo, expo2, expo3,
powlaw, powexp, dexp —
see bkg_fit_lib.FUNCS. A feasibility scan: every grid mass is attempted and
tabulated (no early stop).

Outputs (co-located, namespaced by {channel}_{topology}):
  chi2_ndf_vs_mass/{ch}_{topo}_k{k}.{png,pdf}        χ²/ndf vs m_WR per function
  fit_uncertainty_vs_mass/{ch}_{topo}_k{k}.{png,pdf} δB_fit/B_fit vs m_WR
  params_vs_mass/{function}/{ch}_{topo}_k{k}.*       coefficients vs m_WR (±err)
  diagnostics/{ch}_{topo}/{function}/m{mWR}.*        spectrum + fit + stat box
  in_window_table_{ch}_{topo}.csv                    everything incl. the checks

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
Usage:
  python in_window_fit.py -v --diagnostics
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # bkg_fit_lib

from wrplotter.cli_utils import setup_logging                   # noqa: E402
from wrplotter.config import load_lumi                          # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root       # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter     # noqa: E402

from bkg_fit_lib import (                                        # noqa: E402
    COEF_SYMS, FUNCS, MASS_VAR, MASS_LABEL, CH_LAB, TOPO_LAB,
    func_label, coef_text, fit_model, band, b_window,
    load_grid_widths, grid_widths_from_params, load_summed_background,
)

logger = logging.getLogger(__name__)

# The four fit checks, in reporting order, with stat-box short names.
# valid_minimum folds in EDM/call-limit/Hesse; cov_ok demands an accurate (not
# forced) covariance; limit = a parameter railed against the slope<=0 / dexp
# fence; mono = the fitted background falls across the window with no local rise.
CHECK_KEYS = ["valid_minimum", "cov_ok", "no_param_at_limit", "monotonic"]
CHECK_SHORT = {"valid_minimum": "min", "cov_ok": "cov",
               "no_param_at_limit": "limit", "monotonic": "mono"}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _info_box(ax, channel, topology, era, k, win_desc):
    return ax.text(0.05, 0.95,
                   f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n{era}\n"
                   rf"window $m_{{c}}\pm{k:g}\sigma$" "\n"
                   f"{win_desc}",
                   transform=ax.transAxes, fontsize=13, va="top")


def _legend(ax):
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=11)


def plot_chi2_vs_mass(results, out_path, *, channel, topology, k, com, lumi,
                      era, win_desc, ymax=10.0):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    for name, (color, _, _) in FUNCS.items():
        pts = [p for p in results.get(name, []) if np.isfinite(p[1])]
        if not pts:
            continue
        ms, cs = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(ms, cs, marker="o", markersize=4.5, linewidth=1.6, color=color,
                label=func_label(name), markeredgecolor="black",
                markeredgewidth=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    ax.set_ylim(0.0, ymax)   # fixed so ee/μμ (and resolved/boosted) compare
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"in-window fit  $\chi^2/\mathrm{ndf}$")
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=18)
    info = _info_box(ax, channel, topology, era, k, win_desc)
    # function legend inside the plot, directly under the left info label
    fig.canvas.draw()
    bb = info.get_window_extent().transformed(ax.transAxes.inverted())
    ax.legend(loc="upper left", bbox_to_anchor=(0.05, bb.y0 - 0.03),
              fontsize=11.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("  wrote %s", out_path.with_suffix(".png"))


def plot_fit_unc_vs_mass(results, out_path, *, channel, topology, k, com, lumi,
                         era, win_desc):
    """Relative fit uncertainty on the in-window yield, δB_fit/B_fit, vs m_WR."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    hi = 1.0
    for name, (color, _, _) in FUNCS.items():
        pts = [p for p in results.get(name, []) if np.isfinite(p[1])]
        if not pts:
            continue
        ms, us = [p[0] for p in pts], [100.0 * p[1] for p in pts]
        hi = max(hi, max(us))
        ax.plot(ms, us, marker="o", markersize=4.5, linewidth=1.6, color=color,
                label=func_label(name), markeredgecolor="black",
                markeredgewidth=0.3)
    ax.set_ylim(0.0, min(hi * 1.15, 150.0))
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"fit uncertainty  $\delta B_{\mathrm{fit}}/B_{\mathrm{fit}}$ [%]")
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=18)
    _info_box(ax, channel, topology, era, k, win_desc)
    _legend(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("  wrote %s", out_path.with_suffix(".png"))


def plot_params_vs_mass(pts, name, out_path, *, channel, topology, k, com, lumi,
                        era, win_desc):
    """Each fitted coefficient of one function vs m_WR (with its Minuit error).
    pts: [(mWR, params, perr), ...]. One panel per parameter; robust y-range."""
    npar = FUNCS[name][2]
    color = FUNCS[name][0]
    hep.style.use("CMS")
    fig, axes = plt.subplots(npar, 1, sharex=True,
                             figsize=(7.4, 2.0 * npar + 0.8))
    axes = np.atleast_1d(axes)
    ms = np.array([p[0] for p in pts])
    for i in range(npar):
        ax = axes[i]
        vals = np.array([p[1][i] for p in pts])
        errs = np.array([p[2][i] for p in pts])
        bar = np.where(np.isfinite(errs), errs, 0.0)
        ax.errorbar(ms, vals, yerr=bar, marker="o", markersize=4, linewidth=1.4,
                    color=color, capsize=2, elinewidth=0.9,
                    markeredgecolor="black", markeredgewidth=0.3)
        fin = vals[np.isfinite(vals)]
        if fin.size:
            lo, hi = np.percentile(fin, [4, 96])
            pad = 0.25 * (hi - lo) + 1e-9
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(rf"${COEF_SYMS[i]}$", fontsize=15)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel(r"$m_{W_R}$ [GeV]")
    hep.cms.label(loc=0, ax=axes[0], data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    axes[0].text(0.97, 0.90,
                 f"{CH_LAB[channel]}  {TOPO_LAB[topology]}\n"
                 rf"{func_label(name)},  $m_{{c}}\pm{k:g}\sigma$",
                 transform=axes[0].transAxes, fontsize=12, va="top", ha="right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("  wrote %s", out_path.with_suffix(".png"))


def plot_diagnostic(edges, values, variances, win_mask, name, res, *,
                    mWR, m_c, sigma, k, win_lo, win_hi, out_path, channel,
                    topology, com, lumi, era):
    """One mass/function: local spectrum + the in-window fit (±1σ band)."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = win_hi - win_lo
    lo, hi = win_lo - 1.3 * width, win_hi + 1.3 * width
    view = (centers >= lo) & (centers <= hi)
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.stairs(values, edges, color="#888888", linewidth=1.3,
              label="MC background")
    ax.errorbar(centers[win_mask], values[win_mask],
                yerr=np.sqrt(variances[win_mask]), fmt="o", color="black",
                markersize=4, elinewidth=0.8, capsize=1.5,
                label="in-window (fit)")
    grid = np.linspace(win_lo, win_hi, 400)
    f_grid, s_grid = band(name, res.params, res.cov, grid, m_c)
    if np.all(np.isfinite(s_grid)):
        ax.fill_between(grid, f_grid * np.exp(-s_grid), f_grid * np.exp(s_grid),
                        color="#e42536", alpha=0.25, lw=0, zorder=1,
                        label=r"fit $\pm1\sigma$")
    ax.plot(grid, f_grid, color="#e42536", linewidth=2, zorder=2,
            label=f"{name} fit")
    ax.axvspan(win_lo, win_hi, color="#5790fc", alpha=0.15, zorder=0,
               label=r"$m_{c}\pm k\sigma$")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_xlim(lo, hi)
    pos = values[view][values[view] > 0]
    if pos.size:
        ax.set_ylim(pos.min() / 3.0, pos.max() * 30.0)
    ax.set_xlabel(MASS_LABEL[topology])
    ax.set_ylabel(f"Events / {edges[1] - edges[0]:.0f} GeV")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=16)
    ax.text(0.05, 0.95,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            rf"$m_{{W_R}}={mWR:.0f}$ GeV" "\n"
            rf"$m_{{c}}={m_c:.0f}$ GeV" "\n"
            rf"$\sigma_{{\mathrm{{fit}}}}={sigma:.0f}$ GeV" "\n"
            rf"$[m_{{c}}-{k:g}\sigma,\,m_{{c}}+{k:g}\sigma]=[{win_lo:.0f},\,{win_hi:.0f}]$",
            transform=ax.transAxes, fontsize=14, va="top")

    chi2_str = (rf"$\chi^2/\mathrm{{ndf}}={res.chi2/res.ndf:.2f}$"
                if res.ndf > 0 else r"$\chi^2/\mathrm{ndf}=$n/a")
    if res.passed:
        flag = "fit: PASSED"
    else:
        failed = [CHECK_SHORT[c] for c in CHECK_KEYS if not res.checks[c]]
        flag = "fit: FAILED (" + ", ".join(failed) + ")"
    stat = (rf"{func_label(name)}" "\n"
            + coef_text(res.params, res.perr) + "\n"
            + chi2_str + "\n"
            + flag)
    # legend top-right (nudged down); the stat box directly below it
    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=10.5)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(ax.transAxes.inverted())
    ax.text(bb.x1 - 0.025, bb.y0 - 0.03, stat, transform=ax.transAxes,
            fontsize=10.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20260317_lo_dy")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved", choices=["resolved", "boosted"])
    p.add_argument("--k", type=float, default=3.0)
    p.add_argument("--window-source", default="param",
                   choices=["param", "measured"],
                   help="'param' (default): window mu/sigma from the Stage-2 "
                        "linear parameterization (window_params.json). "
                        "'measured': per-m_WR aggregate of the Stage-1 widths.")
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
    p.add_argument("--functions", nargs="+", default=list(FUNCS),
                   choices=list(FUNCS), help="Subset of functions to test.")
    p.add_argument("--chi2-ymax", type=float, default=10.0,
                   help="Fixed y-axis max for the chi2/ndf-vs-mass plot "
                        "(shared across categories). Default 10.")
    p.add_argument("--width-csv", type=Path,
                   default=Path(__file__).resolve().parents[2]
                   / "1_signal_widths" / "gaussian" / "gauss_fit_table.csv")
    p.add_argument("--window-params", type=Path,
                   default=Path(__file__).resolve().parents[2]
                   / "2_width_parameterization" / "wr" / "window_params.json",
                   help="Stage-2 linear window parameterization (window_source param).")
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--diag-func", default=None, choices=list(FUNCS))
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <script dir>/<run2|run3>, chosen by --era.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    era, channel, topology, k = args.era, args.channel, args.topology, args.k
    info = load_lumi(era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    if args.output_dir is None:
        args.output_dir = (Path(__file__).resolve().parent
                           / {"RunII": "run2", "Run3": "run3"}[str(info["run"])])
    funcs = [f for f in FUNCS if f in args.functions]

    if not args.width_csv.exists():
        logger.error("Width table not found: %s (run Stage 1 first)", args.width_csv)
        sys.exit(1)
    # The Stage-1 table defines which grid masses exist; the window (mu, sigma)
    # comes from the Stage-2 linear parameterization unless --window-source measured.
    measured = load_grid_widths(args.width_csv, channel, topology, args.sigma_agg)
    masses = sorted(m for m in measured if args.mass_min <= m <= args.mass_max)
    if not masses:
        logger.error("No grid masses in [%g, %g]", args.mass_min, args.mass_max)
        sys.exit(1)
    if args.window_source == "param":
        if not args.window_params.exists():
            logger.error("Window params not found: %s (run Stage 2 wr/ first)",
                         args.window_params)
            sys.exit(1)
        grid_widths = grid_widths_from_params(
            args.window_params, channel, topology, masses, args.sigma_kind)
        logger.info("Window source: Stage-2 linear fit (%s sigma) <- %s",
                    args.sigma_kind, args.window_params)
        win_desc = rf"$m_{{c}},\sigma$: Stage-2 linear fit ({args.sigma_kind})"
    else:
        grid_widths = measured
        logger.info("Window source: measured per-m_WR %s over M_N", args.sigma_agg)
        win_desc = rf"$m_{{c}},\sigma$: {args.sigma_agg} over $M_N$"
    logger.info("Category %s %s — %d grid masses, functions: %s",
                channel, topology, len(masses), funcs)

    input_dirs, _ = input_dirs_for_era(era, repo_root(), args.dir)
    region = f"wr_{channel}_{topology}_sr"
    factor = max(1, round(args.bin_width / 10.0))
    edges, values, variances = load_summed_background(
        input_dirs, region, MASS_VAR[topology], factor)
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_fit = (centers >= args.fit_min) & (centers <= args.fit_max)

    diag_funcs = [args.diag_func] if args.diag_func else funcs
    chi2_res: dict[str, list] = {n: [] for n in funcs}
    unc_res: dict[str, list] = {n: [] for n in funcs}
    param_res: dict[str, list] = {n: [] for n in funcs}
    n_pass: dict[str, list] = {n: [0, 0] for n in funcs}   # [passed, attempted]
    rows: list[dict] = []

    empty_checks = {c: "" for c in CHECK_KEYS}
    empty_coefs = {}
    for i in range(4):
        empty_coefs[f"coef_{COEF_SYMS[i]}"] = ""
        empty_coefs[f"coef_{COEF_SYMS[i]}_err"] = ""

    for mWR in masses:
        m_c, sigma = grid_widths[mWR]
        win_lo, win_hi = m_c - k * sigma, m_c + k * sigma
        fit_lo, fit_hi = max(win_lo, args.fit_min), min(win_hi, args.fit_max)
        in_win = (centers >= win_lo) & (centers <= win_hi)
        b_mc = float(values[in_win].sum())
        b_mc_err = float(np.sqrt(variances[in_win].sum()))
        win_mask = in_fit & in_win & (values > 0) & (variances > 0)
        n_win = int(win_mask.sum())

        for name in funcs:
            base = {"channel": channel, "topology": topology, "k": k,
                    "function": name, "mWR": mWR, "m_c": round(m_c, 1),
                    "sigma": round(sigma, 2), "win_lo": round(win_lo, 1),
                    "win_hi": round(win_hi, 1), "n_window_bins": n_win,
                    "B_MC": round(b_mc, 4), "B_MC_err": round(b_mc_err, 4)}
            res = (fit_model(name, edges, values, variances, fit_lo, fit_hi, m_c)
                   if b_mc > 0 else None)
            if res is None:
                rows.append({**base, "fit_ok": False, "fit_passed": "",
                             **empty_checks, "status": "", "cov_status": "",
                             "edm": "", "ncalls": "", "ndf": "", "chi2_ndf": "",
                             "B_fit": "", "B_fit_err": "", **empty_coefs})
                logger.info("  m_WR=%.0f %-6s n_win=%d B_MC=%.2f -> not fittable",
                            mWR, name, n_win, b_mc)
                continue

            b_fit, b_fit_err = b_window(name, res.params, res.cov, centers[in_win], m_c)
            cn = res.chi2 / res.ndf if res.ndf > 0 else float("nan")
            npar = FUNCS[name][2]
            n_pass[name][1] += 1
            n_pass[name][0] += int(res.passed)

            def _num(x, nd=6):               # "inf" flags an unconstrained fit
                return round(float(x), nd) if np.isfinite(x) else "inf"

            chi2_res[name].append((mWR, cn))
            param_res[name].append((mWR, res.params, res.perr))
            if np.isfinite(b_fit_err) and b_fit > 0:
                unc_res[name].append((mWR, b_fit_err / b_fit))
            coef_cols = {}
            for i in range(4):
                coef_cols[f"coef_{COEF_SYMS[i]}"] = (
                    _num(res.params[i]) if i < npar else "")
                coef_cols[f"coef_{COEF_SYMS[i]}_err"] = (
                    _num(res.perr[i]) if i < npar else "")
            rows.append({**base, "fit_ok": True, "fit_passed": res.passed,
                         **{c: res.checks[c] for c in CHECK_KEYS},
                         "status": res.status, "cov_status": res.cov_status,
                         "edm": f"{res.edm:.2e}", "ncalls": res.ncalls,
                         "ndf": res.ndf,
                         "chi2_ndf": round(cn, 3) if res.ndf > 0 else "",
                         "B_fit": round(b_fit, 4), "B_fit_err": _num(b_fit_err, 4),
                         **coef_cols})
            if res.passed:
                flag = "PASS"
            else:
                flag = "FAIL(" + ",".join(CHECK_SHORT[c] for c in CHECK_KEYS
                                          if not res.checks[c]) + ")"
            logger.info("  m_WR=%.0f %-6s n_win=%d chi2/ndf=%s B_fit=%.2f "
                        "B_MC=%.2f %s", mWR, name, n_win,
                        f"{cn:.2f}" if res.ndf > 0 else "n/a", b_fit, b_mc, flag)

            if args.diagnostics and name in diag_funcs:
                plot_diagnostic(
                    edges, values, variances, win_mask, name, res,
                    mWR=mWR, m_c=m_c, sigma=sigma, k=k, win_lo=win_lo,
                    win_hi=win_hi,
                    out_path=args.output_dir / "diagnostics"
                    / f"{channel}_{topology}" / name / f"m{int(mWR)}",
                    channel=channel, topology=topology, com=com, lumi=lumi,
                    era=era)

    if not rows:
        logger.error("No results produced.")
        sys.exit(1)

    plot_chi2_vs_mass(
        chi2_res, args.output_dir / "chi2_ndf_vs_mass" / f"{channel}_{topology}_k{k:g}",
        channel=channel, topology=topology, k=k, com=com, lumi=lumi, era=era,
        win_desc=win_desc, ymax=args.chi2_ymax)
    plot_fit_unc_vs_mass(
        unc_res, args.output_dir / "fit_uncertainty_vs_mass"
        / f"{channel}_{topology}_k{k:g}",
        channel=channel, topology=topology, k=k, com=com, lumi=lumi, era=era,
        win_desc=win_desc)
    for name in funcs:
        if param_res[name]:
            plot_params_vs_mass(
                param_res[name], name, args.output_dir / "params_vs_mass" / name
                / f"{channel}_{topology}_k{k:g}",
                channel=channel, topology=topology, k=k, com=com, lumi=lumi,
                era=era, win_desc=win_desc)

    csv_path = args.output_dir / f"in_window_table_{channel}_{topology}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s", csv_path)

    print(f"\n=== fit checks summary ({channel} {topology}, k={k:g}) ===")
    for name in funcs:
        npass, ntot = n_pass[name]
        print(f"  {name:8} {npass:>3}/{ntot} fits passed all 4 checks")
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
