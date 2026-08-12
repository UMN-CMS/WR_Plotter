#!/usr/bin/env python3
"""Parameterize on-shell FWHM as a function of (M_N/M_WR, M_WR) using ROOT.

Recomputes FWHM directly from signal histograms (no JSON intermediate),
using the same routines as measure_fwhm.py, then fits four candidate
functional forms separately for ee and mumu with ROOT's Minuit2 (the
standard CMS fitting tool). The full parameter covariance is saved so
downstream code can propagate the constraint to the S+B datacard.

Models (x = M_N/M_WR, M = M_WR):
    (a) FWHM = a*x + b*M
    (b) FWHM = a*x + b*M + c
    (c) FWHM = M * (a + b*x)
    (d) FWHM = a*x*sqrt(M) + b*sqrt(M)

Setup (ROOT required):
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Outputs (per channel):
    - fit_<channel>.{pdf,png}      overlay of best model on data
    - per_mass_<channel>.{pdf,png} per-mass linear (a + b*x) fit results
    - results.{json,csv}           parameters + cov + chi2/ndf for all models
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

try:
    import ROOT
except ImportError:
    sys.exit(
        "ERROR: PyROOT is not available. Please set up a ROOT-enabled environment.\n"
        "  Options: cmsenv (CMSSW), conda install root, or source LCG views\n"
        "  e.g. source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"
    )

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

from measure_fwhm import (
    ONSHELL_WINDOW_LO_FRAC,
    ONSHELL_WINDOW_HI_FRAC,
    _compute_fwhm,
    build_hist_key,
    build_region_name,
    compute_shape_params,
    discover_signal_grid,
    load_and_combine_signal,
    parse_masses,
    rebin_histogram,
)

logger = logging.getLogger(__name__)

X_LO_DEFAULT = 0.10
# Per-point upper cut: x <= (M_WR - X_UPPER_GAP_GEV) / M_WR.
# Drops the near-mass-degenerate endpoint at M_N = M_WR - 100 GeV (already
# excluded in measure_fwhm.discover_signal_grid via EXCLUDE_MASS_GAP_GEV).
X_UPPER_GAP_GEV_DEFAULT = 100
WR_MAX_DEFAULT = 6000
N_TOYS_DEFAULT = 100
MIN_VALID_TOYS = 30

# Irreducible FWHM uncertainty from the binned half-max crossings. Each
# crossing has position uncertainty bin_width/sqrt(12) from a uniform-within-
# bin prior (classic "strip detector" resolution). FWHM = x_hi - x_lo, with
# the two crossings independent, so the variance adds in quadrature.
SQRT12 = float(np.sqrt(12.0))
SQRT2 = float(np.sqrt(2.0))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def model_a(X, a, b):
    x, M = X
    return a * x + b * M


def model_b(X, a, b, c):
    x, M = X
    return a * x + b * M + c


def model_c(X, a, b):
    x, M = X
    return M * (a + b * x)


def model_d(X, a, b):
    x, M = X
    return a * x * np.sqrt(M) + b * np.sqrt(M)


def model_e(X, a, b):
    x, M = X
    return a * x * M + b * np.sqrt(M)


def model_f(X, a, b):
    x, M = X
    return a * x + b * np.sqrt(M)


def model_g(X, a, b):
    x, M = X
    return a * x * np.sqrt(M) + b * M


# In ROOT TF2 formulas, x = M_N/M_WR, y = M_WR.
# `latex` is the plot-rendered form of the model.
MODELS = {
    "a_linear": {
        "py": model_a, "root": "[0]*x + [1]*y",
        "pnames": ["a", "b"], "expr": "a·x + b·M_WR",
        "latex": r"$\mathrm{FWHM} = a\,x + b\,M_{W_R}$",
    },
    "b_linear_int": {
        "py": model_b, "root": "[0]*x + [1]*y + [2]",
        "pnames": ["a", "b", "c"], "expr": "a·x + b·M_WR + c",
        "latex": r"$\mathrm{FWHM} = a\,x + b\,M_{W_R} + c$",
    },
    "c_mult": {
        "py": model_c, "root": "y * ([0] + [1]*x)",
        "pnames": ["a", "b"], "expr": "M_WR·(a + b·x)",
        "latex": r"$\mathrm{FWHM} = M_{W_R}\,(a + b\,x)$",
    },
    "d_sqrtM": {
        "py": model_d, "root": "[0]*x*sqrt(y) + [1]*sqrt(y)",
        "pnames": ["a", "b"], "expr": "a·x·√M_WR + b·√M_WR",
        "latex": r"$\mathrm{FWHM} = a\,x\,\sqrt{M_{W_R}} + b\,\sqrt{M_{W_R}}$",
    },
    "e_slopeM_floorSqrtM": {
        "py": model_e, "root": "[0]*x*y + [1]*sqrt(y)",
        "pnames": ["a", "b"], "expr": "a·x·M_WR + b·√M_WR",
        "latex": r"$\mathrm{FWHM} = a\,x\,M_{W_R} + b\,\sqrt{M_{W_R}}$",
    },
    "f_slopeX_floorSqrtM": {
        "py": model_f, "root": "[0]*x + [1]*sqrt(y)",
        "pnames": ["a", "b"], "expr": "a·x + b·√M_WR",
        "latex": r"$\mathrm{FWHM} = a\,x + b\,\sqrt{M_{W_R}}$",
    },
    "g_slopeSqrtM_floorM": {
        "py": model_g, "root": "[0]*x*sqrt(y) + [1]*y",
        "pnames": ["a", "b"], "expr": "a·x·√M_WR + b·M_WR",
        "latex": r"$\mathrm{FWHM} = a\,x\,\sqrt{M_{W_R}} + b\,M_{W_R}$",
    },
}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _fwhm_onshell_from_arrays(edges, values, wr_mass):
    """Recompute fwhm_onshell from a (possibly resampled) histogram."""
    centers = (edges[:-1] + edges[1:]) / 2.0
    onshell_lo = ONSHELL_WINDOW_LO_FRAC * wr_mass
    onshell_hi = ONSHELL_WINDOW_HI_FRAC * wr_mass
    pmask = (centers >= onshell_lo) & (centers <= onshell_hi)
    if pmask.sum() < 3:
        return np.nan
    pvals = values[pmask]
    pcent = centers[pmask]
    if pvals.sum() <= 0:
        return np.nan
    fwhm, _, _, _ = _compute_fwhm(pcent, pvals)
    return float(fwhm)


def _bootstrap_fwhm_err(edges, vals, var, wr_mass, n_toys, rng):
    """Poisson-resample bin contents (unweighted MC); return std of FWHM."""
    lam = np.maximum(vals, 0.0)
    samples = []
    for _ in range(n_toys):
        toy_vals = rng.poisson(lam=lam).astype(float)
        f = _fwhm_onshell_from_arrays(edges, toy_vals, wr_mass)
        if np.isfinite(f):
            samples.append(f)
    if len(samples) < MIN_VALID_TOYS:
        return np.nan, len(samples)
    return float(np.std(samples, ddof=1)), len(samples)


def collect_fwhm_table(
    era: str,
    channel: str,
    topology: str,
    rebin: int,
    input_dirs: list[Path],
    wr_max: int,
    n_toys: int,
    rng: np.random.Generator,
    binning_floor: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (x, M_WR, FWHM, FWHM_err, tags); FWHM_err from bootstrap.

    If `binning_floor` is True (default), add the irreducible binning-resolution
    contribution `Δm/√6` in quadrature on each per-point FWHM uncertainty.
    Set False to reproduce the legacy bootstrap-only error bars.
    """
    region = build_region_name(channel, topology)
    mass_var = "mass_twoobject" if topology == "boosted" else "mass_fourobject"
    hist_key = build_hist_key(region, mass_var)

    grid = discover_signal_grid(input_dirs)
    xs, ms, fws, errs, tags = [], [], [], [], []
    for wr_label, sig_tags in grid.items():
        wr = int(wr_label[2:])
        if wr > wr_max:
            continue
        for sig in sig_tags:
            try:
                edges, vals, var = load_and_combine_signal(input_dirs, hist_key, sig)
                if rebin > 1:
                    edges, vals, var = rebin_histogram(edges, vals, var, rebin)
                sp = compute_shape_params(edges, vals, sig, channel, topology)
            except Exception as e:
                logger.warning("Skip %s/%s: %s", channel, sig, e)
                continue
            if not np.isfinite(sp.fwhm_onshell):
                continue
            wr_m, n_m = parse_masses(sig)
            err_boot, n_ok = _bootstrap_fwhm_err(edges, vals, var, float(wr_m),
                                                 n_toys, rng)
            if not np.isfinite(err_boot):
                logger.warning("  %s: bootstrap failed (%d/%d toys ok), skip",
                               sig, n_ok, n_toys)
                continue
            # Irreducible binning-resolution floor: two independent half-max
            # crossings each have position uncertainty bin_width/sqrt(12).
            bin_width = float(edges[1] - edges[0])
            err_bin = (SQRT2 * bin_width / SQRT12) if binning_floor else 0.0
            err = float(np.sqrt(err_boot ** 2 + err_bin ** 2))
            xs.append(n_m / wr_m)
            ms.append(float(wr_m))
            fws.append(float(sp.fwhm_onshell))
            errs.append(err)
            tags.append(sig)
    return (np.asarray(xs), np.asarray(ms), np.asarray(fws),
            np.asarray(errs), tags)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_model(name, x, M, y, yerr):
    """ROOT (Minuit2) weighted chi2 fit on TGraph2DErrors.

    Returns a dict including the full parameter covariance matrix.
    """
    spec = MODELS[name]
    pnames = spec["pnames"]
    func_py = spec["py"]

    n = len(y)
    g = ROOT.TGraph2DErrors(n)
    for i in range(n):
        g.SetPoint(i, float(x[i]), float(M[i]), float(y[i]))
        g.SetPointError(i, 0.0, 0.0, float(yerr[i]))

    f = ROOT.TF2(
        f"f_{name}", spec["root"],
        float(x.min()), float(x.max()),
        float(M.min()), float(M.max()),
    )
    for i in range(len(pnames)):
        f.SetParameter(i, 1.0)

    # Q = quiet, S = return TFitResultPtr, EX0 = use only z-errors (none on x, y)
    fit_result = g.Fit(f, "QSEX0")
    status = int(fit_result)
    if status != 0:
        logger.warning("Fit %s ended with non-zero Minuit status %d", name, status)

    popt = np.array([f.GetParameter(i) for i in range(len(pnames))])
    perr = np.array([f.GetParError(i)  for i in range(len(pnames))])
    cov = np.array([[fit_result.CovMatrix(i, j)
                     for j in range(len(pnames))]
                    for i in range(len(pnames))])

    # Residuals (unweighted) for diagnostics
    y_pred = func_py((x, M), *popt)
    resid = y - y_pred

    return {
        "model": name,
        "expression": spec["expr"],
        "params": dict(zip(pnames, popt.tolist())),
        "errors": dict(zip(pnames, perr.tolist())),
        "covariance": cov.tolist(),
        "param_order": list(pnames),
        "chi2": float(fit_result.Chi2()),
        "ndf":  int(fit_result.Ndf()),
        "minuit_status": status,
        "rss_per_point": float(np.sqrt(np.mean(resid ** 2))),
        "max_abs_resid": float(np.max(np.abs(resid))),
        "max_rel_resid": float(np.max(np.abs(resid / y))),
    }


def per_mass_linear_fits(x, M, y, yerr):
    """Fit FWHM(x) = a + b*x at each unique M_WR with ROOT.

    Returns list of dicts.
    """
    out = []
    for wr in sorted(set(M.tolist())):
        mask = M == wr
        if mask.sum() < 3:
            continue
        xi, yi, ei = x[mask], y[mask], yerr[mask]

        n = len(xi)
        g = ROOT.TGraphErrors(n)
        for i in range(n):
            g.SetPoint(i, float(xi[i]), float(yi[i]))
            g.SetPointError(i, 0.0, float(ei[i]))

        f = ROOT.TF1(f"f_pm_{int(wr)}", "[0] + [1]*x",
                     float(xi.min()), float(xi.max()))
        f.SetParameter(0, 1.0); f.SetParameter(1, 1.0)
        fit_result = g.Fit(f, "QSEX0")
        if int(fit_result) != 0:
            continue

        out.append({
            "M_WR": wr,
            "a":     f.GetParameter(0), "b":     f.GetParameter(1),
            "a_err": f.GetParError(0),  "b_err": f.GetParError(1),
            "cov_ab": float(fit_result.CovMatrix(0, 1)),
            "n_points": int(mask.sum()),
        })
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fmt_val_err(val: float, err: float) -> str:
    """PDG-style: round error to 2 sig figs, value to the same decimal place.

    Gives consistent precision between channels regardless of magnitude.
    """
    import math
    if not (math.isfinite(err) and err > 0):
        return f"{val:g} \\pm {err:g}"
    exp = int(math.floor(math.log10(abs(err))))
    dec = max(0, 1 - exp)
    return f"{val:.{dec}f} \\pm {err:.{dec}f}"


def plot_overlay(x, M, y, yerr, best, output_path, *, channel, era, com):
    spec = MODELS[best["model"]]
    func, pnames = spec["py"], spec["pnames"]
    popt = [best["params"][p] for p in pnames]

    hep.style.use("CMS")
    fig, ax = plt.subplots()

    wrs = sorted(set(M.tolist()))
    cmap = mcm.viridis
    norm = mcolors.Normalize(vmin=min(wrs), vmax=max(wrs))

    xfit = np.linspace(0.05, 0.98, 120)
    for wr in wrs:
        mask = M == wr
        c = cmap(norm(wr))
        order = np.argsort(x[mask])
        ax.errorbar(x[mask][order], y[mask][order],
                    yerr=yerr[mask][order],
                    marker="o", linestyle="", color=c, markersize=4,
                    markeredgecolor="black", markeredgewidth=0.3,
                    elinewidth=0.8, capsize=0)
        ax.plot(xfit, func((xfit, np.full_like(xfit, wr)), *popt),
                color=c, linestyle="--", linewidth=1.5)

    ax.set_xlabel(r"$M_N / M_{W_R}$")
    ax.set_ylabel("FWHM [GeV]")
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=20)

    ch = {"ee": "ee", "mumu": r"$\mu\mu$"}[channel]
    param_str = ",  ".join(
        rf"${p} = {_fmt_val_err(best['params'][p], best['errors'][p])}$"
        for p in pnames
    )
    chi2_ndf = best["chi2"] / best["ndf"]
    ax.text(
        0.05, 0.96,
        f"{ch}\nResolved SR\n{era}\n"
        f"{spec['latex']}\n"
        f"{param_str}\n"
        rf"$\chi^{{2}}/\mathrm{{ndf}} = "
        rf"{best['chi2']:.1f} / {best['ndf']} = {chi2_ndf:.2f}$"
        "\n(one dashed line per $M_{W_R}$, same $a$, $b$)",
        transform=ax.transAxes, fontsize=13, verticalalignment="top",
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=14)

    # Fixed y-axis so ee/mumu plots are directly comparable.
    # Largest FWHM+err across both channels is ~1080 GeV; this covers both
    # with headroom for the text box.
    ax.set_ylim(0, 1500)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_global_fit(x, M, y, yerr, best, output_path, *, channel, era, com):
    """Predicted-vs-measured scatter — the global fit as a single line.

    Collapses the 2-D (x, M_WR) input onto a single 1-D prediction axis so
    the global fit appears as one y=x reference line. Each point sits where
    its measured FWHM meets the model's prediction for its (x, M_WR).
    """
    spec = MODELS[best["model"]]
    func, pnames = spec["py"], spec["pnames"]
    popt = [best["params"][p] for p in pnames]
    y_pred = func((x, M), *popt)

    hep.style.use("CMS")
    fig, ax = plt.subplots()

    cmap = mcm.viridis
    norm = mcolors.Normalize(vmin=float(M.min()), vmax=float(M.max()))

    # Gray error bars (no marker); colored scatter on top.
    ax.errorbar(y_pred, y, yerr=yerr, fmt="none", ecolor="gray",
                elinewidth=0.6, capsize=0, alpha=0.55, zorder=1)
    sc = ax.scatter(y_pred, y, c=M, cmap=cmap, norm=norm,
                    s=28, edgecolor="black", linewidth=0.3, zorder=2)

    # y = x reference
    lo = float(min(y.min(), y_pred.min()))
    hi = float(max(y.max(), y_pred.max()))
    pad = 0.04 * (hi - lo)
    line_lo, line_hi = lo - pad, hi + pad
    ax.plot([line_lo, line_hi], [line_lo, line_hi],
            color="red", linestyle="--", linewidth=2, zorder=3,
            label=r"$y = x$ (perfect fit)")
    ax.set_xlim(line_lo, line_hi)
    ax.set_ylim(line_lo, line_hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("Predicted FWHM [GeV]")
    ax.set_ylabel("Measured FWHM [GeV]")
    ax.grid(alpha=0.3)

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=20)

    ch = {"ee": "ee", "mumu": r"$\mu\mu$"}[channel]
    param_str = ",  ".join(
        rf"${p} = {_fmt_val_err(best['params'][p], best['errors'][p])}$"
        for p in pnames
    )
    chi2_ndf = best["chi2"] / best["ndf"]
    ax.text(
        0.05, 0.96,
        f"{ch}\nResolved SR\n{era}\n"
        f"{spec['latex']}\n"
        f"{param_str}\n"
        rf"$\chi^{{2}}/\mathrm{{ndf}} = "
        rf"{best['chi2']:.1f} / {best['ndf']} = {chi2_ndf:.2f}$",
        transform=ax.transAxes, fontsize=13, verticalalignment="top",
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_per_mass(per_mass, output_path, *, channel, era, com):
    if not per_mass:
        return
    hep.style.use("CMS")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ms = np.array([d["M_WR"] for d in per_mass])
    a = np.array([d["a"] for d in per_mass])
    b = np.array([d["b"] for d in per_mass])
    ae = np.array([d["a_err"] for d in per_mass])
    be = np.array([d["b_err"] for d in per_mass])

    ax1.errorbar(ms, a, yerr=ae, marker="o", linestyle="", color="#3f90da")
    ax1.set_xlabel(r"$M_{W_R}$ [GeV]")
    ax1.set_ylabel(r"intercept $a$ [GeV]  (FWHM = $a + b\,x$)")
    ax1.grid(alpha=0.3)

    ax2.errorbar(ms, b, yerr=be, marker="o", linestyle="", color="#bd1f01")
    ax2.set_xlabel(r"$M_{W_R}$ [GeV]")
    ax2.set_ylabel(r"slope $b$ [GeV]")
    ax2.grid(alpha=0.3)

    ch = {"ee": "ee", "mumu": r"$\mu\mu$"}[channel]
    fig.suptitle(f"Per-mass FWHM(x) linear fits — {ch}, {era}", fontsize=16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="")
    p.add_argument("--topology", choices=["resolved", "boosted"], default="resolved")
    p.add_argument("--rebin", type=int, default=6)
    p.add_argument("--x-min", type=float, default=X_LO_DEFAULT)
    p.add_argument(
        "--x-upper-gap-gev", type=float, default=X_UPPER_GAP_GEV_DEFAULT,
        help="Per-point upper cut: x <= (M_WR - gap)/M_WR. Default: 100 GeV.",
    )
    p.add_argument("--wr-max", type=int, default=WR_MAX_DEFAULT)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--n-toys", type=int, default=N_TOYS_DEFAULT,
                   help="Bootstrap toys for FWHM uncertainty (default: 100)")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--no-binning-floor", action="store_true",
        help="Disable the binning-resolution floor on per-point FWHM errors. "
             "Reproduces the legacy bootstrap-only uncertainty model.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    era = args.era
    info = load_lumi(era)
    com = info.get("com", 13.0)

    input_dirs, _ = input_dirs_for_era(era, repo_root(), args.dir)
    out_dir = Path(args.output_dir or str(
        repo_root() / "signal_fitting" / "outputs" / era /
        "fwhm" / "fits"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_results: dict = {}

    for channel in ("ee", "mumu"):
        logger.info("=" * 60)
        logger.info("Channel: %s  (bootstrap n_toys=%d)", channel, args.n_toys)

        x, M, y, yerr, tags = collect_fwhm_table(
            era, channel, args.topology, args.rebin, input_dirs, args.wr_max,
            args.n_toys, rng,
            binning_floor=not args.no_binning_floor,
        )
        logger.info("Loaded %d (x, M_WR, FWHM±err) points", len(x))
        if len(yerr):
            logger.info("  median FWHM err = %.1f GeV (range %.1f–%.1f)",
                        float(np.median(yerr)), float(yerr.min()),
                        float(yerr.max()))

        # Restrict fit range.
        # Upper bound is per-point: x <= (M_WR - gap)/M_WR.
        x_upper_per_point = (M - args.x_upper_gap_gev) / M
        sel = (x >= args.x_min) & (x <= x_upper_per_point)
        xs, Ms, ys, es = x[sel], M[sel], y[sel], yerr[sel]
        logger.info(
            "After %g <= x and x <= (M_WR - %g)/M_WR: %d points",
            args.x_min, args.x_upper_gap_gev, sel.sum(),
        )

        # Per-mass linear fits
        per_mass = per_mass_linear_fits(xs, Ms, ys, es)

        # Global fits
        fits = {}
        for name in MODELS:
            r = fit_model(name, xs, Ms, ys, es)
            if r is None:
                continue
            fits[name] = r
            logger.info("  %-15s chi2/ndf = %8.2f / %d   RMS=%.1f GeV   max_rel=%.2f%%",
                        name, r["chi2"], r["ndf"],
                        r["rss_per_point"], 100 * r["max_rel_resid"])

        best_chi2 = min(fits, key=lambda k: fits[k]["chi2"] / fits[k]["ndf"])
        logger.info("  Best by chi2/ndf: %s (chi2/ndf=%.2f)",
                    best_chi2, fits[best_chi2]["chi2"] / fits[best_chi2]["ndf"])

        # Pin downstream to the 2-parameter linear model regardless of which
        # candidate has the lowest chi2/ndf. Differences are small (a_linear
        # vs b_linear_int are typically within ~0.1 in chi2/ndf), and the
        # 2-parameter form is simpler to reason about as a width prior.
        # `best` here drives the plotted overlay and the saved best_model key.
        best_name = "a_linear"
        best = fits[best_name]

        # Plots
        plot_overlay(
            xs, Ms, ys, es, best,
            out_dir / f"fit_{channel}.pdf",
            channel=channel, era=era, com=com,
        )
        plot_global_fit(
            xs, Ms, ys, es, best,
            out_dir / f"global_fit_{channel}.pdf",
            channel=channel, era=era, com=com,
        )
        plot_per_mass(
            per_mass, out_dir / f"per_mass_{channel}.pdf",
            channel=channel, era=era, com=com,
        )

        all_results[channel] = {
            "n_points_fit": int(sel.sum()),
            "n_toys": args.n_toys,
            "x_min": args.x_min,
            "x_upper_gap_gev": args.x_upper_gap_gev,
            "wr_max": args.wr_max,
            "best_model": best_name,
            "models": fits,
            "per_mass_linear": per_mass,
            "points": [
                {"tag": t, "x": float(x_), "M_WR": float(m_),
                 "fwhm": float(y_), "fwhm_err": float(e_)}
                for t, x_, m_, y_, e_ in zip(tags, x, M, y, yerr)
            ],
        }

    # JSON
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # CSV (one row per channel x model)
    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "model", "expression",
                    "a", "a_err", "b", "b_err", "c", "c_err",
                    "chi2", "ndf", "chi2_per_ndf",
                    "rss_per_point_GeV", "max_rel_resid"])
        for ch, res in all_results.items():
            for name, r in res["models"].items():
                p_ = r["params"]
                e_ = r["errors"]
                w.writerow([
                    ch, name, r["expression"],
                    p_.get("a"), e_.get("a"),
                    p_.get("b"), e_.get("b"),
                    p_.get("c", ""), e_.get("c", ""),
                    r["chi2"], r["ndf"], r["chi2"] / r["ndf"],
                    r["rss_per_point"], r["max_rel_resid"],
                ])

    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
