#!/usr/bin/env python3
"""Stage 4 — blind-window sideband closure: background-fit bias vs m_WR.

The essential first test of a background fit function: can it predict the
background under a blinded signal window from the sidebands alone? For one
category (default ee resolved) and each signal-grid mass hypothesis m_WR:

  1. Define a trial signal window [μ - kσ, μ + kσ] centered on the *fitted*
     Gaussian peak μ (mu_gaus, not the nominal m_WR), with σ the on-shell
     Gaussian-core width — both at that grid point (median over M_N, from
     Stage 1's gauss_fit_table.csv). k defaults to 2.
  2. Blind that window.
  3. Fit the summed MC background (DY + tt/tW + nonprompt + other) OUTSIDE the
     window (the sidebands, over the fit range).
  4. Predict the background INSIDE the window by integrating the fit.
  5. Compare to the true MC background in the window:
        bias = (B_pred - B_true) / B_true
  6. Plot bias vs m_WR, one curve per candidate fit function.

Candidate functions are all linear-in-parameters in log space, so each is fit by
weighted linear least squares on ln(counts) (weights from the MC sumw2):

  expo    ln f = a + b·t                      (t = m/1000)
  expo2   ln f = a + b·t + c·t²
  expo3   ln f = a + b·t + c·t² + d·t³
  powexp  ln f = a + b·ln t + c·t             (power law × exponential)
  dijet   ln f = a + b·ln(1-x) + c·ln x       (x = m/√s, √s = 13.6 TeV)

Mass hypotheses are every actual signal grid point (from the width table) between
--mass-min and --mass-max (default 2-6 TeV). The scan walks up in mass and stops
once the background runs out for the high-tail fit — i.e. when fewer than
--min-upper-bins populated bins remain above the window. The closure is only
honest while a real upper sideband exists.

Outputs (co-located with this script):
  bias_vs_mass/{channel}_{topology}_k{k}.{png,pdf}   bias vs m_WR per function
  diagnostics/{function}/m{mWR}.{png,pdf}            per-mass fit + blinded window
  bias_table.csv                                     all numbers

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Usage:
  python blind_window_bias.py -v
  python blind_window_bias.py --channel ee --topology resolved --k 2 --diagnostics
"""
from __future__ import annotations

import argparse
import collections
import csv
import logging
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root
from wrplotter.plotting_helpers import custom_log_formatter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BKG_SAMPLES = ["DYJets", "tt_tW", "Nonprompt", "Other"]

MASS_VAR = {"resolved": "mass_fourobject", "boosted": "mass_twoobject"}
MASS_LABEL = {
    "resolved": r"$m_{\ell\ell jj}$ [GeV]",
    "boosted": r"$m_{\ell J}$ [GeV]",
}
CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}
TOPO_LAB = {"resolved": "Resolved SR", "boosted": "Boosted SR"}

SQRT_S = 13600.0  # GeV, for the dijet variable x = m / sqrt(s)

# Candidate fit functions, each a colored entry. The design-matrix columns are
# the basis of ln f; fitting is weighted linear LS on ln(counts).
FUNCTIONS = {
    "expo":   "#000000",
    "expo2":  "#5790fc",
    "expo3":  "#e42536",
    "powexp": "#f89c20",
    "dijet":  "#964a8b",
}

# Full fit form shown on the plots. The fit variable is t = m/1000 (GeV) for the
# poly/powexp forms and x = m/√s for the dijet form, so the printed coefficients
# below match these formulae directly.
FUNC_FORMULA = {
    "expo":   r"$e^{\,a+bt}$",
    "expo2":  r"$e^{\,a+bt+ct^{2}}$",
    "expo3":  r"$e^{\,a+bt+ct^{2}+dt^{3}}$",
    "powexp": r"$e^{a}\,t^{\,b}\,e^{ct}$",
    "dijet":  r"$e^{a}\,(1-x)^{b}\,x^{c}$",
}

COEF_SYMS = "abcd"


def func_label(name: str) -> str:
    return f"{name}:  {FUNC_FORMULA[name]}"


def coef_text(coef) -> str:
    """Multi-line 'a = …\\nb = …' for the fitted coefficients (fit basis)."""
    return "\n".join(rf"${COEF_SYMS[i]}={c:.3g}$" for i, c in enumerate(coef))

# Baseline on-shell window used for the Stage-1 width table lookup.
BASELINE_FIT_RANGE = "[0.7,1.3]"


def design_matrix(name: str, m: np.ndarray) -> np.ndarray:
    """Basis columns for ln f(m) for the named function."""
    t = m / 1000.0
    ones = np.ones_like(t)
    if name == "expo":
        return np.column_stack([ones, t])
    if name == "expo2":
        return np.column_stack([ones, t, t ** 2])
    if name == "expo3":
        return np.column_stack([ones, t, t ** 2, t ** 3])
    if name == "powexp":
        return np.column_stack([ones, np.log(t), t])
    if name == "dijet":
        x = m / SQRT_S
        return np.column_stack([ones, np.log(1.0 - x), np.log(x)])
    raise ValueError(f"unknown function {name!r}")


# ---------------------------------------------------------------------------
# Inputs: signal widths (Stage 1) and summed background (this stage)
# ---------------------------------------------------------------------------

def load_grid_widths(width_csv: Path, channel: str, topology: str,
                     agg: str) -> dict[float, tuple[float, float]]:
    """{m_WR: (mu, sigma)} from the Stage-1 Gaussian width table.

    Both the fitted peak position mu_gaus and the width sigma_gaus are aggregated
    (median/mean/...) over the M_N points at the baseline on-shell window. The
    window is centered on the fitted peak mu, not the nominal m_WR.
    """
    mu_by_mwr: dict[float, list[float]] = collections.defaultdict(list)
    sig_by_mwr: dict[float, list[float]] = collections.defaultdict(list)
    with open(width_csv) as f:
        for r in csv.DictReader(f):
            if (r["channel"] == channel and r["category"] == topology
                    and r["fit_range"] == BASELINE_FIT_RANGE):
                m = float(r["mWR"])
                mu_by_mwr[m].append(float(r["mu_gaus"]))
                sig_by_mwr[m].append(float(r["sigma_gaus"]))
    aggfn = {"median": statistics.median, "mean": statistics.mean,
             "max": max, "min": min}[agg]
    return {m: (aggfn(mu_by_mwr[m]), aggfn(sig_by_mwr[m]))
            for m in sorted(mu_by_mwr)}


def load_summed_background(input_dirs, region, variable, factor):
    """Sum the four background MC samples across sub-era dirs; rebin by factor."""
    hist_key = f"{region}/{variable}_{region}"
    edges = values = variances = None
    for d in input_dirs:
        for s in BKG_SAMPLES:
            fpath = d / f"WRAnalyzer_{s}.root"
            if not fpath.exists():
                logger.warning("Missing file: %s", fpath)
                continue
            h = uproot.open(fpath)[hist_key]
            e, v, var = h.axes[0].edges(), h.values(), h.variances()
            if edges is None:
                edges, values, variances = e, v.copy(), var.copy()
            else:
                values += v
                variances += var
    if edges is None:
        raise RuntimeError("No background histograms found")
    if factor > 1:
        n = (len(values) // factor) * factor
        values = values[:n].reshape(-1, factor).sum(axis=1)
        variances = variances[:n].reshape(-1, factor).sum(axis=1)
        edges = edges[::factor][: len(values) + 1]
    return edges, values, variances


# ---------------------------------------------------------------------------
# Fit + prediction
# ---------------------------------------------------------------------------

def fit_loglinear(name, m, n, var):
    """Weighted linear LS of ln(n) on the function basis.

    Weights are 1/σ_lnN² = (n/σ_N)² with σ_N = sqrt(var), so the coefficient
    covariance is (ΦᵀWΦ)⁻¹ = (AᵀA)⁻¹ — the statistical prediction uncertainty
    propagated from the MC sumw2. Returns (coef, cov), or (None, None) if the
    system is underdetermined / singular.
    """
    Phi = design_matrix(name, m)
    if Phi.shape[0] <= Phi.shape[1]:
        return None, None
    y = np.log(n)
    sw = n / np.sqrt(var)            # sqrt of the per-point weight
    A = Phi * sw[:, None]
    b = y * sw
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        cov = np.linalg.pinv(A.T @ A)
    except np.linalg.LinAlgError:
        return None, None
    if not (np.all(np.isfinite(coef)) and np.all(np.isfinite(cov))):
        return None, None
    return coef, cov


def predict(name, coef, m):
    return np.exp(design_matrix(name, m) @ coef)


def predict_band(name, coef, cov, m):
    """Central f(m) and the 1σ log-uncertainty s(m); band = f·exp(±s)."""
    Phi = design_matrix(name, m)
    f = np.exp(Phi @ coef)
    var_lnf = np.einsum("ij,jk,ik->i", Phi, cov, Phi)
    return f, np.sqrt(np.clip(var_lnf, 0.0, None))


def b_pred_with_err(name, coef, cov, m_window):
    """Predicted yield in the window and its uncertainty from the fit covariance.

    B = Σ_j f_j with f_j = exp(φ_j·θ); Var(B) = gᵀ C g, g = Σ_j f_j φ_j.
    """
    Phi = design_matrix(name, m_window)
    f = np.exp(Phi @ coef)
    g = Phi.T @ f
    return float(f.sum()), float(np.sqrt(max(g @ cov @ g, 0.0)))


def chi2_ndf(name, coef, m, n, var):
    pred = predict(name, coef, m)
    w = (n / np.sqrt(var)) ** 2
    chi2 = float(np.sum(w * (np.log(n) - np.log(pred)) ** 2))
    ndf = len(m) - design_matrix(name, m).shape[1]
    return chi2, ndf


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bias_vs_mass(results, out_path, *, channel, topology, k, com, lumi, era,
                      sigma_agg):
    """results: {function: [(mWR, bias, bias_err), ...]} (fractions)."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()

    all_bias = []
    n_func = len(FUNCTIONS)
    for i, (name, color) in enumerate(FUNCTIONS.items()):
        pts = results.get(name, [])
        if not pts:
            continue
        off = (i - (n_func - 1) / 2.0) * 18.0   # small x-dither so bars don't overlap
        ms = [p[0] + off for p in pts]
        bs = [100.0 * p[1] for p in pts]
        es = [100.0 * p[2] if np.isfinite(p[2]) else 0.0 for p in pts]
        all_bias.extend(bs)
        ax.errorbar(ms, bs, yerr=es, marker="o", markersize=5, linewidth=1.8,
                    color=color, label=func_label(name), capsize=2, elinewidth=1.0,
                    markeredgecolor="black", markeredgewidth=0.3)

    ax.axhline(0.0, color="black", linewidth=1.0)
    for guide in (10.0, -10.0):
        ax.axhline(guide, color="gray", linestyle=":", linewidth=1.0)

    cap = min(100.0, max(15.0, 1.2 * max((abs(b) for b in all_bias), default=15.0)))
    ax.set_ylim(-cap, cap)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"Closure bias $(B_{\mathrm{pred}}-B_{\mathrm{true}})/B_{\mathrm{true}}$ [%]")
    ax.grid(alpha=0.3)

    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=18)
    ax.text(0.05, 0.95,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n{era}\n"
            rf"window $\mu\pm{k:g}\sigma$ (Gauss fit)"
            "\n"
            rf"$\mu,\sigma$: {sigma_agg} over $M_N$"
            "\n"
            rf"$t=m/1000$, $x=m/\sqrt{{s}}$",
            transform=ax.transAxes, fontsize=13, va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=12,
              title=r"fit function  $f(m)$")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("  wrote %s", out_path.with_suffix(".png"))


def plot_diagnostic(edges, values, variances, fit_mask, win_mask, name, coef, cov, *,
                    mWR, mu, win_lo, win_hi, b_pred, b_pred_err, b_true, b_true_err,
                    bias, bias_err, fit_lo, fit_hi,
                    out_path, channel, topology, com, lumi, era):
    """One mass: spectrum + sideband fit (±1σ band) + blinded window."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    hep.style.use("CMS")
    fig, ax = plt.subplots()

    ax.stairs(values, edges, color="#888888", linewidth=1.3, label="MC background")
    # sideband bins actually used by the fit, with their MC stat uncertainty
    ax.errorbar(centers[fit_mask], values[fit_mask],
                yerr=np.sqrt(variances[fit_mask]), fmt="o", color="black",
                markersize=3, elinewidth=0.8, capsize=1.5, label="sideband (fit)")
    # fit curve + 1σ band across the fit range
    grid = np.linspace(fit_lo, fit_hi, 600)
    f_grid, s_grid = predict_band(name, coef, cov, grid)
    ax.fill_between(grid, f_grid * np.exp(-s_grid), f_grid * np.exp(s_grid),
                    color="#e42536", alpha=0.25, lw=0, zorder=1,
                    label=r"fit $\pm1\sigma$")
    ax.plot(grid, f_grid, color="#e42536", linewidth=2, zorder=2, label=f"{name} fit")
    ax.axvspan(win_lo, win_hi, color="#5790fc", alpha=0.18, zorder=0,
               label=r"blinded $\mu\pm k\sigma$")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_xlim(fit_lo, fit_hi)
    pos = values[values > 0]
    if pos.size:
        ax.set_ylim(pos.min() / 3.0, pos.max() * 30.0)
    ax.set_xlabel(MASS_LABEL[topology])
    ax.set_ylabel(f"Events / {edges[1] - edges[0]:.0f} GeV")

    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=16)
    var_note = r"$x=m/\sqrt{s}$" if name == "dijet" else r"$t=m/1000$"
    ax.text(0.05, 0.95,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            rf"$m_{{W_R}}={mWR:.0f}$ GeV" "\n"
            rf"$\mu_{{\mathrm{{fit}}}}={mu:.0f}$ GeV" "\n"
            + var_note,
            transform=ax.transAxes, fontsize=14, va="top")
    bias_str = (rf"$({100 * bias:+.1f}\pm{100 * bias_err:.1f})\%$"
                if np.isfinite(bias_err) else rf"${100 * bias:+.1f}\%$")
    ax.text(0.97, 0.95,
            rf"$B_{{\mathrm{{pred}}}}={b_pred:.2f}\pm{b_pred_err:.2f}$" "\n"
            rf"$B_{{\mathrm{{true}}}}={b_true:.2f}\pm{b_true_err:.2f}$" "\n"
            rf"bias $=${bias_str}" "\n"
            rf"{func_label(name)}" "\n"
            + coef_text(coef),
            transform=ax.transAxes, fontsize=12, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85))
    ax.legend(loc="lower left", fontsize=12)

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
    p.add_argument("--dir", default="20260317_lo_dy",
                   help="Subdir under rootfiles/<run>/<year>/<era>/.")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved", choices=["resolved", "boosted"])
    p.add_argument("--k", type=float, default=2.0,
                   help="Blind-window half-width in sigma. Default 2.")
    p.add_argument("--sigma-agg", default="median",
                   choices=["median", "mean", "max", "min"],
                   help="How to aggregate the per-m_WR signal sigma over M_N.")
    p.add_argument("--bin-width", type=float, default=100.0,
                   help="Display/fit bin width in GeV (rebinned from native).")
    p.add_argument("--fit-min", type=float, default=800.0,
                   help="Lower edge of the fit range (GeV). Default 800 (SR threshold).")
    p.add_argument("--fit-max", type=float, default=6000.0,
                   help="Upper edge of the fit range (GeV).")
    p.add_argument("--mass-min", type=float, default=2000.0,
                   help="Smallest grid m_WR to test.")
    p.add_argument("--mass-max", type=float, default=6000.0,
                   help="Largest grid m_WR to test (the scan auto-stops earlier "
                        "when the background runs out; see --min-upper-bins).")
    p.add_argument("--min-upper-bins", type=int, default=2,
                   help="Stop the ascending mass scan once fewer than this many "
                        "populated bins remain above the window (the high tail "
                        "needed for the fit). Default 2.")
    p.add_argument("--width-csv", type=Path,
                   default=Path(__file__).resolve().parents[1].parent
                   / "1_signal_widths" / "gaussian" / "gauss_fit_table.csv")
    p.add_argument("--diagnostics", action="store_true",
                   help="Also write per-mass fit diagnostic plots, one per "
                        "function (all functions unless --diag-func restricts).")
    p.add_argument("--diag-func", default=None, choices=list(FUNCTIONS),
                   help="Restrict diagnostics to a single function (default: all).")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    era, channel, topology, k = args.era, args.channel, args.topology, args.k
    info = load_lumi(era)
    lumi, com = info["lumi"], info.get("com", 13.6)

    if not args.width_csv.exists():
        logger.error("Width table not found: %s (run Stage 1 first)", args.width_csv)
        sys.exit(1)
    grid_widths = load_grid_widths(args.width_csv, channel, topology, args.sigma_agg)
    masses = sorted(m for m in grid_widths if args.mass_min <= m <= args.mass_max)
    if not masses:
        logger.error("No grid masses in [%g, %g]", args.mass_min, args.mass_max)
        sys.exit(1)
    logger.info("Category %s %s — %d candidate grid masses: %s",
                channel, topology, len(masses), [int(m) for m in masses])

    input_dirs, _ = input_dirs_for_era(era, repo_root(), args.dir)
    region = f"wr_{channel}_{topology}_sr"
    variable = MASS_VAR[topology]
    factor = max(1, round(args.bin_width / 10.0))
    edges, values, variances = load_summed_background(input_dirs, region, variable, factor)
    centers = 0.5 * (edges[:-1] + edges[1:])
    in_fit = (centers >= args.fit_min) & (centers <= args.fit_max)

    diag_funcs = [args.diag_func] if args.diag_func else list(FUNCTIONS)
    results: dict[str, list[tuple[float, float]]] = {n: [] for n in FUNCTIONS}
    rows: list[dict] = []

    for mWR in masses:  # ascending
        mu, sigma = grid_widths[mWR]
        win_lo, win_hi = mu - k * sigma, mu + k * sigma
        in_win = (centers >= win_lo) & (centers <= win_hi)
        b_true = float(values[in_win].sum())
        b_true_err = float(np.sqrt(variances[in_win].sum()))

        # The high-tail fit needs populated bins ABOVE the window (the upper
        # sideband). When those run out, we've run out of background — stop.
        n_upper = int((in_fit & (centers > win_hi) & (values > 0)).sum())
        if b_true <= 0 or n_upper < args.min_upper_bins:
            logger.info("  m_WR=%.0f: ran out of background for the high-tail fit "
                        "(B_true=%.3f, upper-sideband bins=%d < %d) — stopping",
                        mWR, b_true, n_upper, args.min_upper_bins)
            break

        # sideband bins used for the fit: in fit range, outside window, populated.
        fit_mask = in_fit & ~in_win & (values > 0) & (variances > 0)

        for name in FUNCTIONS:
            coef, cov = fit_loglinear(name, centers[fit_mask], values[fit_mask],
                                      variances[fit_mask])
            if coef is None:
                logger.warning("  m_WR=%.0f %s: fit failed", mWR, name)
                continue
            b_pred, b_pred_err = b_pred_with_err(name, coef, cov, centers[in_win])
            bias = (b_pred - b_true) / b_true
            # error bar on the bias: fit-prediction (B_pred) ⊕ MC-stat (B_true).
            if b_pred > 0 and b_true > 0:
                bias_err = abs(b_pred / b_true) * np.hypot(
                    b_pred_err / b_pred, b_true_err / b_true)
            else:
                bias_err = float("nan")
            c2, ndf = chi2_ndf(name, coef, centers[fit_mask],
                               values[fit_mask], variances[fit_mask])
            results[name].append((mWR, bias, bias_err))
            coef_cols = {f"coef_{COEF_SYMS[i]}": "" for i in range(4)}
            coef_cols.update({f"coef_{COEF_SYMS[i]}": round(float(c), 6)
                              for i, c in enumerate(coef)})
            rows.append({
                "channel": channel, "topology": topology, "k": k, "function": name,
                "mWR": mWR, "mu": round(mu, 1), "sigma": round(sigma, 2),
                "win_lo": round(win_lo, 1), "win_hi": round(win_hi, 1),
                "n_sideband_bins": int(fit_mask.sum()), "n_upper_bins": n_upper,
                "B_pred": round(b_pred, 4), "B_pred_err": round(b_pred_err, 4),
                "B_true": round(b_true, 4), "B_true_err": round(b_true_err, 4),
                "bias": round(bias, 5),
                "bias_err": round(bias_err, 5) if np.isfinite(bias_err) else "",
                "chi2_ndf": round(c2 / ndf, 3) if ndf > 0 else None,
                **coef_cols,
            })
            logger.info("  m_WR=%.0f %-7s B_pred=%.2f±%.2f B_true=%.2f±%.2f bias=(%+.1f±%.1f)%%",
                        mWR, name, b_pred, b_pred_err, b_true, b_true_err,
                        100 * bias, 100 * bias_err)

            if args.diagnostics and name in diag_funcs:
                plot_diagnostic(
                    edges, values, variances, fit_mask, in_win, name, coef, cov,
                    mWR=mWR, mu=mu, win_lo=win_lo, win_hi=win_hi,
                    b_pred=b_pred, b_pred_err=b_pred_err,
                    b_true=b_true, b_true_err=b_true_err,
                    bias=bias, bias_err=bias_err,
                    fit_lo=args.fit_min, fit_hi=args.fit_max,
                    out_path=args.output_dir / "diagnostics" / name / f"m{int(mWR)}",
                    channel=channel, topology=topology, com=com, lumi=lumi, era=era,
                )

    if not rows:
        logger.error("No closure results produced.")
        sys.exit(1)

    plot_bias_vs_mass(
        results, args.output_dir / "bias_vs_mass" / f"{channel}_{topology}_k{k:g}",
        channel=channel, topology=topology, k=k, com=com, lumi=lumi, era=era,
        sigma_agg=args.sigma_agg,
    )

    csv_path = args.output_dir / "bias_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s", csv_path)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
