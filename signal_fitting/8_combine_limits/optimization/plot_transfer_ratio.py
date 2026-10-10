#!/usr/bin/env python3
"""Transfer factor R = N_SR/N_CR for tt+tW MC: how flat is it vs mass?

Justifies the fcr_shape global-R choice with actual MC statistics:

  - per-window R (the k5 card windows) with errors propagated from the
    WRAnalyzer sumw2 variances,
  - the flatness verdict on INDEPENDENT per-bin ratios SR_i/CR_i
    (adjacent bins share no events, unlike the overlapping windows) over
    the trusted range: a constant fit (chi2/ndf) and a linear fit whose
    slope, quoted per TeV with its error and pull from zero, is the
    flatness piece of the transfer-factor systematic. Both are drawn on
    the plots (transfer_ratio_vs_mass* and the dedicated
    transfer_ratio_flatness).

    Verdict bins (--flat-bins): "variable" (default) = FLAT_BINS_VAR,
    built by a rule: 200 GeV bins merged upward until every bin holds
    >= FLAT_NEFF_MIN effective SR tt MC events (the usual threshold for
    treating a weighted count as Gaussian) = 200 GeV to 2.4 TeV, then
    one bin to the end of the MC; ratio errors <= 24%; "fixed" = REBIN_GEV-wide
    bins from 800 to the last bin with MC in both regions (FLAT_MAX is
    only a ceiling; ee runs out at 4.0 TeV, mumu at 4.4 TeV -- above
    3.2 TeV the 200 GeV bins hold 1-3 effective MC events each and their
    ratio errors exceed 100%). Plain chi2 with the observed (sumw2)
    errors -- the conventional fit. Both give R = 0.424; the ee slope is
    -1.2 sigma (variable) vs -1.8 sigma (fixed), the difference being
    the few-event bias described below. Above 2.4 TeV the ee SR holds
    ~40 effective MC events in total: every binning tried (2.4-2.8/2.8-3.2
    split, 2.4-3.2/3.2-4.4 split, one bin) gives R = 0.424 and a slope
    error ~0.02 and differs only in the pull (0.7-1.8 sigma) -- noise.
    The width is chosen so the low-count bias stays small: with errors
    from the observed counts, few-event bins that fluctuate low get small
    errors and too much weight (Neyman bias), which at 100 GeV steepens
    the ee slope to -3 sigma. Two cross-checks are logged, not drawn:
    the same bins with PEARSON errors (SR variance from the fitted
    expectation R_fit(m) * C_i, iterated), which removes the bias, and
    the 100 GeV Neyman fit, which shows it.

  python plot_transfer_ratio.py -v      (outputs in explain_fcr/)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent          # 8_combine_limits/optimization
SIGFIT = HERE.parents[1]
sys.path.insert(0, str(SIGFIT.parent))          # repo root (wrplotter)

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402

import uproot                                                           # noqa: E402
import ROOT                                                             # noqa: E402

ROOT.gROOT.SetBatch(True)
logger = logging.getLogger("plot_transfer_ratio")

import os                                                               # noqa: E402
if os.environ.get("WR_SEMINAR_ERA", "2018") == "2024":
    ERA, BKG_DIR = "RunIII2024Summer24", "20261004_full_systs"
    INPUTS, OUT_DEFAULT = HERE / "inputs_2024", HERE / "explain_fcr_2024"
else:
    ERA, BKG_DIR = "RunIISummer20UL18", "20260714_run2_bkgs"
    INPUTS, OUT_DEFAULT = INPUTS, HERE / "explain_fcr"
FCR_REGION = "wr_resolved_flavor_cr"
# (label, SR region, color, marker)
CHANNELS = [("ee", "wr_ee_resolved_sr", "#e42536", "o"),
            (r"\mu\mu", "wr_mumu_resolved_sr", "#5790fc", "s")]
REBIN_GEV = 200.0            # --flat-bins fixed: bin width
FINE_GEV = 100.0             # cross-check width (logged only)
FLAT_MAX = 6000              # flatness-fit ceiling; the MC ends first
# --flat-bins variable: 200 GeV bins merged upward until each holds
# >= FLAT_NEFF_MIN effective ee SR tt MC events (ratio errors 2% ... 24%).
# 200 GeV bins above 2.4 TeV hold 17, 9, 5, 3, 3, 2, 1, 2 events.
FLAT_NEFF_MIN = 20
FLAT_BINS_VAR = [800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400,
                 4400]
TRUSTED_MAX = 3200           # reliability edge of the ee analysis
M_REF = 2000.0               # linear fit pivot: R(m) = R_ref + s (m - M_REF)/TeV
PEARSON_ITER = 5             # error-from-expectation iterations


def load_tt(bkg_dirs, region):
    """Summed tt_tW (edges, values, variances) at native 10 GeV."""
    key = f"{region}/mass_fourobject_{region}"
    edges = values = variances = None
    for d in bkg_dirs:
        fp = d / "WRAnalyzer_tt_tW.root"
        if not fp.exists():
            continue
        h = uproot.open(fp)[key]
        e, v, w2 = h.axes[0].edges(), h.values(), h.variances()
        if edges is None:
            edges, values, variances = e, v.copy(), w2.copy()
        else:
            values += v
            variances += w2
    keep = np.isfinite(values) & (values > 0)
    return edges, np.where(keep, values, 0.0), np.where(keep, variances, 0.0)


def window_sums(edges, values, variances, lo, hi):
    centers = 0.5 * (edges[:-1] + edges[1:])
    m = (centers >= lo) & (centers <= hi)
    return float(values[m].sum()), float(variances[m].sum())


def k5_window(mass, wpar, edges, bw=50.0, k=5.0):
    """[m_c - k sigma, m_c + k sigma] from the Stage-2 linear window
    parameterization (sigma_median, the ee-window convention), snapped to
    the card grid with the center-in-range rule."""
    m_c = wpar["mu"][0] * mass + wpar["mu"][1]
    sig = wpar["sigma_median"][0] * mass + wpar["sigma_median"][1]
    lo, hi = m_c - k * sig, m_c + k * sig
    f = max(1, round(bw / float(edges[1] - edges[0])))
    n = (len(edges) - 1) // f * f
    e = edges[0:n + 1:f]
    centers = 0.5 * (e[:-1] + e[1:])
    sel = (centers >= lo) & (centers <= hi)
    return float(e[:-1][sel][0]), float(e[1:][sel][-1])


def independent_bin_counts(edges, v_sr, w_sr, v_cr, w_cr,
                           width=REBIN_GEV, m_max=FLAT_MAX, bins=None):
    """Summed (S, sumw2_S, C, sumw2_C) per independent bin (adjacent
    bins share no events): `bins` = explicit edge list, else fixed-width
    bins from the spectrum start to m_max. Empty bins are dropped.
    Returns (centers, half_widths, S, S2, C, C2)."""
    if bins is None:
        bins = np.arange(float(edges[0]), m_max + 0.5 * width, width)
    cen = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (cen > a) & (cen < b)
        S, S2 = v_sr[m].sum(), w_sr[m].sum()
        C, C2 = v_cr[m].sum(), w_cr[m].sum()
        if S > 0 and C > 0:
            rows.append((0.5 * (a + b), 0.5 * (b - a), S, S2, C, C2))
    return tuple(np.array(rows).T)


def ratio_errors(S, S2, C, C2, S_exp=None):
    """Error on r = S/C. Neyman (default): sumw2 of the observed counts.
    Pearson (S_exp given): SR variance rescaled to the expected yield,
    S2 * S_exp / S (same mean weight), evaluated at r = S_exp / C."""
    if S_exp is None:
        return (S / C) * np.sqrt(S2 / S**2 + C2 / C**2)
    S2_exp = S2 * S_exp / S
    return (S_exp / C) * np.sqrt(S2_exp / S_exp**2 + C2 / C**2)


def _chi2_fit(centers, r_i, s_i, formula, start, tag, lo, hi):
    g = ROOT.TGraphErrors(len(centers))
    for i, (m, r, e) in enumerate(zip(centers, r_i, s_i)):
        g.SetPoint(i, float(m), float(r))
        g.SetPointError(i, 0.0, float(e))
    f = ROOT.TF1(tag, formula, lo - 1, hi + 1)
    for k, v in enumerate(start):
        f.SetParameter(k, v)
    res = g.Fit(f, "QNS")
    n = f.GetNpar()
    cov = np.array([[res.CovMatrix(i, j) for j in range(n)]
                    for i in range(n)])
    return f, cov


def fit_flatness(centers, hw, S, S2, C, C2, tag, m_ref=M_REF,
                 pearson=False, n_iter=PEARSON_ITER):
    """Constant and linear chi2 fits (ROOT) to the independent-bin
    ratios r_i = S_i/C_i. Linear form R(m) = R_ref + slope*(m - m_ref)/TeV,
    so R_ref is the value at the pivot and the slope is per TeV with
    little correlation to it. pearson=True: each fit's errors are taken
    from its own fitted expectation (iterated n_iter times from the
    Neyman start); False: plain Neyman errors. Returns a dict with both
    fits, the slope pull from zero, the excursion |slope|*half-range/R_ref
    (fractional change of R across the range: the flatness term of the
    lnN) and the per-bin ratios + errors (constant-model errors) to draw."""
    lo, hi = float(centers.min()), float(centers.max())
    r_i = S / C
    s_neyman = ratio_errors(S, S2, C, C2)
    lin = f"[0] + [1]*(x - {m_ref})/1000."

    s0 = s_neyman
    for it in range(n_iter if pearson else 1):
        f0, _ = _chi2_fit(centers, r_i, s0, "[0]", [float(r_i.mean())],
                          f"flat0_{tag}_{it}", lo, hi)
        if pearson:
            s0 = ratio_errors(S, S2, C, C2, S_exp=f0.GetParameter(0) * C)
    s1 = s_neyman
    for it in range(n_iter if pearson else 1):
        f1, cov1 = _chi2_fit(centers, r_i, s1, lin,
                             [float(r_i.mean()), 0.0],
                             f"flat1_{tag}_{it}", lo, hi)
        if pearson:
            r_exp = f1.GetParameter(0) + f1.GetParameter(1) * (centers - m_ref) / 1e3
            s1 = ratio_errors(S, S2, C, C2, S_exp=r_exp * C)

    slope, e_slope = f1.GetParameter(1), f1.GetParError(1)
    r_ref = f1.GetParameter(0)
    return dict(
        lo=lo, hi=hi, n=len(centers), r_i=r_i, s_i=s0, centers=centers,
        hw=hw, lo_edge=lo - hw[0], hi_edge=hi + hw[-1],
        r0=f0.GetParameter(0), e0=f0.GetParError(0),
        chi2_0=f0.GetChisquare(), ndf_0=f0.GetNDF(),
        r_ref=r_ref, e_ref=f1.GetParError(0),
        slope=slope, e_slope=e_slope,
        pull=slope / e_slope if e_slope > 0 else float("nan"),
        chi2_1=f1.GetChisquare(), ndf_1=f1.GetNDF(), cov1=cov1,
        excursion=abs(slope) * 0.5 * (hi - lo) / 1000.0 / r_ref,
        m_ref=m_ref)


def linear_band(fl, xs):
    """Central value and 1-sigma error of the linear fit at xs, from the
    (R_ref, slope) covariance."""
    t = (xs - fl["m_ref"]) / 1e3
    y = fl["r_ref"] + fl["slope"] * t
    var = (fl["cov1"][0, 0] + t**2 * fl["cov1"][1, 1]
           + 2 * t * fl["cov1"][0, 1])
    return y, np.sqrt(np.clip(var, 0, None))


def draw_fit_bands(ax, fl, xs, color, constant=True, linear=True):
    """1-sigma bands of the constant (R0 +- e0) and linear fits."""
    if constant:
        ax.fill_between(xs, fl["r0"] - fl["e0"], fl["r0"] + fl["e0"],
                        color=color, alpha=0.12, lw=0)
    if linear:
        y, e = linear_band(fl, xs)
        ax.fill_between(xs, y - e, y + e, color=color, alpha=0.12, lw=0)


def flatness_text(lab, fl):
    """Two-line verdict for the plots."""
    return (fr"${lab}$: const $R = {fl['r0']:.3f} \pm {fl['e0']:.3f}$, "
            fr"$\chi^2$/ndf = {fl['chi2_0']:.0f}/{fl['ndf_0']}"
            fr"$\;=\;{fl['chi2_0'] / fl['ndf_0']:.2f}$" "\n"
            fr"      slope = ${fl['slope']:+.4f} \pm {fl['e_slope']:.4f}$ "
            fr"/TeV (${fl['pull']:+.1f}\sigma$), "
            fr"$\chi^2$/ndf = {fl['chi2_1']:.0f}/{fl['ndf_1']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--flat-bins", choices=["variable", "fixed"],
                   default="variable",
                   help="flatness-verdict binning: variable = "
                        f"{FLAT_BINS_VAR}; fixed = {REBIN_GEV:.0f} GeV")
    p.add_argument("--no-stats-box", action="store_true",
                   help="flatness plot: omit the fit-numbers box (they are "
                        "always logged)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    bkg_dirs, _ = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    e_cr, v_cr, w_cr = load_tt(bkg_dirs, FCR_REGION)
    with open(INPUTS / "ee_resolved_fcr.json") as fh:
        fcr = json.load(fh)
    with open(SIGFIT / "2_width_parameterization" / "wr"
              / "window_params.json") as fh:
        wpars = json.load(fh)

    # --- collect per-channel points once ------------------------------------
    data = []
    for lab, sr_region, color, marker in CHANNELS:
        e_sr, v_sr, w_sr = load_tt(bkg_dirs, sr_region)
        wpar = wpars["mumu" if "mumu" in sr_region else "ee"]["resolved"]

        # per-window R with MC-stat errors, each channel in ITS OWN k5
        # windows (Stage-2 parameterization; ee reproduces the card json).
        # Extends past the card mass list -- the windows are analytic and
        # the MC spectrum is the limit, not the json.
        masses, ratios, errs = [], [], []
        for mass in range(1400, 6001, 200):
            slo, shi = k5_window(float(mass), wpar, e_sr)
            s, s2 = window_sums(e_sr, v_sr, w_sr, slo, shi)
            c, c2 = window_sums(e_cr, v_cr, w_cr, slo, shi)
            if s <= 0 or c <= 0:
                logger.info("%s m=%d: empty window (SR %.2f / CR %.2f), "
                            "dropped", lab, mass, s, c)
                continue
            r = s / c
            masses.append(mass)
            ratios.append(r)
            errs.append(r * np.sqrt(s2 / s**2 + c2 / c**2))

        # global (all events summed) -- the model's R. The flatness
        # verdict comes from the independent 100 GeV bins in the trusted
        # range: constant fit (chi2/ndf) + linear fit (slope, pull).
        r_glob = float(v_sr.sum() / v_cr.sum())
        cnt = independent_bin_counts(
            e_sr, v_sr, w_sr, v_cr, w_cr,
            bins=FLAT_BINS_VAR if args.flat_bins == "variable" else None)
        fl = fit_flatness(*cnt, lab.strip("\\"))
        # end of the usable MC: upper edge of the last fixed-width bin
        # with tt MC in BOTH regions (a lone SR event with no CR partner
        # cannot make a ratio)
        fx_c, fx_hw = independent_bin_counts(e_sr, v_sr, w_sr, v_cr, w_cr)[:2]
        fl["mc_end"] = float(fx_c[-1] + fx_hw[-1])
        logger.info("%s: global R = %.4f; verdict on %d independent %s "
                    "bins [%.0f, %.0f], observed errors; rel. errors %s",
                    lab, r_glob, fl["n"], args.flat_bins, fl["lo_edge"],
                    fl["hi_edge"],
                    " ".join(f"{c:.0f}:{100 * e / r:.0f}%" for c, r, e in
                             zip(fl["centers"], fl["r_i"], fl["s_i"])))
        logger.info("%s: constant fit R = %.4f +- %.4f, chi2/ndf = "
                    "%.1f/%d = %.2f", lab, fl["r0"], fl["e0"],
                    fl["chi2_0"], fl["ndf_0"], fl["chi2_0"] / fl["ndf_0"])
        logger.info("%s: linear fit R(%.0f) = %.4f +- %.4f, slope = "
                    "%+.4f +- %.4f /TeV (%+.1f sigma from zero), chi2/ndf "
                    "= %.1f/%d = %.2f; excursion across range = %.1f%% "
                    "of R", lab, fl["m_ref"], fl["r_ref"], fl["e_ref"],
                    fl["slope"], fl["e_slope"], fl["pull"], fl["chi2_1"],
                    fl["ndf_1"], fl["chi2_1"] / fl["ndf_1"],
                    100 * fl["excursion"])
        # cross-checks (see module docstring): Pearson errors on the same
        # bins, and the fine-binned Neyman fit that shows the bias
        for name, cnt_x, kw in (
                ("same bins, Pearson errors", cnt, dict(pearson=True)),
                (f"{REBIN_GEV:.0f} GeV bins, observed errors",
                 independent_bin_counts(e_sr, v_sr, w_sr, v_cr, w_cr), {}),
                (f"{FINE_GEV:.0f} GeV bins, observed errors",
                 independent_bin_counts(e_sr, v_sr, w_sr, v_cr, w_cr,
                                        width=FINE_GEV), {})):
            fx = fit_flatness(*cnt_x, lab.strip("\\") + "_x", **kw)
            logger.info("%s: cross-check, %s: constant R = %.4f +- %.4f "
                        "chi2/ndf = %.2f; slope = %+.4f +- %.4f /TeV "
                        "(%+.1f sigma) -- not used", lab, name, fx["r0"],
                        fx["e0"], fx["chi2_0"] / fx["ndf_0"], fx["slope"],
                        fx["e_slope"], fx["pull"])
        data.append((lab, color, marker, np.array(masses),
                     np.array(ratios), np.array(errs), r_glob, fl))

    # --- draw: trusted range (the analysis plot) + full extension -----------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, m_max, ylim in (("", TRUSTED_MAX, (0.30, 0.90)),
                                ("_full", 6000, (0.20, 1.10))):
        hep.style.use("CMS")
        fig, ax = plt.subplots()
        for lab, color, marker, masses, ratios, errs, r_glob, flat in data:
            sel = masses <= m_max
            ax.axhline(r_glob, color=color, lw=1.6, ls="--")
            ax.errorbar(masses[sel], ratios[sel], yerr=errs[sel], fmt=marker,
                        color=color, ms=7, capsize=4, lw=1.6,
                        label=fr"${lab}$: per-window, "
                              fr"global $R = {r_glob:.3f}$")
        # the fits live on transfer_ratio_flatness, with the independent
        # points they were made from; here only the windows and the
        # global R the model uses
        ax.set_xlabel(r"$m_{W_R}$ [GeV]")
        ax.set_ylabel(
            r"$R = N^{\mathrm{SR}}_{t\bar{t}} / N^{\mathrm{CR}}_{t\bar{t}}$")
        ax.set_ylim(*ylim)
        ax.text(0.03, 0.97,
                "$t\\bar{t}$+tW MC, resolved\n" f"{ERA}\n"
                "SR: $ee$ or $\\mu\\mu$ / CR: $e\\mu$",
                transform=ax.transAxes, va="top", fontsize=14)
        ax.legend(fontsize=12, loc="upper right")
        hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                      lumi=f"{_LUMI:.1f}", com=_COM, fontsize=13)
        out = args.output_dir / f"transfer_ratio_vs_mass{suffix}"
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("wrote %s", out)

    # --- third plot: constant chi2 fit to the full-range window points.
    # CAVEAT: adjacent windows overlap (shared events), so the points are
    # correlated and the chi2 values are qualitative.
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ytxt = 0.47
    for lab, color, marker, masses, ratios, errs, r_glob, _flat in data:
        g = ROOT.TGraphErrors(len(masses))
        for i, (m, r, e) in enumerate(zip(masses, ratios, errs)):
            g.SetPoint(i, float(m), float(r))
            g.SetPointError(i, 0.0, float(e))
        f0 = ROOT.TF1(f"f0_{lab}", "pol0", 1300.0, 6100.0)
        g.Fit(f0, "QN")
        logger.info("%s window-point constant fit: %.4f +- %.4f "
                    "(chi2/ndf %.1f/%d)", lab,
                    f0.GetParameter(0), f0.GetParError(0),
                    f0.GetChisquare(), f0.GetNDF())

        ax.errorbar(masses, ratios, yerr=errs, fmt=marker, color=color,
                    ms=7, capsize=4, lw=1.6, label=fr"${lab}$ per-window")
        xs = np.linspace(float(masses.min()), float(masses.max()), 200)
        ax.plot(xs, np.full_like(xs, f0.GetParameter(0)),
                color=color, ls="-", lw=1.8)
        ax.text(0.22, ytxt,
                fr"${lab}$: $R = {f0.GetParameter(0):.3f} \pm "
                fr"{f0.GetParError(0):.3f}$, $\chi^2$/ndf = "
                fr"{f0.GetChisquare():.0f}/{f0.GetNDF()}"
                fr"$\;=\;{f0.GetChisquare() / f0.GetNDF():.2f}$",
                transform=ax.transAxes, va="top", fontsize=13,
                color=color)
        ytxt -= 0.08
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(
        r"$R = N^{\mathrm{SR}}_{t\bar{t}} / N^{\mathrm{CR}}_{t\bar{t}}$")
    ax.set_ylim(0.20, 1.10)
    ax.text(0.03, 0.97,
            "$t\\bar{t}$+tW MC, resolved\n" f"{ERA}\n"
            "constant fit to the window points",
            transform=ax.transAxes, va="top", fontsize=14)
    ax.legend(fontsize=13, loc="upper right")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{_LUMI:.1f}", com=_COM, fontsize=13)
    out = args.output_dir / "transfer_ratio_vs_mass_fits"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)

    # --- fourth plot: the flatness verdict itself -- independent-bin
    # ratios (no shared events) with the constant and linear fits drawn.
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    verdict = []
    for lab, color, marker, masses, ratios, errs, r_glob, fl in data:
        ax.errorbar(fl["centers"], fl["r_i"], yerr=fl["s_i"],
                    xerr=fl["hw"], fmt=marker, color=color, ms=5,
                    capsize=0, lw=1.2, label=fr"${lab}$: independent bins")
        xs = np.linspace(fl["lo_edge"], fl["hi_edge"], 50)
        ax.plot(xs, np.full_like(xs, fl["r0"]), color=color, lw=1.6,
                ls="--", label=fr"${lab}$: constant fit $\pm 1\sigma$")
        ax.plot(xs, fl["r_ref"] + fl["slope"] * (xs - fl["m_ref"]) / 1e3,
                color=color, lw=1.6, ls="-",
                label=fr"${lab}$: linear fit $\pm 1\sigma$")
        draw_fit_bands(ax, fl, xs, color)
        verdict.append(flatness_text(lab, fl))
    fl0 = data[0][7]
    ax.set_xlabel(r"$m_{lljj}$ [GeV]")
    ax.set_ylabel(
        r"$R = N^{\mathrm{SR}}_{t\bar{t}} / N^{\mathrm{CR}}_{t\bar{t}}$")
    ax.set_ylim(0.20, 1.25)
    ax.set_xlim(fl0["lo_edge"] - 100, max(d[7]["hi_edge"] for d in data) + 100)
    binlab = (f"{REBIN_GEV:.0f} GeV bins" if args.flat_bins == "fixed"
              else f"200 GeV bins, merged above 2.4 TeV "
                   f"($\\geq {FLAT_NEFF_MIN}$ eff. SR MC events per bin)")
    ax.text(0.03, 0.97,
            "$t\\bar{t}$+tW MC, resolved\n" f"{ERA}\n"
            "SR: $ee$ or $\\mu\\mu$ / CR: $e\\mu$\n"
            f"{binlab}\n"
            f"fit range [{fl0['lo_edge']:.0f}, {fl0['hi_edge']:.0f}] GeV; "
            "last MC in both regions: "
            + ", ".join(fr"${d[0]}$ {d[7]['mc_end'] / 1000:.1f} TeV"
                        for d in data),
            transform=ax.transAxes, va="top", fontsize=13)
    if not args.no_stats_box:
        ax.text(0.03, 0.70, "\n".join(verdict),
                transform=ax.transAxes, va="top", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray",
                          lw=0.8, alpha=0.9))
    ax.legend(fontsize=11, loc="upper right", ncol=2)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{_LUMI:.1f}", com=_COM, fontsize=13)
    out = args.output_dir / "transfer_ratio_flatness"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
