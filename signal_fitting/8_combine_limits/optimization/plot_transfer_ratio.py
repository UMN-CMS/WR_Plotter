#!/usr/bin/env python3
"""Transfer factor R = N_SR/N_CR for tt+tW MC: how flat is it vs mass?

Justifies the fcr_shape global-R choice with actual MC statistics:

  - per-window R (the k5 card windows) with errors propagated from the
    WRAnalyzer sumw2 variances,
  - a weighted constant fit to the INDEPENDENT per-bin ratios
    SR_i/CR_i (coarse bins, so adjacent points share no events, unlike
    the overlapping windows), with its chi2/ndf as the flatness verdict.

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

ERA, BKG_DIR = "RunIISummer20UL18", "20260714_run2_bkgs"
FCR_REGION = "wr_resolved_flavor_cr"
# (label, SR region, color, marker)
CHANNELS = [("ee", "wr_ee_resolved_sr", "#e42536", "o"),
            (r"\mu\mu", "wr_mumu_resolved_sr", "#5790fc", "s")]
REBIN_GEV = 100.0            # per-bin ratio granularity (independent bins)
TRUSTED_MAX = 3200           # reliability edge of the ee analysis


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


def fit_constant_ratio(edges, v_sr, w_sr, v_cr, w_cr, rebin_gev=REBIN_GEV):
    """Weighted constant fit to the independent per-bin ratios SR_i/CR_i
    (coarse bins of `rebin_gev`). Returns (r_fit, err, chi2, ndf)."""
    f = max(1, round(rebin_gev / float(edges[1] - edges[0])))
    n = (len(v_sr) // f) * f
    S = v_sr[:n].reshape(-1, f).sum(axis=1)
    S2 = w_sr[:n].reshape(-1, f).sum(axis=1)
    C = v_cr[:n].reshape(-1, f).sum(axis=1)
    C2 = w_cr[:n].reshape(-1, f).sum(axis=1)
    good = (S > 0) & (C > 0)
    r_i = S[good] / C[good]
    s_i = r_i * np.sqrt(S2[good] / S[good]**2 + C2[good] / C[good]**2)
    wgt = 1.0 / s_i**2
    r_fit = float((r_i * wgt).sum() / wgt.sum())
    err = float(1.0 / np.sqrt(wgt.sum()))
    chi2 = float((((r_i - r_fit) / s_i)**2).sum())
    return r_fit, err, chi2, len(r_i) - 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=HERE / "explain_fcr")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    bkg_dirs, _ = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    e_cr, v_cr, w_cr = load_tt(bkg_dirs, FCR_REGION)
    with open(HERE / "inputs" / "ee_resolved_fcr.json") as fh:
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

        # global (all events summed) -- the model's R; log the
        # independent-bin constant fit as a cross-check only
        r_glob = float(v_sr.sum() / v_cr.sum())
        r_fit, s_fit, chi2, ndf = fit_constant_ratio(e_sr, v_sr, w_sr,
                                                     v_cr, w_cr)
        logger.info("%s: global R = %.4f; cross-check constant fit: "
                    "R = %.4f +- %.4f, chi2/ndf = %.1f/%d = %.2f",
                    lab, r_glob, r_fit, s_fit, chi2, ndf, chi2 / ndf)
        data.append((lab, color, marker, np.array(masses),
                     np.array(ratios), np.array(errs), r_glob))

    # --- draw: trusted range (the analysis plot) + full extension -----------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, m_max, ylim in (("", TRUSTED_MAX, (0.30, 0.90)),
                                ("_full", 6000, (0.20, 1.10))):
        hep.style.use("CMS")
        fig, ax = plt.subplots()
        for lab, color, marker, masses, ratios, errs, r_glob in data:
            sel = masses <= m_max
            ax.axhline(r_glob, color=color, lw=1.6, ls="--")
            ax.errorbar(masses[sel], ratios[sel], yerr=errs[sel], fmt=marker,
                        color=color, ms=7, capsize=4, lw=1.6,
                        label=fr"${lab}$: per-window, "
                              fr"global $R = {r_glob:.3f}$")
        ax.set_xlabel(r"$m_{W_R}$ [GeV]")
        ax.set_ylabel(
            r"$R = N^{\mathrm{SR}}_{t\bar{t}} / N^{\mathrm{CR}}_{t\bar{t}}$")
        ax.set_ylim(*ylim)
        ax.text(0.03, 0.97,
                "$t\\bar{t}$+tW MC, resolved\n" f"{ERA}\n"
                "SR: $ee$ or $\\mu\\mu$ / CR: $e\\mu$",
                transform=ax.transAxes, va="top", fontsize=14)
        ax.legend(fontsize=13, loc="upper right")
        hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                      lumi="59.8", com=13, fontsize=13)
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
    for lab, color, marker, masses, ratios, errs, r_glob in data:
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
                  lumi="59.8", com=13, fontsize=13)
    out = args.output_dir / "transfer_ratio_vs_mass_fits"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
