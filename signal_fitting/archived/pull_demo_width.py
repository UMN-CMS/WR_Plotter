#!/usr/bin/env python3
"""Pedagogy plot: one toy fit + parameter-pull visualization for the width Σ.

Companion to pull_demo.py (which visualizes the µ pull). Shows the same
two-panel structure but for the bifurcated Gaussian's total width Σ = σ_L + σ_R:

  Top panel:    histogram + fitted PDF + a "truth-width" PDF (same µ, same Δ,
                but with Σ replaced by Σ_truth) so the visible width comparison
                is purely about Σ.
  Bottom panel: the fit's claimed posterior on Σ (Gaussian centered at Σ_fit
                with width σ_Σ,fit). Dashed line at Σ_truth. The arrow
                visualizes pull = (Σ_fit − Σ_truth) / σ_Σ,fit — number of
                widths of the bottom Gaussian between its centre and truth.

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Usage:
  python signal_fitting/pull_demo_width.py --wr 4000 --n 2000 --channel ee
  python signal_fitting/pull_demo_width.py --wr 4000 --n 2000 --seed 28
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

from measure_fwhm import (
    ONSHELL_WINDOW_LO_FRAC, ONSHELL_WINDOW_HI_FRAC,
    build_hist_key, build_region_name,
    load_and_combine_signal, rebin_histogram,
)
from fit_signal_toy import (
    FWHM_TO_BIFUR_SIGMA,
    MU_PRIOR_ALPHA,
    WIDTH_PRIOR_ALPHA,
    bootstrap_fwhm_estimate,
    bootstrap_peak_estimate,
    predict_fwhm,
    run_fit,
    sample_from_hist,
)
from pull_study import compute_truth

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--channel", choices=["ee", "mumu"], default="ee")
    p.add_argument("--wr", type=int, default=4000)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--n-events", type=int, default=20)
    p.add_argument("--seed", type=int, default=28)
    p.add_argument("--bin-width", type=float, default=200.0)
    p.add_argument("--topology", choices=["resolved", "boosted"], default="resolved")
    p.add_argument("--output-dir", default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def bifur_pdf(xs, mu, sigmaL, sigmaR):
    """Unnormalized bifurcated Gaussian shape."""
    left = xs < mu
    y = np.empty_like(xs)
    y[left]  = np.exp(-0.5 * ((xs[left]  - mu) / sigmaL) ** 2)
    y[~left] = np.exp(-0.5 * ((xs[~left] - mu) / sigmaR) ** 2)
    return y / np.trapz(y, xs)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    M_WR = float(args.wr); M_N = float(args.n)
    sig_tag = f"WR{args.wr}_N{args.n}"

    info = load_lumi(args.era)
    com = info.get("com", 13.0)

    # FWHM prior central + uncertainty
    rj = (repo_root() / "signal_fitting" / "outputs" / args.era /
          "fwhm" / "fits" / "results.json")
    with open(rj) as f:
        results = json.load(f)
    fwhm_pred, fwhm_err = predict_fwhm(
        results[args.channel]["models"]["a_linear"], M_N / M_WR, M_WR,
    )

    # MC histogram
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    region = build_region_name(args.channel, args.topology)
    mass_var = "mass_twoobject" if args.topology == "boosted" else "mass_fourobject"
    hist_key = build_hist_key(region, mass_var)
    edges_n, vals_n, var_n = load_and_combine_signal(input_dirs, hist_key, sig_tag)
    edges, vals, _ = rebin_histogram(edges_n, vals_n, var_n, 6)

    fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
    fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR

    # Truth (matches the pull-study definition: Sigma_truth from per-point FWHM,
    # Delta_truth from a high-stats bifur fit at this cell).
    truth = compute_truth(
        edges_n, vals_n, edges, vals, sig_tag, args.channel, args.topology,
        M_WR, fwhm_pred, fwhm_err, fit_lo, fit_hi,
    )
    Sigma_truth = truth["Sigma_truth"]
    Delta_truth = truth["Delta_truth"]
    mu_truth    = truth["mu_truth"]
    logger.info("Truth: mu=%.1f, Sigma=%.1f, Delta=%.1f",
                mu_truth, Sigma_truth, Delta_truth)

    # Sample toy events
    centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
    in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
    where = np.where(in_win)[0]
    edges_win = edges_n[where[0]:where[-1] + 2]
    vals_win = vals_n[where]
    rng = np.random.default_rng(args.seed)
    events = sample_from_hist(edges_win, vals_win, args.n_events, rng)

    # Bootstrap on rebinned histogram for the µ-prior central (matches pull-study).
    mu_boot, _sigma_peak_boot = bootstrap_peak_estimate(
        edges, vals, sig_tag, args.channel, args.topology,
        n_toys=100, seed=args.seed,
    )
    fwhm_boot, _ = bootstrap_fwhm_estimate(
        edges, vals, sig_tag, args.channel, args.topology, n_toys=100, seed=0,
    )
    mu_prior_sigma     = MU_PRIOR_ALPHA[args.channel]    * fwhm_pred
    width_prior_sigma  = WIDTH_PRIOR_ALPHA[args.channel] * fwhm_pred

    # Production fit: bifur, both constrained.
    fit = run_fit(
        "bifur", "constrained", events, M_WR,
        fwhm_boot, width_prior_sigma,
        fit_lo, fit_hi,
        mu_mode="constrained", mu_central=mu_boot, mu_sigma=mu_prior_sigma,
    )
    Sigma_fit = fit["params"]["Sigma"]
    Sigma_err = fit["errors"]["Sigma"]
    Delta_fit = fit["params"]["Delta"]
    mu_fit    = fit["params"]["mu"]
    sigmaL_fit = fit["params"]["sigmaL"]
    sigmaR_fit = fit["params"]["sigmaR"]
    pull_Sigma = (Sigma_fit - Sigma_truth) / Sigma_err
    logger.info("Sigma_fit = %.1f ± %.1f, Sigma_truth = %.1f, pull = %.2f sigma",
                Sigma_fit, Sigma_err, Sigma_truth, pull_Sigma)

    # ---- Plot ----
    n_bins = int(round((fit_hi - fit_lo) / args.bin_width))
    plot_edges = np.linspace(fit_lo, fit_hi, n_bins + 1)
    plot_centers = 0.5 * (plot_edges[:-1] + plot_edges[1:])

    h_obs, _ = np.histogram(events, bins=plot_edges)
    err_obs = np.sqrt(np.maximum(h_obs, 1.0))
    n_sig_fit = fit["params"]["n_sig"]

    xs_dense = np.linspace(fit_lo, fit_hi, 2000)
    pdf_fit = bifur_pdf(xs_dense, mu_fit, sigmaL_fit, sigmaR_fit)

    # "Truth-width" PDF: same mu_fit, but Sigma replaced by Sigma_truth.
    # Use Delta_truth so the asymmetry is also matched.
    sigmaL_truth = (Sigma_truth - Delta_truth) / 2.0
    sigmaR_truth = (Sigma_truth + Delta_truth) / 2.0
    pdf_truth_w = bifur_pdf(xs_dense, mu_fit, sigmaL_truth, sigmaR_truth)

    hep.style.use("CMS")
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 11),
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.30},
    )

    # ---- Top panel: data + fit PDF + truth-width PDF for visual comparison
    ax_top.errorbar(plot_centers, h_obs, yerr=err_obs, marker="o", linestyle="",
                    color="black", markersize=7, label=f"toy ({args.n_events} events)")
    ax_top.plot(xs_dense, pdf_fit * n_sig_fit * args.bin_width,
                color="red", linewidth=2.2,
                label=rf"fit  ($\Sigma_{{\rm fit}}={Sigma_fit:.0f}\pm{Sigma_err:.0f}$ GeV)")
    ax_top.plot(xs_dense, pdf_truth_w * n_sig_fit * args.bin_width,
                color="black", linestyle="--", linewidth=2.0,
                label=rf"truth-width  ($\Sigma_{{\rm truth}}={Sigma_truth:.0f}$ GeV)")

    ax_top.set_ylabel(f"Events / {args.bin_width:.0f} GeV", fontsize=18)
    ax_top.set_xlim(fit_lo, fit_hi)
    ax_top.set_ylim(bottom=0)
    ax_top.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax_top, data=False, label="Work in Progress",
                  com=com, fontsize=18)

    ch_lab = {"ee": "ee", "mumu": r"$\mu\mu$"}[args.channel]
    ax_top.text(
        0.04, 0.96,
        f"{ch_lab}\nResolved SR\n{args.era}\n"
        rf"$M_{{W_R}}={M_WR:.0f}$ GeV, $M_N={M_N:.0f}$ GeV"
        f"\nBifurcated Gaussian / Both Constrained\nseed = {args.seed}",
        transform=ax_top.transAxes, fontsize=12, verticalalignment="top",
    )

    ax_top.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98),
                  fontsize=11, framealpha=0.90)
    pull_color = ("tab:green" if abs(pull_Sigma) < 1
                  else "tab:orange" if abs(pull_Sigma) < 2
                  else "tab:red")
    ax_top.text(
        0.98, 0.55,
        "Parameter pull\n"
        rf"pull$_\Sigma$ = $(\Sigma_{{\rm fit}} - \Sigma_{{\rm truth}})/\sigma_\Sigma$"
        "\n"
        rf"        = $({Sigma_fit:.0f} - {Sigma_truth:.0f})/{Sigma_err:.0f}$"
        "\n"
        rf"        = $\mathbf{{{pull_Sigma:+.2f}\,\sigma}}$",
        transform=ax_top.transAxes, fontsize=12,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=pull_color, linewidth=2.0, alpha=0.95),
    )

    # ---- Bottom panel: posterior on Sigma ----
    half_pull = max(abs(pull_Sigma) + 1.0, 3.5)
    x_lo = Sigma_fit - half_pull * Sigma_err
    x_hi = Sigma_fit + half_pull * Sigma_err
    xs_g = np.linspace(x_lo, x_hi, 1000)
    gauss = np.exp(-0.5 * ((xs_g - Sigma_fit) / Sigma_err) ** 2)
    ax_bot.fill_between(xs_g, 0, gauss, color="red", alpha=0.18)
    ax_bot.plot(xs_g, gauss, color="red", linewidth=2.0,
                label=r"fit posterior on $\Sigma$:  $\mathcal{N}(\Sigma_{\rm fit}, \sigma_\Sigma)$")
    ax_bot.axvline(Sigma_truth, color="black", linestyle="--", linewidth=1.5,
                   label=r"truth $\Sigma_{\rm truth}$")
    ax_bot.axvline(Sigma_fit, color="red", linestyle="-", linewidth=1.2, alpha=0.7)

    for k in range(-4, 5):
        x = Sigma_fit + k * Sigma_err
        if not (x_lo <= x <= x_hi): continue
        ax_bot.axvline(x, color="red", alpha=0.20, linewidth=0.8)
        ax_bot.text(x, 1.05, f"{k:+d}σ" if k != 0 else r"$\Sigma_{\rm fit}$",
                    color="red", fontsize=11, ha="center", va="bottom",
                    transform=ax_bot.get_xaxis_transform())

    arrow_y = 0.45
    ax_bot.annotate(
        "",
        xy=(Sigma_truth, arrow_y), xytext=(Sigma_fit, arrow_y),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.6),
    )
    midx = 0.5 * (Sigma_fit + Sigma_truth)
    ax_bot.text(
        midx, arrow_y + 0.03,
        rf"pull = $\mathbf{{{pull_Sigma:+.2f}\,\sigma}}$",
        ha="center", va="bottom", fontsize=13,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor="black", alpha=0.95),
    )

    ax_bot.set_xlabel(r"$\Sigma$ [GeV]", fontsize=18)
    ax_bot.set_ylabel("Fit posterior  (a.u.)", fontsize=14)
    ax_bot.set_xlim(x_lo, x_hi)
    ax_bot.set_ylim(0.0, 1.18)
    ax_bot.set_yticks([])
    ax_bot.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
                  fontsize=11, framealpha=0.9)

    out_dir = Path(args.output_dir or
                   repo_root() / "signal_fitting" / "outputs" / args.era /
                   "pull_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir /
           f"pull_demo_width_{sig_tag}_{args.channel}_seed{args.seed}_n{args.n_events}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
