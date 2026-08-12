#!/usr/bin/env python3
"""Pedagogy plot: one toy fit + parameter-pull visualization for mu.

Top panel: the fitted PDF on the toy data, with truth mu and fit mu drawn as
vertical lines so their distance is visible directly in GeV.

Bottom panel: the fit's claimed Gaussian posterior on mu (centered at mu_fit,
width = the reported error sigma_mu). A dashed line marks truth. The parameter
pull is *literally* how many widths of this Gaussian sit between its centre and
the truth line. Tick marks at multiples of sigma_mu spell that out.

The aggregate pull plots in the rest of the pipeline (pull_bias_*,
pull_spread_*) summarise per-toy values of this same number across many toys
and many masses; this script renders one toy so the abstraction is concrete.

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Usage:
  python signal_fitting/pull_demo.py --wr 4000 --n 2000 --channel ee
  python signal_fitting/pull_demo.py --wr 4000 --n 2000 --n-events 20 --seed 42
"""
from __future__ import annotations

import argparse
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
    ONSHELL_WINDOW_LO_FRAC,
    ONSHELL_WINDOW_HI_FRAC,
    build_hist_key,
    build_region_name,
    compute_shape_params,
    load_and_combine_signal,
    rebin_histogram,
)
from fit_signal_toy import (
    FWHM_TO_BIFUR_SIGMA,
    MU_PRIOR_ALPHA,
    WIDTH_PRIOR_ALPHA,
    bootstrap_fwhm_estimate,
    bootstrap_peak_estimate,
    evaluate_fit_curve,
    predict_fwhm,
    run_fit,
    sample_from_hist,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--channel", choices=["ee", "mumu"], default="ee")
    p.add_argument("--wr", type=int, default=4000)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--n-events", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--bin-width", type=float, default=200.0,
                   help="Plot bin width in GeV. Default 200.")
    p.add_argument("--topology", choices=["resolved", "boosted"], default="resolved")
    p.add_argument("--output-dir", default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    M_WR = float(args.wr); M_N = float(args.n)
    x = M_N / M_WR
    sig_tag = f"WR{args.wr}_N{args.n}"

    info = load_lumi(args.era)
    com = info.get("com", 13.0)

    # FWHM prior central + uncertainty from the global parameterization.
    import json
    results_json = (repo_root() / "signal_fitting" / "outputs" / args.era /
                    "fwhm" / "fits" / "results.json")
    with open(results_json) as f:
        results = json.load(f)
    fwhm_pred, fwhm_err = predict_fwhm(
        results[args.channel]["models"]["a_linear"], x, M_WR,
    )

    # Histogram + toy events
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    region = build_region_name(args.channel, args.topology)
    mass_var = "mass_twoobject" if args.topology == "boosted" else "mass_fourobject"
    hist_key = build_hist_key(region, mass_var)
    edges_n, vals_n, var_n = load_and_combine_signal(input_dirs, hist_key, sig_tag)
    edges, vals, _ = rebin_histogram(edges_n, vals_n, var_n, 6)

    fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
    fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR

    # Sample toy events from the native-binned histogram for finer resolution.
    centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
    in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
    where = np.where(in_win)[0]
    edges_win = edges_n[where[0]:where[-1] + 2]
    vals_win = vals_n[where]
    rng = np.random.default_rng(args.seed)
    events = sample_from_hist(edges_win, vals_win, args.n_events, rng)

    # mu prior central + uncertainty (bootstrap on the rebinned MC histogram).
    sp = compute_shape_params(edges, vals, sig_tag, args.channel, args.topology)
    mu_boot, sigma_peak_boot = bootstrap_peak_estimate(
        edges, vals, sig_tag, args.channel, args.topology,
        n_toys=100, seed=args.seed,
    )
    mu_truth = float(mu_boot)  # treat the bootstrap-mean peak as the truth
    logger.info("Truth mu (bootstrap mean) = %.1f, sigma_peak_boot = %.1f",
                mu_truth, sigma_peak_boot)

    # The fit: bifurcated Gaussian, both constrained — production setting.
    # µ prior:    central = mu_boot, sigma = MU_PRIOR_ALPHA[ch] * FWHM_param
    # Width prior: central = FWHM_boot, sigma = WIDTH_PRIOR_ALPHA[ch] * FWHM_boot
    fwhm_boot, _ = bootstrap_fwhm_estimate(
        edges, vals, sig_tag, args.channel, args.topology, n_toys=100, seed=0,
    )
    mu_prior_sigma     = MU_PRIOR_ALPHA[args.channel]    * fwhm_pred
    width_prior_sigma  = WIDTH_PRIOR_ALPHA[args.channel] * fwhm_pred
    fit = run_fit(
        "bifur", "constrained", events, M_WR,
        fwhm_boot, width_prior_sigma,           # width prior central + σ
        fit_lo, fit_hi,
        mu_mode="constrained", mu_central=mu_boot, mu_sigma=mu_prior_sigma,
    )
    mu_fit = fit["params"]["mu"]
    mu_err = fit["errors"]["mu"]
    mu_pull = (mu_fit - mu_truth) / mu_err
    logger.info("mu_fit = %.1f ± %.1f, truth = %.1f, pull = %.2f sigma",
                mu_fit, mu_err, mu_truth, mu_pull)

    # ---- Plot ----
    n_bins = int(round((fit_hi - fit_lo) / args.bin_width))
    plot_edges = np.linspace(fit_lo, fit_hi, n_bins + 1)
    plot_centers = 0.5 * (plot_edges[:-1] + plot_edges[1:])

    h_obs, _ = np.histogram(events, bins=plot_edges)
    err_obs = np.sqrt(np.maximum(h_obs, 1.0))

    xs_dense = np.linspace(fit_lo, fit_hi, 2000)
    pdf_dense = evaluate_fit_curve(fit, xs_dense, M_WR)
    n_sig_fit = fit["params"]["n_sig"]

    hep.style.use("CMS")
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 11),
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.30},
    )

    # ---- Top panel: data + PDF + truth/fit µ visualization ----
    # MC truth shape (rebinned to plot grid)
    mc_per_plot = np.zeros(n_bins)
    centers_in = 0.5 * (edges[:-1] + edges[1:])
    for i in range(n_bins):
        m = (centers_in >= plot_edges[i]) & (centers_in < plot_edges[i + 1])
        mc_per_plot[i] = float(vals[m].sum())
    if mc_per_plot.sum() > 0:
        ax_top.stairs(mc_per_plot / mc_per_plot.sum() * args.n_events, plot_edges,
                      color="black", alpha=0.30, linewidth=1.5,
                      label=f"MC shape (scaled to {args.n_events} events)")

    # Toy data
    ax_top.errorbar(plot_centers, h_obs, yerr=err_obs, marker="o", linestyle="",
                    color="black", markersize=7, label=f"toy ({args.n_events} events)")

    # Fitted PDF
    ax_top.plot(xs_dense, pdf_dense * n_sig_fit * args.bin_width,
                color="red", linewidth=2.2, label="Bifurcated Gaussian fit")

    # Truth and fit mu lines
    y_top = ax_top.get_ylim()[1]
    ax_top.axvline(mu_truth, color="black", linestyle="--", linewidth=1.5,
                   label=rf"truth $\mu_{{\rm truth}}={mu_truth:.0f}$ GeV")
    ax_top.axvline(mu_fit, color="red", linestyle="-", linewidth=1.5)
    ax_top.axvspan(mu_fit - mu_err, mu_fit + mu_err, color="red", alpha=0.15,
                   label=rf"fit $\mu_{{\rm fit}}={mu_fit:.0f}\pm{mu_err:.0f}$ GeV")

    ax_top.set_ylabel(f"Events / {args.bin_width:.0f} GeV", fontsize=18)
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

    # Legend top-right, pull box stacked just below it.
    ax_top.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98),
                  fontsize=11, framealpha=0.90)
    pull_color = "tab:green" if abs(mu_pull) < 1 else (
        "tab:orange" if abs(mu_pull) < 2 else "tab:red")
    ax_top.text(
        0.98, 0.55,
        "Parameter pull\n"
        rf"pull$_\mu$ = $(\mu_{{\rm fit}} - \mu_{{\rm truth}})/\sigma_\mu$"
        "\n"
        rf"        = $({mu_fit:.0f} - {mu_truth:.0f})/{mu_err:.0f}$"
        "\n"
        rf"        = $\mathbf{{{mu_pull:+.2f}\,\sigma}}$",
        transform=ax_top.transAxes, fontsize=12,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=pull_color, linewidth=2.0, alpha=0.95),
    )

    # ---- Bottom panel: parameter-pull visualization (zoomed parameter space) ----
    # The fit's claimed posterior on mu is N(mu_fit, sigma_mu). The bottom
    # panel zooms its x-axis around mu_fit so the Gaussian renders at its
    # true relative width and the +-1, +-2, +-3 sigma ticks are well-spaced
    # and readable. The pull is the distance from the centre to the truth
    # line, in units of sigma_mu.
    half_pull = max(abs(mu_pull) + 1.0, 3.5)  # always show at least +-3 sigma
    x_lo = mu_fit - half_pull * mu_err
    x_hi = mu_fit + half_pull * mu_err
    xs_g = np.linspace(x_lo, x_hi, 1000)
    gauss = np.exp(-0.5 * ((xs_g - mu_fit) / mu_err) ** 2)
    ax_bot.fill_between(xs_g, 0, gauss, color="red", alpha=0.18)
    ax_bot.plot(xs_g, gauss, color="red", linewidth=2.0,
                label=r"fit posterior on $\mu$:  $\mathcal{N}(\mu_{\rm fit}, \sigma_\mu)$")
    ax_bot.axvline(mu_truth, color="black", linestyle="--", linewidth=1.5,
                   label=r"truth $\mu_{\rm truth}$")
    ax_bot.axvline(mu_fit, color="red", linestyle="-", linewidth=1.2, alpha=0.7)

    # sigma tick marks: red vertical guides + "kσ" labels along the top of the
    # panel where they don't collide with the truth line or arrow.
    for k in range(-3, 4):
        x = mu_fit + k * mu_err
        if not (x_lo <= x <= x_hi):
            continue
        ax_bot.axvline(x, color="red", alpha=0.20, linewidth=0.8)
        ax_bot.text(x, 1.05, f"{k:+d}σ" if k != 0 else r"$\mu_{\rm fit}$",
                    color="red", fontsize=11, ha="center", va="bottom",
                    transform=ax_bot.get_xaxis_transform())

    # Arrow showing the pull distance, drawn at a y above the Gaussian peak
    arrow_y = 0.45
    ax_bot.annotate(
        "",
        xy=(mu_truth, arrow_y), xytext=(mu_fit, arrow_y),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.6),
    )
    midx = 0.5 * (mu_fit + mu_truth)
    ax_bot.text(
        midx, arrow_y + 0.03,
        rf"pull = $\mathbf{{{mu_pull:+.2f}\,\sigma}}$",
        ha="center", va="bottom", fontsize=13,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor="black", alpha=0.95),
    )

    ax_bot.set_xlabel(r"$\mu$ [GeV]", fontsize=18)
    ax_bot.set_ylabel(r"Fit posterior  (a.u.)", fontsize=14)
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
           f"pull_demo_{sig_tag}_{args.channel}_seed{args.seed}_n{args.n_events}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
