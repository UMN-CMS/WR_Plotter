#!/usr/bin/env python3
"""Scan prior widths for the single-Gaussian signal fit.

The new prior-width function:
    sigma_mu_prior    = alpha_mu    * FWHM_boot   (mass units, applied to µ)
    sigma_sigma_prior = alpha_sigma * FWHM_boot   (FWHM units, converted to σ
                                                   inside fit_single_gaussian)

Centrals are the existing bootstrap values:
    mu_prior_central    = mu_boot
    sigma_prior_central = FWHM_boot / 2.3548   (FWHM_TO_GAUSS_SIGMA)

Both µ and σ priors are always on. Truth for pulls:
    mu_truth    = mu_boot
    sigma_truth = FWHM_boot / 2.3548

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Example:
  python scan_priors.py --alpha-mu 0.10,0.20,0.50,1.0 --alpha-sigma 0.07 \
      --n-events 5,20,100 --n-toys 50 \
      --output outputs/scan_alpha_mu.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

try:
    import ROOT
except ImportError:
    sys.exit("ERROR: PyROOT unavailable. Source LCG_106 first.")
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

from wrplotter.cli_utils import setup_logging
from wrplotter.paths import input_dirs_for_era, repo_root
from measure_fwhm import (
    ONSHELL_WINDOW_LO_FRAC, ONSHELL_WINDOW_HI_FRAC,
    build_hist_key, build_region_name,
    load_and_combine_signal, parse_masses, rebin_histogram,
)
from fit_signal_toy import (
    FWHM_TO_GAUSS_SIGMA,
    bootstrap_fwhm_estimate, bootstrap_peak_estimate,
    run_fit, sample_from_hist_root,
)

logger = logging.getLogger(__name__)

# 8 bulk cells per channel (M_WR ∈ {3..6} TeV × M_N/M_WR ∈ {≈0.3, ≈0.5}).
BULK_MASSES = [
    "WR3000_N1000", "WR3000_N1400",
    "WR4000_N1200", "WR4000_N2000",
    "WR5000_N1400", "WR5000_N2400",
    "WR6000_N1800", "WR6000_N3000",
]

CSV_FIELDS = [
    "channel", "mass", "n_events", "alpha_mu", "alpha_sigma", "seed",
    "status", "covqual",
    "mu_truth", "mu_fit", "mu_err",
    "sigma_truth", "sigma_fit", "sigma_err",
    "n_sig_fit", "n_sig_err", "min_nll",
    "mu_boot", "fwhm_boot",
    "mu_prior_sigma", "sigma_prior_sigma",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--channels", nargs="+", default=["ee", "mumu"])
    p.add_argument("--masses", nargs="+", default=BULK_MASSES,
                   help="Mass tags to run. Default: 8 bulk cells.")
    p.add_argument("--alpha-mu", default="0.10,0.20,0.50,1.0",
                   help="Comma-separated alpha_mu values. "
                        "Prior sigma on mu = alpha_mu * FWHM_boot.")
    p.add_argument("--alpha-sigma", default="0.07",
                   help="Comma-separated alpha_sigma values. "
                        "Prior sigma on sigma = alpha_sigma * FWHM_boot / 2.3548.")
    p.add_argument("--n-events", default="5,20,100",
                   help="Comma-separated N values.")
    p.add_argument("--n-toys", type=int, default=50,
                   help="Toys per (cell, N, alpha pair). Default: 50.")
    p.add_argument("--output", required=True, type=Path,
                   help="Output CSV path. Appends if file exists.")
    p.add_argument("--use-moments", action="store_true",
                   help="Use windowed (mean, RMS) for truth + prior centrals.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    alpha_mus = [float(x) for x in args.alpha_mu.split(",")]
    alpha_sigmas = [float(x) for x in args.alpha_sigma.split(",")]
    n_events_list = [int(x) for x in args.n_events.split(",")]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)

    n_per_cell_per_n = args.n_toys * len(alpha_mus) * len(alpha_sigmas)
    n_per_cell = n_per_cell_per_n * len(n_events_list)
    n_total = n_per_cell * len(args.masses) * len(args.channels)
    logger.info(
        "Plan: %d channels × %d masses × %d N × %d toys × %d alpha_mu × %d alpha_sigma = %d fits",
        len(args.channels), len(args.masses), len(n_events_list), args.n_toys,
        len(alpha_mus), len(alpha_sigmas), n_total,
    )

    write_header = not args.output.exists() or args.output.stat().st_size == 0
    out_fh = open(args.output, "a", buffering=1)
    csv_writer = csv.DictWriter(out_fh, fieldnames=CSV_FIELDS)
    if write_header:
        csv_writer.writeheader()

    n_done = 0
    t0 = time.time()

    for channel in args.channels:
        region = build_region_name(channel, args.topology)
        mass_var = "mass_twoobject" if args.topology == "boosted" else "mass_fourobject"
        hist_key = build_hist_key(region, mass_var)

        for sig_tag in args.masses:
            try:
                M_WR, M_N = parse_masses(sig_tag)
                M_WR, M_N = float(M_WR), float(M_N)
            except Exception:
                logger.warning("Bad mass tag %r, skipping", sig_tag)
                continue

            try:
                edges_n, vals_n, var_n = load_and_combine_signal(
                    input_dirs, hist_key, sig_tag,
                )
            except Exception as e:
                logger.warning("Failed to load %s/%s: %s", channel, sig_tag, e)
                continue
            edges, vals, _ = rebin_histogram(edges_n, vals_n, var_n, 6)

            fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
            fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR
            centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
            in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
            where = np.where(in_win)[0]
            edges_win = edges_n[where[0]:where[-1] + 2]
            vals_win = vals_n[where]

            if args.use_moments:
                # Windowed mean & RMS via ROOT TH1::GetMean / GetStdDev.
                n_b = len(vals_n)
                ed_arr = np.ascontiguousarray(edges_n, dtype=np.float64)
                h = ROOT.TH1D("", "", n_b, ed_arr)
                h.SetDirectory(0)
                for i in range(n_b):
                    h.SetBinContent(i + 1, max(float(vals_n[i]), 0.0))
                h.GetXaxis().SetRangeUser(fit_lo, fit_hi)
                mu_boot = float(h.GetMean())
                sigma_truth = float(h.GetStdDev())
                # Prior-width yardstick is RMS itself, not FWHM. run_fit takes
                # (fwhm_pred, fwhm_err) and internally divides by 2.355 → so we
                # pass RMS × 2.355 to make the internal σ-prior central = RMS.
                fwhm_boot = sigma_truth * FWHM_TO_GAUSS_SIGMA
                logger.info(
                    "[%s] %s  mean=%.0f  RMS=%.0f  [moment truth (ROOT) + α × RMS priors]",
                    channel, sig_tag, mu_boot, sigma_truth,
                )
            else:
                mu_boot, _ = bootstrap_peak_estimate(
                    edges, vals, sig_tag, channel, args.topology,
                    n_toys=100, seed=0,
                )
                fwhm_boot, _ = bootstrap_fwhm_estimate(
                    edges, vals, sig_tag, channel, args.topology,
                    n_toys=100, seed=0,
                )
                sigma_truth = fwhm_boot / FWHM_TO_GAUSS_SIGMA
                logger.info(
                    "[%s] %s  mu_boot=%.0f  FWHM_boot=%.0f  sigma_truth=%.0f",
                    channel, sig_tag, mu_boot, fwhm_boot, sigma_truth,
                )

            for n_events in n_events_list:
                for seed in range(1, args.n_toys + 1):
                    events = sample_from_hist_root(edges_win, vals_win,
                                                    n_events, seed)
                    for alpha_mu in alpha_mus:
                        # Prior-width convention:
                        #   --use-moments: σ_µ_prior = α_µ × RMS  (RMS-based)
                        #   else:          σ_µ_prior = α_µ × FWHM_boot  (legacy)
                        if args.use_moments:
                            mu_prior_sigma = alpha_mu * sigma_truth
                        else:
                            mu_prior_sigma = alpha_mu * fwhm_boot
                        for alpha_sigma in alpha_sigmas:
                            # σ_σ_prior = α_σ × σ_truth (= α_σ × RMS) in both modes.
                            # run_fit internally divides fwhm_err by 2.355 to get the
                            # σ-prior σ, so we pass α_σ × σ_truth × 2.355 to make the
                            # post-division value α_σ × σ_truth.
                            sigma_prior_sigma_fwhm = alpha_sigma * sigma_truth * FWHM_TO_GAUSS_SIGMA
                            try:
                                fit = run_fit(
                                    "gauss", "constrained", events, M_WR,
                                    fwhm_boot, sigma_prior_sigma_fwhm,
                                    fit_lo, fit_hi,
                                    mu_mode="constrained",
                                    mu_central=mu_boot, mu_sigma=mu_prior_sigma,
                                    suffix_extra=(f"_s{seed}_n{n_events}"
                                                  f"_am{alpha_mu:.3f}"
                                                  f"_as{alpha_sigma:.3f}"),
                                )
                            except Exception as e:
                                logger.debug(
                                    "fit exception %s/%s s=%d n=%d am=%.3f as=%.3f: %s",
                                    channel, sig_tag, seed, n_events,
                                    alpha_mu, alpha_sigma, e,
                                )
                                n_done += 1
                                continue

                            csv_writer.writerow({
                                "channel": channel,
                                "mass": sig_tag,
                                "n_events": n_events,
                                "alpha_mu": alpha_mu,
                                "alpha_sigma": alpha_sigma,
                                "seed": seed,
                                "status": int(fit["minuit_status"]),
                                "covqual": int(fit["covqual"]),
                                "mu_truth": mu_boot,
                                "mu_fit": fit["params"]["mu"],
                                "mu_err": fit["errors"]["mu"],
                                "sigma_truth": sigma_truth,
                                "sigma_fit": fit["params"]["sigma"],
                                "sigma_err": fit["errors"]["sigma"],
                                "n_sig_fit": fit["params"]["n_sig"],
                                "n_sig_err": fit["errors"]["n_sig"],
                                "min_nll": fit["min_nll"],
                                "mu_boot": mu_boot,
                                "fwhm_boot": fwhm_boot,
                                "mu_prior_sigma": mu_prior_sigma,
                                "sigma_prior_sigma": (sigma_prior_sigma_fwhm
                                                       / FWHM_TO_GAUSS_SIGMA),
                            })
                            n_done += 1

            dt = time.time() - t0
            rate = n_done / dt if dt > 0 else 0.0
            logger.info(
                "  %s/%s done  rows=%d  rate=%.1f fits/s  eta=%.0fs",
                channel, sig_tag, n_done, rate,
                (n_total - n_done) / rate if rate > 0 else 0.0,
            )

    out_fh.close()
    logger.info("Done. %d rows written to %s in %.1f min",
                n_done, args.output, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
