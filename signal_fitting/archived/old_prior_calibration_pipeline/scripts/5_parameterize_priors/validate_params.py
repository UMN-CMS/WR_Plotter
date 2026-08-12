#!/usr/bin/env python3
"""Validate the truth parameterization by running toys with parameterized priors.

For each test cell:
  - Load the MC histogram and compute MC (mean, RMS) — the ground truth.
  - Compute parameterized (mean_pred, RMS_pred) from truth_params.json.
  - Sample N events from the MC and run the gauss fit with priors centered
    at (mean_pred, RMS_pred), widths (α_µ × RMS_pred, α_σ × RMS_pred).
  - Pulls are measured against MC truth: (θ_fit − θ_MC) / θ_err.

The test: are the per-cell pull biases close to 0 and spreads close to 1,
the same as in scan_full.py (which uses MC truth as the prior central)?

If yes → the parameterization is production-ready for arbitrary off-grid
mass points.

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wrplotter.cli_utils import setup_logging
from wrplotter.paths import input_dirs_for_era, repo_root
from measure_fwhm import (
    ONSHELL_WINDOW_LO_FRAC, ONSHELL_WINDOW_HI_FRAC,
    build_hist_key, build_region_name,
    load_and_combine_signal, parse_masses,
)
from fit_signal_toy import FWHM_TO_GAUSS_SIGMA, run_fit, sample_from_hist_root
from fit_truth import load_params, predict_priors

logger = logging.getLogger(__name__)

# Use the same 15 curated cells as the 1d_pulls plots — these span the grid.
CURATED_MASSES = [
    "WR2000_N600",  "WR2000_N1000", "WR2000_N1600",
    "WR3000_N1000", "WR3000_N1600", "WR3000_N2400",
    "WR4000_N1200", "WR4000_N2000", "WR4000_N3200",
    "WR5000_N1400", "WR5000_N2400", "WR5000_N4000",
    "WR6000_N1800", "WR6000_N3000", "WR6000_N4800",
]

CSV_FIELDS = [
    "channel", "mass", "M_WR", "M_N", "n_events", "seed",
    "status", "covqual",
    "mu_mc", "rms_mc",                      # MC ground truth
    "mu_pred", "rms_pred",                  # parameterization prediction
    "mu_fit", "mu_err",
    "sigma_fit", "sigma_err",
    "pull_mu", "pull_sigma",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--channels", nargs="+", default=["ee", "mumu"])
    p.add_argument("--masses", nargs="+", default=CURATED_MASSES)
    p.add_argument("--params", type=Path,
                   default=Path("signal_fitting/outputs/truth_params.root"))
    p.add_argument("--alpha-mu", type=float, default=1.0)
    p.add_argument("--alpha-sigma", type=float, default=0.20)
    p.add_argument("--n-events", default="5,10,20,50,100",
                   help="Comma-separated N values.")
    p.add_argument("--n-toys", type=int, default=100)
    p.add_argument("--output", type=Path,
                   default=Path("signal_fitting/outputs/validate_params.csv"))
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def windowed_moments(edges, vals, fit_lo, fit_hi):
    """ROOT TH1::GetMean / GetStdDev within [fit_lo, fit_hi]."""
    n_bins = len(vals)
    ed_arr = np.ascontiguousarray(edges, dtype=np.float64)
    h = ROOT.TH1D("", "", n_bins, ed_arr)
    h.SetDirectory(0)
    for i in range(n_bins):
        h.SetBinContent(i + 1, max(float(vals[i]), 0.0))
    h.GetXaxis().SetRangeUser(fit_lo, fit_hi)
    return float(h.GetMean()), float(h.GetStdDev())


def main():
    args = parse_args()
    setup_logging(args.verbose)
    params = load_params(args.params)
    n_events_list = [int(x) for x in args.n_events.split(",")]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    out_fh = open(args.output, "w", buffering=1)
    csv_writer = csv.DictWriter(out_fh, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()

    n_done = 0
    t0 = time.time()
    for channel in args.channels:
        region = build_region_name(channel, args.topology)
        mass_var = ("mass_twoobject" if args.topology == "boosted"
                    else "mass_fourobject")
        hist_key = build_hist_key(region, mass_var)
        for mass in args.masses:
            M_WR, M_N = parse_masses(mass)
            M_WR, M_N = float(M_WR), float(M_N)
            fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
            fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR
            try:
                edges_n, vals_n, _ = load_and_combine_signal(input_dirs, hist_key, mass)
            except Exception as e:
                logger.warning("Load fail %s/%s: %s", channel, mass, e); continue
            mu_mc, rms_mc = windowed_moments(edges_n, vals_n, fit_lo, fit_hi)
            mu_pred, rms_pred = predict_priors(channel, M_WR, M_N, params)
            logger.info("[%s] %s  MC=(%.0f, %.0f)  PRED=(%.0f, %.0f)  "
                        "Δmean=%+.1f  Δrms=%+.1f",
                        channel, mass, mu_mc, rms_mc, mu_pred, rms_pred,
                        mu_pred - mu_mc, rms_pred - rms_mc)

            centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
            in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
            where = np.where(in_win)[0]
            edges_win = edges_n[where[0]:where[-1] + 2]
            vals_win = vals_n[where]
            fwhm_pred = rms_pred * FWHM_TO_GAUSS_SIGMA
            sigma_prior_sigma_fwhm = (args.alpha_sigma * rms_pred
                                       * FWHM_TO_GAUSS_SIGMA)
            mu_prior_sigma = args.alpha_mu * rms_pred

            for n_events in n_events_list:
                for seed in range(1, args.n_toys + 1):
                    events = sample_from_hist_root(edges_win, vals_win,
                                                    n_events, seed)
                    try:
                        fit = run_fit(
                            "gauss", "constrained", events, M_WR,
                            fwhm_pred, sigma_prior_sigma_fwhm,
                            fit_lo, fit_hi,
                            mu_mode="constrained",
                            mu_central=mu_pred, mu_sigma=mu_prior_sigma,
                            suffix_extra=f"_v_s{seed}_n{n_events}",
                        )
                    except Exception as e:
                        logger.debug("fit exception: %s", e); n_done += 1; continue
                    mu_fit = float(fit["params"]["mu"])
                    mu_err = float(fit["errors"]["mu"])
                    sigma_fit = float(fit["params"]["sigma"])
                    sigma_err = float(fit["errors"]["sigma"])
                    pull_mu = ((mu_fit - mu_mc) / mu_err
                               if mu_err > 0 and np.isfinite(mu_err) else np.nan)
                    pull_sigma = ((sigma_fit - rms_mc) / sigma_err
                                  if sigma_err > 0 and np.isfinite(sigma_err)
                                  else np.nan)
                    csv_writer.writerow({
                        "channel": channel, "mass": mass,
                        "M_WR": M_WR, "M_N": M_N,
                        "n_events": n_events, "seed": seed,
                        "status": int(fit["minuit_status"]),
                        "covqual": int(fit["covqual"]),
                        "mu_mc": mu_mc, "rms_mc": rms_mc,
                        "mu_pred": mu_pred, "rms_pred": rms_pred,
                        "mu_fit": mu_fit, "mu_err": mu_err,
                        "sigma_fit": sigma_fit, "sigma_err": sigma_err,
                        "pull_mu": pull_mu, "pull_sigma": pull_sigma,
                    })
                    n_done += 1

    out_fh.close()
    logger.info("Done: %d rows in %.1f min -> %s",
                n_done, (time.time() - t0) / 60.0, args.output)


if __name__ == "__main__":
    main()
