#!/usr/bin/env python3
"""Quick scan: rerun bifur/both at one cell with sigma_mu_prior overridden.

Compares the existing CSV result (sigma_mu_prior = sigma_peak_boot ~ 38 GeV)
with a deliberately loose prior (default --mu-prior-sigma 200 GeV), keeping
mu_prior_central at mu_boot. Reports convergence, median pull, half-68% range
for both, so we can see whether loosening the mu prior really moves the spread
toward 1 without introducing bias.

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Usage:
  python signal_fitting/scan_loose_mu_prior.py
  python signal_fitting/scan_loose_mu_prior.py --mu-prior-sigma 100
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
import numpy as np

try:
    import ROOT
except ImportError:
    sys.exit("ERROR: PyROOT unavailable. Source LCG_106 first.")
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wrplotter.cli_utils import setup_logging
from wrplotter.paths import input_dirs_for_era, repo_root

from measure_fwhm import (
    ONSHELL_WINDOW_LO_FRAC, ONSHELL_WINDOW_HI_FRAC,
    build_hist_key, build_region_name,
    load_and_combine_signal, rebin_histogram,
)
from fit_signal_toy import (
    bootstrap_peak_estimate, predict_fwhm, run_fit, sample_from_hist,
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
    p.add_argument("--n-toys", type=int, default=100)
    p.add_argument("--model", choices=["gauss", "bifur"], default="bifur",
                   help="Signal PDF to fit (default bifur).")
    p.add_argument("--mu-prior-sigma", type=float, default=200.0,
                   help="Override sigma_mu_prior (GeV). Default 200.")
    p.add_argument("--mu-prior-alpha", type=float, default=None,
                   help="If set, override sigma_mu_prior = alpha * FWHM_pred for "
                        "this cell (FWHM-proportional mode). Overrides "
                        "--mu-prior-sigma when given.")
    p.add_argument("--topology", default="resolved")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def summarize(label, pulls, n_attempted):
    arr = np.asarray(pulls)
    if not len(arr):
        print(f"{label:30s}  no converged toys")
        return
    p16, p84 = np.percentile(arr, [16, 84])
    print(f"{label:30s}  conv={len(arr):>3d}/{n_attempted:<3d}  "
          f"median={np.median(arr):+.3f}  half-68%={0.5*(p84-p16):.3f}")


def main():
    args = parse_args()
    setup_logging(args.verbose)

    M_WR = float(args.wr); M_N = float(args.n)
    sig_tag = f"WR{args.wr}_N{args.n}"

    # MC histogram
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    region = build_region_name(args.channel, args.topology)
    hist_key = build_hist_key(region, "mass_fourobject")
    edges_n, vals_n, var_n = load_and_combine_signal(input_dirs, hist_key, sig_tag)
    edges, vals, _ = rebin_histogram(edges_n, vals_n, var_n, 6)

    # FWHM prior
    rj = repo_root() / "signal_fitting/outputs" / args.era / "fwhm/fits/results.json"
    with open(rj) as f:
        results = json.load(f)
    fwhm_pred, fwhm_err = predict_fwhm(
        results[args.channel]["models"]["a_linear"], M_N / M_WR, M_WR,
    )

    fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
    fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR
    centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
    in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
    where = np.where(in_win)[0]
    edges_win = edges_n[where[0]:where[-1] + 2]
    vals_win = vals_n[where]

    # truth and the production-default mu-prior width
    mu_boot, sigma_peak_boot = bootstrap_peak_estimate(
        edges, vals, sig_tag, args.channel, args.topology, n_toys=100, seed=12345,
    )
    if args.mu_prior_alpha is not None:
        loose_sigma = float(args.mu_prior_alpha) * fwhm_pred
        loose_label = (f"alpha={args.mu_prior_alpha:.3f} * FWHM_pred"
                       f" = {loose_sigma:.1f} GeV")
    else:
        loose_sigma = float(args.mu_prior_sigma)
        loose_label = f"{loose_sigma:.1f} GeV (override)"

    print(f"\nCell: {sig_tag} {args.channel}, N={args.n_events}, {args.n_toys} toys")
    print(f"  mu_boot (truth)       = {mu_boot:.1f} GeV")
    print(f"  FWHM_pred             = {fwhm_pred:.1f} GeV")
    print(f"  default sigma_mu      = {sigma_peak_boot:.1f} GeV (sigma_peak_boot)")
    print(f"  loose   sigma_mu      = {loose_label}\n")

    pulls_default = []
    pulls_loose   = []
    n_def_attempt = 0
    n_lo_attempt  = 0

    for seed in range(1, args.n_toys + 1):
        rng = np.random.default_rng(seed)
        events = sample_from_hist(edges_win, vals_win, args.n_events, rng)

        for label, sigma_mu, pulls_list in (
            ("default", sigma_peak_boot, pulls_default),
            ("loose",   loose_sigma,     pulls_loose),
        ):
            if label == "default":
                n_def_attempt += 1
            else:
                n_lo_attempt += 1
            fit = run_fit(
                args.model, "constrained", events, M_WR, fwhm_pred, fwhm_err,
                fit_lo, fit_hi,
                mu_mode="constrained", mu_central=mu_boot, mu_sigma=sigma_mu,
                suffix_extra=f"_{label}_s{seed}",
            )
            if fit["minuit_status"] != 0 or fit["covqual"] < 3:
                continue
            mu_fit = fit["params"]["mu"]; mu_err = fit["errors"]["mu"]
            if mu_err <= 0 or not np.isfinite(mu_err): continue
            pulls_list.append((mu_fit - mu_boot) / mu_err)

    print(f"{'config':30s}  {'convergence':>11s}  {'median pull':>12s}  {'half-68%':>10s}")
    summarize(f"sigma_mu={sigma_peak_boot:.0f} GeV (default)",
              pulls_default, n_def_attempt)
    summarize(f"sigma_mu={loose_sigma:.0f} GeV (loose)",
              pulls_loose,   n_lo_attempt)

    # Also report mean reported sigma_mu in the loose case so we can see if
    # the data is now driving the error rather than the prior.
    # (We didn't store it; just note that any change in spread tells us so.)


if __name__ == "__main__":
    main()
