#!/usr/bin/env python3
"""Quick scan: rerun bifur/both at one cell with the width-prior central +
width replaced by (bootstrap-mean MC FWHM, alpha * bootstrap_mean).

Compares:
  (a) current production: FWHM_pred (from a_linear parameterization) ± parametric error
  (b) proposed: FWHM_boot (bootstrap mean of MC peak FWHM) ± alpha * FWHM_boot

Both are converted to bifur Sigma via the / FWHM_TO_BIFUR_SIGMA factor.
The mu prior is unchanged (already calibrated at MU_PRIOR_ALPHA = 0.24 * FWHM_pred).

Usage:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python signal_fitting/scan_loose_width_prior.py
  python signal_fitting/scan_loose_width_prior.py --width-prior-alpha 0.10
"""
from __future__ import annotations

import argparse
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
    compute_shape_params, load_and_combine_signal, rebin_histogram,
)
from fit_signal_toy import (
    FWHM_TO_BIFUR_SIGMA,
    MU_PRIOR_ALPHA,
    bootstrap_fwhm_estimate,
    bootstrap_peak_estimate,
    fit_bifurcated_gaussian,
    predict_fwhm,
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
    p.add_argument("--n-toys", type=int, default=100)
    p.add_argument("--width-prior-alpha", type=float, default=0.10,
                   help="sigma_FWHM_prior = alpha * FWHM_boot. Default 0.10.")
    p.add_argument("--topology", default="resolved")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def summarize(label, pulls, n_attempted):
    arr = np.asarray(pulls)
    if not len(arr):
        print(f"  {label:55s}  no converged toys")
        return
    p16, p84 = np.percentile(arr, [16, 84])
    print(f"  {label:55s}  conv={len(arr):>3d}/{n_attempted:<3d}  "
          f"median={np.median(arr):+.3f}  half-68%={0.5*(p84-p16):.3f}")


def run_one(events, M_WR, fit_lo, fit_hi, mu_central, mu_sigma,
            fwhm_central, fwhm_unc, suffix):
    """Run a bifur/both fit with explicit (FWHM_central, FWHM_unc) plumbed in.

    fit_bifurcated_gaussian's signature already takes (fwhm_pred, fwhm_err)
    in this role — we just pass whatever values we want.
    """
    return fit_bifurcated_gaussian(
        events, M_WR, fwhm_central, fwhm_unc, fit_lo, fit_hi,
        width_mode="constrained",
        mu_mode="constrained",
        mu_central=mu_central, mu_sigma=mu_sigma,
        suffix_extra=suffix,
    )


def main():
    args = parse_args()
    setup_logging(args.verbose)

    M_WR = float(args.wr); M_N = float(args.n)
    sig_tag = f"WR{args.wr}_N{args.n}"
    x = M_N / M_WR

    # MC histogram
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    hist_key = build_hist_key(build_region_name(args.channel, args.topology),
                              "mass_fourobject")
    edges_n, vals_n, var_n = load_and_combine_signal(input_dirs, hist_key, sig_tag)
    edges, vals, _ = rebin_histogram(edges_n, vals_n, var_n, 6)

    # Current production prior: parameterization
    rj = repo_root() / "signal_fitting/outputs" / args.era / "fwhm/fits/results.json"
    with open(rj) as f:
        results = json.load(f)
    fwhm_pred, fwhm_err_param = predict_fwhm(
        results[args.channel]["models"]["a_linear"], x, M_WR,
    )

    # Proposed prior: bootstrap mean of per-point MC FWHM
    fwhm_boot, fwhm_boot_unc = bootstrap_fwhm_estimate(
        edges, vals, sig_tag, args.channel, args.topology,
        n_toys=100, seed=12345,
    )
    # Prior σ scales against the parameterization FWHM (matching the µ-prior
    # σ convention), even though the prior central is the bootstrap mean.
    fwhm_alpha_unc = args.width_prior_alpha * fwhm_pred

    fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
    fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR

    # Truth (matches pull-study definition)
    truth = compute_truth(
        edges_n, vals_n, edges, vals, sig_tag, args.channel, args.topology,
        M_WR, fwhm_pred, fwhm_err_param, fit_lo, fit_hi,
    )
    Sigma_truth = truth["Sigma_truth"]
    mu_boot = truth["mu_truth"]
    mu_prior_sigma = MU_PRIOR_ALPHA[args.channel] * fwhm_pred

    print(f"\nCell: {sig_tag} {args.channel}, N={args.n_events}, {args.n_toys} toys")
    print(f"  Sigma_truth (from MC FWHM)         = {Sigma_truth:.1f} GeV")
    print(f"  ---")
    print(f"  (a) FWHM_pred  (parameterization)  = {fwhm_pred:.1f} ± {fwhm_err_param:.1f} GeV")
    print(f"      Sigma_pred (a)                  = {fwhm_pred/FWHM_TO_BIFUR_SIGMA:.1f} ± {fwhm_err_param/FWHM_TO_BIFUR_SIGMA:.1f} GeV")
    print(f"  (b) FWHM_boot  (bootstrap mean)    = {fwhm_boot:.1f} ± {fwhm_boot_unc:.1f} GeV   "
          f"(propose central=FWHM_boot, σ=alpha*FWHM_param={fwhm_alpha_unc:.1f} GeV)")
    print(f"      Sigma_pred (b)                  = {fwhm_boot/FWHM_TO_BIFUR_SIGMA:.1f} ± {fwhm_alpha_unc/FWHM_TO_BIFUR_SIGMA:.1f} GeV")
    print()

    # Sample toys
    centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
    in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
    where = np.where(in_win)[0]
    edges_win = edges_n[where[0]:where[-1] + 2]
    vals_win = vals_n[where]

    pulls_a, pulls_b = [], []
    n_a_att = n_b_att = 0
    for seed in range(1, args.n_toys + 1):
        rng = np.random.default_rng(seed)
        events = sample_from_hist(edges_win, vals_win, args.n_events, rng)
        for label, fwhm_central, fwhm_unc, pulls_list in (
            ("a", fwhm_pred, fwhm_err_param, pulls_a),
            ("b", fwhm_boot, fwhm_alpha_unc, pulls_b),
        ):
            if label == "a": n_a_att += 1
            else:            n_b_att += 1
            try:
                fit = run_one(events, M_WR, fit_lo, fit_hi,
                              mu_boot, mu_prior_sigma,
                              fwhm_central, fwhm_unc, f"_{label}_s{seed}")
            except Exception:
                continue
            if fit["minuit_status"] != 0 or fit["covqual"] < 3:
                continue
            S_fit = fit["params"]["Sigma"]; S_err = fit["errors"]["Sigma"]
            if S_err <= 0 or not np.isfinite(S_err): continue
            pulls_list.append((S_fit - Sigma_truth) / S_err)

    print(f"{'config':57s}  {'convergence':>11s}  {'median pull':>12s}  {'half-68%':>10s}")
    summarize(f"(a) FWHM_pred ± parametric ({fwhm_err_param:.0f} GeV)",
              pulls_a, n_a_att)
    summarize(f"(b) FWHM_boot ± {args.width_prior_alpha}*FWHM_param ({fwhm_alpha_unc:.0f} GeV)",
              pulls_b, n_b_att)


if __name__ == "__main__":
    main()
