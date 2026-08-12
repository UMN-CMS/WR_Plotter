#!/usr/bin/env python3
"""Regenerate the pull histogram (100 toys) AND single-toy pull demo
for one (channel, mass, N) cell at a custom α_σ value.

Defaults reproduce the example in the chat:
  channel=ee, mass=WR4000_N2000, N=20, seed=12345, α_µ=1.0

Pass --alpha-sigma 0.10 (or 0.30) to see what changes when the σ prior
is tightened or loosened relative to the calibrated 0.25.

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
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
from plots import plot_cell_offsets_one, make_pull_demo_one

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--channel", default="ee")
    p.add_argument("--mass", default="WR4000_N2000")
    p.add_argument("--n-events", type=int, default=20)
    p.add_argument("--n-toys", type=int, default=100)
    p.add_argument("--alpha-mu", type=float, default=1.0)
    p.add_argument("--alpha-sigma", type=float, default=0.25,
                   help="Ignored if --no-sigma-prior is passed.")
    p.add_argument("--no-sigma-prior", action="store_true",
                   help="Run the fit with σ unconstrained (width_mode='free').")
    p.add_argument("--seed", type=int, default=12345,
                   help="Seed for the single-toy demo. Default 12345.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("signal_fitting/outputs/plots/alpha_comparison"))
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def windowed_moments(edges, vals, fit_lo, fit_hi):
    n_bins = len(vals)
    ed_arr = np.ascontiguousarray(edges, dtype=np.float64)
    h = ROOT.TH1D("", "", n_bins, ed_arr); h.SetDirectory(0)
    for i in range(n_bins):
        h.SetBinContent(i + 1, max(float(vals[i]), 0.0))
    h.GetXaxis().SetRangeUser(fit_lo, fit_hi)
    return float(h.GetMean()), float(h.GetStdDev())


def main():
    args = parse_args()
    setup_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    M_WR, _ = parse_masses(args.mass)
    M_WR = float(M_WR)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    region = build_region_name(args.channel, args.topology)
    mass_var = ("mass_twoobject" if args.topology == "boosted"
                else "mass_fourobject")
    hist_key = build_hist_key(region, mass_var)
    edges_n, vals_n, _ = load_and_combine_signal(input_dirs, hist_key, args.mass)

    fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
    fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR
    mu_mc, rms_mc = windowed_moments(edges_n, vals_n, fit_lo, fit_hi)
    fwhm_pred = rms_mc * FWHM_TO_GAUSS_SIGMA
    mu_prior_sigma = args.alpha_mu * rms_mc
    sigma_prior_sigma_fwhm = args.alpha_sigma * rms_mc * FWHM_TO_GAUSS_SIGMA
    width_mode = "free" if args.no_sigma_prior else "constrained"
    a_sigma_for_label = None if args.no_sigma_prior else args.alpha_sigma
    a_str = ("no_sigma_prior" if args.no_sigma_prior
             else f"alpha_sigma_{args.alpha_sigma:.2f}".replace(".", "p"))
    logger.info("Cell %s/%s  N=%d  α_µ=%.2f  width_mode=%s  "
                "MC mean=%.0f  RMS=%.0f",
                args.channel, args.mass, args.n_events,
                args.alpha_mu, width_mode, mu_mc, rms_mc)

    centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
    in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
    where = np.where(in_win)[0]
    edges_win = edges_n[where[0]:where[-1] + 2]
    vals_win = vals_n[where]

    # 1) Run n_toys toys, build a cell_rows list compatible with plot_cell_offsets_one.
    cell_rows = []
    t0 = time.time()
    for seed in range(1, args.n_toys + 1):
        events = sample_from_hist_root(edges_win, vals_win, args.n_events, seed)
        try:
            fit = run_fit(
                "gauss", width_mode, events, M_WR,
                fwhm_pred, sigma_prior_sigma_fwhm,
                fit_lo, fit_hi,
                mu_mode="constrained",
                mu_central=mu_mc, mu_sigma=mu_prior_sigma,
                suffix_extra=f"_alpha_s{seed}_n{args.n_events}",
            )
        except Exception as e:
            logger.debug("fit %d failed: %s", seed, e); continue
        cell_rows.append({
            "status":      int(fit["minuit_status"]),
            "covqual":     int(fit["covqual"]),
            "mu_fit":      float(fit["params"]["mu"]),
            "mu_err":      float(fit["errors"]["mu"]),
            "mu_truth":    float(mu_mc),
            "width_fit":   float(fit["params"]["sigma"]),
            "width_err":   float(fit["errors"]["sigma"]),
            "width_truth": float(rms_mc),
        })
    logger.info("Ran %d toys in %.1fs", len(cell_rows), time.time() - t0)

    # 2) Pull histogram (mu and width).
    for param in ("mu", "width"):
        out = (args.output_dir
               / f"{args.mass.lower()}_{args.channel}_{param}_n{args.n_events}_{a_str}.png")
        ok = plot_cell_offsets_one(
            args.channel, args.mass, args.n_events, param, cell_rows,
            config="both", era=args.era, out_path=out,
            alpha_mu=args.alpha_mu, alpha_sigma=a_sigma_for_label,
        )
        if ok:
            logger.info("Wrote pull histogram: %s", out)

    # 3) Single-toy pull demo at the requested seed.
    for param in ("mu", "width"):
        out = (args.output_dir / f"{args.mass.lower()}_{args.channel}_{param}_demo"
                                  f"_n{args.n_events}_seed{args.seed}_{a_str}.png")
        ok = make_pull_demo_one(
            args.channel, args.mass, args.n_events, param,
            args.alpha_mu, args.alpha_sigma, out.parent,
            args.era, args.dir, args.topology,
            seed=args.seed, use_moments=True, out_path=out,
            width_mode=width_mode,
        )
        if ok:
            logger.info("Wrote pull demo:      %s", out)


if __name__ == "__main__":
    main()
