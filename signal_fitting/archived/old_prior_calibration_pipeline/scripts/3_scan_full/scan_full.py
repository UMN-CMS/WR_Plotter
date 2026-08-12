#!/usr/bin/env python3
"""Full-grid gauss pull study with the new prior-width function.

Writes a CSV in the same schema as pull_study.py so existing plotters
(make_pull_xy, pull_cell_demo) can read it directly:

    channel,mass,n_events,model,config,seed,status,covqual,
    mu_truth,mu_fit,mu_err,width_truth,width_fit,width_err,
    delta_truth,delta_fit,delta_err,n_sig_fit,n_sig_err,min_nll

Gauss only, both µ and σ priors always on, prior widths from the new function:

    sigma_mu_prior    = alpha_mu    * FWHM_boot
    sigma_sigma_prior = alpha_sigma * FWHM_boot   (FWHM units; fit_signal_toy
                                                   divides by 2.3548 internally)

Centrals = bootstrap (mu_boot, FWHM_boot). Truth = mu_boot, FWHM_boot / 2.3548.

The CSV `model` column is "gauss"; the `config` column is set from
--config-tag (default "both") so the existing plot scripts will accept it.
delta_* columns are NaN (no Delta parameter for gauss).

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Example:
  python scan_full.py --alpha-mu 0.10 --alpha-sigma 0.05 --n-toys 100 \\
      --n-events 5,10,20,50,100 --output outputs/results.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import re
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


def windowed_moments(edges, vals, fit_lo, fit_hi):
    """Return (mean, std-dev) of the histogram in [fit_lo, fit_hi] via ROOT TH1."""
    n_bins = len(vals)
    edges_arr = np.ascontiguousarray(edges, dtype=np.float64)
    h = ROOT.TH1D("", "", n_bins, edges_arr)
    h.SetDirectory(0)  # detach from gDirectory so name collisions don't matter
    for i in range(n_bins):
        h.SetBinContent(i + 1, max(float(vals[i]), 0.0))
    h.GetXaxis().SetRangeUser(fit_lo, fit_hi)
    if h.Integral() <= 0:
        return float("nan"), float("nan")
    return float(h.GetMean()), float(h.GetStdDev())

logger = logging.getLogger(__name__)

# Schema matches signal_fitting/outputs/.../pull_study/results.csv so existing
# plot machinery (pull_study.make_pull_xy, pull_cell_demo) works unmodified.
CSV_FIELDS = [
    "channel", "mass", "n_events", "model", "config", "seed",
    "status", "covqual",
    "mu_truth", "mu_fit", "mu_err",
    "width_truth", "width_fit", "width_err",
    "delta_truth", "delta_fit", "delta_err",
    "n_sig_fit", "n_sig_err", "min_nll",
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
    p.add_argument("--masses", nargs="+", default=None,
                   help="Specific mass tags. Default: all masses listed in "
                        "fwhm/fits/results.json for each channel.")
    p.add_argument("--alpha-mu", type=float, default=0.10)
    p.add_argument("--alpha-sigma", type=float, default=0.25)
    p.add_argument("--use-moments", action="store_true",
                   help="Use windowed (mean, RMS) of the MC histogram as both "
                        "the truth values *and* the prior centrals. Replaces "
                        "the bootstrap peak / FWHM_boot conventions.")
    p.add_argument("--n-events", default="5,10,20,50,100",
                   help="Comma-separated N values.")
    p.add_argument("--n-toys", type=int, default=100)
    p.add_argument("--config-tag", default="both",
                   help="String written to the 'config' column. Default "
                        "'both' so existing plotters that filter on "
                        "config='both' pick up these rows.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resume", action="store_true",
                   help="Skip (channel, mass, n_events) cells already in the "
                        "output CSV.")
    p.add_argument("--max-masses-per-channel", type=int, default=None,
                   help="Limit masses per channel (useful for chunking around "
                        "the RooFit memory leak).")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def load_mass_list(channel, era, dir_="20260406_signals", min_ratio=0.10):
    """Return all mass tags on disk with M_N/M_WR >= min_ratio, sorted."""
    from measure_fwhm import MIN_RATIO
    input_dirs, _ = input_dirs_for_era(era, repo_root(), dir_)
    tags = set()
    pat = re.compile(r"WRAnalyzer_signal_(WR\d+_N\d+)\.root$")
    for d in input_dirs:
        for f in d.glob("WRAnalyzer_signal_WR*_N*.root"):
            m = pat.search(f.name)
            if m: tags.add(m.group(1))
    filtered = []
    for t in sorted(tags):
        try:
            wr, n = parse_masses(t)
            if n / wr >= min_ratio:
                filtered.append(t)
        except Exception:
            pass
    return filtered


def completed_cells(csv_path):
    """Returns set of (channel, mass, n_events) already in csv_path."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()
    completed = set()
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            completed.add((r["channel"], r["mass"], int(r["n_events"])))
    return completed


def main():
    args = parse_args()
    setup_logging(args.verbose)

    n_events_list = [int(x) for x in args.n_events.split(",")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)

    completed = completed_cells(args.output) if args.resume else set()
    if completed:
        logger.info("Resume: %d (channel,mass,N) cells already in %s",
                    len(completed), args.output)

    write_header = (not args.output.exists()
                    or args.output.stat().st_size == 0)
    out_fh = open(args.output, "a", buffering=1)
    csv_writer = csv.DictWriter(out_fh, fieldnames=CSV_FIELDS)
    if write_header:
        csv_writer.writeheader()

    t0 = time.time()
    n_done = 0

    for channel in args.channels:
        if args.masses is None:
            mass_list = load_mass_list(channel, args.era)
        else:
            mass_list = list(args.masses)
        if args.max_masses_per_channel:
            mass_list = mass_list[: args.max_masses_per_channel]

        logger.info("[%s] %d masses × %d N × %d toys = %d fits planned",
                    channel, len(mass_list), len(n_events_list), args.n_toys,
                    len(mass_list) * len(n_events_list) * args.n_toys)

        region = build_region_name(channel, args.topology)
        mass_var = "mass_twoobject" if args.topology == "boosted" else "mass_fourobject"
        hist_key = build_hist_key(region, mass_var)

        for outer_i, sig_tag in enumerate(mass_list, start=1):
            try:
                M_WR, M_N = parse_masses(sig_tag)
                M_WR, M_N = float(M_WR), float(M_N)
            except Exception:
                logger.warning("Bad mass tag %r, skipping", sig_tag)
                continue

            missing_ns = [n for n in n_events_list
                          if (channel, sig_tag, n) not in completed]
            if not missing_ns:
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
                # Moment-based truth: windowed mean and RMS of MC.
                # Prior widths are also moment-based: α × RMS for both µ and σ.
                mu_center, sigma_truth = windowed_moments(
                    edges_n, vals_n, fit_lo, fit_hi,
                )
                mu_truth_val = mu_center
                # σ-prior central passed to run_fit is in FWHM units; gets
                # divided by 2.355 internally → σ-prior central = RMS.
                fwhm_for_priors = sigma_truth * FWHM_TO_GAUSS_SIGMA
                # σ_µ_prior = α_µ × RMS  (NEW: RMS scale, not FWHM)
                mu_prior_sigma = args.alpha_mu * sigma_truth
                # σ_σ_prior = α_σ × RMS  (pass × 2.355 so internal /2.355 → α × RMS)
                sigma_prior_sigma_fwhm = args.alpha_sigma * sigma_truth * FWHM_TO_GAUSS_SIGMA
                logger.info(
                    "[%s][%d/%d] %s  mean=%.0f  RMS=%.0f  "
                    "[moment truth + α × RMS priors]",
                    channel, outer_i, len(mass_list), sig_tag,
                    mu_center, sigma_truth,
                )
            else:
                mu_center, _ = bootstrap_peak_estimate(
                    edges, vals, sig_tag, channel, args.topology,
                    n_toys=100, seed=0,
                )
                fwhm_boot, _ = bootstrap_fwhm_estimate(
                    edges, vals, sig_tag, channel, args.topology,
                    n_toys=100, seed=0,
                )
                sigma_truth = fwhm_boot / FWHM_TO_GAUSS_SIGMA
                fwhm_for_priors = fwhm_boot
                mu_truth_val = mu_center
                # Legacy convention: σ_µ_prior = α_µ × FWHM_boot
                mu_prior_sigma = args.alpha_mu * fwhm_for_priors
                sigma_prior_sigma_fwhm = args.alpha_sigma * fwhm_for_priors
                logger.info(
                    "[%s][%d/%d] %s  mu_boot=%.0f  FWHM_boot=%.0f  "
                    "[peak/FWHM truth + α × FWHM priors]",
                    channel, outer_i, len(mass_list), sig_tag,
                    mu_center, fwhm_boot,
                )

            for n_events in missing_ns:
                for seed in range(1, args.n_toys + 1):
                    events = sample_from_hist_root(edges_win, vals_win,
                                                    n_events, seed)
                    try:
                        fit = run_fit(
                            "gauss", "constrained", events, M_WR,
                            fwhm_for_priors, sigma_prior_sigma_fwhm,
                            fit_lo, fit_hi,
                            mu_mode="constrained",
                            mu_central=mu_center, mu_sigma=mu_prior_sigma,
                            suffix_extra=f"_n{n_events}_s{seed}",
                        )
                    except Exception as e:
                        logger.debug("fit exception %s/%s n=%d s=%d: %s",
                                     channel, sig_tag, n_events, seed, e)
                        continue
                    csv_writer.writerow({
                        "channel": channel, "mass": sig_tag,
                        "n_events": n_events,
                        "model": "gauss",
                        "config": args.config_tag,
                        "seed": seed,
                        "status": int(fit["minuit_status"]),
                        "covqual": int(fit["covqual"]),
                        "mu_truth": mu_truth_val,
                        "mu_fit": fit["params"]["mu"],
                        "mu_err": fit["errors"]["mu"],
                        "width_truth": sigma_truth,
                        "width_fit": fit["params"]["sigma"],
                        "width_err": fit["errors"]["sigma"],
                        "delta_truth": math.nan,
                        "delta_fit": math.nan,
                        "delta_err": math.nan,
                        "n_sig_fit": fit["params"]["n_sig"],
                        "n_sig_err": fit["errors"]["n_sig"],
                        "min_nll": fit["min_nll"],
                    })
                    n_done += 1

            dt = time.time() - t0
            rate = n_done / dt if dt > 0 else 0.0
            logger.info("  %s/%s done — total rows=%d  rate=%.0f fits/s",
                        channel, sig_tag, n_done, rate)

    out_fh.close()
    logger.info("Done. %d rows written to %s in %.1f min",
                n_done, args.output, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
