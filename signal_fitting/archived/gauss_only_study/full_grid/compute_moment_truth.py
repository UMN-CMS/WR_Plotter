#!/usr/bin/env python3
"""Recompute mu_truth and width_truth as the windowed first two moments of the
MC mass histogram (instead of peak-finding + FWHM).

Per (channel, mass):
    centers, contents = native MC histogram in [fit_lo, fit_hi] = [LO_FRAC, HI_FRAC] × M_WR
    mu_truth    = Σ N_i × x_i / Σ N_i                  (mean in window)
    width_truth = sqrt( Σ N_i × (x_i − mu)² / Σ N_i )  (RMS in window)

These are the natural truth targets for an unbinned Gaussian MLE fit (whose
equilibrium µ and σ are the sample mean and sample RMS).

Input:   results_final.csv (or any pull-study-format CSV)
Output:  results_moments.csv (same schema, with mu_truth and width_truth
                              replaced by the moment-truth values)

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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wrplotter.cli_utils import setup_logging
from wrplotter.paths import input_dirs_for_era, repo_root
from measure_fwhm import (
    ONSHELL_WINDOW_LO_FRAC, ONSHELL_WINDOW_HI_FRAC,
    build_hist_key, build_region_name,
    load_and_combine_signal, parse_masses,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--topology", default="resolved")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def windowed_moments(edges, vals, fit_lo, fit_hi):
    """Return (mean, std-dev) of the histogram in [fit_lo, fit_hi] via ROOT TH1."""
    import ROOT  # lazy: only needed if this script is actually invoked
    n_bins = len(vals)
    edges_arr = np.ascontiguousarray(edges, dtype=np.float64)
    h = ROOT.TH1D("", "", n_bins, edges_arr)
    h.SetDirectory(0)
    for i in range(n_bins):
        h.SetBinContent(i + 1, max(float(vals[i]), 0.0))
    h.GetXaxis().SetRangeUser(fit_lo, fit_hi)
    if h.Integral() <= 0:
        return float("nan"), float("nan")
    return float(h.GetMean()), float(h.GetStdDev())


def main():
    args = parse_args()
    setup_logging(args.verbose)
    df = pd.read_csv(args.input)
    logger.info("Read %d rows from %s", len(df), args.input)

    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    unique_cells = df[["channel", "mass"]].drop_duplicates()
    logger.info("Computing moment-truth for %d (channel, mass) cells",
                len(unique_cells))

    truth_map = {}
    t0 = time.time()
    for i, (ch, mass) in enumerate(
        zip(unique_cells["channel"], unique_cells["mass"]), start=1
    ):
        try:
            M_WR, M_N = parse_masses(mass)
            M_WR = float(M_WR)
        except Exception:
            logger.warning("Bad mass tag %r, skipping", mass)
            truth_map[(ch, mass)] = (np.nan, np.nan)
            continue
        region = build_region_name(ch, args.topology)
        mass_var = ("mass_twoobject" if args.topology == "boosted"
                    else "mass_fourobject")
        hist_key = build_hist_key(region, mass_var)
        try:
            edges, vals, _ = load_and_combine_signal(input_dirs, hist_key, mass)
        except Exception as e:
            logger.warning("Load fail %s/%s: %s", ch, mass, e)
            truth_map[(ch, mass)] = (np.nan, np.nan)
            continue
        fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
        fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR
        mean, rms = windowed_moments(edges, vals, fit_lo, fit_hi)
        truth_map[(ch, mass)] = (mean, rms)
        if i % 50 == 0:
            dt = time.time() - t0
            logger.info("  %d / %d cells  (%.1f s)", i, len(unique_cells), dt)

    # Sanity sample
    sample_keys = list(truth_map.keys())[:5]
    logger.info("Sample truth values:")
    for k in sample_keys:
        logger.info("  %s: mean=%.1f  RMS=%.1f", k, *truth_map[k])

    # Apply: replace mu_truth and width_truth columns.
    df["mu_truth"] = df.apply(
        lambda r: truth_map.get((r["channel"], r["mass"]), (np.nan, np.nan))[0],
        axis=1,
    )
    df["width_truth"] = df.apply(
        lambda r: truth_map.get((r["channel"], r["mass"]), (np.nan, np.nan))[1],
        axis=1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info("Wrote %s (%d rows)", args.output, len(df))

    # Quick diff against the previous truth.
    df_orig = pd.read_csv(args.input)
    diff = df["mu_truth"] - df_orig["mu_truth"]
    diff_w = df["width_truth"] - df_orig["width_truth"]
    logger.info("Δ mu_truth (new − old): median=%+.1f, p16=%+.1f, p84=%+.1f GeV",
                float(np.nanmedian(diff)),
                float(np.nanpercentile(diff, 16)),
                float(np.nanpercentile(diff, 84)))
    logger.info("Δ width_truth (new − old): median=%+.1f, p16=%+.1f, p84=%+.1f GeV",
                float(np.nanmedian(diff_w)),
                float(np.nanpercentile(diff_w, 16)),
                float(np.nanpercentile(diff_w, 84)))


if __name__ == "__main__":
    main()
