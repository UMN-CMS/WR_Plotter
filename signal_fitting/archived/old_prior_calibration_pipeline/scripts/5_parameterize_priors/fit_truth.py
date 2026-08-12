#!/usr/bin/env python3
"""Parameterize (mean, RMS) truth as TGraph2D interpolators in (M_WR, M_N).

For each channel, scan the MC for all available signal mass cells, compute
the windowed (mean, RMS) of m_lljj, and store the (M_WR, M_N) → mean and
(M_WR, M_N) → RMS data as ROOT TGraph2D objects. At predict time,
TGraph2D::Interpolate does Delaunay-triangulation-based interpolation.

The MIN_RATIO = 0.10 filter is applied (boosted-N regime excluded).

Inputs:
    Signal MC under rootfiles/<run>/<year>/<era>/<dir>/
Outputs:
    signal_fitting/outputs/truth_params.root   (4 TGraph2D: {ch}_{mean,rms})

Library use:
    from fit_truth import load_params, predict_priors
    p = load_params("signal_fitting/outputs/truth_params.root")
    mu, rms = predict_priors("ee", M_WR=4321, M_N=1234, params=p)

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    MIN_RATIO,
    ONSHELL_WINDOW_LO_FRAC, ONSHELL_WINDOW_HI_FRAC,
    build_hist_key, build_region_name,
    load_and_combine_signal, parse_masses,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--channels", nargs="+", default=["ee", "mumu"])
    p.add_argument("--min-ratio", type=float, default=MIN_RATIO,
                   help=f"Filter: drop cells with M_N/M_WR < min_ratio. "
                        f"Default: {MIN_RATIO} (boosted-N filter).")
    p.add_argument("--output", type=Path,
                   default=Path("signal_fitting/outputs/truth_params.root"))
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


def load_truth_from_mc(input_dirs, channels, topology, min_ratio):
    """Scan MC for all WR*_N* signal files, compute windowed (mean, RMS) per
    (channel, mass), apply MIN_RATIO filter. Returns a DataFrame."""
    tags = set()
    pat = re.compile(r"WRAnalyzer_signal_(WR\d+_N\d+)\.root$")
    for d in input_dirs:
        for f in d.glob("WRAnalyzer_signal_WR*_N*.root"):
            m = pat.search(f.name)
            if m: tags.add(m.group(1))
    tags = sorted(tags)
    logger.info("Found %d unique mass tags on disk", len(tags))

    mass_var = ("mass_twoobject" if topology == "boosted"
                else "mass_fourobject")
    rows, n_kept, n_filt, n_fail = [], 0, 0, 0
    for channel in channels:
        region = build_region_name(channel, topology)
        hist_key = build_hist_key(region, mass_var)
        for tag in tags:
            try:
                M_WR, M_N = parse_masses(tag)
                M_WR, M_N = float(M_WR), float(M_N)
            except Exception:
                continue
            if M_N / M_WR < min_ratio:
                n_filt += 1; continue
            try:
                edges, vals, _ = load_and_combine_signal(input_dirs, hist_key, tag)
            except Exception as e:
                logger.warning("Load fail %s/%s: %s", channel, tag, e)
                n_fail += 1; continue
            fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
            fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR
            mu, rms = windowed_moments(edges, vals, fit_lo, fit_hi)
            rows.append({
                "channel": channel, "mass": tag,
                "M_WR": M_WR, "M_N": M_N,
                "mean": mu, "rms": rms,
            })
            n_kept += 1
    logger.info("Kept %d (channel, mass) cells; filtered %d (x < %.2f); "
                "failed-load %d", n_kept, n_filt, min_ratio, n_fail)
    return pd.DataFrame(rows)


def build_graphs(df_ch, channel):
    """Build (g_mean, g_rms) TGraph2D objects for one channel."""
    n = len(df_ch)
    x = np.ascontiguousarray(df_ch["M_WR"].values, dtype=np.float64)
    y = np.ascontiguousarray(df_ch["M_N"].values,  dtype=np.float64)
    z_mean = np.ascontiguousarray(df_ch["mean"].values, dtype=np.float64)
    z_rms  = np.ascontiguousarray(df_ch["rms"].values,  dtype=np.float64)
    g_mean = ROOT.TGraph2D(n, x, y, z_mean)
    g_rms  = ROOT.TGraph2D(n, x, y, z_rms)
    g_mean.SetName(f"{channel}_mean")
    g_rms .SetName(f"{channel}_rms")
    g_mean.SetTitle(
        f"{channel} windowed mean of m_lljj in [0.7, 1.3]*M_WR;"
        f"M_WR [GeV];M_N [GeV];mean [GeV]")
    g_rms .SetTitle(
        f"{channel} windowed RMS of m_lljj in [0.7, 1.3]*M_WR;"
        f"M_WR [GeV];M_N [GeV];RMS [GeV]")
    return g_mean, g_rms


def load_params(root_path):
    """Load the TGraph2D interpolators back from disk.

    Returns {channel: {"mean": TGraph2D, "rms": TGraph2D}}.
    The TFile is kept open and owned by the returned dict so the graphs
    stay alive — close it manually if needed via params['_file'].Close().
    """
    f = ROOT.TFile.Open(str(root_path), "READ")
    if not f or f.IsZombie():
        raise IOError(f"Cannot open {root_path}")
    params = {"_file": f}
    for key in f.GetListOfKeys():
        name = key.GetName()
        if "_" not in name:
            continue
        ch, kind = name.rsplit("_", 1)
        if kind not in ("mean", "rms"):
            continue
        params.setdefault(ch, {})[kind] = f.Get(name)
    return params


def predict_priors(channel, M_WR, M_N, params):
    """Return (mean_pred, rms_pred) at (M_WR, M_N) via Delaunay interpolation.

    Raises ValueError if (M_WR, M_N) sits outside the training envelope
    (TGraph2D::Interpolate returns exactly 0 in that case).
    """
    if channel not in params:
        raise KeyError(f"channel {channel!r} not in params (keys: "
                       f"{[k for k in params if not k.startswith('_')]})")
    mean = float(params[channel]["mean"].Interpolate(float(M_WR), float(M_N)))
    rms  = float(params[channel]["rms" ].Interpolate(float(M_WR), float(M_N)))
    if mean == 0.0 or rms == 0.0:
        raise ValueError(
            f"({M_WR}, {M_N}) is outside the training envelope for "
            f"channel {channel!r} (TGraph2D::Interpolate returned 0)")
    return mean, rms


def main():
    args = parse_args()
    setup_logging(args.verbose)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    logger.info("Scanning MC in %s", input_dirs)
    df = load_truth_from_mc(input_dirs, args.channels, args.topology,
                             args.min_ratio)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    f_out = ROOT.TFile.Open(str(args.output), "RECREATE")
    for channel, sub in df.groupby("channel"):
        g_mean, g_rms = build_graphs(sub.reset_index(drop=True), channel)
        g_mean.Write()
        g_rms.Write()
        logger.info("  %s: wrote TGraph2D mean (%d points) and rms (%d points)",
                    channel, g_mean.GetN(), g_rms.GetN())
    f_out.Close()
    logger.info("Wrote interpolators to %s", args.output)

    # Sanity demo at an off-grid point (re-open as a user would).
    params = load_params(args.output)
    demo_wr, demo_n = 4321, 1234
    for ch in args.channels:
        mu_p, rms_p = predict_priors(ch, demo_wr, demo_n, params)
        logger.info("Demo (%s) M_WR=%d, M_N=%d  ->  mean=%.1f  RMS=%.1f",
                    ch, demo_wr, demo_n, mu_p, rms_p)
    params["_file"].Close()


if __name__ == "__main__":
    main()
