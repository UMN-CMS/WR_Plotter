#!/usr/bin/env python3
"""Stage 10.9, step 1 -- variant inputs for the float-region optimization.

Goal: improve the EXPECTED limit at 2000-3200 GeV (the trusted float regime)
by scanning card configurations, all on the no-data (MC Asimov) observation:

  window   k3 (baseline mu+-3sigma) / k4 / k5 / a53 (asymmetric [mu-5s, mu+3s]
           -- more LEFT sideband where the background lives, short right tail)
  binning  100 GeV (baseline) / 50 / 20  (signal sigma ~ 160-220 GeV, so the
           baseline bins are ~0.6 sigma wide -- coarse for a shape fit)
  bkg      float (norm+slope free, baseline) / bconstr (slope Gaussian-
           constrained to the trusted-spectrum anchor fit) / bfixed (slope
           constant, norm free) / anch (both constant -- the bound on what
           background knowledge can buy)

This step (LCG_106) writes, per (mass, window, binning): the window edges and
the per-channel efficiency (recomputed for THAT window and binning via the
Stage-7b machinery, bfrac = 0.5), plus the native 10 GeV summed-background
TH1 the workspace builder rebins per card.

  python prepare_opt.py -v
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # 8_combine_limits/optimization
SIGFIT = HERE.parents[1]                        # signal_fitting
sys.path.insert(0, str(SIGFIT.parent))          # repo root (wrplotter)
sys.path.insert(0, str(SIGFIT / "4_background_fits"))
sys.path.insert(0, str(SIGFIT / "shared"))
sys.path.insert(0, str(SIGFIT / "7_limit_plots"))

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402

import numpy as np                                                      # noqa: E402
import ROOT                                                             # noqa: E402
from bkg_fit_lib import MASS_VAR, load_summed_background                # noqa: E402
from measure_fwhm import build_hist_key, build_region_name              # noqa: E402
from xsec_limit import default_signal_config, signal_efficiency         # noqa: E402

logger = logging.getLogger("prepare_opt")

ERA, BKG_DIR = "RunIISummer20UL18", "20260714_run2_bkgs"
SIGNAL_ERA, SIGNAL_DIR = "RunIISummer20UL18", "20260624_signals"
CHANNEL, TOPOLOGY = "ee", "resolved"
# 2000-3200 was the optimization scan; 1400-1800 were added afterwards so the
# Stage-10.8 float regime (1400-3200) can adopt the winner configuration
# (the k5 windows there clamp against the 800 GeV floor -- handled below).
MASSES = [1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200]

# window variants: (k_left, k_right) in units of sigma
WINDOWS = {"k3": (3.0, 3.0), "k4": (4.0, 4.0), "k5": (5.0, 5.0),
           "a53": (5.0, 3.0)}
BINWIDTHS = [100.0, 50.0, 20.0]
XMIN, XMAX = 800.0, 6000.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=HERE / "inputs")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    tag = f"{CHANNEL}_{TOPOLOGY}"
    lumi = load_lumi(ERA)["lumi"]

    # geometry (m_c, sigma) from the Stage-6 run2 table
    import csv
    stage6 = (SIGFIT / "6_spurious_signal_toys" / "run2"
              / f"spurious_toy_table_{tag}.csv")
    geo = {}
    with open(stage6, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["function"] == "expo":
                geo[int(float(r["mWR"]))] = (float(r["m_c"]),
                                             float(r["sigma_win"]),
                                             r["signal_tag"])

    # native 10 GeV summed background -> TH1 for the builder
    bkg_dirs, _ = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    region = build_region_name(CHANNEL, TOPOLOGY)
    edges, values, _ = load_summed_background(bkg_dirs, region,
                                              MASS_VAR[TOPOLOGY], 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fout = ROOT.TFile(str(args.output_dir / f"{tag}.root"), "RECREATE")
    h = ROOT.TH1D("bkg_native", ";m [GeV];events", len(values),
                  np.ascontiguousarray(edges, np.float64))
    for i, v in enumerate(values, start=1):
        h.SetBinContent(i, float(v) if (np.isfinite(v) and v > 0) else 0.0)
    h.Write()
    fout.Close()

    cfg_path = default_signal_config(SIGNAL_ERA)
    with open(cfg_path) as fh:
        sig_cfg = {v["dataset"]: v for v in json.load(fh).values()}
    sig_dirs, _ = input_dirs_for_era(SIGNAL_ERA, repo_root(), SIGNAL_DIR)
    hist_key = build_hist_key(region, MASS_VAR[TOPOLOGY])
    centers = 0.5 * (edges[:-1] + edges[1:])

    out = {"lumi": lumi, "era": ERA, "bkg_dir": BKG_DIR, "masses": {}}
    for mass in MASSES:
        m_c, sigma, stag = geo[mass]
        entry = {"m_c": m_c, "sigma": sigma, "signal_tag": stag, "vars": {}}
        for wname, (kl, kr) in WINDOWS.items():
            lo = max(m_c - kl * sigma, XMIN)
            hi = min(m_c + kr * sigma, XMAX)
            bsel = (centers >= lo) & (centers <= hi)
            b_win = float(values[bsel].sum())
            for bw in BINWIDTHS:
                eff = signal_efficiency(sig_dirs, hist_key, stag, lo, hi,
                                        sig_cfg, bin_width=bw)
                key = f"{wname}_bw{bw:.0f}"
                entry["vars"][key] = {
                    "fit_lo": lo, "fit_hi": hi, "bw": bw,
                    "eff": eff["eff"], "rate_per_fb": lumi * eff["eff"],
                    "B_window": b_win,
                }
        logger.info("m=%d: %s  B(k3)=%.1f B(k5)=%.1f  eff(k3,100)=%.3f "
                    "eff(k5,100)=%.3f", mass, stag,
                    entry["vars"]["k3_bw100"]["B_window"],
                    entry["vars"]["k5_bw100"]["B_window"],
                    entry["vars"]["k3_bw100"]["eff"],
                    entry["vars"]["k5_bw100"]["eff"])
        out["masses"][str(mass)] = entry

    with open(args.output_dir / f"{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    logger.info("wrote %s", args.output_dir / f"{tag}.json")


if __name__ == "__main__":
    main()
