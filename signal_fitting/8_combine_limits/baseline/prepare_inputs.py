#!/usr/bin/env python3
"""Stage 9, step 1 -- dump combine inputs (background hist + per-mass metadata).

Everything the workspace builder needs, written as plain TH1D + JSON so that
`make_workspaces.py` can run inside the combine container with no wrplotter /
uproot / LCG dependencies (and no RooWorkspace version mismatch: workspaces are
built by the container's own ROOT).

Per (channel, topology):
  inputs/{tag}.root   `data_obs_full` -- the summed background MC (same four
                      samples, era, dir and 100 GeV binning as Stage 6), with
                      negative / non-finite bins clipped to 0 (they live in the
                      sparse tail, exactly as in the Stage-6 toy means).
  inputs/{tag}.json   lumi + per-mass window (m_c, sigma, fit_lo/hi), signal
                      tag, efficiency and rate (events per fb of sigma x BR),
                      theory xsec, and the Stage-6 toy moments per function
                      (kept as the homemade reference for the parity check).

Windows, signal tags and efficiencies are read/derived exactly as in Stage 7b:
the Stage-6 CSV is the single source, `eff = S_fit / genEventSumw` on the raw
signal histograms.

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                        # repo root
sys.path.insert(0, str(HERE.parents[1] / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(HERE.parents[1] / "shared"))             # loaders
sys.path.insert(0, str(HERE.parents[1] / "7_limit_plots"))      # xsec_limit

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402

import ROOT                                                             # noqa: E402
from bkg_fit_lib import MASS_VAR, load_summed_background                # noqa: E402
from measure_fwhm import build_hist_key, build_region_name              # noqa: E402
from xsec_limit import (                                                # noqa: E402
    default_signal_config, read_stage6_table, signal_efficiency,
)

logger = logging.getLogger("prepare_inputs")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24",
                   help="background era (lumi the limit is computed at)")
    p.add_argument("--dir", default="20260317_lo_dy", help="background dir")
    p.add_argument("--signal-era", default="RunIISummer20UL18")
    p.add_argument("--signal-dir", default="20260624_signals")
    p.add_argument("--signal-config", type=Path, default=None)
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--table", type=Path, default=None,
                   help="Stage-6 summary CSV (default: the {ch}_{topo} one)")
    p.add_argument("--with-data", action="store_true",
                   help="also dump the real data histogram (EGamma for ee, "
                        "Muon for mumu) as 'data_obs_real', for "
                        "make_workspaces --observed data")
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--min-mass", type=float, default=1200.0)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <script dir>/<run2|run3>/inputs, by --era.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    channel, topology = args.channel, args.topology
    tag = f"{channel}_{topology}"
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    run_sub = {"RunII": "run2", "Run3": "run3"}[str(info["run"])]
    if args.output_dir is None:
        args.output_dir = HERE / run_sub / "inputs"

    # Background MC -> plain TH1D (full range; the range cut happens per mass
    # in make_workspaces so one hist serves all masses).
    bkg_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    region = build_region_name(channel, topology)
    factor = max(1, round(args.bin_width / 10.0))
    edges, values, _ = load_summed_background(
        bkg_dirs, region, MASS_VAR[topology], factor)
    clipped = sum(1 for v in values if not (math.isfinite(v) and v >= 0.0))
    data = [float(v) if (math.isfinite(v) and v > 0.0) else 0.0 for v in values]
    if clipped:
        logger.info("Clipped %d negative/non-finite background bins to 0", clipped)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    fout = ROOT.TFile(str(args.output_dir / f"{tag}.root"), "RECREATE")
    h = ROOT.TH1D("data_obs_full", ";m [GeV];events", len(data),
                  np.ascontiguousarray(edges, np.float64))
    for i, v in enumerate(data, start=1):
        h.SetBinContent(i, v)
    h.Write()

    data_sample = None
    if args.with_data:
        import uproot
        data_sample = {"ee": "EGamma", "mumu": "Muon"}[channel]
        hist_key_d = f"{region}/{MASS_VAR[topology]}_{region}"
        dvals = None
        for d in bkg_dirs:
            hd = uproot.open(d / f"WRAnalyzer_{data_sample}.root")[hist_key_d]
            v = hd.values()
            dvals = v.copy() if dvals is None else dvals + v
        n = (len(dvals) // factor) * factor
        dvals = dvals[:n].reshape(-1, factor).sum(axis=1)
        hd_out = ROOT.TH1D("data_obs_real", ";m [GeV];events", len(dvals),
                           np.ascontiguousarray(edges, np.float64))
        for i, v in enumerate(dvals, start=1):
            hd_out.SetBinContent(i, float(v))
        hd_out.Write()
        logger.info("Real data (%s): %.0f events total", data_sample,
                    float(dvals.sum()))
    fout.Close()

    # Per-mass metadata from the Stage-6 CSV + the Stage-7b efficiency.
    cfg_path = args.signal_config or default_signal_config(args.signal_era)
    with open(cfg_path) as fh:
        sig_cfg = {v["dataset"]: v for v in json.load(fh).values()}
    sig_dirs, _ = input_dirs_for_era(args.signal_era, repo_root(),
                                     args.signal_dir)
    hist_key = build_hist_key(region, MASS_VAR[topology])

    table = args.table or (HERE.parents[1] / "6_spurious_signal_toys"
                           / run_sub / f"spurious_toy_table_{tag}.csv")
    rows = read_stage6_table(table, channel, topology)

    masses = {}
    for r in rows:
        mWR = r["mWR"]
        if mWR < args.min_mass or not r["signal_tag"]:
            continue
        key = f"{mWR:.0f}"
        if key not in masses:
            # geometry columns are per-mass (identical across functions)
            with open(table, newline="") as fh:      # grab m_c/sigma once
                import csv as _csv
                for rr in _csv.DictReader(fh):
                    if (rr["channel"] == channel and rr["topology"] == topology
                            and float(rr["mWR"]) == mWR):
                        geo = rr
                        break
            eff = signal_efficiency(sig_dirs, hist_key, r["signal_tag"],
                                    r["fit_lo"], r["fit_hi"], sig_cfg)
            if eff is None or not (eff["eff"] > 0.0):
                continue
            masses[key] = {
                "signal_tag": r["signal_tag"], "m_N": eff["m_N"],
                "m_c": float(geo["m_c"]), "sigma": float(geo["sigma_win"]),
                "fit_lo": r["fit_lo"], "fit_hi": r["fit_hi"],
                "eff": eff["eff"], "rate_per_fb": lumi * eff["eff"],
                "xsec_pb": eff["xsec_pb"], "ref": {},
            }
            logger.info("m=%s %s: eff=%.4f rate=%.2f ev/fb",
                        key, r["signal_tag"], eff["eff"], lumi * eff["eff"])
        if key in masses:
            masses[key]["ref"][r["function"]] = {
                "mean_Nsp": r["mean_Nsp"], "rms_Nsp": r["rms_Nsp"],
                "nsp_asimov": r["nsp_asimov"],
                "nsp_asimov_err": r["nsp_asimov_err"],
            }

    meta = {"channel": channel, "topology": topology, "era": args.era,
            "lumi": lumi, "com": com, "bin_width": args.bin_width,
            "bkg_dir": args.dir, "signal_era": args.signal_era,
            "signal_dir": args.signal_dir, "data_sample": data_sample,
            "masses": masses}
    out_json = args.output_dir / f"{tag}.json"
    with open(out_json, "w") as fh:
        json.dump(meta, fh, indent=1)
    logger.info("Wrote %s (%d masses) and %s.root", out_json, len(masses), tag)


if __name__ == "__main__":
    main()
