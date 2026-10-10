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
from measure_fwhm import load_and_combine_signal                        # noqa: E402

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


def signal_efficiency_run3(sig_dirs, hist_key, tag, lo, hi, sig_cfg, bw, lumi):
    """Run-3 signal samples are flavour-specific (WRtoNEltoElElJJ / WRtoNMutoMuMuJJ).
    The analyzer fills each with genWeight*xsec*lumi/genEventSumw, so the window
    yield of the channel's signal-region histogram is already sigma*L*eps of the
    matching flavour sample, and eps = yield / (xsec * lumi).

    Do NOT divide by the file's weighted cutflow total: the signal_<tag> file
    merges the electron AND the muon sample of that mass point, and its no_cuts
    row counts only the events that survived the skim (2026-10-06 bug: that gave
    eps = 0.25 instead of 0.40 for WR2000_N1100 ee)."""
    m, n = tag[2:].split("_N")
    pref = {"ee": "WRtoNEltoElElJJ", "mumu": "WRtoNMutoMuMuJJ"}[CHANNEL] + f"_MWR{m}_N{n}_"
    entry = next((v for k, v in sig_cfg.items() if k.startswith(pref)), None)
    if entry is None:
        logger.warning("  %s: no '%s*' in the signal config, skipping", tag, pref)
        return None
    edges, values, _ = load_and_combine_signal(sig_dirs, hist_key, tag)
    f = max(1, round(bw / float(edges[1] - edges[0])))
    nb = (len(values) // f) * f
    e_r = edges[0:nb + 1:f]
    v_r = values[:nb].reshape(-1, f).sum(axis=1)
    c_r = 0.5 * (e_r[:-1] + e_r[1:])
    s_fit = float(v_r[(c_r >= lo) & (c_r <= hi)].sum())
    sigma_l = float(entry["xsec"]) * 1000.0 * float(lumi)     # pb -> fb, times fb^-1
    eff = s_fit / sigma_l if sigma_l > 0 else float("nan")
    logger.info("  %s: window yield %.1f / (xsec %.5f pb x L %.2f fb^-1 = %.1f) -> eff %.4f",
                tag, s_fit, float(entry["xsec"]), float(lumi), sigma_l, eff)
    return {"eff": eff, "xsec_pb": float(entry["xsec"]), "s_fitrange": s_fit,
            "sigma_L": sigma_l}


def fit_signal_core(sig_dirs, hist_key, tag, mwr):
    """(m_c, sigma) from the Stage-1 iterative Gaussian-core fit of the
    raw signal histogram (seed range [0.7, 1.3] m_WR)."""
    from shape_estimators import gaussian_core_fit
    edges, values, _ = load_and_combine_signal(sig_dirs, hist_key, tag)
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    r = gaussian_core_fit(edges, values, float(mwr), 0.7, 1.3)
    logger.info("signal core fit %s: mu=%.1f sigma=%.1f (+-%.1f) status %d "
                "converged %s", tag, r["mu"], r["sigma"], r["sigma_err"],
                r["fit_status"], r["converged"])
    return float(r["mu"]), float(r["sigma"])


def main():
    global ERA, BKG_DIR, SIGNAL_ERA, SIGNAL_DIR, MASSES, CHANNEL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=HERE / "inputs")
    p.add_argument("--era", default=ERA, help="background era (lumi, dirs)")
    p.add_argument("--bkg-dir", default=BKG_DIR)
    p.add_argument("--signal-era", default=SIGNAL_ERA,
                   help="era of the signal samples used for the efficiency")
    p.add_argument("--signal-dir", default=SIGNAL_DIR)
    p.add_argument("--masses", type=int, nargs="+", default=None,
                   help="m_WR list (default: the optimization list)")
    p.add_argument("--channel", default=CHANNEL, choices=["ee", "mumu"],
                   help="lepton channel (output tag <channel>_<topology>)")
    p.add_argument("--signal-config", type=Path, default=None,
                   help="signal config JSON for the Run-2 efficiencies (default: the era's "
                        "data/configs file; the chain expects the WRtoNLtoLLJJ_MWR*_MN* naming)")
    p.add_argument("--geo-from-signal", nargs="+", default=[],
                   metavar="ERA:DIR:TAG",
                   help="for the mass of TAG (e.g. WR2000_N1100), take m_c and "
                        "sigma from a Gaussian-core fit of that sample in "
                        "ERA/DIR, and the efficiency from it too, instead of "
                        "the Run-2 table/samples")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    ERA, BKG_DIR = args.era, args.bkg_dir
    SIGNAL_ERA, SIGNAL_DIR = args.signal_era, args.signal_dir
    CHANNEL = args.channel
    if args.masses:
        MASSES = list(args.masses)
    tag = f"{CHANNEL}_{TOPOLOGY}"
    lumi = load_lumi(ERA)["lumi"]

    # geometry (m_c, sigma) from the Stage-6 run2 table
    import csv
    stage6 = (SIGFIT / "6_spurious_signal_toys" / "run2"
              / f"spurious_toy_table_{tag}.csv")
    if not stage6.exists():
        # the Stage-6 table exists for ee only; the mumu Gaussian cores differ
        # by a few percent, and the running example takes its geometry from
        # --geo-from-signal anyway, so fall back to the ee table with a warning
        fallback = stage6.with_name(f"spurious_toy_table_ee_{TOPOLOGY}.csv")
        logger.warning("no %s; using the ee geometry table %s", stage6.name, fallback.name)
        stage6 = fallback
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

    cfg_path = args.signal_config or default_signal_config(SIGNAL_ERA)
    logger.info("Run-2 signal config: %s", cfg_path)
    with open(cfg_path) as fh:
        sig_cfg = {v["dataset"]: v for v in json.load(fh).values()}
    sig_dirs, _ = input_dirs_for_era(SIGNAL_ERA, repo_root(), SIGNAL_DIR)
    hist_key = build_hist_key(region, MASS_VAR[TOPOLOGY])
    centers = 0.5 * (edges[:-1] + edges[1:])

    # per-mass overrides: geometry + efficiency from a specific (Run-3) sample
    override = {}
    for spec in args.geo_from_signal:
        o_era, o_dir, o_tag = spec.split(":")
        o_mass = int(o_tag[2:].split("_N")[0])
        o_dirs, _ = input_dirs_for_era(o_era, repo_root(), o_dir)
        with open(default_signal_config(o_era)) as fh:
            o_cfg = {v["dataset"]: v for v in json.load(fh).values()}
        m_c, sigma = fit_signal_core(o_dirs, hist_key, o_tag, o_mass)
        override[o_mass] = dict(era=o_era, dirs=o_dirs, cfg=o_cfg, tag=o_tag,
                                m_c=m_c, sigma=sigma)
        geo[o_mass] = (m_c, sigma, o_tag)

    out = {"lumi": lumi, "era": ERA, "bkg_dir": BKG_DIR,
           "signal_era": SIGNAL_ERA, "signal_dir": SIGNAL_DIR,
           "overrides": {str(m): dict(era=o["era"], tag=o["tag"], m_c=o["m_c"],
                                      sigma=o["sigma"])
                         for m, o in override.items()},
           "masses": {}}
    for mass in MASSES:
        m_c, sigma, stag = geo[mass]
        entry = {"m_c": m_c, "sigma": sigma, "signal_tag": stag, "vars": {}}
        if mass in override:
            entry["signal_era"] = override[mass]["era"]
        for wname, (kl, kr) in WINDOWS.items():
            lo = max(m_c - kl * sigma, XMIN)
            hi = min(m_c + kr * sigma, XMAX)
            bsel = (centers >= lo) & (centers <= hi)
            b_win = float(values[bsel].sum())
            for bw in BINWIDTHS:
                if mass in override:
                    o = override[mass]
                    eff = signal_efficiency_run3(o["dirs"], hist_key, o["tag"],
                                                 lo, hi, o["cfg"], bw,
                                                 load_lumi(o["era"])["lumi"])
                else:
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
