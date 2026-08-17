#!/usr/bin/env python3
"""Flavor-CR variant, step 1 -- split-background inputs for the CR-anchored card.

The `fcr` card variant constrains the tt+tW normalization from the flavor
(e-mu) control region instead of letting it float freely inside the summed
background. That needs the SR background SPLIT into the CR-constrained part
(tt+tW) and the rest (DYJets + Nonprompt + Other, still a floating expo), plus
the flavor-CR yields the counting channel is built from.

This step (LCG_106) reads the geometry (m_c, sigma, window edges per mass and
window/binning variant) from the prepare_opt inputs and writes, per (mass,
variant):

  T_tt      tt+tW yield inside the snapped SR window        (win-channel rate)
  B_rest    DY+Nonprompt+Other yield inside the same window (rest-norm init)
  T_cr      tt+tW yield in the flavor CR, same mass window  (fcr-channel rate)
  C_other   non-tt flavor-CR yield, same window             (fixed fcr rate)
  b_tt      expo slope of the SR tt+tW shape, binned-likelihood fit inside the
            snapped window at the card binning (recentered at m_c, /1000) --
            the ONE tt-shape number both the combine workspace and the
            homemade card model use

plus the native 10 GeV split TH1s (tt_native, rest_native, fcr_tt_native,
fcr_rest_native) the workspace builder rebins per card.

  python prepare_fcr.py -v
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

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402

import numpy as np                                                      # noqa: E402
import ROOT                                                             # noqa: E402
from bkg_fit_lib import MASS_VAR, load_summed_background                # noqa: E402

logger = logging.getLogger("prepare_fcr")
ROOT.gROOT.SetBatch(True)

ERA, BKG_DIR = "RunIISummer20UL18", "20260714_run2_bkgs"
CHANNEL, TOPOLOGY = "ee", "resolved"
TT, REST = ["tt_tW"], ["DYJets", "Nonprompt", "Other"]
FCR_REGION = "wr_resolved_flavor_cr"


def load_split(bkg_dirs, region, samples):
    """Summed (edges, values) for `samples` in `region`, native 10 GeV bins,
    negative / non-finite MC bins clipped to 0 (as prepare_opt does)."""
    import bkg_fit_lib
    orig = bkg_fit_lib.BKG_SAMPLES
    try:
        bkg_fit_lib.BKG_SAMPLES = samples
        edges, values, _ = load_summed_background(
            bkg_dirs, region, MASS_VAR[TOPOLOGY], 1)
    finally:
        bkg_fit_lib.BKG_SAMPLES = orig
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    return edges, values


def to_th1(name, edges, values):
    h = ROOT.TH1D(name, ";m [GeV];events", len(values),
                  np.ascontiguousarray(edges, np.float64))
    for i, v in enumerate(values, start=1):
        h.SetBinContent(i, float(v))
    return h


def rebin_snap(edges, values, bw, fit_lo, fit_hi):
    """Rebin native -> bw, then select bins whose CENTER lies in
    [fit_lo, fit_hi] (the TH1::Fit / make_opt_workspaces.snap rule).
    Returns (snapped edges, snapped values)."""
    f = max(1, round(bw / float(edges[1] - edges[0])))
    n = (len(values) // f) * f
    vals = values[:n].reshape(-1, f).sum(axis=1)
    e = edges[0:n + 1:f]
    centers = 0.5 * (e[:-1] + e[1:])
    sel = (centers >= fit_lo) & (centers <= fit_hi)
    lo_edges = e[:-1][sel]
    return np.append(lo_edges, e[1:][sel][-1]), vals[sel]


def fit_tt_slope(edges, values, m_c):
    """Binned-likelihood expo fit A*exp(b*(x-m_c)/1000) to the snapped window
    histogram. Returns (b, b_err)."""
    h = to_th1("h_ttfit", edges, values)
    f = ROOT.TF1("f_tt", f"[0]*exp([1]*((x-{m_c:.10g})/1000.))",
                 float(edges[0]), float(edges[-1]))
    f.SetParameter(0, max(values.max(), 0.1))
    f.SetParameter(1, -3.0)
    f.SetParLimits(1, -100.0, 0.0)
    h.Fit(f, "LRSQN")
    return float(f.GetParameter(1)), float(f.GetParError(1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=HERE / "inputs",
                   help="where the prepare_opt geometry json lives")
    p.add_argument("--output-dir", type=Path, default=HERE / "inputs")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    tag = f"{CHANNEL}_{TOPOLOGY}"

    with open(args.input_dir / f"{tag}.json") as fh:
        geo = json.load(fh)

    bkg_dirs, _ = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    sr = f"wr_{CHANNEL}_{TOPOLOGY}_sr"
    e_tt, v_tt = load_split(bkg_dirs, sr, TT)
    e_rest, v_rest = load_split(bkg_dirs, sr, REST)
    e_ftt, v_ftt = load_split(bkg_dirs, FCR_REGION, TT)
    e_frest, v_frest = load_split(bkg_dirs, FCR_REGION, REST)
    # flavor-CR DATA (the CR is unblinded and signal-free; the SR stays blind).
    # The e-mu CR is recorded in both primary datasets -- keep both counts.
    fcr_data = {d: load_split(bkg_dirs, FCR_REGION, [d])
                for d in ("EGamma", "Muon")}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fout = ROOT.TFile(str(args.output_dir / f"{tag}_fcr.root"), "RECREATE")
    for name, (e, v) in [("tt_native", (e_tt, v_tt)),
                         ("rest_native", (e_rest, v_rest)),
                         ("fcr_tt_native", (e_ftt, v_ftt)),
                         ("fcr_rest_native", (e_frest, v_frest))]:
        to_th1(name, e, v).Write()
    fout.Close()

    out = {"era": ERA, "bkg_dir": BKG_DIR, "fcr_region": FCR_REGION,
           "tt_samples": TT, "rest_samples": REST, "masses": {}}
    for mass_s, m in sorted(geo["masses"].items(), key=lambda kv: int(kv[0])):
        m_c = m["m_c"]
        entry = {"m_c": m_c, "sigma": m["sigma"], "vars": {}}
        for key, v in m["vars"].items():
            se, sv_tt = rebin_snap(e_tt, v_tt, v["bw"], v["fit_lo"], v["fit_hi"])
            _, sv_rest = rebin_snap(e_rest, v_rest, v["bw"],
                                    v["fit_lo"], v["fit_hi"])
            _, sv_ftt = rebin_snap(e_ftt, v_ftt, v["bw"],
                                   v["fit_lo"], v["fit_hi"])
            _, sv_frest = rebin_snap(e_frest, v_frest, v["bw"],
                                     v["fit_lo"], v["fit_hi"])
            b_tt, b_tt_err = fit_tt_slope(se, sv_tt, m_c)
            entry["vars"][key] = {
                "T_tt": float(sv_tt.sum()), "B_rest": float(sv_rest.sum()),
                "T_cr": float(sv_ftt.sum()), "C_other": float(sv_frest.sum()),
                "b_tt": b_tt, "b_tt_err": b_tt_err,
                "slo": float(se[0]), "shi": float(se[-1]),
            }
            for dname, (ed, vd) in fcr_data.items():
                _, svd = rebin_snap(ed, vd, v["bw"], v["fit_lo"], v["fit_hi"])
                entry["vars"][key][f"N_data_{dname}"] = float(svd.sum())
        vk = entry["vars"].get("k5_bw50") or next(iter(entry["vars"].values()))
        logger.info("m=%s: T_tt=%.1f B_rest=%.1f | T_cr=%.1f C_other=%.1f "
                    "purity=%.2f b_tt=%.3f  (k5_bw50)", mass_s, vk["T_tt"],
                    vk["B_rest"], vk["T_cr"], vk["C_other"],
                    vk["T_cr"] / max(vk["T_cr"] + vk["C_other"], 1e-9),
                    vk["b_tt"])
        out["masses"][mass_s] = entry

    with open(args.output_dir / f"{tag}_fcr.json", "w") as fh:
        json.dump(out, fh, indent=1)
    logger.info("wrote %s", args.output_dir / f"{tag}_fcr.json")


if __name__ == "__main__":
    main()
