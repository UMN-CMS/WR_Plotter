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
sys.path.insert(0, str(SIGFIT / "shared"))

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


def transfer_factor_systs(bkg_dirs, sr, exclude=("muscale", "musmear")):
    """R = N_SR/N_CR of tt+tW (full spectrum) per systematic variation.
    Returns (R_nom, {source: dict(up, down, sr_up, sr_down, cr_up, cr_down)},
    dR_total): per source the larger |R_var - R_nom| of the two directions,
    added in quadrature. A source absent in one region (e.g. muon scale
    factors in the ee SR) varies the other region only. Lumi cancels exactly;
    the muon scale/smear placeholders are excluded. The source list is the
    plotter's systematics.yaml plus the electron ID/trigger scale factors the
    analyzer also provides."""
    from syst_band import window_yield_variations
    from wrplotter.config import load_systematics
    sources = list(load_systematics())
    for extra in ("electronidsf", "electrontrigsf", "looseelectronrecosf"):
        if extra not in sources:
            sources.append(extra)
    sources = [x for x in sources if x not in set(exclude)]
    n_sr, v_sr = window_yield_variations(bkg_dirs, sr, "mass_fourobject", TT,
                                         -1.0, 1e9, sources=sources, exclude=set())
    n_cr, v_cr = window_yield_variations(bkg_dirs, FCR_REGION, "mass_fourobject",
                                         TT, -1.0, 1e9, sources=sources, exclude=set())
    r_nom = n_sr / n_cr
    per = {}
    tot2 = 0.0
    for src in sources:
        if src not in v_sr and src not in v_cr:
            continue
        s_up, s_dn = v_sr.get(src, (n_sr, n_sr))
        c_up, c_dn = v_cr.get(src, (n_cr, n_cr))
        r_up, r_dn = s_up / c_up, s_dn / c_dn
        d = max(abs(r_up - r_nom), abs(r_dn - r_nom))
        per[src] = dict(up=r_up, down=r_dn, sr_up=s_up, sr_down=s_dn,
                        cr_up=c_up, cr_down=c_dn, in_sr=src in v_sr,
                        in_cr=src in v_cr, rel=d / r_nom)
        tot2 += d * d
        logger.info("  R syst %-20s SR %5.2f%%/%5.2f%%  CR %5.2f%%/%5.2f%%  ->  |dR|/R = %.2f%%%s",
                    src, 100 * (s_up / n_sr - 1), 100 * (s_dn / n_sr - 1),
                    100 * (c_up / n_cr - 1), 100 * (c_dn / n_cr - 1), 100 * d / r_nom,
                    "" if (src in v_sr and src in v_cr) else
                    ("  [SR only]" if src in v_sr else "  [CR only]"))
    return r_nom, per, tot2 ** 0.5


def main():
    global ERA, BKG_DIR, CHANNEL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=HERE / "inputs",
                   help="where the prepare_opt geometry json lives")
    p.add_argument("--output-dir", type=Path, default=HERE / "inputs")
    p.add_argument("--era", default=ERA)
    p.add_argument("--bkg-dir", default=BKG_DIR)
    p.add_argument("--systs", action="store_true",
                   help="also evaluate the transfer factor R under every "
                        "systematic variation (writes R_global_err, R_systs "
                        "and a per-mass R_err_rel the fit models read)")
    p.add_argument("--channel", default=CHANNEL, choices=["ee", "mumu"],
                   help="lepton channel (input/output tag <channel>_<topology>)")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    ERA, BKG_DIR = args.era, args.bkg_dir
    CHANNEL = args.channel
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
    for dname, (ed, vd) in fcr_data.items():
        to_th1(f"fcr_data_{dname}_native", ed, vd).Write()
    fout.Close()

    # global transfer factor: full-spectrum tt SR/CR ratio (all events,
    # summed). The ratio is flat vs mass -- per-window ratios agree within
    # MC statistics (see plot_transfer_ratio.py) -- so one global number
    # serves every window.
    r_global = float(v_tt.sum() / v_ftt.sum())
    logger.info("global transfer factor R = %.4f  (SR %.1f / CR %.1f)",
                r_global, v_tt.sum(), v_ftt.sum())

    out = {"era": ERA, "bkg_dir": BKG_DIR, "fcr_region": FCR_REGION,
           "tt_samples": TT, "rest_samples": REST, "R_global": r_global,
           "masses": {}}
    r_err_rel = None
    if args.systs:
        r_nom, per, dr = transfer_factor_systs(bkg_dirs, sr)
        r_err_rel = dr / r_nom
        logger.info("transfer factor R = %.4f +- %.4f (syst, %.1f%%)",
                    r_nom, dr, 100.0 * r_err_rel)
        out["R_global_err"] = dr
        out["R_err_rel"] = r_err_rel
        out["R_systs"] = per
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
            if r_err_rel is not None:
                entry["vars"][key]["R_err_rel"] = r_err_rel
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
