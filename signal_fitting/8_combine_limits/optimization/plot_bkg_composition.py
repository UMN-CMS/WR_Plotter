#!/usr/bin/env python3
"""Background composition vs m_lljj: MC vs the fcr_shape fit.

Question: does the CR-anchored fit split the SR background into tt+tW and
"rest" (DY + Nonprompt + Other) the same way the MC does, at low AND high
mass? Three figures in explain_fcr/:

  composition_mc        tt+tW / rest from the ee-resolved SR MC, 100 GeV
                        bins, sumw2 errors
  composition_fit       the same ratio from the fcr_shape fit, one curve per
                        card window: R T e_i(b_tt) / B f_i(b) across the
                        window bins. Every input is the official 2018 MC
                        (20260714_run2_bkgs): SR = MC Asimov, CR = MC Asimov
                        (tt+tW + rest in the flavor CR) -- a closure test of
                        the fit's decomposition. --cr data swaps the CR
                        channel for the EGamma data spectrum instead.
  composition_mc_vs_fit both on one axis + a lower panel with the
                        window-integrated fit/MC ratio per mass point
  composition_windows   one point per card window: tt+tW / rest integrated
                        over the window, fit vs MC (same bins), x-bars =
                        window extent; lower panel fit/MC

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python \
      plot_bkg_composition.py -v
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent          # 8_combine_limits/optimization
SIGFIT = HERE.parents[1]
sys.path.insert(0, str(SIGFIT.parent))          # repo root
sys.path.insert(0, str(SIGFIT / "shared"))      # card_sb_fit

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402

import uproot                                                           # noqa: E402
import ROOT                                                             # noqa: E402
from card_sb_fit import CardSB, rebin_snap                              # noqa: E402

ROOT.gROOT.SetBatch(True)
logger = logging.getLogger("plot_bkg_composition")

ERA, BKG_DIR, VAR = "RunIISummer20UL18", "20260714_run2_bkgs", "k5_bw50"
SR_REGION = "wr_ee_resolved_sr"
TT, REST = ["tt_tW"], ["DYJets", "Nonprompt", "Other"]
C_MC, C_FIT = "#9c9ca1", "#e42536"
MC_BW, MC_RANGE = 100.0, (800.0, 3600.0)


def load(bkg_dirs, region, samples):
    """Summed native (edges, values, variances) over the sample list."""
    key = f"{region}/mass_fourobject_{region}"
    edges = values = w2 = None
    for s in samples:
        for d in bkg_dirs:
            fp = d / f"WRAnalyzer_{s}.root"
            if not fp.exists():
                continue
            h = uproot.open(fp)[key]
            e, v, var = h.axes[0].edges(), h.values(), h.variances()
            if edges is None:
                edges, values, w2 = e, v.copy(), var.copy()
            else:
                values += v
                w2 += var
    keep = np.isfinite(values) & (values > 0)
    return edges, np.where(keep, values, 0.0), np.where(keep, w2, 0.0)


def rebin(edges, vals, bw, lo, hi):
    f = max(1, round(bw / float(edges[1] - edges[0])))
    n = (len(vals) // f) * f
    v = vals[:n].reshape(-1, f).sum(axis=1)
    e = edges[0:n + 1:f]
    c = 0.5 * (e[:-1] + e[1:])
    sel = (c >= lo) & (c <= hi)
    return c[sel], v[sel]


def mc_ratio(bkg_dirs):
    e, t, t2 = load(bkg_dirs, SR_REGION, TT)
    _, o, o2 = load(bkg_dirs, SR_REGION, REST)
    c, T = rebin(e, t, MC_BW, *MC_RANGE)
    _, T2 = rebin(e, t2, MC_BW, *MC_RANGE)
    _, O = rebin(e, o, MC_BW, *MC_RANGE)
    _, O2 = rebin(e, o2, MC_BW, *MC_RANGE)
    good = (T > 0) & (O > 0)
    r = np.where(good, T / np.where(good, O, 1), np.nan)
    err = r * np.sqrt(np.where(good, T2 / T**2 + O2 / O**2, 0))
    return c, r, err


def fit_windows(bkg_dirs, cr="mc"):
    """Run the fcr_shape fit (r profiled at 0) per card mass; return per-mass
    dicts with the bin-wise fit ratio and the window-integrated numbers."""
    with open(HERE / "inputs" / "ee_resolved.json") as fh:
        meta_all = json.load(fh)["masses"]
    with open(HERE / "inputs" / "ee_resolved_fcr.json") as fh:
        fcr_all = json.load(fh)
    r_global = fcr_all["R_global"]
    hsr = uproot.open(HERE / "inputs" / "ee_resolved.root")["bkg_native"]
    hcr = uproot.open(HERE / "inputs" / "ee_resolved_fcr.root")
    e_tt, v_tt, _ = load(bkg_dirs, SR_REGION, TT)
    _, v_rest, _ = load(bkg_dirs, SR_REGION, REST)

    out = []
    for mass, meta in sorted(meta_all.items(), key=lambda kv: int(kv[0])):
        if str(mass) not in fcr_all["masses"]:
            continue
        v = meta["vars"][VAR]
        f = fcr_all["masses"][str(mass)]["vars"][VAR]
        edges, vals = rebin_snap(hsr.axes[0].edges(), hsr.values(), v["bw"],
                                 v["fit_lo"], v["fit_hi"])
        mu_sr = np.where(np.isfinite(vals) & (vals > 0), vals, 0.0)
        _, cr_other = rebin_snap(hcr["fcr_rest_native"].axes[0].edges(),
                                 hcr["fcr_rest_native"].values(),
                                 v["bw"], v["fit_lo"], v["fit_hi"])
        if cr == "data":
            _, cr_data = rebin_snap(
                hcr["fcr_data_EGamma_native"].axes[0].edges(),
                hcr["fcr_data_EGamma_native"].values(),
                v["bw"], v["fit_lo"], v["fit_hi"])
        else:   # MC Asimov CR: tt+tW + rest from the same official MC files
            _, cr_tt = rebin_snap(hcr["fcr_tt_native"].axes[0].edges(),
                                  hcr["fcr_tt_native"].values(),
                                  v["bw"], v["fit_lo"], v["fit_hi"])
            cr_data = cr_tt + cr_other
        sb = CardSB(edges, meta["m_c"], meta["sigma"], mode="fcr_shape",
                    fcr=dict(R=r_global, T_cr=f["T_cr"], b_tt=f["b_tt"],
                             cr_other=cr_other))
        res = sb.fit_profiled(0.0, mu_sr, data_cr=cr_data)
        T, b_tt, b, B = res["params"]
        eT, _, _, eB = res["perr"]
        tt_i = r_global * T * sb._expo_shape(b_tt)
        rest_i = B * sb._expo_shape(b)
        # MC in the same window bins (card counting)
        _, tt_mc = rebin_snap(e_tt, v_tt, v["bw"], v["fit_lo"], v["fit_hi"])
        _, rest_mc = rebin_snap(e_tt, v_rest, v["bw"], v["fit_lo"], v["fit_hi"])
        fit_int = r_global * T / B
        mc_int = float(tt_mc.sum() / rest_mc.sum())
        out.append(dict(mass=int(mass), m_c=meta["m_c"], centers=sb.centers,
                        tt_i=tt_i, rest_i=rest_i, eB=eB,
                        ratio=tt_i / rest_i, fit_int=fit_int,
                        fit_int_err=fit_int * np.hypot(eT / T, eB / B),
                        mc_int=mc_int, T=T, eT=eT, b_tt=b_tt, b=b, B=B,
                        status=res["status"]))
        logger.info("m=%s: T=%.1f+-%.1f b_tt=%.2f b=%.2f B=%.1f | tt/rest "
                    "window: fit %.3f  MC %.3f  (status %d)", mass, T, eT,
                    b_tt, b, B, fit_int, mc_int, res["status"])
    return out


def style(ax, com, lumi, ylab=r"$N_{t\bar{t}+tW}\,/\,N_{\mathrm{rest}}$"):
    ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    ax.set_ylabel(ylab)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=13)


def draw_mc(ax, c, r, err):
    ax.errorbar(c, r, yerr=err, fmt="o", color=C_MC, ms=5, capsize=3,
                lw=1.4, label="MC: tt+tW / (DY+Nonprompt+Other)")


def draw_fit(ax, wins, cmap):
    for i, w in enumerate(wins):
        col = cmap(i / max(1, len(wins) - 1))
        ax.plot(w["centers"], w["ratio"], color=col, lw=2.0,
                label=fr"fit, $m_{{W_R}}={w['mass']}$ window")
        ax.errorbar([w["m_c"]], [w["fit_int"]], yerr=[w["fit_int_err"]],
                    fmt="s", color=col, ms=6, capsize=3, lw=1.4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=HERE / "explain_fcr")
    p.add_argument("--cr", choices=["mc", "data"], default="mc",
                   help="flavor-CR channel: MC Asimov (default) or EGamma data")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    info = load_lumi(ERA)
    lumi, com = info["lumi"], info.get("com", 13)
    bkg_dirs, _ = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    c, r, err = mc_ratio(bkg_dirs)
    wins = fit_windows(bkg_dirs, cr=args.cr)
    cr_lab = ("CR = EGamma data" if args.cr == "data"
              else "CR = MC Asimov (same files)")
    tag = "" if args.cr == "mc" else "_crdata"
    cmap = plt.get_cmap("viridis")
    hep.style.use("CMS")
    note = ("ee resolved SR, " f"{ERA}\n"
            "rest = DY + Nonprompt + Other")

    # --- MC only -------------------------------------------------------------
    fig, ax = plt.subplots()
    draw_mc(ax, c, r, err)
    style(ax, com, lumi)
    ax.set_xlim(*MC_RANGE)
    ax.set_ylim(0, 3.0)
    ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", fontsize=14)
    ax.legend(fontsize=12, loc="upper right")
    fig.savefig(out / "composition_mc.pdf", bbox_inches="tight")
    fig.savefig(out / "composition_mc.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # --- fit only (MC faint for reference) ----------------------------------
    fig, ax = plt.subplots()
    ax.errorbar(c, r, yerr=err, fmt="o", color=C_MC, ms=4, capsize=2,
                lw=1.0, alpha=0.5, label="MC (reference)")
    draw_fit(ax, wins, cmap)
    style(ax, com, lumi)
    ax.set_xlim(*MC_RANGE)
    ax.set_ylim(0, 3.0)
    ax.text(0.03, 0.97, note + "\nfcr_shape fit per k5 window: "
            r"$R\,T\,e_i(b_{t\bar t})\,/\,B\,f_i(b)$" "\n"
            "SR = MC Asimov, " + cr_lab,
            transform=ax.transAxes, va="top", fontsize=12)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    fig.savefig(out / f"composition_fit{tag}.pdf", bbox_inches="tight")
    fig.savefig(out / f"composition_fit{tag}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # --- overlay + window-integrated fit/MC ---------------------------------
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(10, 11),
                                  gridspec_kw=dict(height_ratios=[3, 1],
                                                   hspace=0.05))
    draw_mc(ax, c, r, err)
    draw_fit(ax, wins, cmap)
    style(ax, com, lumi)
    ax.set_xlabel("")
    ax.set_ylim(0, 3.0)
    ax.text(0.03, 0.97, note + "\nSR = MC Asimov, " + cr_lab
            + "\nsquares: window-integrated fit ratio "
            r"$R\,\hat T/\hat B$", transform=ax.transAxes, va="top",
            fontsize=12)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    m = np.array([w["m_c"] for w in wins])
    q = np.array([w["fit_int"] / w["mc_int"] for w in wins])
    qe = np.array([w["fit_int_err"] / w["mc_int"] for w in wins])
    axr.errorbar(m, q, yerr=qe, fmt="s", color=C_FIT, ms=6, capsize=3, lw=1.4)
    axr.axhline(1.0, color="k", lw=1.0, ls="--")
    axr.set_ylim(0.6, 1.4)
    axr.set_ylabel("fit / MC\n(window)")
    axr.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    axr.set_xlim(*MC_RANGE)
    fig.savefig(out / f"composition_mc_vs_fit{tag}.pdf", bbox_inches="tight")
    fig.savefig(out / f"composition_mc_vs_fit{tag}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    # --- one point per window: integrated fit vs integrated MC --------------
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(10, 11),
                                  gridspec_kw=dict(height_ratios=[3, 1],
                                                   hspace=0.05))
    wlo = np.array([w["centers"][0] - 25.0 for w in wins]) / 1000
    whi = np.array([w["centers"][-1] + 25.0 for w in wins]) / 1000
    mc_i = np.array([w["mc_int"] for w in wins])
    fi = np.array([w["fit_int"] for w in wins])
    fe = np.array([w["fit_int_err"] for w in wins])
    xm = m / 1000
    ax.errorbar(xm, mc_i, xerr=[xm - wlo, whi - xm], fmt="o", color="black",
                ms=6, lw=1.6, capsize=3, label="MC, integrated over the window")
    ax.errorbar(xm, fi, yerr=fe, fmt="s", color=C_FIT, ms=6, lw=1.6, capsize=3,
                label="fcr_shape fit, integrated over the window")
    ax.axhline(1.0, color="black", ls="--", lw=1.0)
    ax.set_ylabel(r"$N_{t\bar{t}+tW}\,/\,N_{\mathrm{rest}}$")
    ax.set_ylim(0, 2.0)
    ax.text(0.03, 0.97, note + "\nSR = MC Asimov, " + cr_lab
            + "\none point per card window (x-bar = window extent)",
            transform=ax.transAxes, va="top", fontsize=12)
    ax.legend(fontsize=12, loc="upper right")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=13)
    axr.errorbar(xm, q, yerr=qe, fmt="s", color=C_FIT, ms=6, capsize=3, lw=1.4)
    axr.axhline(1.0, color="k", lw=1.0, ls="--")
    axr.set_ylim(0.6, 1.4)
    axr.set_ylabel("fit / MC")
    axr.set_xlabel(r"$m_{\ell\ell jj}$ [TeV]")
    axr.set_xlim(0.8, 3.8)
    fig.savefig(out / f"composition_windows{tag}.pdf", bbox_inches="tight")
    fig.savefig(out / f"composition_windows{tag}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("wrote composition_{mc,fit,mc_vs_fit,windows} -> %s", out)


if __name__ == "__main__":
    main()
