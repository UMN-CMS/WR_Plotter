#!/usr/bin/env python3
"""Explainer plots for the flavor-CR anchored card (`k5_bw50_fcr`).

Four figures, one idea each (running example m_WR = 2000, k5 window
[1150, 2700], 50 GeV bins -- the card binning):

  1a/1b_region   WHY e-mu isolates tt: SR (ee) and flavor CR (e-mu) drawn in
                 the analysis house style (Stage-3 build_stack + plot_data_mc:
                 config sample colors/labels, hatched MC stat band, Data/Sim.
                 ratio where unblinded), at the CARD binning (50 GeV), with
                 the fit window shaded. DY -> ee/mumu never gives e-mu and the
                 W_R signal is ee, so the e-mu selection keeps tt/tW (which
                 decays to ee : e-mu : mumu = 1 : 2 : 1) and drops the rest.
  2_model        WHAT the fit is: the SR window model split into the tt
                 component (expo, slope FIXED from MC, norm = mu_tt x T_tt)
                 and the floating rest expo, next to the one-bin CR channel
                 whose expectation mu_tt x T_cr + C_other carries the SAME
                 mu_tt -- the rateParam tie.
  3_constraint   WHERE the constraint comes from: -2 Delta lnL vs mu_tt for
                 the SR alone (tt/rest nearly degenerate -> shallow), the CR
                 bin alone (Poisson count of ~500 -> +-4.5%), and both.
  4_effect       WHAT it buys: fcr / float expected-limit ratio vs m_WR,
                 combine and the homemade card-model toys overlaid.

  python explain_fcr.py -v          (LCG_106; outputs in explain_fcr/)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
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
sys.path.insert(0, str(SIGFIT / "3_background_plots" / "cr_plots"))

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import (load_lumi, load_plot_settings,            # noqa: E402
                              index_plot_settings, get_var_cfg,
                              load_kfactors)
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402
from wrplotter.regions import regions_for_era                           # noqa: E402
from wrplotter.sample_groups import load_sample_groups                  # noqa: E402
from wrplotter.variables import build_variables                         # noqa: E402

import uproot                                                           # noqa: E402
import ROOT                                                             # noqa: E402
from card_sb_fit import CardSB, rebin_snap                              # noqa: E402
from plot_control_regions import build_stack, plot_data_mc              # noqa: E402

logger = logging.getLogger("explain_fcr")

ERA, BKG_DIR = "RunIISummer20UL18", "20260714_run2_bkgs"
MASS, VAR = 2000, "k5_bw50"
SR_REGION, FCR_REGION = "wr_ee_resolved_sr", "wr_resolved_flavor_cr"
C_TT, C_DY, C_OTH = "#e42536", "#5790fc", "#9c9ca1"   # tt / DY / NP+Other


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _cmslabel(ax, com, lumi):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=13)


def load_sample(bkg_dirs, region, sample):
    """Native 10 GeV (edges, values) for one WRAnalyzer sample, negatives
    clipped (as everywhere in the card inputs)."""
    key = f"{region}/mass_fourobject_{region}"
    edges = values = None
    for d in bkg_dirs:
        fp = d / f"WRAnalyzer_{sample}.root"
        if not fp.exists():
            continue
        h = uproot.open(fp)[key]
        e, v = h.axes[0].edges(), h.values()
        if edges is None:
            edges, values = e, v.copy()
        else:
            values += v
    values = np.where(np.isfinite(values) & (values > 0), values, 0.0)
    return edges, values


# ---------------------------------------------------------------------------
# 1 -- the two regions
# ---------------------------------------------------------------------------

def plot_regions(bkg_dirs, input_lumis, v, wins, out_dir, com, lumi,
                 fcr_var=None):
    """SR (1a) and flavor CR (1b) in the analysis house style -- the Stage-3
    build_stack + plot_data_mc pipeline (config sample colors/labels, hatched
    MC stat band, Data/Sim. ratio where unblinded) -- at the CARD binning
    (50 GeV), with the snapped fit window shaded and its yields annotated.
    `wins` = {region: {label: yield}} window yields (card counting, no
    density), computed by the caller."""
    groups, order = load_sample_groups()
    all_groups = [groups[k] for k in order if k in groups]
    mc_groups = [g for g in all_groups if getattr(g, "kind", "mc") != "data"]
    scales = load_kfactors()
    name_to_var = {vv.name: vv for vv in build_variables()}
    region_cfgs, common_vars = index_plot_settings(load_plot_settings(ERA))
    variable = name_to_var["mass_fourobject"]
    rebin = max(1, round(v["bw"] / 10.0))           # card binning, not the
                                                    # plot-settings default
    regions = {}
    for r in regions_for_era(ERA):
        if r.name == SR_REGION and SR_REGION not in regions:
            regions[SR_REGION] = r
        if (r.name == FCR_REGION and (FCR_REGION not in regions
                or "egamma" in str(r.primary_dataset).lower())):
            regions[FCR_REGION] = r

    slo, shi = wins["slo"], wins["shi"]
    for suffix, rname in (("1a_region_sr", SR_REGION),
                          ("1b_region_fcr", FCR_REGION)):
        region = regions[rname]
        vcfg = get_var_cfg(region_cfgs, common_vars,
                           region.name, variable.name) or {}
        ylim = tuple(map(float, vcfg.get("ylim", (1e-2, 1e6))))
        ratio_ylim = tuple(map(float, vcfg.get("ratio_ylim", (0.5, 2.0))))
        sgs = mc_groups if rname == SR_REGION else all_groups  # SR stays blind
        stack_list, stack_colors, stack_labels, data_hist = build_stack(
            region, variable, rebin=rebin, xmax=4000.0,
            input_dirs=bkg_dirs, input_lumis=input_lumis,
            era=ERA, scales=scales, sample_groups=sgs, density=True)
        fig = plot_data_mc(
            region, variable, stack_list=stack_list,
            stack_colors=stack_colors, stack_labels=stack_labels,
            data_hist=data_hist, xlim=(800.0, 4000.0), ylim=ylim,
            ratio_ylim=ratio_ylim, lumi=lumi, com=com)
        ax = fig.axes[0]
        for a in fig.axes[:2]:
            a.axvspan(slo, shi, color="gold", alpha=0.15, zorder=0)
        y = wins[rname]
        tot = sum(y.values())
        ind = "    "
        text = f"window [{slo:.0f}, {shi:.0f}]:\n"
        if rname == FCR_REGION and fcr_var.get("N_data_EGamma"):
            nd, tc = fcr_var["N_data_EGamma"], fcr_var["T_cr"]
            co = fcr_var["C_other"]
            mu, err = (nd - co) / tc, math.sqrt(nd) / tc
            text += (f"{ind}data  {nd:.0f}\n"
                     fr"{ind}non-$t\bar{{t}}$  {y['dy'] + y['oth']:.0f}" "\n"
                     fr"{ind}$t\bar{{t}}$+tW  {y['tt']:.0f}  "
                     fr"({y['tt'] / tot:.0%})" "\n"
                     fr"$\mu_{{tt}} = \frac{{{nd:.0f} - {co:.0f}}}"
                     fr"{{{tc:.0f}}} = {mu:.3f} \pm {err:.3f}$")
        else:
            text += (fr"{ind}non-$t\bar{{t}}$  {y['dy'] + y['oth']:.0f}" "\n"
                     fr"{ind}$t\bar{{t}}$+tW  {y['tt']:.0f}  "
                     fr"({y['tt'] / tot:.0%})")
        ax.text(0.60, 0.58, text,
                transform=ax.transAxes, va="top", ha="left", fontsize=15)
        _save(fig, out_dir / suffix)


# ---------------------------------------------------------------------------
# 2 -- the two-channel model
# ---------------------------------------------------------------------------

def plot_split(sb, mu_sr, mu_cr, f, out, com, lumi):
    """2a -- ONLY the background split in the SR window: MC points + the two
    fitted components stacked like a stack plot (tt below, rest on top).
    No signal curve, no CR panel, no equations."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    prof = sb.fit_profiled(0.0, mu_sr, data_cr=mu_cr)
    mu_tt_hat, b_hat, B_hat = prof["params"]
    tt_comp = mu_tt_hat * sb.T_tt * sb.t
    rest_comp = B_hat * sb._expo_shape(b_hat)
    hep.histplot([tt_comp, rest_comp], bins=sb.edges, stack=True,
                 histtype="fill", alpha=0.8, color=[C_TT, C_DY],
                 label=[fr"$t\bar{{t}}$+tW: size = $\mu_{{tt}}\times{sb.T_tt:.0f}$,"
                        " shape frozen from MC",
                        "rest: size and shape free (as before)"], ax=ax)
    ax.errorbar(sb.centers, mu_sr, yerr=np.sqrt(mu_sr), fmt="ko", ms=5,
                label="MC (Asimov data)")
    ax.set_xlabel(r"$m_{lljj}$ [GeV]")
    ax.set_ylabel("Events / 50 GeV")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 300.0)
    ax.set_xlim(sb.lo, sb.hi)
    ax.text(0.03, 0.97,
            "ee\nResolved SR\n" f"{ERA}\n"
            fr"$m_{{W_R}}={MASS}$" "\n"
            f"window [{sb.lo:.0f}, {sb.hi:.0f}]",
            transform=ax.transAxes, va="top", fontsize=15)
    ax.legend(fontsize=14, loc="upper right")
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_cr_bin(sb, mu_cr, f, out, com, lumi):
    """2b -- the flavor-CR counting channel alone: one bin, expectation
    mu_tt x T_cr + C_other, observed count with sqrt(N)."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.bar([0], [f["T_cr"]], width=0.5, color=C_TT)
    ax.bar([0], [f["C_other"]], width=0.5, bottom=[f["T_cr"]], color=C_OTH)
    ax.errorbar([0], [mu_cr], yerr=[math.sqrt(mu_cr)], fmt="ko", ms=7,
                capsize=6)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=C_TT,
              label=fr"$t\bar{{t}}$+tW: $\mu_{{tt}}\times{f['T_cr']:.0f}$"),
        Patch(facecolor=C_OTH, label=f"other: {f['C_other']:.0f} (fixed)"),
        Line2D([0], [0], marker="o", color="k", ls="",
               label=r"observed $\pm\sqrt{N}$"),
    ]
    ax.set_xticks([0])
    ax.set_xticklabels(["flavor-CR bin\n(same mass window)"], fontsize=15)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylabel("Events")
    ax.set_ylim(0, 1.5 * mu_cr)
    rel = math.sqrt(mu_cr) / f["T_cr"]
    ax.text(0.04, 0.97,
            r"e$\mu$" "\nResolved Flavor CR\n" f"{ERA}\n"
            fr"window [{sb.lo:.0f}, {sb.hi:.0f}]",
            transform=ax.transAxes, va="top", fontsize=15)
    ax.text(0.04, 0.74,
            fr"count $\approx$ {mu_cr:.0f} $\pm$ {math.sqrt(mu_cr):.0f}" "\n"
            fr"$\Rightarrow$ tt size known to {rel:.1%}",
            transform=ax.transAxes, va="top", fontsize=15)
    ax.legend(handles=handles, fontsize=13, loc="upper right",
              bbox_to_anchor=(0.97, 0.97))
    _cmslabel(ax, com, lumi)
    _save(fig, out)


# ---------------------------------------------------------------------------
# 3 -- where the constraint comes from
# ---------------------------------------------------------------------------

def profile_mu_tt(sb, mu_sr, mu_cr, mu_grid, use_sr=True, use_cr=True):
    """-2 Delta lnL vs mu_tt with (r, b_rest, B_rest) profiled at each point
    (only the terms selected)."""
    def nll_at(mu_tt):
        if not use_sr:
            vc = mu_tt * sb.T_cr + sb.C_other
            return 2.0 * (vc - mu_cr * math.log(vc))

        def fcn(p):
            v = sb.nll2(p[0], (mu_tt, p[1], p[2]), mu_sr,
                        data_cr=mu_cr if use_cr else None) \
                if use_cr else sb.nll2(p[0], (mu_tt, p[1], p[2]), mu_sr,
                                       data_cr=None)
            return v if math.isfinite(v) else 1e30
        ntot = float(mu_sr.sum())
        _, _, nll, _ = sb._minimize(
            fcn, [0.0, -3.0, max(ntot - sb.T_tt, 1.0)],
            [(-2000.0, 2000.0), (-100.0, 0.0), (0.0, 10.0 * ntot + 50.0)])
        return nll
    vals = np.array([nll_at(m) for m in mu_grid])
    return vals - vals.min()


def plot_constraint(sb, mu_sr, mu_cr, out, com, lumi):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    grid = np.linspace(0.55, 1.45, 61)
    # SR-only: drop the CR term (mu_tt only identified by the tt-vs-rest
    # slope difference -> nearly flat)
    sb_nocr = CardSB(sb.edges, sb.m_c, sb.sigma, mode="fcr",
                     fcr=dict(T_tt=sb.T_tt, T_cr=sb.T_cr,
                              C_other=sb.C_other, b_tt=sb.b_tt))
    # combined first (wide black), CR-alone on top (thin red dots): the two
    # coincide -- seeing red on black IS the message
    curves = [
        (profile_mu_tt(sb, mu_sr, mu_cr, grid, use_sr=True, use_cr=True),
         "SR + flavor CR (the fcr card)", "black", "-", 3.2),
        (profile_mu_tt(sb, mu_sr, mu_cr, grid, use_sr=False, use_cr=True),
         "flavor-CR bin alone", C_TT, ":", 2.0),
        (profile_mu_tt(sb_nocr, mu_sr, mu_cr, grid, use_sr=True, use_cr=False),
         "SR alone (r, rest norm+slope profiled)", C_DY, "--", 2.2),
    ]
    for y, lab, col, ls, lw in curves:
        ax.plot(grid, y, color=col, ls=ls, lw=lw, label=lab)
    ax.axhline(1.0, color="grey", lw=0.9, ls="-.")
    ax.text(grid[1], 1.06, r"$-2\Delta\ln L = 1$  ($\pm1\sigma$)",
            color="grey", fontsize=12)
    ax.set_xlabel(r"$\mu_{tt}$ (tt+tW normalization scale)")
    ax.set_ylabel(r"$-2\,\Delta\ln L$")
    ax.set_ylim(0, 8)
    ax.legend(fontsize=13, loc="center right")
    ax.text(0.03, 0.97,
            r"ee Resolved SR + e$\mu$ Flavor CR" "\n"
            f"{ERA}\n"
            fr"$m_{{W_R}}={MASS}$, window [{sb.lo:.0f}, {sb.hi:.0f}]" "\n"
            r"the SR can barely tell $t\bar{t}$ from DY" "\n"
            r"$\Rightarrow$ the constraint IS the CR" "\n"
            "(black = red: adding the SR\n"
            r"changes nothing about $\mu_{tt}$)",
            transform=ax.transAxes, va="top", fontsize=14)
    _cmslabel(ax, com, lumi)
    _save(fig, out)


# ---------------------------------------------------------------------------
# 4 -- what it buys
# ---------------------------------------------------------------------------

def plot_effect(out, com, lumi):
    res = HERE / "results" / "ee_resolved"

    def med(lab, m):
        try:
            with uproot.open(
                    res / f"higgsCombine_{lab}.AsymptoticLimits.mH{m}.root") as fh:
                t = fh["limit"]
                for li, qi in zip(t["limit"].array(library="np"),
                                  t["quantileExpected"].array(library="np")):
                    if abs(qi - 0.5) < 1e-3:
                        return float(li)
        except Exception:
            return None

    import csv
    hm = {}
    tab = (SIGFIT / "7_limit_plots" / "run2" / "card"
           / "expected_limit_table_ee_resolved.csv")
    if tab.exists():
        with open(tab, newline="") as fh:
            for r in csv.DictReader(fh):
                hm[(r["function"], int(float(r["mWR"])))] = float(r["ul_med"])

    masses = list(range(1400, 3400, 200))
    mc, rc = [], []
    for m in masses:
        fl, fc = med("k5_bw50_float", m), med("k5_bw50_fcr", m)
        if fl and fc:
            mc.append(m)
            rc.append(fc / fl)
    mh, rh = [], []
    for m in masses:
        if m > 2800:            # runaway-tail RMS points, ratio meaningless
            continue
        a, b = hm.get(("card_float", m)), hm.get(("card_fcr", m))
        if a and b:
            mh.append(m)
            rh.append(b / a)

    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhline(1.0, color="grey", lw=1.0, ls="--")
    ax.plot(mc, rc, "o-", color="black", lw=1.6, ms=5,
            label="combine AsymptoticLimits")
    ax.plot(mh, rh, "s--", color=C_TT, lw=1.4, ms=5,
            label="homemade card-model toys")
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel("expected limit ratio  fcr / float")
    ax.set_ylim(0.85, 1.10)
    ax.legend(fontsize=13, loc="lower right")
    ax.text(0.03, 0.97,
            "ee resolved, k5 window, 50 GeV bins\n"
            r"gain where background is large ($t\bar{t}$-rich, low mass);"
            "\nstatistics dominate above ~2400",
            transform=ax.transAxes, va="top", fontsize=13)
    _cmslabel(ax, com, lumi)
    _save(fig, out)


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=HERE / "explain_fcr")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    info = load_lumi(ERA)
    lumi, com = info["lumi"], info.get("com", 13.6)

    with open(HERE / "inputs" / "ee_resolved.json") as fh:
        meta = json.load(fh)["masses"][str(MASS)]
    with open(HERE / "inputs" / "ee_resolved_fcr.json") as fh:
        f = json.load(fh)["masses"][str(MASS)]["vars"][VAR]
    v = meta["vars"][VAR]

    bkg_dirs, input_lumis = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    h = uproot.open(HERE / "inputs" / "ee_resolved.root")["bkg_native"]
    edges, vals = rebin_snap(h.axes[0].edges(), h.values(), v["bw"],
                             v["fit_lo"], v["fit_hi"])
    mu_sr = np.where(np.isfinite(vals) & (vals > 0), vals, 0.0)
    mu_cr = f["T_cr"] + f["C_other"]
    sb = CardSB(edges, meta["m_c"], meta["sigma"], mode="fcr", fcr=f)

    # analysis sample-group colors, so every figure speaks the house palette
    global C_TT, C_DY
    groups, _ = load_sample_groups()
    for g in groups.values():
        if "tt_tW" in getattr(g, "samples", ()):
            C_TT = getattr(g, "color", C_TT)
        if "DYJets" in getattr(g, "samples", ()):
            C_DY = getattr(g, "color", C_DY)

    # window yields in card counting (raw sums, exactly the T_tt/B_rest logic)
    wins = {"slo": float(edges[0]), "shi": float(edges[-1])}
    for rname in (SR_REGION, FCR_REGION):
        y = {}
        for lab, samples in (("tt", ["tt_tW"]), ("dy", ["DYJets"]),
                             ("oth", ["Nonprompt", "Other"])):
            tot = 0.0
            for s in samples:
                e, val = load_sample(bkg_dirs, rname, s)
                _, vw = rebin_snap(e, val, v["bw"], v["fit_lo"], v["fit_hi"])
                tot += float(vw.sum())
            y[lab] = tot
        wins[rname] = y

    out = args.output_dir
    plot_regions(bkg_dirs, input_lumis, v, wins, out, com, lumi, fcr_var=f)
    logger.info("1a/1b regions done")
    plot_split(sb, mu_sr, mu_cr, f, out / "2a_sr_split", com, lumi)
    plot_cr_bin(sb, mu_cr, f, out / "2b_cr_bin", com, lumi)
    logger.info("2a/2b model done")
    plot_constraint(sb, mu_sr, mu_cr, out / "3_constraint", com, lumi)
    logger.info("3_constraint done")
    plot_effect(out / "4_effect", com, lumi)
    logger.info("4_effect done -> %s", out)


if __name__ == "__main__":
    main()
