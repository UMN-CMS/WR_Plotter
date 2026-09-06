#!/usr/bin/env python3
"""Explainer plots for the flavor-CR anchored fit (fcr_shape, transfer-factor
form: the CR-window tt yield T and slope b_tt are measured by the binned e-mu
CR and carried into the ee SR as R*T, R = global full-spectrum SR/CR ratio).

Running example m_WR = 2000, k5 window [1150, 2700], 50 GeV bins (the card
binning). All panels show the SAME simultaneous fit (r = 0 profiled, SR
Asimov + real EGamma CR data):

  1a/1b_region   WHY e-mu isolates tt, and the fit on both canvases: SR (ee)
                 and flavor CR (e-mu) in the analysis house style at the card
                 binning, window shaded, with the fitted curves overlaid
                 (dashed = the shared tt exponential, solid = tt + rest).
                 tt/tW decays ee : e-mu : mumu = 1 : 2 : 1; DY and the W_R
                 signal never give e-mu.
  2a_sr_split    the SR window model zoomed: tt (norm and slope CR-shared) +
                 free rest expo, with the model equation and fitted values.
  2c_cr_shape    the CR channel as the fit sees it: card bins only, EGamma
                 data vs T*t_i(b_tt) + o_i, fitted T and slope vs the MC
                 expectations.
  3_constraint   WHERE the constraint comes from: -2 Delta lnL vs T for
                 SR alone (flat -- tt hides behind the free rest expo), CR
                 alone (parabola, +-5%), and both (== CR alone).
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
                 fcr_var=None, fit_overlay=None):
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
        if rname == SR_REGION:
            # tt at the visual bottom, so the stack seam matches the fit's
            # tt-vs-rest split (the dashed tt curve = top of the tt band)
            ti = next((i for i, lab in enumerate(stack_labels)
                       if "tW" in lab), None)
            if ti is not None and ti != 0:
                for lst in (stack_list, stack_colors, stack_labels):
                    lst.insert(0, lst.pop(ti))
        fig = plot_data_mc(
            region, variable, stack_list=stack_list,
            stack_colors=stack_colors, stack_labels=stack_labels,
            data_hist=data_hist, xlim=(800.0, 4000.0), ylim=ylim,
            ratio_ylim=ratio_ylim, lumi=lumi, com=com)
        ax = fig.axes[0]
        for a in fig.axes[:2]:
            a.axvspan(slo, shi, color="gold", alpha=0.15, zorder=0)
        if fit_overlay:
            # fcr_shape fit curves, drawn ONLY inside the window where the
            # fit and its normalizations are defined
            fo = fit_overlay
            xs = np.linspace(slo, shi, 300)

            def fitcurve(norm, slope):
                k = slope / 1000.0
                cn = (math.exp(k * (shi - fo["m_c"]))
                      - math.exp(k * (slo - fo["m_c"]))) / k
                return norm * np.exp(k * (xs - fo["m_c"])) * fo["bw"] / cn

            if rname == FCR_REGION:
                # tt component alone (dashed, matching the SR panel's tt
                # line): sits ~10% below the total stack (non-tt not
                # included)
                ax.plot(xs, fitcurve(fo["T"], fo["b_tt"]),
                        color="black", lw=1.8, ls="--", zorder=6)
            else:
                y_tt = fitcurve(fo["R"] * fo["T"], fo["b_tt"])
                ax.plot(xs, y_tt + fitcurve(fo["B"], fo["b"]),
                        color="black", lw=2.2, zorder=6)
                ax.plot(xs, y_tt, color="black", lw=1.8, ls="--", zorder=6)
        y = wins[rname]
        tot = sum(y.values())
        ind = "    "
        text = f"window [{slo:.0f}, {shi:.0f}]:\n"
        if rname == FCR_REGION and fcr_var.get("N_data_EGamma"):
            nd = fcr_var["N_data_EGamma"]
            text += (f"{ind}data  {nd:.0f}\n"
                     fr"{ind}non-$t\bar{{t}}$  {y['dy'] + y['oth']:.0f}" "\n"
                     fr"{ind}$t\bar{{t}}$+tW  {y['tt']:.0f}  "
                     fr"({y['tt'] / tot:.0%})")
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

def plot_split(sb, mu_sr, cr_data, f, out, com, lumi):
    """2a -- ONLY the background split in the SR window: MC points + the two
    fitted components stacked like a stack plot (tt below, rest on top).
    fcr_shape model: the tt slope b_tt is fit, shared with the CR channel."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    prof = sb.fit_profiled(0.0, mu_sr, data_cr=cr_data)
    T_hat, b_tt_hat, b_hat, B_hat = prof["params"]
    T_err, b_tt_err, b_err, B_err = prof["perr"]
    tt_comp = sb.R * T_hat * sb._expo_shape(b_tt_hat)
    rest_comp = B_hat * sb._expo_shape(b_hat)
    hep.histplot([tt_comp, rest_comp], bins=sb.edges, stack=True,
                 histtype="fill", alpha=0.8, color=[C_TT, C_DY],
                 label=[fr"$t\bar{{t}}$+tW: size = $R\times T$,"
                        " slope shared with CR",
                        "rest: size and shape free (as before)"], ax=ax)
    ax.errorbar(sb.centers, mu_sr, yerr=np.sqrt(mu_sr), fmt="ko", ms=5,
                label="MC")

    # the actual fitted function, drawn continuously (both components are
    # recentered expos, so the smooth curve is exact, not an interpolation)
    xs = np.linspace(sb.lo, sb.hi, 400)

    def curve(norm, b):
        k = b / 1000.0
        cn = ((math.exp(k * (sb.hi - sb.m_c))
               - math.exp(k * (sb.lo - sb.m_c))) / k) if abs(b) > 1e-9 \
            else (sb.hi - sb.lo)
        return norm * np.exp(k * (xs - sb.m_c)) * sb.width / cn

    ax.plot(xs, curve(sb.R * T_hat, b_tt_hat) + curve(B_hat, b_hat),
            color="black", lw=2.0, label="total fit ($r = 0$)")
    ax.plot(xs, curve(sb.R * T_hat, b_tt_hat),
            color="black", lw=1.8, ls="--",
            label=r"$t\bar{t}$ component (CR-shared)")
    ax.text(0.53, 0.73,
            fr"$\nu_i = r\,S_i \,+\, R\cdot T"
            r"\cdot t_i(b_{tt}) \,+\, B\,e_i(b)$" "\n"
            "     signal      tt (CR-shared)      rest (free)\n"
            fr"$R = {sb.R:.3f}$ (MC transfer, full spectrum)" "\n"
            "fitted:\n"
            fr"  $\hat{{T}} = {T_hat:.0f} \pm {T_err:.0f}$, "
            fr"$\hat{{b}}_{{tt}} = {b_tt_hat:.2f} \pm {b_tt_err:.2f}$ /TeV" "\n"
            fr"  $\hat{{B}} = {B_hat:.0f} \pm {B_err:.0f}$, "
            fr"$\hat{{b}} = {b_hat:.2f} \pm {b_err:.2f}$ /TeV",
            transform=ax.transAxes, va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black",
                      lw=1.0, alpha=0.9))
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


def plot_cr_shape(sb_shape, cr_data, f, out, com, lumi):
    """2c -- the BINNED flavor-CR channel of the fcr_shape model: EGamma data
    vs the fitted T*t_i(b_tt) + o_i, with T and the shared slope measured
    from this spectrum (r = 0 profiled, simultaneous with the SR)."""
    import card_sb_fit  # noqa: F401  (sb_shape built by caller)
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    prof = sb_shape.fit_profiled(0.0, sb_shape._mu_sr, data_cr=cr_data)
    T_hat, b_tt = prof["params"][0], prof["params"][1]
    e_T, e_btt = prof["perr"][0], prof["perr"][1]
    tt_comp = T_hat * sb_shape._expo_shape(b_tt)
    hep.histplot([tt_comp, sb_shape.cr_other], bins=sb_shape.edges, stack=True,
                 histtype="fill", alpha=0.8, color=[C_TT, C_OTH],
                 label=[fr"$t\bar{{t}}$+tW: $T\times t_i(b_{{tt}})$,"
                        " size and slope fitted",
                        f"other: {sb_shape.C_other:.0f} ev (fixed MC)"], ax=ax)
    ax.errorbar(sb_shape.centers, cr_data, yerr=np.sqrt(cr_data), fmt="ko",
                ms=5, label="EGamma data")

    xs = np.linspace(sb_shape.lo, sb_shape.hi, 400)
    k = b_tt / 1000.0
    cn = (math.exp(k * (sb_shape.hi - sb_shape.m_c))
          - math.exp(k * (sb_shape.lo - sb_shape.m_c))) / k
    ax.plot(xs, T_hat * np.exp(k * (xs - sb_shape.m_c))
            * sb_shape.width / cn,
            color="black", lw=1.8, ls="--",
            label=r"$t\bar{t}$ component (fitted slope)")

    ax.set_xlabel(r"$m_{lljj}$ [GeV]")
    ax.set_ylabel("Events / 50 GeV")
    ax.set_yscale("log")
    ax.set_ylim(0.3, 300.0)
    ax.set_xlim(sb_shape.lo, sb_shape.hi)
    ax.text(0.03, 0.97,
            r"e$\mu$" "\nResolved Flavor CR\n" f"{ERA}\n"
            f"window [{sb_shape.lo:.0f}, {sb_shape.hi:.0f}]",
            transform=ax.transAxes, va="top", fontsize=15)
    ax.text(0.55, 0.71,
            fr"$\nu_{{c,i}} = T\cdot t_i(b_{{tt}}) \,+\, o_i$" "\n"
            "fitted (shared with SR):\n"
            fr"  $\hat{{T}} = {T_hat:.0f} \pm {e_T:.0f}$ ev" "\n"
            fr"  $\hat{{b}}_{{tt}} = {b_tt:.2f} \pm {e_btt:.2f}$ /TeV" "\n"
            "MC (for comparison):\n"
            fr"  $T_{{cr}} = {sb_shape.T_cr:.0f}$ ev, "
            fr"$b_{{tt}} = {f['b_tt']:.2f} \pm {f['b_tt_err']:.2f}$ /TeV",
            transform=ax.transAxes, va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black",
                      lw=1.0, alpha=0.9))
    ax.legend(fontsize=14, loc="upper right")
    hep.cms.label(loc=0, ax=ax, data=True, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=13)
    _save(fig, out)


# ---------------------------------------------------------------------------
# 3 -- where the constraint comes from
# ---------------------------------------------------------------------------

def profile_T(sb, mu_sr, cr_data, t_grid, use_sr=True, use_cr=True):
    """-2 Delta lnL vs T (the CR-window tt yield) for the fcr_shape model,
    profiling (r, b_tt, b_rest, B_rest) at each point (terms selected)."""
    ntot = float(mu_sr.sum())

    def nll_at(T):
        if not use_sr:
            # CR channel alone: profile the shared slope b_tt
            def fcn(p):
                vc = np.clip(T * sb._expo_shape(p[0])
                             + sb.cr_other, 1e-12, None)
                v = 2.0 * float(np.sum(vc) - np.sum(cr_data * np.log(vc)))
                return v if math.isfinite(v) else 1e30
            _, _, nll, _ = sb._minimize(fcn, [-3.0], [(-100.0, 0.0)])
            return nll

        def fcn(p):
            v = sb.nll2(p[0], (T, p[1], p[2], p[3]), mu_sr,
                        data_cr=cr_data if use_cr else None)
            return v if math.isfinite(v) else 1e30
        _, _, nll, _ = sb._minimize(
            fcn, [0.0, -3.0, -3.0, max(ntot - sb.R * sb.T_cr, 1.0)],
            [(-2000.0, 2000.0), (-100.0, 0.0), (-100.0, 0.0),
             (0.0, 10.0 * ntot + 50.0)])
        return nll
    vals = np.array([nll_at(T) for T in t_grid])
    return vals - vals.min()


def plot_constraint(sb, mu_sr, cr_data, out, com, lumi):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    grid = np.linspace(0.55, 1.45, 61) * sb.T_cr
    # combined first (wide black), CR-alone on top (thin red dots): the two
    # coincide -- seeing red on black IS the message
    curves = [
        (profile_T(sb, mu_sr, cr_data, grid, use_sr=True, use_cr=True),
         "SR + flavor CR", "black", "-", 3.2),
        (profile_T(sb, mu_sr, cr_data, grid, use_sr=False, use_cr=True),
         "flavor CR alone", C_TT, ":", 2.0),
        (profile_T(sb, mu_sr, cr_data, grid, use_sr=True, use_cr=False),
         "SR alone", C_DY, "--", 2.2),
    ]
    for y, lab, col, ls, lw in curves:
        ax.plot(grid, y, color=col, ls=ls, lw=lw, label=lab)
    ax.set_xlabel(r"$T$ ($t\bar{t}$+tW yield in the CR window) [events]")
    ax.set_ylabel(r"$-2\,\Delta\ln L$")
    ax.set_ylim(0, 8)
    ax.legend(fontsize=13, loc="upper right")
    ax.text(0.03, 0.97,
            r"ee Resolved SR + e$\mu$ Flavor CR" "\n"
            f"{ERA}\n"
            fr"$m_{{W_R}}={MASS}$, window [{sb.lo:.0f}, {sb.hi:.0f}]",
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
        fcr_all = json.load(fh)
    f = fcr_all["masses"][str(MASS)]["vars"][VAR]
    r_global = fcr_all["R_global"]
    v = meta["vars"][VAR]

    bkg_dirs, input_lumis = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    h = uproot.open(HERE / "inputs" / "ee_resolved.root")["bkg_native"]
    edges, vals = rebin_snap(h.axes[0].edges(), h.values(), v["bw"],
                             v["fit_lo"], v["fit_hi"])
    mu_sr = np.where(np.isfinite(vals) & (vals > 0), vals, 0.0)

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
    # binned CR channel (fcr_shape): data spectrum + fixed non-tt template
    hcr = uproot.open(HERE / "inputs" / "ee_resolved_fcr.root")
    _, cr_data = rebin_snap(hcr["fcr_data_EGamma_native"].axes[0].edges(),
                            hcr["fcr_data_EGamma_native"].values(),
                            v["bw"], v["fit_lo"], v["fit_hi"])
    _, cr_other = rebin_snap(hcr["fcr_rest_native"].axes[0].edges(),
                             hcr["fcr_rest_native"].values(),
                             v["bw"], v["fit_lo"], v["fit_hi"])
    sb_shape = CardSB(edges, meta["m_c"], meta["sigma"], mode="fcr_shape",
                      fcr=dict(R=r_global, T_cr=f["T_cr"], b_tt=f["b_tt"],
                               cr_other=cr_other))
    sb_shape._mu_sr = mu_sr
    prof_shape = sb_shape.fit_profiled(0.0, mu_sr, data_cr=cr_data)
    fit_overlay = dict(T=prof_shape["params"][0],
                       e_T=prof_shape["perr"][0],
                       b_tt=prof_shape["params"][1],
                       e_btt=prof_shape["perr"][1],
                       b=prof_shape["params"][2],
                       B=prof_shape["params"][3],
                       R=sb_shape.R, m_c=sb_shape.m_c, bw=v["bw"])
    plot_regions(bkg_dirs, input_lumis, v, wins, out, com, lumi, fcr_var=f,
                 fit_overlay=fit_overlay)
    logger.info("1a/1b regions done")
    plot_split(sb_shape, mu_sr, cr_data, f, out / "2a_sr_split", com, lumi)
    plot_cr_shape(sb_shape, cr_data, f, out / "2c_cr_shape", com, lumi)
    logger.info("2a/2c model done")
    plot_constraint(sb_shape, mu_sr, cr_data, out / "3_constraint", com, lumi)
    logger.info("3_constraint done")
    plot_effect(out / "4_effect", com, lumi)
    logger.info("4_effect done -> %s", out)


if __name__ == "__main__":
    main()
