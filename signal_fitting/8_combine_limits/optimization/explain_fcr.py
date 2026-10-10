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
                              index_plot_settings, get_var_cfg)
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402
from wrplotter.regions import regions_for_era                           # noqa: E402
from wrplotter.sample_groups import load_sample_groups                  # noqa: E402
from wrplotter.variables import build_variables                         # noqa: E402

import uproot                                                           # noqa: E402
import ROOT                                                             # noqa: E402
from card_sb_fit import CardSB, rebin_snap                              # noqa: E402
from plot_control_regions import build_stack, plot_data_mc              # noqa: E402

logger = logging.getLogger("explain_fcr")

import os                                                               # noqa: E402
# seminar era switch (see explainer/_common.py): WR_SEMINAR_ERA=2024 moves the
# whole family onto the 2024 inputs/outputs; default stays the run2 chain
SEMINAR_ERA = os.environ.get("WR_SEMINAR_ERA", "2018")
if SEMINAR_ERA == "2024":
    ERA, BKG_DIR = "RunIII2024Summer24", "20261004_full_systs"
    INPUTS, OUT_DEFAULT = HERE / "inputs_2024", HERE / "explain_fcr_2024"
else:
    ERA, BKG_DIR = "RunIISummer20UL18", "20260714_run2_bkgs"
    INPUTS, OUT_DEFAULT = INPUTS, HERE / "explain_fcr"
MASS, VAR = 2000, "k5_bw50"
CHANNEL, TAG, CH_LABEL = "ee", "ee_resolved", "ee"
SR_REGION, FCR_REGION = "wr_ee_resolved_sr", "wr_resolved_flavor_cr"


def set_channel(channel):
    """Switch the module onto one lepton channel (ee or mumu)."""
    global CHANNEL, TAG, CH_LABEL, SR_REGION
    CHANNEL, TAG = channel, f"{channel}_resolved"
    CH_LABEL = "ee" if channel == "ee" else r"$\mu\mu$"
    SR_REGION = f"wr_{channel}_resolved_sr"
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
                 fcr_var=None, fit_overlay=None, suffix_tag="",
                 shade_window=True, tt_bottom=True):
    """SR (1a) and flavor CR (1b) in the analysis house style -- the Stage-3
    build_stack + plot_data_mc pipeline (config sample colors/labels, hatched
    MC stat band, Data/Sim. ratio where unblinded) -- at the CARD binning
    (50 GeV), with the snapped fit window shaded and its yields annotated.
    `wins` = {region: {label: yield}} window yields (card counting, no
    density), computed by the caller."""
    groups, order = load_sample_groups()
    all_groups = [groups[k] for k in order if k in groups]
    mc_groups = [g for g in all_groups if getattr(g, "kind", "mc") != "data"]
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
            era=ERA, sample_groups=sgs, density=True)
        if rname == SR_REGION and tt_bottom:
            # tt at the visual bottom, so the stack seam matches the fit's
            # tt-vs-rest split (the dashed tt curve = top of the tt band)
            ti = next((i for i, lab in enumerate(stack_labels)
                       if "tW" in lab), None)
            if ti is not None and ti != 0:
                for lst in (stack_list, stack_colors, stack_labels):
                    lst.insert(0, lst.pop(ti))
        syst_fn = None
        try:
            from syst_band import total_syst_envelope
            mc_samples = [smp for g in mc_groups for smp in g.samples]
            e_s, n_s, u_s, d_s = total_syst_envelope(
                bkg_dirs, rname, "mass_fourobject", mc_samples, rebin=rebin)
            if u_s.any():
                c_s = 0.5 * (e_s[:-1] + e_s[1:])
                ru = np.divide(u_s, n_s, out=np.zeros_like(u_s), where=n_s > 0)
                rd = np.divide(d_s, n_s, out=np.zeros_like(d_s), where=n_s > 0)
                syst_fn = (lambda c, ru=ru, rd=rd, c_s=c_s:
                           (np.interp(c, c_s, ru), np.interp(c, c_s, rd)))
        except Exception as exc:                        # pragma: no cover
            logger.warning("no systematic band for %s: %s", rname, exc)
        fig = plot_data_mc(
            region, variable, stack_list=stack_list,
            stack_colors=stack_colors, stack_labels=stack_labels,
            data_hist=data_hist, xlim=(800.0, 4000.0), ylim=ylim,
            ratio_ylim=ratio_ylim, lumi=lumi, com=com, syst_rel_fn=syst_fn)
        ax = fig.axes[0]
        if shade_window:
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
        if shade_window:
            ax.text(0.60, 0.58, text,
                    transform=ax.transAxes, va="top", ha="left", fontsize=15)
        _save(fig, out_dir / (suffix + suffix_tag))


# ---------------------------------------------------------------------------
# 2 -- the two-channel model
# ---------------------------------------------------------------------------

def _load_w2(bkg_dirs, region, samples):
    """Summed native (edges, values, variances) over a sample list."""
    key = f"{region}/mass_fourobject_{region}"
    edges = values = w2 = None
    for smp in samples:
        for d in bkg_dirs:
            fp = d / f"WRAnalyzer_{smp}.root"
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


COARSE_BINS = [800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 4500]


def plot_split(sb, mu_sr, cr_data, f, out, com, lumi, bkg_dirs=None,
               coarse=False, stat_box=True, inject=None, float_r=False,
               figsize=(10, 11.5), hspace=0.05, ymax=300.0):
    """2a -- ONLY the background split in the SR window: MC points + the two
    fitted components stacked like a stack plot (tt below, rest on top).
    fcr_shape model: the tt slope b_tt is fit, shared with the CR channel.
    With bkg_dirs, a lower panel compares tt+tW / rest per card bin: MC
    (points, sumw2 errors) vs the fitted components (curve). coarse=True
    additionally marks the composition-plot bins that lie fully inside the
    window and draws the ratio integrated over each (MC bar vs fit square)
    -- the numbers that go onto bkg_fractions_*_m<MASS>.
    inject=N: the points are MC + N signal events (N * S_i, no fluctuation)
    and the fit is the global one (r floating); the fitted signal r_hat*S_i
    is stacked on top of the two background components."""
    hep.style.use("CMS")
    if bkg_dirs is not None:
        fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=figsize,
                                      gridspec_kw=dict(height_ratios=[3, 1],
                                                       hspace=hspace))
    else:
        fig, ax = plt.subplots()
        axr = None
    if inject is None and not float_r:
        prof = sb.fit_profiled(0.0, mu_sr, data_cr=cr_data)
        points, r_hat, r_err = mu_sr, 0.0, 0.0
    elif inject is None:
        # the global fit (r floating) on the signal-free pseudo-data: the
        # same fit as the parameter table on the seminar slide
        points = mu_sr
        prof = sb.fit(points, data_cr=cr_data)
        r_hat, r_err = prof["nsig"], prof["nsig_err"]
    else:
        points = mu_sr + inject * sb.S
        prof = sb.fit(points, data_cr=cr_data)
        r_hat, r_err = prof["nsig"], prof["nsig_err"]
    T_hat, b_tt_hat, b_hat, B_hat = prof["params"][:4]
    T_err, b_tt_err, b_err, B_err = prof["perr"][:4]
    rho_hat = prof["params"][4] if sb.has_rho else 1.0
    logger.info("plot_split %s: inject=%s r=%.2f+-%.2f T=%.1f+-%.1f "
                "b_tt=%.3f+-%.3f b=%.3f+-%.3f B=%.1f+-%.1f", out.name,
                inject, r_hat, r_err, T_hat, T_err, b_tt_hat, b_tt_err,
                b_hat, b_err, B_hat, B_err)
    tt_comp = rho_hat * sb.R * T_hat * sb._expo_shape(b_tt_hat)
    rest_comp = B_hat * sb._expo_shape(b_hat)
    comps = [tt_comp, rest_comp]
    cols = [C_TT, C_DY]
    labs = [r"$t\bar{t}$+tW: size and shape sim. fit with FCR",
            "Rest: size and shape floating"]
    if inject is not None or (float_r and r_hat >= 0.0):
        comps.append(r_hat * sb.S)
        cols.append("crimson")
        labs.append(fr"Signal: $\hat r = {r_hat:.0f}$ events")
    hep.histplot(comps, bins=sb.edges, stack=True,
                 histtype="fill", alpha=0.8, color=cols, label=labs, ax=ax)
    ax.errorbar(sb.centers, points, yerr=np.sqrt(points), fmt="ko", ms=5,
                label="MC" if inject is None else f"MC + {inject:.0f} signal")

    # the actual fitted function, drawn continuously (both components are
    # recentered expos, so the smooth curve is exact, not an interpolation)
    xs = np.linspace(sb.lo, sb.hi, 400)

    def curve(norm, b):
        k = b / 1000.0
        cn = ((math.exp(k * (sb.hi - sb.m_c))
               - math.exp(k * (sb.lo - sb.m_c))) / k) if abs(b) > 1e-9 \
            else (sb.hi - sb.lo)
        return norm * np.exp(k * (xs - sb.m_c)) * sb.width / cn

    tot_curve = curve(rho_hat * sb.R * T_hat, b_tt_hat) + curve(B_hat, b_hat)
    if inject is not None or float_r:
        tot_curve = tot_curve + r_hat * sb.width / (
            math.sqrt(2 * math.pi) * sb.sigma) * np.exp(
            -0.5 * ((xs - sb.m_c) / sb.sigma) ** 2)
    if inject is None and not float_r:
        tot_label = "total fit ($r = 0$)"
    elif inject is None:
        tot_label = fr"total fit, $r$ floating: $\hat r = {r_hat:.0f} \pm {r_err:.0f}$"
    else:
        tot_label = "total fit ($r$ floating)"
    ax.plot(xs, tot_curve, color="black", lw=2.0, label=tot_label)
    ax.plot(xs, curve(rho_hat * sb.R * T_hat, b_tt_hat),
            color="black", lw=1.8, ls="--",
            label=r"$t\bar{t}$ component (sim. fit w/ FCR)")
    if stat_box:
      ax.text(0.53, 0.73,
            fr"$\nu_i = r\,S_i \,+\, R\cdot T"
            r"\cdot t_i(b_{tt}) \,+\, B\,e_i(b)$" "\n"
            "     signal      tt (sim. fit)      rest (free)",
            transform=ax.transAxes, va="top", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black",
                      lw=1.0, alpha=0.9))
    ax.set_xlabel(r"$m_{lljj}$ [GeV]")
    ax.set_ylabel("Events / 50 GeV")
    ax.set_yscale("log")
    ax.set_ylim(0.3, ymax)
    ax.set_xlim(sb.lo, sb.hi)
    ax.text(0.03, 0.97,
            f"{CH_LABEL}\nResolved SR\n" f"{ERA}\n"
            fr"$m_{{W_R}}={MASS}$" "\n"
            f"window [{sb.lo:.0f}, {sb.hi:.0f}]",
            transform=ax.transAxes, va="top", fontsize=15)
    ax.legend(fontsize=14, loc="upper right")
    _cmslabel(ax, com, lumi)
    if axr is not None:
        e, v_tt, w_tt = _load_w2(bkg_dirs, SR_REGION, ["tt_tW"])
        _, v_re, w_re = _load_w2(bkg_dirs, SR_REGION,
                                 ["DYJets", "Nonprompt", "Other"])
        bw, lo, hi = sb.width, sb.lo + 1e-3, sb.hi - 1e-3
        _, tt = rebin_snap(e, v_tt, bw, lo, hi)
        _, tt2 = rebin_snap(e, w_tt, bw, lo, hi)
        _, re = rebin_snap(e, v_re, bw, lo, hi)
        _, re2 = rebin_snap(e, w_re, bw, lo, hi)
        good = (tt > 0) & (re > 0)
        rmc = np.where(good, tt / np.where(good, re, 1), np.nan)
        rerr = rmc * np.sqrt(np.where(good, tt2 / tt**2 + re2 / re**2, 0))
        if not coarse:      # coarse: only the integrated bins in the panel
            axr.errorbar(sb.centers, rmc, yerr=rerr, fmt="ko", ms=5,
                         label="MC")
            axr.plot(xs, curve(rho_hat * sb.R * T_hat, b_tt_hat) / curve(B_hat, b_hat),
                     color="black", lw=2.0, label="fit")
        axr.axhline(1.0, color="gray", lw=1.0, ls="--")
        if coarse:
            cb = [b for b in COARSE_BINS if sb.lo - 1 <= b <= sb.hi + 1]
            for k, (a, b) in enumerate(zip(cb[:-1], cb[1:])):
                for a_ in (ax, axr):
                    a_.axvspan(a, b, color="gray",
                               alpha=0.10 if k % 2 == 0 else 0.02, lw=0)
                    a_.axvline(a, color="gray", lw=1.0, ls=":")
                m = (sb.centers > a) & (sb.centers < b)
                r_mc = tt[m].sum() / re[m].sum()
                e_mc = r_mc * np.sqrt(tt2[m].sum() / tt[m].sum()**2
                                      + re2[m].sum() / re[m].sum()**2)
                r_fit = tt_comp[m].sum() / rest_comp[m].sum()
                e_fit = r_fit * np.hypot(T_err / T_hat, B_err / B_hat)
                xc = 0.5 * (a + b)
                axr.errorbar([xc - 0.08 * (b - a)], [r_mc], yerr=[e_mc],
                             xerr=[[xc - 0.08 * (b - a) - a],
                                   [b - xc + 0.08 * (b - a)]],
                             fmt="o", color="black", ms=7, lw=2.2, capsize=3,
                             label="MC, integrated" if k == 0 else None)
                axr.errorbar([xc + 0.08 * (b - a)], [r_fit], yerr=[e_fit],
                             fmt="s", mfc="white", color=C_TT, ms=9, lw=2.2,
                             capsize=3, zorder=5,
                             label="fit, integrated" if k == 0 else None)
            for a_ in (ax, axr):
                a_.axvline(cb[-1], color="gray", lw=1.0, ls=":")
        axr.set_ylim(0, 2.6)
        axr.set_yticks([0, 1, 2])
        axr.set_ylabel(r"$t\bar{t}$+tW / rest")
        axr.set_xlabel(r"$m_{lljj}$ [GeV]")
        axr.legend(fontsize=13, loc="upper right", ncol=2)
        ax.set_xlabel("")
    _save(fig, out)


def plot_cr_shape(sb_shape, cr_data, f, out, com, lumi, other_bottom=False, float_r=False):
    """2c -- the BINNED flavor-CR channel of the fcr_shape model: EGamma data
    vs the fitted T*t_i(b_tt) + o_i, with T and the shared slope measured
    from this spectrum (r = 0 profiled, simultaneous with the SR)."""
    import card_sb_fit  # noqa: F401  (sb_shape built by caller)
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    # float_r: the global fit with r floating (as 2a_sr_split_nobox --float-r),
    # otherwise the background-only (r = 0) profile
    prof = (sb_shape.fit(sb_shape._mu_sr, data_cr=cr_data) if float_r
            else sb_shape.fit_profiled(0.0, sb_shape._mu_sr, data_cr=cr_data))
    T_hat, b_tt = prof["params"][0], prof["params"][1]
    e_T, e_btt = prof["perr"][0], prof["perr"][1]
    tt_comp = T_hat * sb_shape._expo_shape(b_tt)
    comps = [tt_comp, sb_shape.cr_other]
    cols = [C_TT, C_OTH]
    labs = [fr"$t\bar{{t}}$+tW: $T\times t_i(b_{{tt}})$," " size and slope fitted",
            f"other: {sb_shape.C_other:.0f} ev (fixed MC)"]
    if other_bottom:      # frozen non-tt at the bottom, fitted tt on top
        comps, cols, labs = comps[::-1], cols[::-1], labs[::-1]
    hep.histplot(comps, bins=sb_shape.edges, stack=True,
                 histtype="fill", alpha=0.8, color=cols, label=labs, ax=ax)
    ax.errorbar(sb_shape.centers, cr_data, yerr=np.sqrt(cr_data), fmt="ko",
                ms=5, label="EGamma data")

    xs = np.linspace(sb_shape.lo, sb_shape.hi, 400)
    k = b_tt / 1000.0
    cn = (math.exp(k * (sb_shape.hi - sb_shape.m_c))
          - math.exp(k * (sb_shape.lo - sb_shape.m_c))) / k
    if not other_bottom:
        ax.plot(xs, T_hat * np.exp(k * (xs - sb_shape.m_c))
                * sb_shape.width / cn,
                color="black", lw=1.8, ls="--",
                label=r"$t\bar{t}$ component (fitted slope)")

    ax.set_xlabel(r"$m_{lljj}$ [GeV]")
    ax.set_ylabel("Events / 50 GeV")
    ax.set_yscale("log")
    ax.set_ylim(1.0 if other_bottom else 0.3, 300.0)
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
            pars = (T, p[1], p[2], p[3]) + ((p[4],) if sb.has_rho else ())
            v = sb.nll2(p[0], pars, mu_sr,
                        data_cr=cr_data if use_cr else None)
            return v if math.isfinite(v) else 1e30
        s0 = [0.0, -3.0, -3.0, max(ntot - sb.R * sb.T_cr, 1.0)]
        lim = [(-2000.0, 2000.0), (-100.0, 0.0), (-100.0, 0.0),
               (0.0, 10.0 * ntot + 50.0)]
        if sb.has_rho:
            s0.append(1.0)
            lim.append((0.2, 3.0))
        _, _, nll, _ = sb._minimize(fcn, s0, lim)
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
            f"{CH_LABEL} Resolved SR + e$\\mu$ Flavor CR" "\n"
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

def pseudo_data_sr(inputs, v, f, tt_to_cr=False, dataset="EGamma", tag=None):
    """(edges, mu_sr, mu_tt) -- the SR pseudo-data in the snapped window.
    Default: the summed MC. tt_to_cr=True: the MC with its tt+tW part scaled
    by mu_tt = (N_data - C_other)/T_cr, i.e. to what the flavor-CR DATA says
    (what card_model_toys does for the CR-data toys); without it a CR
    data/MC offset shows up as a fake r in the anchored fit of the MC."""
    tag = tag or TAG
    h = uproot.open(inputs / f"{tag}.root")["bkg_native"]
    e10, v10 = h.axes[0].edges(), h.values()
    mu_tt = 1.0
    if tt_to_cr:
        fs = uproot.open(inputs / f"{tag}_fcr.root")
        tt10, rest10 = fs["tt_native"].values(), fs["rest_native"].values()
        mu_tt = (f[f"N_data_{dataset}"] - f["C_other"]) / f["T_cr"]
        v10 = mu_tt * tt10 + rest10
        logger.info("SR pseudo-data: MC with tt scaled by mu_tt = %.3f "
                    "(flavor-CR %s data %.0f, MC %.1f)", mu_tt, dataset,
                    f[f"N_data_{dataset}"], f["T_cr"] + f["C_other"])
    edges, vals = rebin_snap(e10, v10, v["bw"], v["fit_lo"], v["fit_hi"])
    return edges, np.where(np.isfinite(vals) & (vals > 0), vals, 0.0), mu_tt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--float-r", action="store_true",
                   help="2a_sr_split_nobox: draw the global fit (r floating) "
                        "instead of the r = 0 profile")
    p.add_argument("--inject", type=float, nargs="+", default=[10.0],
                   help="signal events injected for the 2a_sr_split_inj<N> plots")
    p.add_argument("--tt-to-cr", action="store_true",
                   help="SR pseudo-data = MC with tt+tW scaled to the "
                        "flavor-CR data (see pseudo_data_sr)")
    p.add_argument("--channel", choices=["ee", "mumu"], default="ee",
                   help="lepton channel of the signal region (inputs <channel>_resolved*)")
    p.add_argument("--cr", choices=["data", "mc"], default="data",
                   help="flavor-CR channel: EGamma data (default) or MC "
                        "Asimov (tt+tW + rest); mc writes ONLY 2a_sr_split_crmc")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    info = load_lumi(ERA)
    lumi, com = info["lumi"], info.get("com", 13.6)

    set_channel(args.channel)
    with open(INPUTS / f"{TAG}.json") as fh:
        meta = json.load(fh)["masses"][str(MASS)]
    with open(INPUTS / f"{TAG}_fcr.json") as fh:
        fcr_all = json.load(fh)
    f = fcr_all["masses"][str(MASS)]["vars"][VAR]
    r_global = fcr_all["R_global"]
    v = meta["vars"][VAR]

    bkg_dirs, input_lumis = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    edges, mu_sr, mu_tt_pd = pseudo_data_sr(INPUTS, v, f, args.tt_to_cr)

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
        if rname == SR_REGION and args.tt_to_cr:
            y["tt"] *= mu_tt_pd              # the pseudo-data's tt+tW
        wins[rname] = y

    out = args.output_dir
    # binned CR channel (fcr_shape): data spectrum + fixed non-tt template
    hcr = uproot.open(INPUTS / f"{TAG}_fcr.root")
    _, cr_data = rebin_snap(hcr["fcr_data_EGamma_native"].axes[0].edges(),
                            hcr["fcr_data_EGamma_native"].values(),
                            v["bw"], v["fit_lo"], v["fit_hi"])
    _, cr_other = rebin_snap(hcr["fcr_rest_native"].axes[0].edges(),
                             hcr["fcr_rest_native"].values(),
                             v["bw"], v["fit_lo"], v["fit_hi"])
    if args.cr == "mc":     # same official MC files in the CR channel too
        _, cr_tt = rebin_snap(hcr["fcr_tt_native"].axes[0].edges(),
                              hcr["fcr_tt_native"].values(),
                              v["bw"], v["fit_lo"], v["fit_hi"])
        cr_data = cr_tt + cr_other
    sb_shape = CardSB(edges, meta["m_c"], meta["sigma"], mode="fcr_shape",
                      fcr=dict(R=r_global, T_cr=f["T_cr"], b_tt=f["b_tt"],
                               cr_other=cr_other,
                               R_err_rel=f.get("R_err_rel", 0.0)))
    sb_shape._mu_sr = mu_sr
    if args.cr == "mc":
        plot_split(sb_shape, mu_sr, cr_data, f, out / "2a_sr_split_crmc",
                   com, lumi, bkg_dirs=bkg_dirs)
        plot_split(sb_shape, mu_sr, cr_data, f,
                   out / "2a_sr_split_crmc_noratio", com, lumi)
        plot_split(sb_shape, mu_sr, cr_data, f,
                   out / "2a_sr_split_crmc_bins", com, lumi,
                   bkg_dirs=bkg_dirs, coarse=True)
        logger.info("2a (CR = MC Asimov) done -> %s", out)
        return
    prof_shape = sb_shape.fit_profiled(0.0, mu_sr, data_cr=cr_data)
    fit_overlay = dict(T=prof_shape["params"][0],
                       e_T=prof_shape["perr"][0],
                       b_tt=prof_shape["params"][1],
                       e_btt=prof_shape["perr"][1],
                       b=prof_shape["params"][2],
                       B=prof_shape["params"][3],
                       R=sb_shape.R * (prof_shape["params"][4]
                                       if sb_shape.has_rho else 1.0),
                       m_c=sb_shape.m_c, bw=v["bw"])
    plot_regions(bkg_dirs, input_lumis, v, wins, out, com, lumi, fcr_var=f,
                 fit_overlay=fit_overlay)
    # seminar variant: the two region plots with the window but no fit curves
    plot_regions(bkg_dirs, input_lumis, v, wins, out, com, lumi, fcr_var=f,
                 fit_overlay=None, suffix_tag="_nofit")
    # and with no window and the usual stack order (global-R slide)
    plot_regions(bkg_dirs, input_lumis, v, wins, out, com, lumi, fcr_var=f,
                 fit_overlay=None, suffix_tag="_plain", shade_window=False, tt_bottom=False)
    # same, with the fitted tt exponential drawn inside the window
    plot_regions(bkg_dirs, input_lumis, v, wins, out, com, lumi, fcr_var=f,
                 fit_overlay=fit_overlay, suffix_tag="_plainfit", shade_window=False, tt_bottom=False)
    if args.float_r:
        # the same plot from the r-floating two-region fit, to sit beside
        # 2a_sr_split_nobox(_sq) --float-r: one likelihood, one fit
        glob_shape = sb_shape.fit(mu_sr, data_cr=cr_data)
        fit_overlay_f = dict(fit_overlay,
                             T=glob_shape["params"][0], e_T=glob_shape["perr"][0],
                             b_tt=glob_shape["params"][1], e_btt=glob_shape["perr"][1],
                             b=glob_shape["params"][2], B=glob_shape["params"][3],
                             R=sb_shape.R * (glob_shape["params"][4]
                                             if sb_shape.has_rho else 1.0))
        logger.info("plainfit_floatr: r = %.2f, T = %.1f, b_tt = %.3f (r = 0 profile: T = %.1f, b_tt = %.3f)",
                    glob_shape["nsig"], fit_overlay_f["T"], fit_overlay_f["b_tt"],
                    fit_overlay["T"], fit_overlay["b_tt"])
        plot_regions(bkg_dirs, input_lumis, v, wins, out, com, lumi, fcr_var=f,
                     fit_overlay=fit_overlay_f, suffix_tag="_plainfit_floatr",
                     shade_window=False, tt_bottom=False)
    logger.info("1a/1b regions done")
    plot_split(sb_shape, mu_sr, cr_data, f, out / "2a_sr_split", com, lumi,
               bkg_dirs=bkg_dirs)
    plot_split(sb_shape, mu_sr, cr_data, f, out / "2a_sr_split_nobox", com, lumi,
               bkg_dirs=bkg_dirs, stat_box=False, float_r=args.float_r)
    # same figure on the canvas of the control-region plots (plot_data_mc: 10 x 10,
    # 3:1 panels, hspace 0.1), to sit beside 1b_region_fcr_plainfit at equal size
    plot_split(sb_shape, mu_sr, cr_data, f, out / "2a_sr_split_nobox_sq", com, lumi,
               bkg_dirs=bkg_dirs, stat_box=False, float_r=args.float_r,
               figsize=(10.35, 10), hspace=0.1, ymax=3000.0)  # headroom: curves clear the labels
    # seminar: same, with N signal events added to the pseudo-data, r floating
    for n_inj in args.inject:
        plot_split(sb_shape, mu_sr, cr_data, f, out / f"2a_sr_split_inj{n_inj:.0f}",
                   com, lumi, stat_box=False, inject=float(n_inj), ymax=3000.0)
    plot_cr_shape(sb_shape, cr_data, f, out / "2c_cr_shape", com, lumi)
    if args.float_r:     # the CR half of the same r-floating fit as 2a_sr_split_nobox
        plot_cr_shape(sb_shape, cr_data, f, out / "2c_cr_shape_floatr", com, lumi,
                      float_r=True)
        plot_cr_shape(sb_shape, cr_data, f, out / "2c_cr_shape_otherbottom_floatr", com,
                      lumi, other_bottom=True, float_r=True)
    plot_cr_shape(sb_shape, cr_data, f, out / "2c_cr_shape_otherbottom", com, lumi,
                  other_bottom=True)
    logger.info("2a/2c model done")
    plot_constraint(sb_shape, mu_sr, cr_data, out / "3_constraint", com, lumi)
    logger.info("3_constraint done")
    try:
        plot_effect(out / "4_effect", com, lumi)
        logger.info("4_effect done -> %s", out)
    except Exception as exc:                            # 2024: no scan yet
        logger.warning("4_effect skipped: %s", exc)


if __name__ == "__main__":
    main()
