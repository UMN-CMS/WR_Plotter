#!/usr/bin/env python3
"""SR background fractions vs m_lljj -- the 2024 `background_fractions.py`
presentation ported to the WRAnalyzer files. MC only, full spectrum.

  top    fraction of each background (Z+jets, tt+tW, Nonprompt, Other)
         relative to the total MC, variable bins across the full spectrum
  bottom tt+tW / Z+jets from MC per bin

`--rest` collapses the four groups to two -- tt+tW vs rest (DY + Nonprompt +
Other) -- and the ratio panel becomes tt+tW / rest (output *_rest).
`--fit-mass M` overlays ONE fit (the m_WR = M card window) on the MC points:
its tt and rest integrated over each MC bin that lies fully inside the
window, drawn as open markers at the bin centres (output *_mM).
`--fit-all` does that for EVERY card window at once: each fit's per-bin
prediction is a horizontal line across the bin, coloured by m_WR (colorbar),
no error bars (output *_fitall).
`--rest --fit-only` draws the SAME axes from the fcr_shape fit instead of MC:
one point per card window (x = window centre, x-bar = window extent), top =
R T / (R T + B) and B / (R T + B), bottom = R T / B (output *_rest_fitonly).

Inputs: the official 2018 MC (20260714_run2_bkgs). Opt-in `--fit` overlays
the fcr_shape fit's tt/DY per k5 window on the ratio panel (fit DY = rest x
the MC DY share of rest in that window; --cr picks the CR channel).

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python \
      plot_bkg_fractions.py -v
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_bkg_composition import (ERA, BKG_DIR, SR_REGION, load, OUT_DEFAULT,  # noqa: E402
                                  fit_windows)
from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402
from card_sb_fit import rebin_snap                                      # noqa: E402

logger = logging.getLogger("plot_bkg_fractions")

BINS = np.array([800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000.])
GROUPS = [("Other backgrounds", ["Other"], "#9c9ca1"),
          ("Nonprompt", ["Nonprompt"], "#964a8b"),
          (r"$t\bar{t}+tW$", ["tt_tW"], "#e42536"),
          ("Z+jets", ["DYJets"], "#f89c20")]
GROUPS_REST = [("Rest (Z+jets, Nonprompt, Other)",
                ["DYJets", "Nonprompt", "Other"], "#5790fc"),
               (r"$t\bar{t}+tW$", ["tt_tW"], "#e42536")]


def rebin_var(edges, vals, new_edges):
    """Sum native bins into variable-width bins (edges must align)."""
    idx = np.searchsorted(edges, new_edges)
    return np.array([vals[idx[i]:idx[i + 1]].sum() for i in range(len(new_edges) - 1)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--fit", action="store_true",
                   help="overlay the fcr_shape fit tt/DY per window (opt-in)")
    p.add_argument("--rest", action="store_true",
                   help="two groups only: tt+tW vs rest; ratio = tt+tW / rest")
    p.add_argument("--fit-mass", type=int, default=None,
                   help="overlay one fit's per-bin prediction (e.g. 2000)")
    p.add_argument("--fit-all", action="store_true",
                   help="overlay every fit's per-bin prediction, coloured by m_WR")
    p.add_argument("--fit-only", action="store_true",
                   help="with --rest: fit fractions per window instead of MC")
    p.add_argument("--shade-window", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="shade this m_lljj window [GeV] on both panels (output suffix _win)")
    p.add_argument("--xmax", type=float, default=8000.0,
                   help="x-axis max [GeV]; the last MC bin edge is moved here")
    p.add_argument("--cr", choices=["mc", "data"], default="mc")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    info = load_lumi(ERA)
    lumi, com = info["lumi"], info.get("com", 13)
    bkg_dirs, _ = input_dirs_for_era(ERA, repo_root(), BKG_DIR)

    # --- MC fractions in the variable bins ----------------------------------
    bins = BINS.copy()
    bins[-1] = min(bins[-1], args.xmax)     # last bin ends at the axis edge
    groups = GROUPS_REST if args.rest else GROUPS
    denom_lab = groups[0][0] if args.rest else "Z+jets"
    yields, errs = {}, {}
    for lab, samples, _ in groups:
        e, v, w2 = load(bkg_dirs, SR_REGION, samples)
        yields[lab] = rebin_var(e, v, bins)
        errs[lab] = np.sqrt(rebin_var(e, w2, bins))
    total = sum(yields.values())
    total_err = np.sqrt(sum(x**2 for x in errs.values()))
    frac, frac_err = {}, {}
    for lab in yields:
        y, ye = yields[lab], errs[lab]
        frac[lab] = y / total
        frac_err[lab] = frac[lab] * np.sqrt((ye / np.where(y > 0, y, 1))**2
                                            + (total_err / total)**2)
    tt, dy = yields[r"$t\bar{t}+tW$"], yields[denom_lab]
    ratio = tt / dy
    ratio_err = ratio * np.hypot(errs[r"$t\bar{t}+tW$"] / tt, errs[denom_lab] / dy)
    centers, half = 0.5 * (bins[:-1] + bins[1:]), 0.5 * np.diff(bins)

    # --- optional fit per window: tt/DY = tt_i / (rest_i * DY share) --------
    wins = (fit_windows(bkg_dirs, cr=args.cr)
            if (args.fit or args.fit_only or args.fit_mass or args.fit_all)
            else [])
    e_dy, v_dy, _ = load(bkg_dirs, SR_REGION, ["DYJets"])
    _, v_rest, _ = load(bkg_dirs, SR_REGION, ["DYJets", "Nonprompt", "Other"])
    for w in wins:
        lo, hi = w["centers"][0] - 25.0, w["centers"][-1] + 25.0
        _, dy_w = rebin_snap(e_dy, v_dy, 50.0, lo, hi)
        _, rest_w = rebin_snap(e_dy, v_rest, 50.0, lo, hi)
        share = float(dy_w.sum() / rest_w.sum())
        w["ratio_dy"] = w["ratio"] / share
        w["fit_int_dy"] = w["fit_int"] / share
        w["fit_int_dy_err"] = w["fit_int_err"] / share
        w["mc_int_dy"] = float(rebin_snap(e_dy, load(bkg_dirs, SR_REGION, ["tt_tW"])[1],
                                          50.0, lo, hi)[1].sum() / dy_w.sum())
        logger.info("m=%d: DY share of rest %.3f | tt/DY window: fit %.3f+-%.3f  MC %.3f",
                    w["mass"], share, w["fit_int_dy"], w["fit_int_dy_err"], w["mc_int_dy"])

    # --- draw ----------------------------------------------------------------
    hep.style.use("CMS")
    fig, (ax, axr) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [5, 2]},
                                  figsize=(10, 11), sharex=True)
    if args.fit_only:
        # fit per window: a = R T (tt), b = B (rest); fractions of a+b
        xm = np.array([w["m_c"] for w in wins])
        xlo = np.array([w["centers"][0] - 25.0 for w in wins])
        xhi = np.array([w["centers"][-1] + 25.0 for w in wins])
        a = np.array([w["fit_int"] * w["B"] for w in wins])      # R T
        b = np.array([w["B"] for w in wins])
        ea = np.array([w["fit_int_err"] * w["B"] for w in wins])  # ~ R eT
        f_tt = a / (a + b)
        f_tt_err = f_tt * (1 - f_tt) * (ea / a)
        for lab, _, col in groups:
            y = f_tt if "tW" in lab else 1 - f_tt
            ax.errorbar(xm, y, yerr=f_tt_err, xerr=[xm - xlo, xhi - xm],
                        fmt="o", color=col, lw=2, ms=6, label=lab)
    else:
        for lab, _, col in groups:
            ax.errorbar(centers, frac[lab], yerr=frac_err[lab], xerr=half,
                        fmt="o", color=col, lw=2, ms=6, label=lab)
    def per_bin(w):
        wlo, whi = w["centers"][0] - 25.0, w["centers"][-1] + 25.0
        idx = [i for i in range(len(bins) - 1)
               if bins[i] >= wlo - 1e-6 and bins[i + 1] <= whi + 1e-6]
        sel = [(w["centers"] > bins[i]) & (w["centers"] < bins[i + 1]) for i in idx]
        return (idx, np.array([w["tt_i"][m].sum() for m in sel]),
                np.array([w["rest_i"][m].sum() for m in sel]))

    allfit = []
    if args.fit_all:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = Normalize(min(w["mass"] for w in wins) / 1000,
                         max(w["mass"] for w in wins) / 1000)
        cm = plt.get_cmap("viridis")
        for w in wins:
            idx, tt_b, re_b = per_bin(w)
            colw = cm(norm(w["mass"] / 1000))
            for i, t, r in zip(idx, tt_b, re_b):
                ft = t / (t + r)
                for lab, _, col in groups:
                    if "tW" in lab:
                        ax.hlines(ft, bins[i], bins[i + 1], color=colw, lw=2.2)
                    elif args.rest:
                        ax.hlines(1 - ft, bins[i], bins[i + 1], color=colw, lw=2.2)
                allfit.append((bins[i], bins[i + 1], t / r, colw))
        ax.plot([], [], color=cm(0.5), lw=2.2,
                label="fcr_shape fit per window, coloured by $m_{W_R}$")
    one = None
    if args.fit_mass:
        one = next(w for w in wins if w["mass"] == args.fit_mass)
        wlo, whi = one["centers"][0] - 25.0, one["centers"][-1] + 25.0
        inside = [i for i in range(len(bins) - 1)
                  if bins[i] >= wlo - 1e-6 and bins[i + 1] <= whi + 1e-6]
        bc = np.array([centers[i] for i in inside])
        sel = [(one["centers"] > bins[i]) & (one["centers"] < bins[i + 1])
               for i in inside]
        tt_b = np.array([one["tt_i"][m].sum() for m in sel])
        re_b = np.array([one["rest_i"][m].sum() for m in sel])
        r_tt, r_re = one["eT"] / one["T"], one["eB"] / one["B"]
        f1 = tt_b / (tt_b + re_b)
        f1_err = f1 * (1 - f1) * np.hypot(r_tt, r_re)
        rat1 = tt_b / re_b
        rat1_err = rat1 * np.hypot(r_tt, r_re)
        for lab, _, col in groups:
            if "tW" in lab:
                ax.errorbar(bc, f1, yerr=f1_err, fmt="s", mfc="white",
                            color=col, lw=2, ms=8, capsize=3, zorder=5)
            elif args.rest:
                ax.errorbar(bc, 1 - f1, yerr=f1_err, fmt="s", mfc="white",
                            color=col, lw=2, ms=8, capsize=3, zorder=5)
        ax.errorbar([], [], fmt="s", mfc="white", color="black", ms=8,
                    label=fr"fcr_shape fit, $m_{{W_R}}={args.fit_mass}$ "
                          fr"window [{wlo/1000:.2f}, {whi/1000:.2f}] TeV")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=16)
    ax.set_ylabel("Background Fraction")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(800, args.xmax)
    ax.legend(reverse=True,
              fontsize=18 if not (args.fit_mass or args.fit_all) else 14,
              loc="upper right")
    ax.text(0.05, 0.96, "$ee$", transform=ax.transAxes, fontsize=20, va="top")
    ax.text(0.05, 0.90, r"$m_{\ell\ell} > 400$ GeV", transform=ax.transAxes,
            fontsize=20, va="top")
    if args.fit_only:
        cr_lab = "CR = EGamma data" if args.cr == "data" else "CR = MC Asimov"
        ax.text(0.05, 0.84, "fcr_shape fit per window, SR = MC Asimov, "
                + cr_lab, transform=ax.transAxes, fontsize=13, va="top")

    if args.fit_only:
        axr.errorbar(xm, [w["fit_int"] for w in wins],
                     yerr=[w["fit_int_err"] for w in wins],
                     xerr=[xm - xlo, xhi - xm], fmt="o", lw=2, capsize=2,
                     color="black", label="fit")
        wins = []   # no per-window curves on top of the points
    else:
        axr.errorbar(centers, ratio, yerr=ratio_err, xerr=half, fmt="o", lw=2,
                     capsize=2, color="black", label="MC")
        if one is not None:
            axr.errorbar(bc, rat1, yerr=rat1_err, fmt="s", mfc="white",
                         color="#e42536", lw=2, ms=8, capsize=3, zorder=5)
        for lo_, hi_, rr, colw in allfit:
            axr.hlines(rr, lo_, hi_, color=colw, lw=2.2)
    cmap = plt.get_cmap("viridis")
    for i, w in enumerate(wins if args.fit else []):
        col = cmap(i / max(1, len(wins) - 1))
        axr.plot(w["centers"], w["ratio_dy"], color=col, lw=2)
        axr.errorbar([w["m_c"]], [w["fit_int_dy"]], yerr=[w["fit_int_dy_err"]],
                     fmt="s", color=col, ms=5, capsize=2, lw=1.4)
    if args.fit:
        axr.plot([], [], color=cmap(0.5), lw=2,
                 label="fcr_shape fit per window (squares: window-integrated)")
    axr.axhline(1, color="black", ls="--")
    axr.set_ylabel(r"$\frac{t\bar{t}+tW}{\mathrm{rest}}$" if args.rest
                   else r"$\frac{t\bar{t}+tW}{Z+jets}$")
    axr.set_ylim(0, 2.4)
    axr.set_yticks([0, 1, 2])
    axr.set_xlim(800, args.xmax)
    axr.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:g}"))
    axr.set_xlabel(r"$m_{\ell\ell jj}$ [TeV]")
    if args.fit:
        axr.legend(fontsize=11, loc="upper right")
        cr_lab = "CR = EGamma data" if args.cr == "data" else "CR = MC Asimov"
        axr.text(0.98, 0.72, f"SR = MC Asimov, {cr_lab}",
                 transform=axr.transAxes, fontsize=11, ha="right", va="top")
    if args.shade_window:
        for a in (ax, axr):
            a.axvspan(*args.shade_window, color="tab:red", alpha=0.12, zorder=0)
    plt.subplots_adjust(hspace=0.08)
    if args.fit_all:
        sm = ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=[ax, axr], pad=0.02, fraction=0.05)
        cb.set_label(r"fit window $m_{W_R}$ [TeV]")
    tag = ("_rest" if args.rest else "") + ("_fitall" if args.fit_all else "") + (
        f"_m{args.fit_mass}" if args.fit_mass else "") + (
        "" if not args.fit else ("_fit" if args.cr == "mc" else "_fit_crdata")) + (
        "" if not args.fit_only else ("_fitonly" if args.cr == "mc" else "_fitonly_crdata")) + (
        "_win" if args.shade_window else "")
    out = args.output_dir / f"bkg_fractions_ee{tag}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
