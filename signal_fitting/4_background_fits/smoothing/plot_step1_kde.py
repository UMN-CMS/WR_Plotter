#!/usr/bin/env python3
"""Step 1 -- the binned MC histogram with the RooKeysPdf smoothing over it.

One plot per background component and one for the total (50 GeV bins, MC stat
error bars, the RooKeysPdf expectation as a line, a MC / smooth ratio panel
underneath; common x/y ranges), plus
the usual CMS-style stacked MC histogram of the components with the
RooKeysPdf sum drawn over it, and `combined`: every per-process curve
together with their sum over the binned total MC. Reads run2/ee_resolved/smoothed_kde.root from
smooth_background.py.

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python plot_step1_kde.py
    -> run2/ee_resolved/plots/step1/{tt_tW,DYJets,Nonprompt,Other,total,stack}.{png,pdf}
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
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import smooth_lib as sl                                                  # noqa: E402
from smooth_lib import (COMP_LABEL, COMP_COLOR, COMP_LS, STACK_ORDER,     # noqa: E402
                        TOTAL_LABEL, centers_of, rebin)
from smooth_background import XMIN, _cms, _save                            # noqa: E402
from plot_mc_backgrounds import draw_stat_band                            # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter              # noqa: E402

logger = logging.getLogger("plot_step1_kde")
KDE_COLOR = "#e42536"
RATIO_YLIM = (0.0, 2.5)


def _figure():
    hep.style.use("CMS")
    return plt.subplots(2, 1, sharex=True, figsize=(10, 10),
                        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})


def _ratio_panel(axr, c5, v5, var5, s5, vis, color, xmax):
    """MC / smooth per bin with the MC-stat error; line at 1 in the model color."""
    ok = vis & (v5 > 0) & (s5 > 0)
    axr.errorbar(c5[ok], v5[ok] / s5[ok], yerr=np.sqrt(var5[ok]) / s5[ok],
                 fmt="o", color="black", markersize=3, linewidth=0.8)
    axr.axhline(1.0, color=color, linewidth=1.6)
    axr.set_ylim(*RATIO_YLIM)
    axr.set_xlim(XMIN, xmax)
    axr.set_ylabel("MC / smooth")
    axr.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")


def kde_param_text(d, lo, hi, total=None):
    """The RooKeysPdf constructor settings, for the plot:
    RooKeysPdf(name, title, x, data, mirror, rho)."""
    if not d:
        return "RooKeysPdf settings: see smoothed_kde.json"
    x = "ln m" if d.get("var_kind", "logm") == "logm" else "m"
    return (f"RooKeysPdf\n"
            f"x = {x},  [{lo:.0f}, {hi:.0f}] GeV\n"
            f"mirror = {d.get('mirror')}\n"
            f"rho = {d.get('rho')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=HERE / "run2" / "ee_resolved" / "smoothed_kde.root")
    p.add_argument("--output-dir", type=Path,
                   default=HERE / "run2" / "ee_resolved" / "plots" / "step1")
    p.add_argument("--bw", type=float, default=50.0)
    p.add_argument("--xmax", type=float, default=6000.0)
    p.add_argument("--ylim", type=float, nargs=2, default=(2e-5, 2e4),
                   help="common y range of every plot")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # the json written next to the root file: era, KDE settings per component
    meta = {}
    jpath = args.input.with_suffix(".json")
    if jpath.exists():
        with open(jpath) as fh:
            meta = json.load(fh)
    era = meta.get("era", sl.ERA)
    info = load_lumi(era)
    # the luminosity the histograms were NORMALIZED with, when recorded with
    # the smoothed set (it can differ from today's lumi.yaml); else lumi.yaml
    lumi, com = meta.get("lumi", info["lumi"]), info.get("com", 13.0)
    kde_par = meta.get("components", {})
    kde_label = ("RooKeysPdf, one entry per MC event"
                 if meta.get("kwargs", {}).get("entries") == "events" else "RooKeysPdf")

    f = uproot.open(args.input)
    edges = f["bkg_raw"].axes[0].edges()
    comps = [k.split(";")[0] for k in f.keys()
             if not k.split(";")[0].endswith("_raw") and not k.startswith("bkg_")]
    cset = next((k for k, o in STACK_ORDER.items() if set(o) == set(comps)), None)
    order = STACK_ORDER[cset] if cset else comps
    total_label = TOTAL_LABEL.get(cset, "Total background")
    logger.info("components in %s: %s (set %s)", args.input.name, comps, cset)
    items = {c: (f[c].values(), f[f"{c}_raw"].values(), f[f"{c}_raw"].variances())
             for c in comps}
    items["total"] = (f["bkg_smooth"].values(), f["bkg_raw"].values(),
                      f["bkg_raw"].variances())
    fac = int(round(args.bw / sl.NATIVE_BW))
    at50 = {}

    for name, (s, v, var) in items.items():
        e5, v5, var5 = rebin(edges, v, var, fac)
        _, s5, _ = rebin(edges, s, np.zeros_like(s), fac)
        c5 = centers_of(e5)
        at50[name] = (e5, v5, var5, s5)
        vis = (c5 >= XMIN) & (c5 <= args.xmax)
        pos = vis & (v5 > 0)
        sm = vis & (s5 > 0)

        is_total = name == "total"
        fig, (ax, axr) = _figure()
        ax.errorbar(c5[pos], v5[pos], yerr=np.sqrt(var5[pos]), fmt="o",
                    color="black", markersize=4, linewidth=1.0, zorder=4,
                    label=("Sum of the MC (binned) $\\pm$ MC stat" if is_total
                           else "MC (binned) $\\pm$ MC stat"))
        ax.plot(c5[sm], s5[sm], color=KDE_COLOR, linewidth=2.4, zorder=5,
                label=(f"Sum of the {len(comps)} RooKeysPdf curves" if is_total
                       else kde_label))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
        ax.set_ylim(*args.ylim)
        ax.set_xlim(XMIN, args.xmax)
        ax.set_ylabel(f"Events / {args.bw:.0f} GeV")
        _ratio_panel(axr, c5, v5, var5, s5, vis, KDE_COLOR, args.xmax)
        if is_total:
            axr.set_ylabel("MC / sum")
        ax.text(0.04, 0.95,
                f"{total_label if name == 'total' else COMP_LABEL[name]}\n"
                f"ee resolved SR, {era}",
                transform=ax.transAxes, fontsize=14, va="top")
        ax.legend(loc="upper right", fontsize=13)
        c10 = centers_of(edges)
        if not is_total:        # the total's legend already says what it is
            ptxt = kde_param_text(kde_par.get(name, {}), XMIN, float(edges[-1]),
                                  float(v[c10 >= XMIN].sum()))
            ax.text(0.04, 0.04, ptxt, transform=ax.transAxes, fontsize=10.5,
                    va="bottom", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor="0.6", alpha=1.0), zorder=10)
        _cms(ax, lumi, com)
        _save(fig, args.output_dir / name)

    # the combined curve: every per-process curve and their sum, over the
    # binned total MC (legend: SR yield of each process)
    cn = centers_of(edges)
    sr = cn >= XMIN
    yields = {k: float(items[k][1][sr].sum()) for k in comps}
    e5, tv, tvar, ts = at50["total"]
    c5 = centers_of(e5)
    vis = (c5 >= XMIN) & (c5 <= args.xmax)
    fig, (ax, axr) = _figure()
    pos = vis & (tv > 0)
    ax.errorbar(c5[pos], tv[pos], yerr=np.sqrt(tvar[pos]), fmt="o", color="black",
                markersize=4, linewidth=1.0, zorder=6,
                label="MC (binned) $\\pm$ MC stat")
    sm = vis & (ts > 0)
    ax.plot(c5[sm], ts[sm], color="black", linewidth=2.8, zorder=5,
            label=f"sum of the {len(comps)} curves ({sum(yields.values()):.0f})")
    for k in sorted(comps, key=lambda k: -yields[k]):
        sk = at50[k][3]
        ok = vis & (sk > 0)
        ax.plot(c5[ok], sk[ok], color=COMP_COLOR[k], linewidth=1.8,
                linestyle=COMP_LS.get(k, "-"), zorder=4,
                label=f"{COMP_LABEL[k]} ({yields[k]:.3g})")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*args.ylim)
    ax.set_xlim(XMIN, args.xmax)
    ax.set_ylabel(f"Events / {args.bw:.0f} GeV")
    _ratio_panel(axr, c5, tv, tvar, ts, vis, "black", args.xmax)
    axr.set_ylabel("MC / sum")
    ax.text(0.04, 0.95,
            f"RooKeysPdf per process, summed\nee resolved SR, {era}",
            transform=ax.transAxes, fontsize=14, va="top")
    ax.legend(loc="upper right", fontsize=11, handlelength=1.8)
    _cms(ax, lumi, com)
    _save(fig, args.output_dir / "combined")

    # the usual stacked MC histogram (house colors / order), KDE total on top
    e5 = at50["total"][0]
    c5 = centers_of(e5)
    vis = (c5 >= XMIN) & (c5 <= args.xmax)
    stack = [np.where(vis & (at50[k][1] > 0), at50[k][1], 0.0) for k in order]
    fig, (ax, axr) = _figure()
    hep.histplot(stack, e5, stack=True, histtype="fill", alpha=0.7,
                 color=[COMP_COLOR[k] for k in order],
                 label=[COMP_LABEL[k] for k in order], ax=ax)
    tv, tvar, ts = at50["total"][1], at50["total"][2], at50["total"][3]
    draw_stat_band(ax, e5, np.where(vis, tv, 0.0), np.where(vis, tvar, 0.0),
                   label="MC stat")
    sm = vis & (ts > 0)
    ax.plot(c5[sm], ts[sm], color="black", linewidth=2.4, zorder=6,
            label="RooKeysPdf (sum)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*args.ylim)
    ax.set_xlim(XMIN, args.xmax)
    ax.set_ylabel(f"Events / {args.bw:.0f} GeV")
    _ratio_panel(axr, c5, tv, tvar, ts, vis, "black", args.xmax)
    ax.text(0.04, 0.95,
            f"{'Background MC' if total_label.startswith('Total') else total_label + ' MC'}"
            f"\nee resolved SR, {era}",
            transform=ax.transAxes, fontsize=14, va="top")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize=13)
    _cms(ax, lumi, com)
    _save(fig, args.output_dir / "stack")


if __name__ == "__main__":
    main()
