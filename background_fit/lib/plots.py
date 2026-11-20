"""CMS-style figures for background_fit (the plotter's helpers: cms_label, fit_header)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import config as C                                                       # noqa: E402
from wrplotter.plotting_helpers import (FONT_SIZE_LEGEND, cms_label,     # noqa: E402
                                        custom_log_formatter, fit_header)

YMIN = 1e-4
# The spectrum figures end at 8 TeV. Their floor is a decade below the simulation at 8 TeV (3e-4 events per 100 GeV
# in the resolved SR).
XMAX_FULL = 8000.0
YMIN_FULL = 1e-4 if C.BOOSTED else 1e-5
CH = {"ee": "ee", "mumu": r"$\mu\mu$"}
MASS_LABEL = C.MASS_LABEL                 # m_lljj (resolved) or m_lJ (boosted)


def save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def region_text(channel, extra=None):
    t = f"{CH.get(channel, channel)}\n{C.REGION_TEXT}"
    return t + (f"\n{extra}" if extra else "")


def spectrum_figure(height=10):
    """10 in wide, a main panel over a ratio panel; a taller figure (12, 13) keeps the ratio panel's size
    and gives the main panel the room of a long legend."""
    hep.style.use("CMS")
    return plt.subplots(2, 1, sharex=True, figsize=(10, height),
                        gridspec_kw={"height_ratios": [3 * height / 10 + (height - 10) / 4, 1],
                                     "hspace": 0.1 * 10 / height})


def finish(ax, axr, *, channel, xmax, top, extra=None, ylabel=None, ratio_label="Fit / MC",
           ratio_lim=(0.0, 3.0), legend_fontsize=FONT_SIZE_LEGEND, ncol=1, legend_loc="upper right",
           legend_title=None, log=True, xmin=None, ymin=None, ratio_ref=1.0):
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
        ax.set_ylim(YMIN if ymin is None else ymin, 10.0 * top)
    else:
        ax.set_ylim(0.0, 1.3 * top)
    ax.set_xlim(C.THRESHOLD if xmin is None else xmin, xmax)
    ax.set_ylabel(ylabel or f"Events / {C.BIN_WIDTH:.0f} GeV")
    if axr is not None:
        axr.set_xlim(C.THRESHOLD if xmin is None else xmin, xmax)
        axr.set_ylim(*ratio_lim)
        axr.axhline(ratio_ref, color="k", lw=0.8)                 # 1 for a ratio panel, 0 for a pull panel
        axr.set_ylabel(ratio_label)
        axr.set_xlabel(MASS_LABEL)
    else:
        ax.set_xlabel(MASS_LABEL)
    cms_label(ax, C.LUMI, C.COM, data=False, text=region_text(channel, extra), year=C.YEAR)
    ax.legend(loc=legend_loc, fontsize=legend_fontsize, handlelength=1.6, ncol=ncol, columnspacing=0.8,
              title=legend_title, title_fontsize=legend_fontsize, alignment="left")
    fit_header(ax)


def step_hist(ax, edges, values, variances, color, label, errors=True):
    c = 0.5 * (edges[:-1] + edges[1:])
    sel = c >= C.THRESHOLD
    ax.step(edges[:-1][sel], np.maximum(values[sel], YMIN), where="post", color=color, lw=1.5, label=label)
    if errors:
        err = np.sqrt(np.maximum(variances, 0.0))
        ax.errorbar(c[sel], values[sel], yerr=err[sel], fmt="none", ecolor=color, elinewidth=1, capsize=0)


# ---- the scan figures (a quantity versus m_WR): colours, layout, the CMS label -------------------
PETROFF6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
TRUTH_COLOR = {"raw": "#9c9ca1", "kde_expo": "#5790fc", "kde_powexp": "#f89c20", "kernel_expo": "#e42536",
               "kernel_powexp": "#964a8b", "kde": "#7a21dd"}
TRUTH_SHORT = {"raw": "raw MC", "kde_expo": "RooKeysPdf + expo tail", "kde_powexp": "RooKeysPdf + powexp tail",
               "kernel_expo": "kernel + expo tail", "kernel_powexp": "kernel + powexp tail",
               "kde": "RooKeysPdf alone", "kde_sumexpo": "RooKeysPdf + expo tail"}
VARIANT_LS = {"lo_ht": "-", "nlo_mll": "--", "nlo_njet": ":", "nlo_inc": "-.", "lo_inc": (0, (1, 1))}
VARIANT_SHORT = {"lo_ht": "LO HT-binned", "nlo_mll": r"NLO $m(\ell\ell)$-binned", "nlo_njet": "NLO n(jet)-binned",
                 "nlo_inc": "NLO inclusive", "lo_inc": "LO inclusive"}
MWR_LABEL = r"$m_{W_R}$ [GeV]"
SSPUR_LABEL = r"median $N_{\rm sig}$ / RMS (no injection)"


def scan_figure(height=8):
    hep.style.use("CMS")
    return plt.subplots(figsize=(10, height))


def scan_figure2(ratios=(3, 2)):
    hep.style.use("CMS")
    return plt.subplots(2, 1, sharex=True, figsize=(10, 11),
                        gridspec_kw={"height_ratios": list(ratios), "hspace": 0.08})


def bias_lines(ax):
    """The 0.3 target and 0.5 pass lines and the zero line of a spurious-signal panel."""
    for yv, col in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
        ax.axhline(yv, color=col, lw=0.9, ls="--")
        ax.axhline(-yv, color=col, lw=0.9, ls="--")
    ax.axhline(0, color="k", lw=0.8)


def truth_lines(ax, rows, cand, variants, key="bias_median"):
    """One line per (DY variant, truth) of *cand* at zero injection: colour = truth, style = variant.
    Returns the legend handles (the truths by colour, then the variants by style)."""
    from matplotlib.lines import Line2D
    keys = sorted({(r["variant"], r["truth"]) for r in rows if r["variant"] in variants and r["candidate"] == cand})
    seen_t, seen_v = [], []
    for variant, truth in keys:
        rs = sorted([r for r in rows if r["candidate"] == cand and r["variant"] == variant and r["truth"] == truth
                     and r["inject"] == 0.0 and np.isfinite(r[key])], key=lambda r: r["mass"])
        if not rs:
            continue
        ax.plot([r["mass"] for r in rs], [r[key] for r in rs], marker="o", ms=4, lw=1.4,
                color=TRUTH_COLOR.get(truth, "k"), ls=VARIANT_LS.get(variant, "-"))
        if truth not in seen_t:
            seen_t.append(truth)
        if variant not in seen_v:
            seen_v.append(variant)
    handles = [Line2D([], [], color=TRUTH_COLOR.get(t, "k"), lw=1.8, label=TRUTH_SHORT.get(t, t)) for t in seen_t]
    handles += [Line2D([], [], color="k", lw=1.8, ls=VARIANT_LS.get(v, "-"), label=VARIANT_SHORT.get(v, v))
                for v in seen_v]
    return handles


def band_handle(label="band: range over the primary truths"):
    from matplotlib.patches import Patch
    return Patch(facecolor="0.6", alpha=0.4, lw=0, label=label)


def finish_scan(ax, *, channel, extra, ylabel, xlabel=MWR_LABEL, ylim=None, handles=None, loc="upper right",
                ncol=2):
    """Axis labels, the CMS label with the region text, the legend at the plotter's size, fit_header last."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    cms_label(ax, C.LUMI, C.COM, data=False, text=region_text(channel, extra), year=C.YEAR)
    if handles is None:
        ax.legend(loc=loc, fontsize=FONT_SIZE_LEGEND, ncol=ncol, handlelength=1.8)
    else:
        ax.legend(handles=handles, loc=loc, fontsize=FONT_SIZE_LEGEND, ncol=ncol, handlelength=1.8)
    fit_header(ax)
