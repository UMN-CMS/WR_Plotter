#!/usr/bin/env python3
"""The integral from high mass: N(m_lljj >= m) of the binned MC and of the RooKeysPdf, from 12 TeV down.

Every process's RooKeysPdf is scaled to its MC total over [800, 12000) GeV (smooth_lib.smooth_kde), so the
integrals over the whole spectrum agree by construction. This step integrates both from the top of the
histogram downward and draws how N(>= m) evolves on the way down: where the smooth curve parts from the
simulation, and by how much above 3 and 4 TeV. Per dataset (--datasets,
reloading config for each: 2024 = results/, run3 = results_run3/) and channel, on the nine-process LO
HT-binned set:

  cumulative_sum          N(>= m) of the summed MC (staircase, band = sqrt of the summed variances) and of the
                          summed RooKeysPdf alone (3_truths_kde9, the deck's smoothing frame; --truths kde_powexp
                          adds the RooKeysPdf with the fitted power x exponential tail, 3_truths_split9); ratio
                          panel curve / MC with the MC's relative error around 1; the last MC event marked
                          (above it the MC integral is zero and the ratio stops)
  cumulative_difference   N(curve >= m) - N(MC >= m) in events, linear, to 12 TeV: zero at both ends, so the
                          tail's excess is the bulk's deficit (the answer to "how can the integrals be the same")
  slices                  events per m_lljj slice, MC (points, errors) and the curve (steps); below, curve - MC per
                          slice in events with the MC error band
  slices_cumulative       the same with the lower panel as the running sum of the bars from 800 GeV, to 12 TeV
  cumulative_processes_<truth>   the ratio for DY, tt+tW, W+jets and the other processes summed
  cumulative.csv          N(>= m) per 10 GeV edge: MC, its error, every truth, the ratios and the pulls
  summary.md              the table at 2, 2.5, 3, 3.5, 4, 4.5, 5 and 6 TeV; from where on, coming up from
                          800 GeV, each truth stays beyond 5 % / 10 % / 1 sigma / 2 sigma of the MC; the yield
                          each curve puts beyond the last MC event; with --check-integral the midpoint-grid
                          sum against the exact RooKeysPdf integral (createIntegral) above 3 / 4 / 5 TeV

With two datasets the ratios of the sums go on one figure per channel, cumulative_ratio_datasets, under
the last dataset's results.

  cd WR_Plotter && python background_fit/18_cumulative.py \
      [--datasets 2024 run3] [--truths kde] [--check-integral]
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402
import plots as P                                                        # noqa: E402
from spectrum import (COMPONENT_COLOR, COMPONENT_LABEL, COMPONENT_LS, centers,   # noqa: E402
                      components_of, last_populated, load_components)
from truth_sets import load_truth                                        # noqa: E402
from util import fmt, sl, write_csv                                      # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402

logger = logging.getLogger("18_cumulative")
VARIANT = "lo_ht"
COMPONENTS = "split9"
TRUTH_DIR = {"kde": "3_truths_kde9", "kde_powexp": "3_truths_split9", "kde_expo": "3_truths_split9"}
TRUTH_LS = {"kde": "-", "kde_powexp": "--", "kde_expo": ":"}
TRUTH_LABEL = {"kde": "RooKeysPdf", "kde_powexp": r"RooKeysPdf + power $\times$ expo tail",
               "kde_expo": "RooKeysPdf + expo tail"}
MC_COLOR = "#717581"
DATASET_COLOR = {"2024": "#5790fc", "run3": "#e42536", "run3_scaled": "#964a8b"}
REF_MASSES = [2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 6000.0]
CHECK_MASSES = [3000.0, 4000.0, 5000.0]
CHECK_COMPONENTS = ["DYJets", "tt_tW"]
SHOWN = ["DYJets", "tt_tW", "WJets"]       # the processes with MC above 3 TeV; the rest are drawn summed
LEAVE = [("5 %", "rel", 0.05), ("10 %", "rel", 0.10), ("1 sigma", "pull", 1.0), ("2 sigma", "pull", 2.0)]
RATIO_TICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]
RATIO_LIM = (0.05, 50.0)        # the single-panel ratio figures: the legend sits in the empty corner below 1
XMAX = 8000.0
# the mass slices of the slice figure: per range, where the curve is below and above the MC
SLICE_EDGES = [800.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 5000.0, 6000.0, 8000.0, 12000.0]
# axis names: pick m_min on the x axis,
# read the number of events with m_lljj above it on the y axis
XLABEL = r"$m_\mathrm{min}$ [GeV]"
YLABEL = r"Events with $m_{\ell\ell jj} > m_\mathrm{min}$"


def use_dataset(name):
    """Point config (and every module holding it) at another dataset."""
    os.environ["FIT_DATASET"] = name
    importlib.reload(C)


def reverse_cumulative(values):
    """N(>= e_i): the sum of the bins from i on, one value per lower edge."""
    return np.cumsum(np.asarray(values, float)[::-1])[::-1]


def ds_label(name):
    d = C.DATASETS[name]
    return rf"{d['lumi']:.1f} fb$^{{-1}}$ ({d['year'].replace('$', '')})" if name != "run3_scaled" \
        else rf"{d['lumi']:.1f} fb$^{{-1}}$ (2024 MC $\times$ 2.58)"


def load_truths(channel, names):
    """{name: (parts, path)} of the requested truths; the no-tail RooKeysPdf also lives in the split9 set."""
    out = {}
    for name in names:
        cands = [C.RESULTS / channel / TRUTH_DIR[name] / VARIANT / f"truth_{name}"]
        if name == "kde":
            cands.append(C.RESULTS / channel / "3_truths_split9" / VARIANT / f"truth_{name}")
        path = next((p for p in cands if p.with_suffix(".npz").exists()), None)
        if path is None:
            logger.warning("%s %s: no %s truth under %s (build it with 3_truths.py --components split9)",
                           C.DATASET, channel, name, C.RESULTS / channel)
            continue
        edges, parts, _ = load_truth(path)
        out[name] = (edges, parts, path)
    return out


def leave_mass(edges_lo, cond, valid):
    """The lowest edge from which `cond` holds at every higher valid edge (None when it never holds)."""
    idx = np.nonzero(valid)[0]
    if not len(idx):
        return None
    run_start = None
    for i in idx[::-1]:
        if cond[i]:
            run_start = i
        else:
            break
    return float(edges_lo[run_start]) if run_start is not None else None


def kde_exact_above(e, v, thresholds, rho=1.0, mirror="MirrorLeft"):
    """The exact fraction of a process's RooKeysPdf above each threshold (createIntegral, the construction of
    smooth_lib.smooth_kde: entries at the populated bin centres, weight = content, x in [800, 12000))."""
    import ROOT
    kde_exact_above.uid += 1
    u = kde_exact_above.uid
    c = centers(e)
    lo, xmax = float(C.THRESHOLD), float(e[-1])
    x = ROOT.RooRealVar(f"xc{u}", "x", lo, xmax)
    w = ROOT.RooRealVar(f"wc{u}", "w", 0.0, 1e9)
    ds = ROOT.RooDataSet(f"dsc{u}", "", ROOT.RooArgSet(x, w), ROOT.RooFit.WeightVar(w.GetName()))
    inside = (c >= lo) & (c < xmax)
    for i in np.nonzero(inside & (v > 0))[0]:
        x.setVal(float(c[i]))
        ds.add(ROOT.RooArgSet(x), float(v[i]))
    pdf = ROOT.RooKeysPdf(f"kdec{u}", "", x, ds, sl._MIRROR[mirror], rho)
    nset = ROOT.RooArgSet(x)
    out = {}
    for thr in thresholds:
        rname = f"above{int(thr)}_{u}"
        x.setRange(rname, float(thr), xmax)
        integ = pdf.createIntegral(nset, ROOT.RooFit.NormSet(nset), ROOT.RooFit.Range(rname))
        out[thr] = float(integ.getVal())
    return out


kde_exact_above.uid = 0


def ratio_name(curves, op="/"):
    """'RooKeysPdf / MC' when the only curve is the RooKeysPdf alone, else 'curve / MC'."""
    return f"RooKeysPdf {op} MC" if list(curves) == ["kde"] else f"curve {op} MC"


def ratio_axis(ax, lim=(0.5, 30.0)):
    import matplotlib.ticker as mticker
    ax.set_yscale("log")
    ax.set_ylim(*lim)
    ax.yaxis.set_major_locator(mticker.FixedLocator([t for t in RATIO_TICKS if lim[0] <= t <= lim[1]]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def plot_sum(channel, e, N_mc, E_mc, curves, m_last, out):
    """N(>= m) of the summed MC with its band and of every truth; ratio panel curve / MC."""
    lo_edges = e[:-1]
    sel = lo_edges >= C.THRESHOLD
    fig, (ax, axr) = P.spectrum_figure()
    ax.step(lo_edges[sel], np.maximum(N_mc[sel], P.YMIN), where="post", color=MC_COLOR, lw=1.8,
            label=f"MC, {len(C.COMPONENT_SETS[COMPONENTS])} processes")
    ax.fill_between(lo_edges[sel], np.maximum(N_mc[sel] - E_mc[sel], P.YMIN), np.maximum(N_mc[sel] + E_mc[sel], P.YMIN),
                    step="post", color=MC_COLOR, alpha=0.3, lw=0, label="MC stat. error")
    for name, N_t in curves.items():
        ax.plot(lo_edges[sel], np.maximum(N_t[sel], P.YMIN), color=P.TRUTH_COLOR.get(name, "k"), lw=1.8,
                ls=TRUTH_LS.get(name, "-"), label=TRUTH_LABEL.get(name, name))
    for a_ in (ax, axr):
        a_.axvline(m_last, color="0.45", ls=":", lw=1.5)
    ax.plot([], [], color="0.45", ls=":", lw=1.5, label=f"last MC event, {m_last / 1000:.2f} TeV")
    ok = sel & (N_mc > 0)
    axr.fill_between(lo_edges[ok], 1.0 - E_mc[ok] / N_mc[ok], 1.0 + E_mc[ok] / N_mc[ok], step="post",
                     color=MC_COLOR, alpha=0.3, lw=0)
    for name, N_t in curves.items():
        axr.plot(lo_edges[ok], N_t[ok] / N_mc[ok], color=P.TRUTH_COLOR.get(name, "k"), lw=1.8, ls=TRUTH_LS.get(name, "-"))
    P.finish(ax, axr, channel=channel, xmax=XMAX, top=float(N_mc[sel].max()), ymin=1e-2,
             ylabel=YLABEL, extra=C.DY_SHORT[VARIANT], ratio_label=ratio_name(curves), ratio_lim=(0.5, 50.0))
    axr.set_xlabel(XLABEL)
    ratio_axis(axr, (0.5, 50.0))
    P.save(fig, out)


def slice_sums(e, values, edges):
    """The sum of the 10 GeV bins inside each slice [lo, hi) of `edges`."""
    return np.array([float(np.asarray(values)[int(np.searchsorted(e, lo)):int(np.searchsorted(e, hi))].sum())
                     for lo, hi in zip(edges[:-1], edges[1:])])


def plot_slices(channel, e, v, var, curve_values, out):
    """Events per mass slice: the MC (points, error bars) and the curve (steps) on top; below, curve - MC in
    events per slice with the MC error band. Where the bar is negative the MC has more; the bars sum to zero."""
    edges = np.array(SLICE_EDGES[:-1], float)                 # drawn to 8 TeV; the 8-12 TeV remainder is in the summary
    mc, err = slice_sums(e, v, edges), np.sqrt(np.maximum(slice_sums(e, var, edges), 0.0))
    c, w = 0.5 * (edges[:-1] + edges[1:]), edges[1:] - edges[:-1]
    fig, (ax, axr) = P.spectrum_figure()
    ax.errorbar(c, mc, xerr=w / 2, yerr=err, fmt="o", ms=5, color=MC_COLOR, lw=1.4,
                label=f"MC, {len(C.COMPONENT_SETS[COMPONENTS])} processes")
    top = 0.0
    for name, vals in curve_values.items():
        cs = slice_sums(e, vals, edges)
        ax.step(edges, np.append(cs, cs[-1]), where="post", color=P.TRUTH_COLOR.get(name, "k"), lw=1.8,
                ls=TRUTH_LS.get(name, "-"), label=TRUTH_LABEL.get(name, name))
        d = cs - mc
        axr.bar(edges[:-1], d, width=w, align="edge", color=P.TRUTH_COLOR.get(name, "k"), alpha=0.75, lw=0)
        top = max(top, float(np.abs(d).max()))
    axr.fill_between(edges, np.append(-err, -err[-1]), np.append(err, err[-1]), step="post", color=MC_COLOR,
                     alpha=0.3, lw=0)
    ax.fill_between([], [], [], color=MC_COLOR, alpha=0.3, lw=0, label="MC stat. error (lower panel)")
    lim = 1.3 * top
    P.finish(ax, axr, channel=channel, xmax=XMAX, top=float(mc.max()), ymin=1e-2, ylabel="Events in slice",
             extra=C.DY_SHORT[VARIANT], ratio_label=ratio_name(curve_values, "$-$"),
             ratio_lim=(-lim, lim), ratio_ref=0.0)
    axr.set_xlabel(P.MASS_LABEL)
    axr.set_ylabel(ratio_name(curve_values, "$-$"), fontsize=20)   # the CMS floor; the full name fits the short panel
    # legend order: the MC, its band, then the curve (matplotlib lists the step line before the errorbar)
    h, l = ax.get_legend_handles_labels()
    idx = sorted(range(len(l)), key=lambda i: 0 if l[i].startswith("MC,") else 1 if l[i].startswith("MC stat") else 2)
    ax.legend([h[i] for i in idx], [l[i] for i in idx], loc="upper right", fontsize=P.FONT_SIZE_LEGEND, handlelength=1.6)
    P.save(fig, out)


def plot_slices_cumulative(channel, e, v, var, curve_values, out):
    """The slice figure with the lower panel as the running sum of the bars from 800 GeV up:
    a staircase whose level in each slice is the sum of (curve - MC) over this and every lower slice; it starts
    near zero, reaches its minimum where the curve has given up the most and returns to zero at the end of the
    spectrum, which is drawn to 12 TeV here so the return is on the page."""
    edges = np.array(SLICE_EDGES, float)
    mc, err = slice_sums(e, v, edges), np.sqrt(np.maximum(slice_sums(e, var, edges), 0.0))
    c, w = 0.5 * (edges[:-1] + edges[1:]), edges[1:] - edges[:-1]
    fig, (ax, axr) = P.spectrum_figure()
    ax.errorbar(c, mc, xerr=w / 2, yerr=err, fmt="o", ms=5, color=MC_COLOR, lw=1.4,
                label=f"MC, {len(C.COMPONENT_SETS[COMPONENTS])} processes")
    top = 0.0
    for name, vals in curve_values.items():
        cs = slice_sums(e, vals, edges)
        ax.step(edges, np.append(cs, cs[-1]), where="post", color=P.TRUTH_COLOR.get(name, "k"), lw=1.8,
                ls=TRUTH_LS.get(name, "-"), label=TRUTH_LABEL.get(name, name))
        run = np.cumsum(cs - mc)
        axr.step(edges, np.append(run, run[-1]), where="post", color=P.TRUTH_COLOR.get(name, "k"), lw=2.2,
                 ls=TRUTH_LS.get(name, "-"))
        top = max(top, float(np.abs(run).max()))
    lim = 1.3 * top
    P.finish(ax, axr, channel=channel, xmax=float(edges[-1]), top=float(mc.max()), ymin=1e-3, ylabel="Events in slice",
             extra=C.DY_SHORT[VARIANT], ratio_label=ratio_name(curve_values, "$-$"),
             ratio_lim=(-lim, lim), ratio_ref=0.0)
    axr.set_xlabel(P.MASS_LABEL)
    axr.set_ylabel(r"$\sum$ (" + ratio_name(curve_values, "$-$") + ")", fontsize=20)   # the running sum of the bars
    h, l = ax.get_legend_handles_labels()
    idx = sorted(range(len(l)), key=lambda i: 0 if l[i].startswith("MC,") else 1)
    ax.legend([h[i] for i in idx], [l[i] for i in idx], loc="upper right", fontsize=P.FONT_SIZE_LEGEND, handlelength=1.6)
    P.save(fig, out)


def plot_difference(channel, e, N_mc, E_mc, curves, out):
    """N_curve(>= m) - N_MC(>= m) in events on a linear axis, to the end of the histogram: zero at 800 GeV (the
    normalisation) and at 12 TeV; the tail's excess is the bulk's deficit. Band = the MC error on N(>= m)."""
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    lo_edges = e[:-1]
    sel = lo_edges >= C.THRESHOLD
    fig, ax = P.scan_figure()
    ax.fill_between(lo_edges[sel], -E_mc[sel], E_mc[sel], step="post", color=MC_COLOR, alpha=0.3, lw=0,
                    label="MC stat. error")
    top = 0.0
    for name, N_t in curves.items():
        D = N_t - N_mc
        ax.step(lo_edges[sel], D[sel], where="post", color=P.TRUTH_COLOR.get(name, "k"), lw=1.8,
                ls=TRUTH_LS.get(name, "-"), label=TRUTH_LABEL.get(name, name))
        top = max(top, float(np.abs(D[sel]).max()))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlim(C.THRESHOLD, float(e[-1]))
    ax.set_ylim(-1.2 * top, 3.0 * top)                 # room for the label block over the peak near 3 TeV
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ratio_name(curves, "$-$") + " [events]")
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, C.DY_SHORT[VARIANT]), year=C.YEAR)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND, ncol=1, handlelength=1.8)
    fit_header(ax, ignore_above=True)                  # the band fills the frame at low mass: not a reason to raise
    P.save(fig, out)


def plot_processes(channel, name, e, mc, truth_parts, out):
    """The ratio truth / MC of N(>= m) for the sum, the three processes with MC above 3 TeV (DY, tt+tW,
    W+jets) and the other processes summed (log axis)."""
    lo_edges = e[:-1]
    fig, ax = P.scan_figure()
    comps = components_of(mc)
    e_, v_, var_ = mc["sum"]
    N_mc, E_mc = reverse_cumulative(v_), np.sqrt(np.maximum(reverse_cumulative(var_), 0.0))
    ok = (lo_edges >= C.THRESHOLD) & (N_mc > 0)
    ax.fill_between(lo_edges[ok], 1.0 - E_mc[ok] / N_mc[ok], 1.0 + E_mc[ok] / N_mc[ok], step="post",
                    color=MC_COLOR, alpha=0.3, lw=0, label="MC stat. error (sum)")
    ax.plot(lo_edges[ok], reverse_cumulative(truth_parts["sum"])[ok] / N_mc[ok], color="k", lw=2.2, label="sum")
    others = [c for c in comps if c not in SHOWN]
    lines = [(c, mc[c][1], truth_parts[c], COMPONENT_COLOR[c], COMPONENT_LS.get(c, "-"), COMPONENT_LABEL[c])
             for c in SHOWN if c in comps]
    if others:
        lines.append(("others", sum(mc[c][1] for c in others), sum(truth_parts[c] for c in others), "#964a8b", "--",
                      f"{len(others)} other processes summed"))
    for comp, v, t, col, ls, label in lines:
        N = reverse_cumulative(v)
        okc = (lo_edges >= C.THRESHOLD) & (N > 0)
        if not okc.any():
            continue
        ax.plot(lo_edges[okc], reverse_cumulative(t)[okc] / N[okc], color=col, lw=1.6, ls=ls, label=label)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlim(C.THRESHOLD, XMAX)
    ratio_axis(ax, (0.02, RATIO_LIM[1]))              # six legend entries below the curves at 1
    P.finish_scan(ax, channel=channel, extra=f"{C.DY_SHORT[VARIANT]}\n{TRUTH_LABEL.get(name, name)}",
                  ylabel=ratio_name({name: None}), xlabel=XLABEL, loc="lower left", ncol=1)
    P.save(fig, out)


def plot_datasets(channel, store, names, out):
    """The ratio of the summed curves to the summed MC for every dataset on one axis: colour = dataset,
    line style = truth, band = the MC's relative error of each dataset."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    fig, ax = P.scan_figure()
    handles = []
    for ds in names:
        d = store.get((ds, channel))
        if d is None:
            continue
        lo_edges = d["e"][:-1]
        ok = (lo_edges >= C.THRESHOLD) & (d["N_mc"] > 0)
        col = DATASET_COLOR.get(ds, "k")
        ax.fill_between(lo_edges[ok], 1.0 - d["E_mc"][ok] / d["N_mc"][ok], 1.0 + d["E_mc"][ok] / d["N_mc"][ok],
                        step="post", color=col, alpha=0.2, lw=0)
        for name, N_t in d["curves"].items():
            ax.plot(lo_edges[ok], N_t[ok] / d["N_mc"][ok], color=col, lw=1.8, ls=TRUTH_LS.get(name, "-"))
        handles.append(Line2D([], [], color=col, lw=1.8, label=d["label"]))
    truth_names = [n for n in TRUTH_LS if any(n in store[k]["curves"] for k in store if k[1] == channel)]
    for name in truth_names:
        handles.append(Line2D([], [], color="k", lw=1.8, ls=TRUTH_LS[name], label=TRUTH_LABEL[name]))
    handles.append(Patch(facecolor="0.5", alpha=0.3, lw=0, label="MC stat. error"))
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlim(C.THRESHOLD, XMAX)
    ratio_axis(ax, RATIO_LIM)
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ratio_name({n: None for n in truth_names}))
    cms_label(ax, None, C.COM, data=False, text=P.region_text(channel, C.DY_SHORT[VARIANT]))
    ax.legend(handles=handles, loc="lower left", fontsize=FONT_SIZE_LEGEND, ncol=1, handlelength=1.8)
    fit_header(ax)
    P.save(fig, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["2024", "run3"], choices=sorted(C.DATASETS))
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--truths", nargs="+", default=["kde"], choices=sorted(TRUTH_DIR),
                    help="the smoothed curves: the RooKeysPdf alone (3_truths_kde9, the default, no tails) "
                         "and, on request, with fitted tails (kde_powexp / kde_expo from 3_truths_split9)")
    ap.add_argument("--check-integral", action="store_true",
                    help="compare the 10 GeV midpoint grid with the exact RooKeysPdf integral above 3 / 4 / 5 TeV "
                         "for DY and tt+tW (every dataset and channel; a RooKeysPdf rebuild each)")
    ap.add_argument("--out", default="18_cumulative", help="results/<channel>/<out>/")
    ap.add_argument("--slice-width", type=float, default=None,
                    help="uniform slices of this width [GeV] from 800 GeV to 12 TeV for the slice figures, written as "
                         "slices_w<width>; default: the SLICE_EDGES list")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    slice_tag = ""
    if args.slice_width:
        SLICE_EDGES[:] = [float(x) for x in np.arange(C.THRESHOLD, 12000.0 + 0.5 * args.slice_width, args.slice_width)]
        slice_tag = f"_w{args.slice_width:.0f}"
    store = {}
    for ds in args.datasets:
        use_dataset(ds)
        for channel in args.channels:
            out = C.RESULTS / channel / args.out
            out.mkdir(parents=True, exist_ok=True)
            mc = load_components(channel, VARIANT, COMPONENTS)
            e, v, var = mc["sum"]
            lo_edges = e[:-1]
            N_mc = reverse_cumulative(v)
            E_mc = np.sqrt(np.maximum(reverse_cumulative(var), 0.0))
            m_last = last_populated(e, v)
            truths = load_truths(channel, args.truths)
            curves, parts_by = {}, {}
            for name, (edges, parts, path) in truths.items():
                if not np.array_equal(edges, e):
                    raise ValueError(f"{path}: binning differs from the MC's")
                missing = set(components_of(mc)) - set(components_of(parts))
                if missing:
                    raise ValueError(f"{path}: no curve for {sorted(missing)}")
                curves[name] = reverse_cumulative(parts["sum"])
                parts_by[name] = parts
            i800 = int(np.searchsorted(e, C.THRESHOLD))
            L = [f"# {channel} {C.REGION_NAME}, {C.ERA} ({C.LUMI:.1f} fb-1): the integral from high mass", "",
                 f"N(>= m) = the sum of the 10 GeV bins from m to {e[-1] / 1000:.0f} TeV, of the summed MC ({COMPONENTS}, "
                 f"{C.DY_LABEL[VARIANT]} DY) and of the summed smoothed curves. The MC error is the square root of the "
                 f"summed variances. Pull = (MC - curve) / MC error. The curves are scaled to the MC total over "
                 f"[{C.THRESHOLD:.0f}, {e[-1]:.0f}) GeV process by process, so the ratio at {C.THRESHOLD:.0f} GeV is 1 by construction.", ""]
            L += [f"Curves: " + ", ".join(f"{name} = {truths[name][2].relative_to(C.RESULTS)}" for name in curves) + ".", ""]
            L += [f"Normalisation check at {C.THRESHOLD:.0f} GeV: MC {N_mc[i800]:.4f}; "
                  + ", ".join(f"{name} {curves[name][i800]:.4f} ({curves[name][i800] / N_mc[i800] - 1:+.1e})" for name in curves)
                  + f". Last MC event: {m_last:.0f} GeV.", ""]
            # the table at the reference masses
            head = "| m [GeV] | MC N(>= m) | MC error |" + "".join(f" {name} | curve - MC | ratio | pull |" for name in curves)
            L += [head, "|---|---|---|" + "---|---|---|---|" * len(curves)]
            for m in REF_MASSES:
                i = int(np.searchsorted(e, m))
                row = f"| {m:.0f} | {N_mc[i]:.4g} | {E_mc[i]:.2g} |"
                for name in curves:
                    N_t = curves[name][i]
                    r = N_t / N_mc[i] if N_mc[i] > 0 else float("nan")
                    p = (N_mc[i] - N_t) / E_mc[i] if E_mc[i] > 0 else float("nan")
                    row += f" {N_t:.4g} | {N_t - N_mc[i]:+.3g} | {fmt(r)} | {fmt(p, 1)} |"
                L.append(row)
            L.append("")
            # per mass slice: where the curve is below the MC and where above (figure slices)
            sl_edges = np.array(SLICE_EDGES, float)
            mc_sl, err_sl = slice_sums(e, v, sl_edges), np.sqrt(np.maximum(slice_sums(e, var, sl_edges), 0.0))
            L += ["Events per mass slice (figure `slices`; the bars of its lower panel are the curve - MC column):", "",
                  "| range [GeV] | MC | MC error |" + "".join(f" {name} | {name} - MC |" for name in curves),
                  "|---|---|---|" + "---|---|" * len(curves)]
            for k, (lo_, hi_) in enumerate(zip(sl_edges[:-1], sl_edges[1:])):
                row = f"| {lo_:.0f}-{hi_:.0f} | {mc_sl[k]:.4g} | {err_sl[k]:.2g} |"
                for name in curves:
                    cs = slice_sums(e, parts_by[name]["sum"], sl_edges)
                    row += f" {cs[k]:.4g} | {cs[k] - mc_sl[k]:+.2f} |"
                L.append(row)
            L.append("")
            # the same integral: where the tail's excess comes from
            L += ["The difference D(m) = N(curve >= m) - N(MC >= m) in events (figure cumulative_difference): zero at "
                  f"{C.THRESHOLD:.0f} GeV and at {e[-1] / 1000:.0f} TeV; its maximum is the number of events the curve "
                  "moves from below that mass to above it:", ""]
            for name, N_t in curves.items():
                D = N_t - N_mc
                bulk = (lo_edges >= C.THRESHOLD) & (lo_edges < 2000.0)
                tail_ = lo_edges >= 2000.0
                imax = int(np.nonzero(tail_)[0][np.argmax(D[tail_])])
                below = N_mc[i800] - N_mc[imax]
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = np.abs(D[bulk] / E_mc[bulk])
                L.append(f"- {name}: below 2 TeV D wanders between {D[bulk].min():+.2g} and {D[bulk].max():+.2g} events, "
                         f"at most {np.nanmax(z):.1f} sigma_MC (the smoother ironing out the MC's bin-to-bin fluctuations; the "
                         f"first bins at the {C.THRESHOLD:.0f} GeV mirror edge included); its tail peak is {D[imax]:+.3g} events at "
                         f"{lo_edges[imax]:.0f} GeV, i.e. the curve holds {D[imax]:.3g} fewer events than the MC below "
                         f"{lo_edges[imax]:.0f} GeV ({D[imax] / below * 100:.2g} % of the MC's {below:.0f} there) and {D[imax]:.3g} "
                         f"more above it; D at {e[-1] / 1000:.0f} TeV = {D[-1]:+.2g}.")
            L.append("")
            # where each curve leaves the MC on the way up from 800 GeV, and stays away
            valid = (lo_edges >= C.THRESHOLD) & (N_mc > 0)
            L += ["From where on (coming up from 800 GeV) the curve stays beyond the MC by more than:", "",
                  "| curve | 5 % | 10 % | 1 sigma | 2 sigma | yield beyond the last MC event |", "|---|---|---|---|---|---|"]
            for name, N_t in curves.items():
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.abs(N_t / N_mc - 1.0)
                    pull = np.abs((N_mc - N_t) / E_mc)
                cells = []
                for _, kind, x in LEAVE:
                    cond = (rel if kind == "rel" else pull) > x
                    m_x = leave_mass(lo_edges, np.nan_to_num(cond, nan=False), valid)
                    cells.append(f"{m_x:.0f}" if m_x is not None else "-")
                beyond = N_t[int(np.searchsorted(e, m_last))]
                L.append(f"| {name} | " + " | ".join(cells) + f" | {beyond:.3g} ({beyond / N_mc[i800] * 100:.2g} % of the total) |")
            L.append("")
            # the per-process view
            L += ["Per process, curve / MC of N(>= m) above 3 / 4 / 5 TeV (the sum last):", "",
                  "| process | MC > 3 TeV |" + "".join(f" {name} 3 / 4 / 5 TeV |" for name in curves),
                  "|---|---|" + "---|" * len(curves)]
            for comp in components_of(mc) + ["sum"]:
                _, vc, _ = mc[comp]
                Nc = reverse_cumulative(vc)
                row = f"| {comp} | {Nc[int(np.searchsorted(e, 3000.0))]:.3g} |"
                for name in curves:
                    Nt = reverse_cumulative(parts_by[name][comp])
                    vals = []
                    for m in CHECK_MASSES:
                        i = int(np.searchsorted(e, m))
                        vals.append(fmt(Nt[i] / Nc[i]) if Nc[i] > 0 else "-")
                    row += " " + " / ".join(vals) + " |"
                L.append(row)
            L.append("")
            if args.check_integral and "kde" in parts_by:
                L += ["Midpoint grid against the exact RooKeysPdf integral (createIntegral), the no-tail curve:", "",
                      "| process | m [GeV] | grid N(>= m) | exact N(>= m) | grid / exact |", "|---|---|---|---|---|"]
                for comp in CHECK_COMPONENTS:
                    ec, vc, _ = mc[comp]
                    exact = kde_exact_above(ec, vc, CHECK_MASSES)
                    Nt = reverse_cumulative(parts_by["kde"][comp])
                    total = Nt[i800]
                    for m in CHECK_MASSES:
                        i = int(np.searchsorted(e, m))
                        L.append(f"| {comp} | {m:.0f} | {Nt[i]:.4g} | {exact[m] * total:.4g} | {Nt[i] / (exact[m] * total):.4f} |")
                        logger.info("%s %s %s above %.0f: grid %.4g exact %.4g ratio %.4f", ds, channel, comp, m, Nt[i],
                                    exact[m] * total, Nt[i] / (exact[m] * total))
                L.append("")
            rows = []
            for i in np.nonzero(lo_edges >= C.THRESHOLD)[0]:
                r = {"m": float(lo_edges[i]), "N_mc": float(N_mc[i]), "err_mc": float(E_mc[i])}
                for name, N_t in curves.items():
                    r[f"N_{name}"] = float(N_t[i])
                    r[f"ratio_{name}"] = float(N_t[i] / N_mc[i]) if N_mc[i] > 0 else float("nan")
                    r[f"pull_{name}"] = float((N_mc[i] - N_t[i]) / E_mc[i]) if E_mc[i] > 0 else float("nan")
                rows.append(r)
            write_csv(out / "cumulative.csv", rows)
            plot_sum(channel, e, N_mc, E_mc, curves, m_last, out / "cumulative_sum")
            plot_difference(channel, e, N_mc, E_mc, curves, out / "cumulative_difference")
            plot_slices(channel, e, v, var, {n: parts_by[n]["sum"] for n in curves}, out / f"slices{slice_tag}")
            plot_slices_cumulative(channel, e, v, var, {n: parts_by[n]["sum"] for n in curves}, out / f"slices_cumulative{slice_tag}")
            for name in curves:
                plot_processes(channel, name, e, mc, parts_by[name], out / f"cumulative_processes_{name}")
            (out / "summary.md").write_text("\n".join(L) + "\n")
            store[(ds, channel)] = dict(e=e, N_mc=N_mc, E_mc=E_mc, curves=curves, m_last=m_last, label=ds_label(ds))
            for name, N_t in curves.items():
                i3, i4 = int(np.searchsorted(e, 3000.0)), int(np.searchsorted(e, 4000.0))
                logger.info("%s %s %s: curve / MC above 3 / 4 TeV %.2f / %.2f", ds, channel, name,
                            N_t[i3] / N_mc[i3], N_t[i4] / N_mc[i4])
            logger.info("wrote %s", out)
    if len(args.datasets) > 1:
        for channel in args.channels:
            out = C.RESULTS / channel / args.out / "cumulative_ratio_datasets"
            plot_datasets(channel, store, args.datasets, out)
            logger.info("wrote %s (the datasets %s)", out, ", ".join(args.datasets))


if __name__ == "__main__":
    main()
