#!/usr/bin/env python3
"""The raw MC against every smoothed background of the study.

Reads the stored truths of results*/<ch>/ and draws them over the summed MC of the nine processes
(LO HT-binned DY): the RooKeysPdf alone (3_truths_kde9), the RooKeysPdf and the kernel regression
with a fitted expo or powexp tail per component (3_truths_split9, 3_truths), and the power x
exponential and the exp. of a quadratic fitted to the summed MC from 1200 GeV (3_truths_fit).

  spectrum  the 100 GeV spectrum with every curve; below, (MC - curve) / sigma_MC
  slices    events per mass slice; below, curve - MC in events with the MC statistical error

Writes results*/<ch>/20_truths_overlay/{spectrum,slices}.{pdf,png} and slices.md. The default set is the
RooKeysPdf and the two fits; --with-tails adds the four truths with a fitted tail per component, into
20_truths_overlay_tails/. A truth that was not built for the dataset is left out.

  cd WR_Plotter && python background_fit/20_truths_overlay.py [--datasets 2024 run3] [--channels ee mumu]
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402

logger = logging.getLogger("20_truths_overlay")
VARIANT = "lo_ht"
SLICES = [1200.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 5000.0, 6000.0, 8000.0]
# (results sub-directory, file, legend label, colour, line style), in the order of the legend.
# The default set has no fitted tail; --with-tails adds the truths with one.
TRUTHS = [
    ("3_truths_kde9", "truth_kde", "RooKeysPdf", "#7a21dd", ":"),
    ("3_truths_fit", "truth_fitpowexp", r"power $\times$ exponential fit", "#f89c20", "-"),
    ("3_truths_fit", "truth_fitexpo2", "exp. of a quadratic fit", "#e42536", "-"),
]
TAIL_TRUTHS = [
    ("3_truths_split9", "truth_kde_expo", "RooKeysPdf + expo tail", "#5790fc", "--"),
    ("3_truths_split9", "truth_kde_powexp", "RooKeysPdf + powexp tail", "#964a8b", "--"),
    ("3_truths", "truth_kernel_expo", "kernel + expo tail", "#9c9ca1", "-."),
    ("3_truths", "truth_kernel_powexp", "kernel + powexp tail", "#717581", "-."),
]


def use_dataset(ds):
    import os
    os.environ["FIT_DATASET"] = ds
    importlib.reload(C)
    for name in ("spectrum", "plots", "truth_sets"):
        if name in sys.modules:
            importlib.reload(sys.modules[name])


def slice_sums(e10, v10, edges):
    c = 0.5 * (e10[:-1] + e10[1:])
    return np.array([v10[(c >= a) & (c < b)].sum() for a, b in zip(edges[:-1], edges[1:])])


def legend_mc_first(ax, fontsize):
    h, l = ax.get_legend_handles_labels()
    idx = sorted(range(len(l)), key=lambda i: 0 if l[i].startswith("Simulation") or l[i].startswith("MC") else
                 1 if l[i].startswith("MC stat") else 2)
    ax.legend([h[i] for i in idx], [l[i] for i in idx], loc="upper right", fontsize=fontsize, handlelength=1.8)


def load_truths(ch, with_tails=False):
    from truth_sets import load_truth
    out = []
    for sub, name, label, color, ls in TRUTHS + (TAIL_TRUTHS if with_tails else []):
        path = C.RESULTS / ch / sub / VARIANT / name
        if not path.with_suffix(".npz").exists():
            logger.warning("%s %s: no %s, left out", C.DATASET, ch, path)
            continue
        e10, parts, _ = load_truth(path)
        out.append((label, color, ls, e10, np.asarray(parts["sum"], float)))
    return out


def plot_spectrum(P, ch, e, v, var, truths, out):
    from spectrum import centers, rebin
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND
    c = centers(e)
    err = np.sqrt(np.maximum(var, 0.0))
    sel = (c >= C.THRESHOLD) & (c <= 8000.0)
    fig, (ax, axr) = P.spectrum_figure(height=12)
    ax.errorbar(c[sel], v[sel], yerr=err[sel], fmt="o", ms=4, color="k", label="Simulation")
    stair_edges = np.append(e[:-1][sel], e[1:][sel][-1])
    for label, color, ls, e10, t10 in truths:
        _, t100, _ = rebin(e10, t10, np.zeros_like(t10))
        ax.plot(c[sel], np.maximum(t100[sel], P.YMIN), color=color, lw=1.8, ls=ls, label=label)
        pull = np.where(err > 0, (v - t100) / np.where(err > 0, err, 1.0), np.nan)
        axr.stairs(pull[sel], stair_edges, color=color, lw=1.2, ls=ls)
    for y in (-2, 2):
        axr.axhline(y, color="0.5", lw=0.6, ls="--")
    P.finish(ax, axr, channel=ch, xmax=8000.0, top=float(v[sel].max()), ymin=1e-3,
             extra=f"{C.DY_SHORT[VARIANT]}\nsmoothed backgrounds", ratio_label=r"(MC $-$ curve) / $\sigma_{\rm MC}$",
             ratio_lim=(-5, 5), ratio_ref=0.0)
    axr.set_ylabel(r"(MC $-$ curve) / $\sigma_{\rm MC}$", fontsize=20)
    legend_mc_first(ax, FONT_SIZE_LEGEND)
    P.save(fig, out)


def plot_slices(P, ch, edges, mc, err, truths, out):
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND
    c, w = 0.5 * (edges[:-1] + edges[1:]), edges[1:] - edges[:-1]
    fig, (ax, axr) = P.spectrum_figure(height=12)
    ax.errorbar(c, mc, xerr=w / 2, yerr=err, fmt="o", ms=5, color="k", lw=1.4, label="MC, 9 processes")
    top, rows = 0.0, []
    for label, color, ls, e10, t10 in truths:
        cs = slice_sums(e10, t10, edges)
        ax.step(edges, np.append(cs, cs[-1]), where="post", color=color, lw=1.8, ls=ls, label=label)
        d = cs - mc
        axr.stairs(d, edges, color=color, lw=1.8, ls=ls)
        top = max(top, float(np.abs(d).max()))
        rows.append((label, cs))
    axr.fill_between(edges, np.append(-err, -err[-1]), np.append(err, err[-1]), step="post", color="0.5", alpha=0.3, lw=0)
    ax.fill_between([], [], [], color="0.5", alpha=0.3, lw=0, label="MC stat. error (lower panel)")
    lim = 1.3 * top
    P.finish(ax, axr, channel=ch, xmin=edges[0], xmax=edges[-1], top=float(mc.max()), ymin=1e-2, ylabel="Events in slice",
             extra=f"{C.DY_SHORT[VARIANT]}\nsmoothed backgrounds", ratio_label=r"Curve $-$ MC [events]",
             ratio_lim=(-lim, lim), ratio_ref=0.0)
    axr.set_ylabel(r"Curve $-$ MC [events]", fontsize=20)
    legend_mc_first(ax, FONT_SIZE_LEGEND)
    P.save(fig, out)
    L = [f"# Events per mass slice, {C.ERA} {ch} {C.REGION_NAME}, {C.DY_SHORT[VARIANT]} ({C.LUMI:.1f} fb-1)", "",
         "Curve - MC in units of the MC statistical error in brackets.", "",
         "| slice [GeV] | MC | " + " | ".join(r[0].replace("$\\times$", "x") for r in rows) + " |",
         "|---|---|" + "---|" * len(rows)]
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        L.append(f"| {a:.0f}-{b:.0f} | {mc[i]:.3g} +- {err[i]:.2g} | "
                 + " | ".join(f"{cs[i]:.3g} ({(cs[i] - mc[i]) / err[i]:+.1f})" for _, cs in rows) + " |")
    out.with_name("slices.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["2024"], choices=sorted(C.DATASETS))
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--with-tails", action="store_true", help="also the truths with a fitted tail per component")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for ds in args.datasets:
        use_dataset(ds)
        import plots as P                                                # noqa: F811
        from spectrum import load_components, rebin                      # noqa: F811
        for ch in args.channels:
            comps = load_components(ch, VARIANT, "split9")
            e10, v10, var10 = comps["sum"]
            e, v, var = rebin(e10, v10, var10)
            g4 = load_components(ch, VARIANT, "groups4")["sum"][1].sum()
            logger.info("%s %s: MC sum %.2f (nine processes) / %.2f (four groups)", ds, ch, v10.sum(), g4)
            truths = load_truths(ch, args.with_tails)
            out = C.RESULTS / ch / ("20_truths_overlay_tails" if args.with_tails else "20_truths_overlay")
            out.mkdir(parents=True, exist_ok=True)
            plot_spectrum(P, ch, e, v, var, truths, out / "spectrum")
            edges = np.array(SLICES)
            plot_slices(P, ch, edges, slice_sums(e10, v10, edges), np.sqrt(slice_sums(e10, var10, edges)), truths, out / "slices")
            logger.info("%s %s: %d curves -> %s", ds, ch, len(truths), out)


if __name__ == "__main__":
    main()
