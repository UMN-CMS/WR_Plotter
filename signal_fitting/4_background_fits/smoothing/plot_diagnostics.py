#!/usr/bin/env python3
"""Stage 4 / smoothing -- diagnostic plots: the binned MC histogram and its
smoothed expectation, one method at a time.

Reads run2/ee_resolved/smoothed_<method>.root (written by
smooth_background.py) and draws, at the card bin width (50 GeV):

  plots/diag/<method>/<comp>.*        raw MC (points +- MC stat) with the
                                      smoothed expectation, log y, ratio panel
                                      raw / smooth with the MC-stat band
  plots/diag/<method>/sum.*           the same for the summed background
  plots/diag/<method>/grid.*          all components + sum on one page

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python plot_diagnostics.py -v
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
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import smooth_lib as sl                                                  # noqa: E402
from smooth_lib import (COMP_LABEL, METHODS, METHOD_COLOR, METHOD_LABEL,   # noqa: E402
                        centers_of, rebin)
from smooth_background import COMP_COLOR, XMIN, XMAX, _cms, _save         # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter              # noqa: E402

logger = logging.getLogger("plot_diagnostics")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=HERE / "run2" / "ee_resolved")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--bw", type=float, default=50.0, help="plot bin width (card)")
    p.add_argument("--xmax", type=float, default=XMAX)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def read(path):
    f = uproot.open(path)
    comps = [k.rstrip(";1") for k in f.keys()
             if not k.rstrip(";1").endswith("_raw") and not k.startswith("bkg_")]
    out, edges = {}, None
    for c in comps:
        hs, hr = f[c], f[f"{c}_raw"]
        edges = hs.axes[0].edges()
        out[c] = (hs.values(), hr.values(), hr.variances())
    out["sum"] = (sum(v[0] for v in out.values()), sum(v[1] for v in out.values()),
                  sum(v[2] for v in out.values()))
    return out, edges


def draw_panel(ax, axr, name, s, v, var, edges, method, bw, xmax, *, fontsize=13):
    f = int(round(bw / sl.NATIVE_BW))
    e5, v5, var5 = rebin(edges, v, var, f)
    _, s5, _ = rebin(edges, s, np.zeros_like(s), f)
    c5 = centers_of(e5)
    err5 = np.sqrt(var5)
    vis = (c5 >= XMIN) & (c5 <= xmax)
    pos = vis & (v5 > 0)
    # raw MC: points with MC-stat error bars (empty bins drawn at the axis floor)
    ax.errorbar(c5[pos], v5[pos], yerr=err5[pos], fmt="o", color="black",
                markersize=3.5, linewidth=0.9, capsize=0, zorder=4,
                label="MC (binned) $\\pm$ MC stat")
    ax.stairs(np.where(v5 > 0, v5, np.nan), e5, color="black", linewidth=0.8,
              alpha=0.5, zorder=3)
    sm = vis & (s5 > 0)
    ax.plot(c5[sm], s5[sm], color=METHOD_COLOR[method], linewidth=2.2, zorder=5,
            label=f"smoothed: {METHOD_LABEL[method]}")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ymax = max(v5[pos].max(), s5[sm].max()) if pos.any() else 1.0
    ymin = min(v5[pos].min(), s5[sm].min()) if pos.any() else 1e-3
    ax.set_ylim(max(ymin / 5, ymax * 1e-7), ymax * 40)
    ax.set_xlim(XMIN, xmax)
    ax.set_ylabel(f"Events / {bw:.0f} GeV", fontsize=fontsize + 2)
    ax.text(0.04, 0.95, f"{COMP_LABEL[name]}\nee resolved SR, {sl.ERA}",
            transform=ax.transAxes, fontsize=fontsize, va="top")
    ax.legend(loc="upper right", fontsize=fontsize - 2)
    if axr is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((v5 > 0) & (s5 > 0), v5 / s5, np.nan)
            rerr = np.where((v5 > 0) & (s5 > 0), err5 / s5, np.nan)
        axr.errorbar(c5[vis], ratio[vis], yerr=rerr[vis], fmt="o", color="black",
                     markersize=3, linewidth=0.8)
        axr.axhline(1.0, color=METHOD_COLOR[method], linewidth=1.5)
        axr.set_ylim(0.0, 2.5)
        axr.set_ylabel("MC / smooth", fontsize=fontsize + 1)
        axr.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")


def plot_one(name, hvv, edges, method, *, bw, xmax, lumi, com, out):
    s, v, var = hvv
    hep.style.use("CMS")
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(10, 10),
                                  gridspec_kw={"height_ratios": [3, 1],
                                               "hspace": 0.06})
    draw_panel(ax, axr, name, s, v, var, edges, method, bw, xmax)
    _cms(ax, lumi, com)
    _save(fig, out)


def plot_grid(data, edges, method, *, bw, xmax, lumi, com, out):
    names = [n for n in data if n != "sum"] + ["sum"]
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    hep.style.use("CMS")
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 5.6 * nrow))
    axes = np.atleast_2d(axes)
    for ax, name in zip(axes.flat, names):
        s, v, var = data[name]
        draw_panel(ax, None, name, s, v, var, edges, method, bw, xmax, fontsize=11)
        ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]", fontsize=14)
        ax.tick_params(labelsize=12)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    _cms(axes[0, 0], lumi, com, fontsize=12)
    fig.suptitle(f"binned MC vs smoothed expectation -- {METHOD_LABEL[method]}",
                 fontsize=16, y=1.0)
    fig.tight_layout()
    _save(fig, out)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    info = load_lumi(sl.ERA)
    lumi, com = info["lumi"], info.get("com", 13.0)
    for method in args.methods:
        data, edges = read(args.input_dir / f"smoothed_{method}.root")
        pdir = args.input_dir / "plots" / "diag" / method
        for name, hvv in data.items():
            plot_one(name, hvv, edges, method, bw=args.bw, xmax=args.xmax,
                     lumi=lumi, com=com, out=pdir / name)
        plot_grid(data, edges, method, bw=args.bw, xmax=args.xmax, lumi=lumi,
                  com=com, out=pdir / "grid")


if __name__ == "__main__":
    main()
