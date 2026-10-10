#!/usr/bin/env python3
"""Explainer: RooKeysPdf on ln m versus on m, for one process.

Same entries, same settings; only the axis the bumps are drawn on changes.

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python explain_kde_variable.py \\
      --input run3/ee_resolved_split8/smoothed_kde.root --process tt_tW
  -> <input dir>/plots/step1/explain_variable_<process>.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker as mticker
import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import smooth_lib as sl                                                  # noqa: E402
from smooth_lib import COMP_LABEL, centers_of, rebin                     # noqa: E402
from smooth_background import XMIN, _cms, _save                          # noqa: E402
from plot_step1_kde import _figure                                        # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter              # noqa: E402

logger = logging.getLogger("explain_kde_variable")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=HERE / "run3" / "ee_resolved_split8" / "smoothed_kde.root")
    p.add_argument("--process", default="tt_tW")
    p.add_argument("--bw", type=float, default=50.0)
    p.add_argument("--xmax", type=float, default=6000.0)
    p.add_argument("--ylim", type=float, nargs=2, default=(2e-5, 2e4))
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    with open(args.input.with_suffix(".json")) as fh:
        meta = json.load(fh)
    era = meta.get("era", sl.ERA)
    info = load_lumi(era)
    lumi, com = info["lumi"], info.get("com", 13.0)
    kw = {k: meta["kwargs"][k] for k in ("mirror", "rho", "resample")}

    f = uproot.open(args.input)
    edges = f["bkg_raw"].axes[0].edges()
    v, var = f[f"{args.process}_raw"].values(), f[f"{args.process}_raw"].variances()
    curves = {}
    for kind, lab in (("m", "bumps drawn in m (what we use)"),
                      ("logm", "bumps drawn in ln m")):
        s, d = sl.smooth_kde(edges, v, var, var_kind=kind, **kw)
        curves[kind] = (s, lab)
        logger.info("%s: total %.2f", kind, s.sum())

    fac = int(round(args.bw / sl.NATIVE_BW))
    e5, v5, var5 = rebin(edges, v, var, fac)
    c5 = centers_of(e5)
    vis = (c5 >= XMIN) & (c5 <= args.xmax)
    pos = vis & (v5 > 0)
    fig, (ax, axr) = _figure()
    ax.errorbar(c5[pos], v5[pos], yerr=np.sqrt(var5[pos]), fmt="o", color="black",
                markersize=4, linewidth=1.0, zorder=6, label="MC (binned) $\\pm$ MC stat")
    style = {"m": dict(color="#e42536", linestyle="-"),
             "logm": dict(color="#5790fc", linestyle="--")}
    for kind, (s, lab) in curves.items():
        _, s5, _ = rebin(edges, s, np.zeros_like(s), fac)
        ok = vis & (s5 > 0)
        ax.plot(c5[ok], s5[ok], linewidth=2.6, zorder=5, label=lab, **style[kind])
        okr = vis & (v5 > 0) & (s5 > 0)
        axr.errorbar(c5[okr], v5[okr] / s5[okr], yerr=np.sqrt(var5[okr]) / s5[okr],
                     fmt="o" if kind == "m" else "s", color=style[kind]["color"],
                     markersize=3, linewidth=0.8,
                     markerfacecolor=style[kind]["color"] if kind == "m" else "white")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*args.ylim)
    ax.set_xlim(XMIN, args.xmax)
    ax.set_ylabel(f"Events / {args.bw:.0f} GeV")
    ax.text(0.04, 0.95, f"{COMP_LABEL[args.process]}\nee resolved SR, {era}",
            transform=ax.transAxes, fontsize=14, va="top")
    ax.legend(loc="upper right", fontsize=12)
    axr.axhline(1.0, color="0.4", linewidth=1.2)
    axr.set_ylim(0.0, 2.5)
    axr.set_xlim(XMIN, args.xmax)
    axr.set_ylabel("MC / curve")
    axr.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    _cms(ax, lumi, com)
    _save(fig, args.input.parent / "plots" / "step1" / f"explain_variable_{args.process}")


if __name__ == "__main__":
    main()
