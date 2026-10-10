#!/usr/bin/env python3
"""Explainer: one RooKeysPdf setting, two values, one process.

Same entries, everything else as recorded in smoothed_kde.json; only the
chosen setting changes. Settings: var_kind (m | logm), mirror (RooKeysPdf
mirror option), rho (bandwidth multiplier), resample (sub-entries per bin).

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python explain_kde_setting.py \\
      --input run3/ee_resolved_split8/smoothed_kde.root --process tt_tW \\
      --setting mirror --values MirrorLeft NoMirror --xmax 2000 --ylim 1 1000
  -> <input dir>/plots/step1/explain_<setting>_<process>.{png,pdf}
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

logger = logging.getLogger("explain_kde_setting")
STYLES = [dict(color="#e42536", linestyle="-"), dict(color="#5790fc", linestyle="--"),
          dict(color="#2ca02c", linestyle=":")]
LABEL = {("var_kind", "m"): "bumps drawn in m", ("var_kind", "logm"): "bumps drawn in ln m",
         ("mirror", "MirrorLeft"): "MirrorLeft (entries reflected about 800 GeV)",
         ("mirror", "NoMirror"): "NoMirror (nothing below 800 GeV)",
         ("mirror", "MirrorBoth"): "MirrorBoth"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=HERE / "run3" / "ee_resolved_split8" / "smoothed_kde.root")
    p.add_argument("--process", default="tt_tW")
    p.add_argument("--setting", default="mirror",
                   choices=["var_kind", "mirror", "rho", "resample"])
    p.add_argument("--values", nargs="+", default=["MirrorLeft", "NoMirror"])
    p.add_argument("--bw", type=float, default=50.0)
    p.add_argument("--xmax", type=float, default=6000.0)
    p.add_argument("--ylim", type=float, nargs=2, default=(2e-5, 2e4))
    p.add_argument("--legend", default="upper right", help="matplotlib legend loc")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    with open(args.input.with_suffix(".json")) as fh:
        meta = json.load(fh)
    era = meta.get("era", sl.ERA)
    info = load_lumi(era)
    lumi, com = info["lumi"], info.get("com", 13.0)
    base = {k: meta["kwargs"][k] for k in ("var_kind", "mirror", "rho", "resample")}

    f = uproot.open(args.input)
    edges = f["bkg_raw"].axes[0].edges()
    v, var = f[f"{args.process}_raw"].values(), f[f"{args.process}_raw"].variances()
    cast = float if args.setting == "rho" else (int if args.setting == "resample" else str)
    fac = int(round(args.bw / sl.NATIVE_BW))
    e5, v5, var5 = rebin(edges, v, var, fac)
    c5 = centers_of(e5)
    vis = (c5 >= XMIN) & (c5 <= args.xmax)
    pos = vis & (v5 > 0)
    fig, (ax, axr) = _figure()
    ax.errorbar(c5[pos], v5[pos], yerr=np.sqrt(var5[pos]), fmt="o", color="black",
                markersize=4, linewidth=1.0, zorder=6, label="MC (binned) $\\pm$ MC stat")
    for i, val in enumerate(args.values):
        kw = {**base, args.setting: cast(val)}
        s, d = sl.smooth_kde(edges, v, var, **kw)
        lab = LABEL.get((args.setting, str(val)), f"{args.setting} = {val}")
        if val == str(base[args.setting]):
            lab += "  [what we use]"
        logger.info("%s = %s: total %.2f", args.setting, val, s.sum())
        _, s5, _ = rebin(edges, s, np.zeros_like(s), fac)
        ok = vis & (s5 > 0)
        st = STYLES[i % len(STYLES)]
        ax.plot(c5[ok], s5[ok], linewidth=2.6, zorder=5, label=lab, **st)
        okr = vis & (v5 > 0) & (s5 > 0)
        axr.errorbar(c5[okr], v5[okr] / s5[okr], yerr=np.sqrt(var5[okr]) / s5[okr],
                     fmt="o" if i == 0 else "s", color=st["color"], markersize=3.5,
                     linewidth=0.8, markerfacecolor=st["color"] if i == 0 else "white")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*args.ylim)
    ax.set_xlim(XMIN, args.xmax)
    ax.set_ylabel(f"Events / {args.bw:.0f} GeV")
    ax.text(0.04, 0.95, f"{COMP_LABEL[args.process]}\nee resolved SR, {era}",
            transform=ax.transAxes, fontsize=14, va="top")
    ax.legend(loc=args.legend, fontsize=12)
    axr.axhline(1.0, color="0.4", linewidth=1.2)
    axr.set_ylim(0.5, 1.5)
    axr.set_xlim(XMIN, args.xmax)
    axr.set_ylabel("MC / curve")
    axr.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    _cms(ax, lumi, com)
    _save(fig, args.input.parent / "plots" / "step1"
          / f"explain_{args.setting}_{args.process}")


if __name__ == "__main__":
    main()
