#!/usr/bin/env python3
"""RooKeysPdf fed with one entry per populated 10 GeV bin vs one entry per
MC event, for one process (or `total`), over the binned MC.

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python compare_entries.py \\
      --bins run3/ee_resolved_split8/smoothed_kde.root \\
      --events run3/ee_resolved_split8_events/smoothed_kde.root --process DYJets
  -> <events dir>/plots/step1/compare_entries_<process>.{png,pdf}
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

logger = logging.getLogger("compare_entries")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bins", type=Path,
                   default=HERE / "run3" / "ee_resolved_split8" / "smoothed_kde.root")
    p.add_argument("--events", type=Path,
                   default=HERE / "run3" / "ee_resolved_split8_events" / "smoothed_kde.root")
    p.add_argument("--process", default="total")
    p.add_argument("--bw", type=float, default=50.0)
    p.add_argument("--xmax", type=float, default=6000.0)
    p.add_argument("--ylim", type=float, nargs=2, default=(2e-5, 2e4))
    p.add_argument("--legend", default="upper right")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    with open(args.events.with_suffix(".json")) as fh:
        meta = json.load(fh)
    era = meta.get("era", sl.ERA)
    info = load_lumi(era)
    lumi, com = meta.get("lumi", info["lumi"]), info.get("com", 13.0)

    fb, fe = uproot.open(args.bins), uproot.open(args.events)
    key = "bkg_smooth" if args.process == "total" else args.process
    rkey = "bkg_raw" if args.process == "total" else f"{args.process}_raw"
    edges = fb[rkey].axes[0].edges()
    v, var = fb[rkey].values(), fb[rkey].variances()
    assert np.allclose(v, fe[rkey].values()), "the two inputs are not the same MC"
    curves = [(fb[key].values(), "one entry per populated 10 GeV bin",
               dict(color="#5790fc", linestyle="--"), "s"),
              (fe[key].values(), "one entry per MC event",
               dict(color="#e42536", linestyle="-"), "o")]
    n_ev = (meta["components"].get(args.process, {}).get("n_entries")
            if args.process != "total" else
            sum(d.get("n_entries", 0) for d in meta["components"].values()))

    fac = int(round(args.bw / sl.NATIVE_BW))
    e5, v5, var5 = rebin(edges, v, var, fac)
    c5 = centers_of(e5)
    vis = (c5 >= XMIN) & (c5 <= args.xmax)
    pos = vis & (v5 > 0)
    fig, (ax, axr) = _figure()
    ax.errorbar(c5[pos], v5[pos], yerr=np.sqrt(var5[pos]), fmt="o", color="black",
                markersize=4, linewidth=1.0, zorder=6, label="MC (binned) $\\pm$ MC stat")
    for s, lab, st, mk in curves:
        _, s5, _ = rebin(edges, s, np.zeros_like(s), fac)
        ok = vis & (s5 > 0)
        ax.plot(c5[ok], s5[ok], linewidth=2.6, zorder=5, label=f"RooKeysPdf, {lab}", **st)
        okr = vis & (v5 > 0) & (s5 > 0)
        axr.errorbar(c5[okr], v5[okr] / s5[okr], yerr=np.sqrt(var5[okr]) / s5[okr],
                     fmt=mk, color=st["color"], markersize=3.5, linewidth=0.8,
                     markerfacecolor=st["color"] if mk == "o" else "white")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*args.ylim)
    ax.set_xlim(XMIN, args.xmax)
    ax.set_ylabel(f"Events / {args.bw:.0f} GeV")
    label = "Total background" if args.process == "total" else COMP_LABEL[args.process]
    ax.text(0.04, 0.95, f"{label}\nee resolved SR, {era}\n{n_ev} MC events",
            transform=ax.transAxes, fontsize=14, va="top")
    ax.legend(loc=args.legend, fontsize=12)
    axr.axhline(1.0, color="0.4", linewidth=1.2)
    axr.set_ylim(0.0, 2.5)
    axr.set_xlim(XMIN, args.xmax)
    axr.set_ylabel("MC / curve")
    axr.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    _cms(ax, lumi, com)
    _save(fig, args.events.parent / "plots" / "step1" / f"compare_entries_{args.process}")


if __name__ == "__main__":
    main()
