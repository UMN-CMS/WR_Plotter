#!/usr/bin/env python3
"""Expected limit vs m_WR from the flavor-CR anchored cards (AsymptoticLimits
results written by run_fcr.sh), as a Brazil band in fb -- no comparison curve.

  python plot_fcr_limit.py --results results_2024 --inputs inputs_2024 \\
      --label k5_bw50_fcrd --era RunIII2024Summer24 --out explain_fcr_2024/limit_vs_mass
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
import mplhep as hep
import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                  # repo root
from wrplotter.cli_utils import setup_logging             # noqa: E402
from wrplotter.config import load_lumi                    # noqa: E402

logger = logging.getLogger("plot_fcr_limit")
QUANT = {-2: 0.025, -1: 0.16, 0: 0.5, 1: 0.84, 2: 0.975}


def read_limits(results, label, masses):
    out = {}
    for m in masses:
        fp = results / f"higgsCombine_{label}.AsymptoticLimits.mH{m}.root"
        if not fp.exists():
            logger.warning("missing %s", fp.name)
            continue
        f = uproot.open(fp)
        if "limit" not in [k.split(";")[0] for k in f.keys()]:
            logger.warning("no limit tree in %s (combine failed there)", fp.name)
            continue
        t = f["limit"]
        q = t["quantileExpected"].array(library="np")
        l = t["limit"].array(library="np")
        row = {}
        for k, qq in QUANT.items():
            sel = np.isclose(q, qq, atol=2e-3)
            if sel.any():
                row[k] = float(l[sel][0])
        if len(row) == 5:
            out[m] = row
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=HERE / "results_2024")
    p.add_argument("--inputs", type=Path, default=HERE / "inputs_2024")
    p.add_argument("--label", default="k5_bw50_fcrd")
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--tag", default="ee_resolved")
    p.add_argument("--out", type=Path, default=HERE / "explain_fcr_2024" / "limit_vs_mass")
    p.add_argument("--mark", type=int, default=2000,
                   help="mass of the running example, marked on the plot")
    p.add_argument("--xmax", type=float, default=6000.0)
    p.add_argument("--note", default="",
                   help="extra text line under the region label")
    p.add_argument("--overlay", nargs="*", default=[], metavar="RESULTS:LABEL:LEGEND",
                   help="extra median curves, e.g. results_2024_mconly:mc:'background from simulation'")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    with open(args.inputs / f"{args.tag}.json") as fh:
        meta = json.load(fh)
    masses = sorted(int(m) for m in meta["masses"])
    lim = read_limits(args.results / args.tag, args.label, masses)
    ms = np.array(sorted(lim))
    band = {k: np.array([lim[m][k] for m in ms]) for k in QUANT}
    for m in ms:
        logger.info("m=%d  median %.3f fb  [%.3f, %.3f] 1s  [%.3f, %.3f] 2s",
                    m, lim[m][0], lim[m][-1], lim[m][1], lim[m][-2], lim[m][2])

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.fill_between(ms, band[-2], band[2], color="#f5d800", lw=0,
                    label=r"95% expected")
    ax.fill_between(ms, band[-1], band[1], color="#00cc00", lw=0,
                    label=r"68% expected")
    ax.plot(ms, band[0], color="black", lw=2.2, ls="--",
            label="median expected, background from data")
    if args.mark in lim:
        ax.plot([args.mark], [lim[args.mark][0]], "o", color="#e42536", ms=9,
                zorder=6, label=fr"$m_{{W_R}} = {args.mark}$: the running example")
    overlays = {}
    for spec in args.overlay:
        res, lab, legend = spec.split(":", 2)
        o_lim = read_limits(HERE / res / args.tag if not Path(res).is_absolute() else Path(res) / args.tag,
                            lab, masses)
        o_ms = np.array(sorted(o_lim))
        ax.plot(o_ms, [o_lim[m][0] for m in o_ms], color="#e42536", lw=2.4, ls="-",
                zorder=7, label=legend)
        overlays[legend] = {str(m): o_lim[m] for m in o_ms}
        for m in o_ms:
            logger.info("overlay %s m=%d  median %.3f fb  (ratio to main %.2f)", lab, m,
                        o_lim[m][0], o_lim[m][0] / lim[m][0] if m in lim else float("nan"))
    ax.set_yscale("log")
    ax.set_xlim(800.0, args.xmax)
    ax.set_ylim(1e-2, 1e2)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"$\sigma(pp \to W_R)\,\mathcal{B}(W_R \to eeq\bar q')$ [fb]")
    txt = ("ee\nResolved\n" fr"$m_N = m_{{W_R}}/2$" "\n"
           "background: 2024 simulation + $e\\mu$ data\n"
           "signal shapes: Run 2 samples")
    if args.note:
        txt += "\n" + args.note
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", fontsize=14)
    ax.legend(fontsize=14, loc="upper right")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=13)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    with open(args.out.with_suffix(".json"), "w") as fh:
        json.dump({"main": {str(m): lim[m] for m in ms}, "overlays": overlays}, fh, indent=1)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
