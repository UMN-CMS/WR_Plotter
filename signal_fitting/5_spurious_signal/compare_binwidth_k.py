#!/usr/bin/env python3
"""Stage 5 -- overlay the binning study across window widths k.

Reads the per-k binning tables written by rebinning_study.py
(`k{K}/rebinning_table_{ch}_{topo}.csv`) and shows, on ONE axis, how the
spurious result depends on BIN WIDTH for each window half-width k=2,3,4,5. The
mass dimension is collapsed into robust summaries over the trustworthy region
(m_WR <= mass_max), so bin width (x) and k (one curve each) both fit on a plot:

  k_comparison/{ch}_{topo}/binwidth_summary_{fn}.*
    top:    median |pull|        vs bin width, one curve per k
    bottom: median sigma(N_sp)   vs bin width, one curve per k

Passed-only fits (all four quality checks). A (k, bin width) point is dropped if
no mass in the trustworthy region produced a passing fit there.

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import logging
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                        # repo root
sys.path.insert(0, str(HERE.parents[0] / "4_background_fits"))  # bkg_fit_lib

from wrplotter.cli_utils import setup_logging                  # noqa: E402
from wrplotter.config import load_lumi                         # noqa: E402
from bkg_fit_lib import FUNCS, CH_LAB, TOPO_LAB                 # noqa: E402

logger = logging.getLogger("compare_binwidth_k")


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _load_k(output_dir, k, channel, topology, mass_max):
    """{function: {bin_width: [rows]}} of passed fits with m_WR <= mass_max."""
    path = output_dir / f"k{k}" / f"rebinning_table_{channel}_{topology}.csv"
    if not path.exists():
        logger.warning("missing %s", path)
        return {}
    out = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        if r["fit_ok"] != "True" or r.get("fit_passed") != "True":
            continue
        if float(r["mWR"]) > mass_max:
            continue
        out[r["function"]][float(r["bin_width"])].append(
            {"pull": float(r["pull"]), "nsig_err": float(r["N_spur_err"])})
    return out


def _k_colors(ks):
    cmap = matplotlib.colormaps["viridis"]
    n = max(len(ks) - 1, 1)
    return {k: cmap(i / n) for i, k in enumerate(sorted(ks))}


def plot_summary(per_k, name, ks, out, *, channel, topology, com, lumi, mass_max):
    colors = _k_colors(ks)
    hep.style.use("CMS")
    fig, (axp, axs) = plt.subplots(2, 1, sharex=True, figsize=(8.0, 9.0))
    fig.subplots_adjust(hspace=0.08)
    for k in ks:
        bw_map = per_k.get(k, {}).get(name, {})
        bws = sorted(bw_map)
        if not bws:
            continue
        med_pull = [st.median(abs(p["pull"]) for p in bw_map[b]) for b in bws]
        med_sig = [st.median(p["nsig_err"] for p in bw_map[b]) for b in bws]
        axp.plot(bws, med_pull, "o-", color=colors[k], ms=5, lw=1.4,
                 label=fr"$k={k}$")
        axs.plot(bws, med_sig, "o-", color=colors[k], ms=5, lw=1.4)
    axp.axhspan(0, 0.2, color="#74c476", alpha=0.30, lw=0, zorder=0)
    axp.axhspan(0.2, 0.5, color="#f7d600", alpha=0.20, lw=0, zorder=0)
    axp.set_ylabel(r"median $|{\rm pull}|$")
    axp.set_ylim(bottom=0)
    axs.set_ylabel(r"median $\sigma_{N_{\rm sp}}$ [events]")
    axs.set_ylim(bottom=0)
    axs.set_xlabel("bin width [GeV]")
    axp.text(0.03, 0.95,
             f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n{name}: {FUNCS[name][1]}"
             fr"  ($m_{{W_R}}\leq{mass_max:.0f}$)",
             transform=axp.transAxes, va="top", fontsize=12)
    axp.legend(fontsize=10, loc="upper left", bbox_to_anchor=(0.03, 0.80),
               ncol=2, title="window")
    hep.cms.label(loc=0, ax=axp, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    _save(fig, out)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4, 5])
    p.add_argument("--functions", nargs="+", default=["expo", "powlaw"],
                   choices=list(FUNCS))
    p.add_argument("--mass-max", type=float, default=3400.0)
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    channel, topology = args.channel, args.topology
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)

    per_k = {k: _load_k(args.output_dir, k, channel, topology, args.mass_max)
             for k in args.ks}
    ks = [k for k in args.ks if per_k[k]]
    if not ks:
        logger.error("No per-k rebinning tables under %s", args.output_dir)
        sys.exit(1)

    outd = args.output_dir / "k_comparison" / f"{channel}_{topology}"
    for name in args.functions:
        if not any(name in per_k[k] for k in ks):
            continue
        plot_summary(per_k, name, ks, outd / f"binwidth_summary_{name}",
                     channel=channel, topology=topology, com=com, lumi=lumi,
                     mass_max=args.mass_max)
        logger.info("  %s: wrote binwidth-vs-k summary", name)
    logger.info("Done. Outputs in %s", outd)


if __name__ == "__main__":
    main()
