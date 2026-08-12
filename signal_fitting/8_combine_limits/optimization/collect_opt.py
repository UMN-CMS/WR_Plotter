#!/usr/bin/env python3
"""Stage 10.9, step 4 -- rank the variant configurations.

Reads every AsymptoticLimits result, tabulates the expected median (fb) per
(mass, variant), and ranks variants by their GEOMETRIC-MEAN ratio to the
baseline (k3_bw100_float) over 2000-3200. Writes opt_table_ee_resolved.csv
and a ratio-vs-mass plot per variant family.

  python collect_opt.py -v          (LCG_106)
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1].parent))          # repo root

from wrplotter.cli_utils import setup_logging            # noqa: E402

logger = logging.getLogger("collect_opt")
BASELINE = "k3_bw100_float"


def read_median(path):
    with uproot.open(path) as f:
        t = f["limit"]
        lim = t["limit"].array(library="np")
        q = t["quantileExpected"].array(library="np")
    for li, qi in zip(lim, q):
        if abs(qi - 0.5) < 1e-3:
            return float(li)
    return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path,
                   default=HERE / "results" / "ee_resolved")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)

    med = {}                                    # {variant: {mass: fb}}
    for f in sorted(args.results_dir.glob(
            "higgsCombine_*.AsymptoticLimits.mH*.root")):
        label = f.name.split("higgsCombine_")[1].split(".AsymptoticLimits")[0]
        mass = int(f.name.split(".mH")[1].split(".")[0])
        med.setdefault(label, {})[mass] = read_median(f)

    masses = sorted(med[BASELINE])
    base = med[BASELINE]

    rows, ranking = [], []
    for label, vals in med.items():
        ratios = [vals[m] / base[m] for m in masses
                  if m in vals and base.get(m, 0) > 0 and vals[m] > 0]
        gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        ranking.append((gm, label))
        rows.append({"variant": label,
                     **{f"med_{m}": round(vals.get(m, float("nan")), 5)
                        for m in masses},
                     "geomean_ratio": round(gm, 4)})
    ranking.sort()

    out = HERE / "opt_table_ee_resolved.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["geomean_ratio"]))
    logger.info("wrote %s", out)

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = plt.get_cmap("tab20")
    for i, (gm, label) in enumerate(ranking):
        if label == BASELINE:
            continue
        r = [med[label][m] / base[m] for m in masses if m in med[label]]
        mm = [m for m in masses if m in med[label]]
        ax.plot(mm, r, "o-", lw=1.8, ms=5, color=cmap(i % 20),
                label=f"{label}  (gm {gm:.3f})")
    ax.axhline(1, color="black", ls=":", lw=1.5)
    ax.set_xlabel(r"$m_{W_R}$ (GeV)")
    ax.set_ylabel("expected median / baseline (k3_bw100_float)")
    ax.legend(fontsize=10, frameon=False, ncol=2, loc="upper left")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi="59.8", com=13, fontsize=15)
    for ext in (".pdf", ".png"):
        fig.savefig(HERE / f"opt_ratios{ext}", bbox_inches="tight",
                    **({"dpi": 150} if ext == ".png" else {}))
    logger.info("wrote opt_ratios.pdf/.png")

    print(f"\nranking (geometric-mean expected-median ratio, 2000-3200):")
    for gm, label in ranking:
        marker = "  <- baseline" if label == BASELINE else ""
        print(f"  {gm:.3f}  {label}{marker}")
    print(f"\n{'m':>6}" + "".join(f"{m:>9}" for m in masses))
    for gm, label in ranking[:6]:
        print(f"{label:>22}: " + "".join(
            f"{med[label].get(m, float('nan')):>9.3f}" for m in masses))


if __name__ == "__main__":
    main()
