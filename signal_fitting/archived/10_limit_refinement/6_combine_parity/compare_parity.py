#!/usr/bin/env python3
"""Stage 10.6, step 3 -- compare the combine limits with the Stage-10.5 band.

Everything is in EVENTS (the parity cards set rate = 1, so combine's r is the
signal yield; the 10.5 table is a UL on N_sig).

Comparisons made, per mass:
  fit regime     combine AsymptoticLimits (fixed card)  vs  v2 fit band
                 combine AsymptoticLimits (prior card)  vs  same
                 -- expectation: ratio combine/v2 ~ 0.9-1.0 (both are now the
                 same Poisson-ML estimator; residual = per-toy-quantile vs
                 Asimov-asymptotic band construction and any leftover
                 pull-width != 1)
  sparse regime  combine HybridNew (anchored card, toys) vs v2 counting band
                 -- expectation: close; combine may sit slightly BELOW the
                 counting band because the in-window Gaussian shape carries a
                 little more discrimination than a pure count.  combine
                 AsymptoticLimits on the same card is also read out to show
                 the known asymptotic breakdown at a few events.

Outputs: parity_table_{ch}_{topo}.csv + a printed summary + ratio plot.

Setup (NOT the container -- needs uproot/matplotlib):
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from wrplotter.cli_utils import setup_logging                 # noqa: E402

logger = logging.getLogger("compare_parity")

QMAP = {0.025: "ul_m2s", 0.16: "ul_m1s", 0.5: "ul_med",
        0.84: "ul_p1s", 0.975: "ul_p2s"}


def read_asymptotic(path):
    """{quantile_key: limit} from an AsymptoticLimits output file."""
    t = uproot.open(path)["limit"]
    lim, q = t["limit"].array(library="np"), t["quantileExpected"].array(library="np")
    out = {}
    for li, qi in zip(lim, q):
        for qv, key in QMAP.items():
            if abs(qi - qv) < 1e-3:
                out[key] = float(li)
    return out


def read_hybrid(res_dir, variant, mass):
    """{quantile_key: limit} from HybridNew quantile files."""
    out = {}
    for q, key in ((0.16, "ul_m1s"), (0.5, "ul_med"), (0.84, "ul_p1s")):
        pat = f"{res_dir}/higgsCombine_{variant}.HybridNew.mH{mass}.quant{q:.3f}.root"
        hits = glob.glob(pat)
        if not hits:
            continue
        t = uproot.open(hits[0])["limit"]
        lim = t["limit"].array(library="np")
        if len(lim):
            out[key] = float(lim[-1])
    return out


def read_v2(v2_csv):
    """{mass: row} from the 10.5 table."""
    out = {}
    with open(v2_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            out[int(float(r["mWR"]))] = r
    return out


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", default="ee")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--v2-table", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    tag = f"{args.channel}_{args.topology}"
    res_dir = HERE / "results" / tag
    v2_csv = args.v2_table or (HERE.parents[0] / "5_expected_limits_v2"
                               / f"expected_limit_v2_table_{tag}.csv")
    v2 = read_v2(v2_csv)

    rows = []
    for f in sorted(glob.glob(str(res_dir / "higgsCombine_*.AsymptoticLimits.mH*.root"))):
        name = Path(f).name
        variant = name.split("higgsCombine_")[1].split(".")[0]
        mass = int(name.split(".mH")[1].split(".")[0])
        comb = read_asymptotic(f)
        row = {"mWR": mass, "variant": variant, "method": "asymptotic", **{
            k: round(v, 3) for k, v in comb.items()}}
        r2 = v2.get(mass)
        if r2:
            row["v2_med"] = _f(r2["ul_med"])
            row["v2_regime"] = r2["regime"]
            if comb.get("ul_med") and row["v2_med"] > 0:
                row["ratio_med"] = round(comb["ul_med"] / row["v2_med"], 3)
        rows.append(row)
    for f in sorted(glob.glob(str(res_dir / "higgsCombine_*.HybridNew.mH*.quant0.500.root"))):
        name = Path(f).name
        variant = name.split("higgsCombine_")[1].split(".")[0]
        mass = int(name.split(".mH")[1].split(".")[0])
        comb = read_hybrid(res_dir, variant, mass)
        row = {"mWR": mass, "variant": variant, "method": "hybridnew", **{
            k: round(v, 3) for k, v in comb.items()}}
        r2 = v2.get(mass)
        if r2:
            row["v2_med"] = _f(r2["ul_med"])
            row["v2_regime"] = r2["regime"]
            if comb.get("ul_med") and row["v2_med"] > 0:
                row["ratio_med"] = round(comb["ul_med"] / row["v2_med"], 3)
        rows.append(row)

    if not rows:
        sys.exit(f"no combine outputs found under {res_dir}")

    rows.sort(key=lambda r: (r["mWR"], r["variant"], r["method"]))
    logger.info("%6s %-9s %-11s %8s %8s %8s | %8s %8s",
                "mWR", "variant", "method", "med", "-1s", "+1s", "v2 med",
                "ratio")
    for r in rows:
        logger.info("%6d %-9s %-11s %8.2f %8.2f %8.2f | %8.2f %8s",
                    r["mWR"], r["variant"], r["method"],
                    r.get("ul_med", float("nan")),
                    r.get("ul_m1s", float("nan")),
                    r.get("ul_p1s", float("nan")),
                    r.get("v2_med", float("nan")),
                    r.get("ratio_med", "-"))

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    out_csv = args.output_dir / f"parity_table_{tag}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s", out_csv)

    # ratio plot
    fig, ax = plt.subplots(figsize=(8, 5))
    style = {("fixed", "asymptotic"): ("o", "#e42536", "fixed / v2 fit band"),
             ("prior", "asymptotic"): ("s", "#5790fc", "prior / v2 fit band"),
             ("anchored", "hybridnew"): ("D", "#2ca02c",
                                         "anchored HybridNew / v2 counting"),
             ("anchored", "asymptotic"): ("x", "#9c9ca1",
                                          "anchored asymptotic (breakdown)")}
    for (variant, method), (mk, color, label) in style.items():
        pts = [(r["mWR"], r["ratio_med"]) for r in rows
               if r["variant"] == variant and r["method"] == method
               and r.get("ratio_med")]
        if pts:
            ax.plot(*zip(*sorted(pts)), mk, ms=7, color=color, label=label)
    ax.axhline(1.0, color="black", lw=0.9, ls="--")
    ax.axhspan(0.9, 1.1, color="#2ca02c", alpha=0.12)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel("combine median / Stage-10.5 median")
    ax.set_ylim(0.4, 1.6)
    ax.legend(fontsize=9)
    ax.set_title(f"combine parity ({tag}, events convention)")
    out = args.output_dir / f"parity_ratio_{tag}"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Done.")


if __name__ == "__main__":
    main()
