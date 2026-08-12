#!/usr/bin/env python3
"""Stage 9, step 5 -- plot the combine limits (official style) and overlay the
homemade band.

Reads `combine_limit_table_{tag}.csv` (from collect_limits.py) and the inputs
JSON, then produces, per function:

  plots/{tag}/{fn}.*          sigma x BR limit, official style -- identical
                              figure to Stage 7b, drawn with the same
                              `plot_band` imported from ../7_limit_plots
  plots/{tag}/{fn}_mu.*       the same in mu = sigma/sigma_theory
  plots/{tag}/{fn}_overlay.*  combine band vs the homemade closed-form band
                              (center-zero, sigma = toy RMS -- the comparable
                              convention), with a median-ratio subpanel

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                        # repo root
sys.path.insert(0, str(HERE.parents[1] / "7_limit_plots"))      # plot_band

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from xsec_limit import BANDS, cls_band, plot_band, _save, _cmslabel     # noqa: E402
from bkg_fit_lib import CH_LAB, TOPO_LAB                                # noqa: E402

logger = logging.getLogger("plot_combine_limits")

FB_KEYS = {-2: "comb_fb_m2s", -1: "comb_fb_m1s", 0: "comb_fb_med",
           1: "comb_fb_p1s", 2: "comb_fb_p2s"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_table(path, fn):
    pts = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["function"] != fn:
                continue
            xsec = _f(r["xsec_pb"])
            sigma = {N: _f(r[FB_KEYS[N]]) / 1000.0 for N in BANDS}   # fb -> pb
            if not math.isfinite(sigma[0]):
                continue
            obs = _f(r.get("comb_fb_obs")) / 1000.0
            pts.append({
                "mWR": _f(r["mWR"]), "sigma": sigma,
                "mu": {N: sigma[N] / xsec for N in BANDS},
                "sigma_obs": obs, "mu_obs": obs / xsec,
                "xsec_pb": xsec, "rate_per_fb": _f(r["rate_per_fb"]),
            })
    return sorted(pts, key=lambda p: p["mWR"])


def homemade_zero_band(meta, fn, alpha):
    """{mWR: {N: sigma_UL [fb]}} from the Stage-6 toy moments -- the homemade
    center-zero band, the convention comparable to combine's expected band."""
    out = {}
    for key, m in meta["masses"].items():
        ref = m["ref"].get(fn)
        if not ref or not isinstance(ref.get("rms_Nsp"), (int, float)):
            continue
        rms = ref["rms_Nsp"]
        if not (isinstance(rms, float) and math.isfinite(rms) and rms > 0):
            continue
        band = cls_band(0.0, rms, alpha)
        out[float(key)] = {N: band[N] / m["rate_per_fb"] for N in BANDS}
    return out


def plot_overlay(fn, pts, home, out, *, channel, topology, com, lumi, cl):
    """Combine band (filled) vs homemade center-zero band (lines) + median
    ratio subpanel."""
    hep.style.use("CMS")
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, height_ratios=[3, 1],
        gridspec_kw={"hspace": 0.06}, figsize=(10, 11))
    m = [p["mWR"] / 1000.0 for p in pts]
    fb = {N: [p["sigma"][N] * 1000.0 for p in pts] for N in BANDS}
    ax.fill_between(m, fb[-2], fb[2], color="#f5d800", label=r"combine $\pm2\sigma$")
    ax.fill_between(m, fb[-1], fb[1], color="#00cc00", label=r"combine $\pm1\sigma$")
    ax.plot(m, fb[0], "k--", lw=2, label="combine median")

    hm = sorted(k for k in home if any(abs(k - p["mWR"]) < 0.5 for p in pts))
    hx = [k / 1000.0 for k in hm]
    ax.plot(hx, [home[k][0] for k in hm], color="#5790fc", ls="-.", lw=2,
            label="homemade median (center-zero)")
    for N in (-1, 1):
        ax.plot(hx, [home[k][N] for k in hm], color="#5790fc", ls=":", lw=1.5,
                label=r"homemade $\pm1\sigma$" if N == 1 else None)
    ax.set_yscale("log")
    ax.set_ylabel(rf"{cl*100:.0f}% CL UL on $\sigma\,\mathcal{{B}}$ (fb)")
    ax.text(0.55, 0.96, f"{TOPO_LAB[topology]} {CH_LAB[channel]} channel\n"
            f"{fn} background fit, stat-only",
            transform=ax.transAxes, ha="center", va="top", fontsize=15)
    ax.legend(loc="lower left", fontsize=12)
    _cmslabel(ax, com, lumi)

    both = [(k, next(p for p in pts if abs(p["mWR"] - k) < 0.5)) for k in hm]
    axr.plot([k / 1000.0 for k, _ in both],
             [home[k][0] / (p["sigma"][0] * 1000.0) for k, p in both],
             "o-", color="#5790fc", ms=4, lw=1.5)
    axr.axhline(1.0, color="grey", lw=0.8, ls=":")
    axr.set_ylabel("home / comb", fontsize=14)
    axr.set_xlabel(r"$m_{W_R}$ (TeV)")
    axr.set_ylim(0.8, 1.8)
    _save(fig, out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", default="ee")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--functions", nargs="+", default=["expo", "powlaw"])
    p.add_argument("--cl", type=float, default=0.95)
    p.add_argument("--trust-max", type=float, default=None)
    p.add_argument("--run", default="run3", choices=["run2", "run3"],
                   help="run-label subdir for the default in/out dirs")
    p.add_argument("--observed", default="mc", choices=["mc", "data"],
                   help="what the collected results used as observation "
                        "(sets the observed-limit legend label)")
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--table", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    tag = f"{args.channel}_{args.topology}"
    if args.input_dir is None:
        args.input_dir = HERE / args.run / "inputs"
    if args.output_dir is None:
        args.output_dir = HERE / args.run / "plots"

    with open(args.input_dir / f"{tag}.json") as fh:
        meta = json.load(fh)
    lumi, com = meta["lumi"], meta.get("com", 13.6)
    table = args.table or (HERE / args.run / f"combine_limit_table_{tag}.csv")

    ll = {"ee": "ee", "mumu": r"\mu\mu"}[args.channel]
    is_data = args.observed == "data"
    common = dict(channel=args.channel, topology=args.topology, com=com,
                  lumi=lumi, cl=args.cl, trust_max=args.trust_max,
                  center="combine", data=is_data,
                  obs_label="Observed limit" if is_data
                  else "Observed limit (MC Asimov)")
    for fn in args.functions:
        pts = read_table(table, fn)
        if not pts:
            logger.warning("no points for %s", fn)
            continue
        out = args.output_dir / tag
        plot_band(fn, pts, out / fn, ykey="sigma", obskey="sigma_obs",
                  theory=True, scale=1000.0,
                  ylabel=(rf"$\sigma(pp \to W_R)\,\mathcal{{B}}"
                          rf"(W_R \to {ll}q\bar{{q}}\,')$ (fb)"), **common)
        plot_band(fn, pts, out / f"{fn}_mu", ykey="mu", obskey="mu_obs",
                  theory=False,
                  ylabel=(rf"{args.cl*100:.0f}% CL upper limit on "
                          r"$\mu = \sigma/\sigma_{\rm theory}$"), **common)
        home = homemade_zero_band(meta, fn, 1.0 - args.cl)
        plot_overlay(fn, pts, home, out / f"{fn}_overlay",
                     channel=args.channel, topology=args.topology,
                     com=com, lumi=lumi, cl=args.cl)
        logger.info("%s: %d combine points, %d homemade reference points",
                    fn, len(pts), len(home))
    logger.info("Plots in %s", args.output_dir / tag)


if __name__ == "__main__":
    main()
