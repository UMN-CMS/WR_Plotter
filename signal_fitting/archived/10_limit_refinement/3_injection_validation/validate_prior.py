#!/usr/bin/env python3
"""Stage 10.3 -- fresh-seed validation of the chosen prior config.

The 10.2 scan picked its winner on 300 paired toys and mc-template injections
at four masses.  This script re-tests the shortlist on INDEPENDENT seeds with
more toys, more masses, more injection levels, and the control shapes the scan
skipped -- so a winner that was luck, or that only works at N=9, gets caught:

  configs   fixed (0,0)              the Stage 5-8 baseline
            winner                   from chosen_prior_{ch}_{topo}.json
            runners-up               (if any)
            sig1p2 / sig1p35         FIXED shape with sigma0 statically
                                     inflated x1.2 / x1.35 -- the cheap
                                     alternative: if static inflation matches
                                     the winner, the floating machinery is
                                     unnecessary
            free (inf, inf)          bounded-free anchor (should fail)
  N_inj     {0, 9, 20}               9 = the user's benchmark; 20 = does a
                                     loose prior distort a LARGE signal
  shapes    mc                       true MC template (primary), x in
                                     {min, 0.2, 0.5, 0.9}
            gauss_matched            Gaussian exactly at (mu0, sigma0) -- the
                                     harness closure: recovery must be ~exact
                                     for every config (any deficit here is
                                     estimator floor, not shape mismatch)

Toys are PAIRED across configs (one Poisson draw, fit by every config).
Scoreboard: deficit = [mean(N) - mean(null)] - N*W  per cell, W = in-window
template fraction; a validated winner re-passes the 10.2 gates on every
trusted cell and beats fixed and the static inflations on max|deficit|.

Outputs: validation_table_{ch}_{topo}.csv + deficit/pull plots vs mass.

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
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / "4_background_fits"))
sys.path.insert(0, str(HERE.parents[1] / "shared"))

from wrplotter.cli_utils import setup_logging                 # noqa: E402

import ROOT                                                   # noqa: E402
from measure_fwhm import parse_masses                         # noqa: E402
from prior_fit_lib import fit_splusb_v2, gauss_template, gaussianity  # noqa: E402
from toy_engine import Inputs, accepted                       # noqa: E402

logger = logging.getLogger("validate_prior")


def build_configs(chosen_json):
    """[(label, alpha_mu, alpha_sigma, sigma0_scale)] -- alphas may be 'inf'."""
    cfgs = [("fixed", 0.0, 0.0, 1.0)]
    if chosen_json and Path(chosen_json).exists():
        with open(chosen_json) as fh:
            ch = json.load(fh)
        if ch.get("winner"):
            w = ch["winner"]
            cfgs.append(("winner", w["alpha_mu"], w["alpha_sigma"], 1.0))
        for i, r in enumerate(ch.get("runners_up", [])[:1]):
            cfgs.append((f"runner{i+1}", r["alpha_mu"], r["alpha_sigma"], 1.0))
    cfgs += [("sig1p2", 0.0, 0.0, 1.2), ("sig1p35", 0.0, 0.0, 1.35),
             ("free", "inf", "inf", 1.0)]
    return cfgs


def a_width(a, sigma0):
    if a in ("inf", float("inf")):
        return None
    return float(a) * sigma0


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20260317_lo_dy")
    p.add_argument("--signal-era", default="RunIISummer20UL18")
    p.add_argument("--signal-dir", default="20260624_signals")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--chosen", type=Path,
                   default=HERE.parents[0] / "2_prior_scan"
                   / "chosen_prior_ee_resolved.json")
    p.add_argument("--k", type=float, default=3.0)
    p.add_argument("--sigma-kind", default="median",
                   choices=["median", "conservative"])
    p.add_argument("--stat", default="poisson", choices=["poisson", "chi2"])
    p.add_argument("--inject", type=float, nargs="+", default=[0, 9, 20])
    p.add_argument("--x-fracs", nargs="+", default=["min", "0.2", "0.5", "0.9"])
    p.add_argument("--masses", nargs="+", type=float,
                   default=[1200, 1600, 2000, 2400, 2800, 3200, 3400])
    p.add_argument("--gauss-matched", action="store_true", default=True)
    p.add_argument("--no-gauss-matched", dest="gauss_matched",
                   action="store_false")
    p.add_argument("--ntoys", type=int, default=1000)
    p.add_argument("--seed", type=int, default=88031,
                   help="MUST differ from the 10.2 scan seed (fresh toys)")
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    tag = f"{args.channel}_{args.topology}"
    inp = Inputs(era=args.era, bkg_dir=args.dir, signal_era=args.signal_era,
                 signal_dir=args.signal_dir, channel=args.channel,
                 topology=args.topology, bin_width=args.bin_width,
                 sigma_kind=args.sigma_kind)
    configs = build_configs(args.chosen)
    logger.info("Validation %s: configs %s, N %s, x %s, %d toys, seed %d",
                tag, [c[0] for c in configs], args.inject, args.x_fracs,
                args.ntoys, args.seed)

    rows = []
    for mWR in args.masses:
        m_c, sigma_win, lo, hi = inp.fit_range(mWR, args.k, args.fit_min,
                                               args.fit_max)
        win = (inp.centers >= lo) & (inp.centers <= hi)

        # cells: null, mc injections, gauss_matched injections
        cells = [(0.0, "", "none", None, "", 1.0)]
        for n_inj in [n for n in args.inject if n > 0]:
            for xf in args.x_fracs:
                frac = 0.05 if xf == "min" else float(xf)
                stag = inp.signal_tag(mWR, frac)
                shape = inp.signal_shape(stag) if stag else None
                if shape is None:
                    continue
                W = float(shape[win].sum())
                cells.append((n_inj, xf, "mc", shape, stag, W))
            if args.gauss_matched:
                g = gauss_template(inp.centers, m_c, sigma_win, inp.binwidth)
                W = float(g[win].sum())
                cells.append((n_inj, "", "gauss_matched", g, "", W))

        for n_inj, xlab, kind, shape, stag, W in cells:
            mu_expect = inp.mu_bkg + (n_inj * shape if shape is not None else 0.0)
            mu_expect = np.where(np.isfinite(mu_expect) & (mu_expect > 0.0),
                                 mu_expect, 0.0)
            xcode = 0 if xlab == "" else int(round(100 * (0.05 if xlab == "min"
                                                          else float(xlab))))
            kcode = {"none": 0, "mc": 1, "gauss_matched": 2}[kind]
            seed = (args.seed * 1_000_003 + int(mWR) * 1009
                    + int(round(n_inj)) * 101 + xcode * 13 + kcode * 7)
            rng = ROOT.TRandom3(seed)
            per_cfg = {c[0]: [] for c in configs}
            for _ in range(args.ntoys):
                data = np.array([rng.Poisson(m) for m in mu_expect],
                                dtype=float)
                for label, amu, asig, sscale in configs:
                    res = fit_splusb_v2(
                        "expo", inp.edges, data, lo, hi, m_c, sigma_win,
                        m_c, sigma_win * sscale, inp.binwidth, stat=args.stat,
                        s_mu=a_width(amu, sigma_win),
                        s_sigma=a_width(asig, sigma_win))
                    per_cfg[label].append(res)
            for label, amu, asig, sscale in configs:
                ok = [r for r in per_cfg[label] if accepted(r)]
                nsigs = [r["nsig"] for r in ok]
                g = gaussianity(nsigs)
                pulls = [(r["nsig"] - n_inj * W) / r["nsig_err"] for r in ok]
                gp = gaussianity(pulls)
                row = {"channel": args.channel, "topology": args.topology,
                       "cfg": label, "alpha_mu": amu, "alpha_sigma": asig,
                       "sigma0_scale": sscale, "stat": args.stat,
                       "mWR": mWR, "N_inj": n_inj, "x_frac": xlab,
                       "kind": kind, "signal_tag": stag,
                       "W_win": round(W, 4), "target": round(n_inj * W, 3),
                       "ntoys": args.ntoys, "n_ok": len(ok),
                       "conv": round(len(ok) / args.ntoys, 4)}
                if g:
                    row.update({
                        "mean_Nsig": round(g["mean"], 4),
                        "rms_Nsig": round(g["rms"], 4),
                        "half68": round((g["q84"] - g["q16"]) / 2, 4),
                        "q50_Nsig": round(g["q50"], 4),
                        "jb_pvalue": round(g["jb_pvalue"], 5),
                        "pull_mean": round(gp["mean"], 4) if gp else "",
                        "pull_width": round(gp["rms"], 4) if gp else "",
                        "frac_sigma_rail": round(
                            sum(r.get("sigma_railed", 0) for r in ok)
                            / len(ok), 4),
                        "mean_sigma_fit": round(float(np.mean(
                            [r["sigma"] for r in ok])), 2)})
                rows.append(row)
            logger.info("  m=%.0f N=%-3g %-14s x=%-4s done", mWR, n_inj,
                        kind, xlab or "-")

    # deficits vs the per-config null (added as columns)
    null_mean = {(r["cfg"], r["mWR"]): _sf(r.get("mean_Nsig"))
                 for r in rows if r["N_inj"] == 0.0}
    for r in rows:
        if r["N_inj"] > 0 and r.get("mean_Nsig") not in (None, ""):
            nm = null_mean.get((r["cfg"], r["mWR"]))
            if nm is not None and math.isfinite(nm):
                r["deficit"] = round(
                    r["mean_Nsig"] - nm - r["target"], 4)

    out_csv = args.output_dir / f"validation_table_{tag}.csv"
    fields = []
    for r in rows:
        for k_ in r:
            if k_ not in fields:
                fields.append(k_)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", out_csv, len(rows))

    # scoreboard plot: max |deficit| per (config, mass) over mc cells, N=9
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"fixed": "#e42536", "winner": "#5790fc", "runner1": "#9c9ca1",
              "sig1p2": "#f89c20", "sig1p35": "#964a8b", "free": "#2ca02c"}
    for label in {r["cfg"] for r in rows}:
        pts = {}
        for r in rows:
            if (r["cfg"] == label and r["N_inj"] == 9.0 and r["kind"] == "mc"
                    and r.get("deficit") not in (None, "")):
                pts.setdefault(r["mWR"], []).append(abs(r["deficit"]))
        if not pts:
            continue
        m = sorted(pts)
        axs[0].plot(m, [max(pts[x]) for x in m], "o-", ms=4,
                    color=colors.get(label, "black"), label=label)
        nulls = sorted([(r["mWR"], _sf(r.get("half68")))
                        for r in rows if r["cfg"] == label
                        and r["N_inj"] == 0.0 and r.get("half68")])
        if nulls:
            axs[1].plot(*zip(*nulls), "o-", ms=4,
                        color=colors.get(label, "black"), label=label)
    axs[0].axhline(2.0, color="grey", lw=0.8, ls="--")
    axs[0].set_xlabel(r"$m_{W_R}$ [GeV]")
    axs[0].set_ylabel(r"max over x of |deficit| at $N=9$ [events]")
    axs[0].legend(fontsize=9)
    axs[1].set_xlabel(r"$m_{W_R}$ [GeV]")
    axs[1].set_ylabel("null half68 [events]")
    axs[1].legend(fontsize=9)
    fig.suptitle(f"{tag}  (fresh seed {args.seed}, {args.ntoys} toys)")
    out = args.output_dir / f"scoreboard_{tag}"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Done.")


def _sf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
