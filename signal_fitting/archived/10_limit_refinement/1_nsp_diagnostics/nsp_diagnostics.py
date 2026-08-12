#!/usr/bin/env python3
"""Stage 10.1 -- N_sp Gaussianity diagnostics (raw toys saved).

Stage 6 established that only a subset of m_WR points have Gaussian nsp_hist
distributions, but it only stored mean/RMS summaries, so "which masses can the
mean+RMS+Gaussian-CLs chain be trusted at?" could not be answered quantitatively.
This step re-runs the null (background-only) toys, SAVES EVERY RAW FIT RESULT,
and scores each (mass, statistic, function) sample:

  conv        n_ok / ntoys  (the Stage-6 acceptance)
  skew, exkurt, jb_pvalue   Jarque-Bera normality test (TMath.Prob)
  r68, r95    quantile/RMS width ratios ((q84-q16)/2/RMS etc.; 1 for a Gaussian)
  clamped     window truncated by fit-min/fit-max (structural, e.g. m_WR=1000)

and classifies:  TRUSTED   conv >= 0.95 and jb_p >= 0.01 and |r68-1| <= 0.15
                            and not clamped
                 BROKEN    conv < 0.80 or n_ok < 200 or r68 < 0.50 or clamped
                 MARGINAL  everything else

Both statistics are run so the fix can be separated from the diagnosis:
  chi2     the Stage 5-8 convention (populated bins only) -- reproduces Stage 6
  poisson  Baker-Cousins including empty bins (prior_fit_lib) -- the Stage-10
           default, expected to remove the sparse-bin chi2 bias and extend the
           usable mass range

Outputs (namespaced {ch}_{topo}):
  raw/{ch}_{topo}/{stat}_{fn}_m{mass}.csv      every toy (acc + failures)
  nsp_overlay/{ch}_{topo}/{fn}/m{mass}.*       chi2 vs poisson N_sp overlay
  metrics_vs_mass/{ch}_{topo}/{fn}.*           conv/jb/r68/skew vs mass panels
  gaussianity_table_{ch}_{topo}.csv            all metrics + classification

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                          # prior_fit_lib, toy_engine
sys.path.insert(0, str(HERE.parents[2]))                      # repo root
sys.path.insert(0, str(HERE.parents[1] / "4_background_fits"))

from wrplotter.cli_utils import setup_logging                 # noqa: E402
from wrplotter.config import load_lumi                        # noqa: E402

from bkg_fit_lib import FUNCS, CH_LAB, TOPO_LAB               # noqa: E402
from prior_fit_lib import fit_splusb_v2, gaussianity          # noqa: E402
from toy_engine import Inputs, run_toys, write_raw, make_rng  # noqa: E402

logger = logging.getLogger("nsp_diagnostics")

STATS = ["chi2", "poisson"]
STAT_COLOR = {"chi2": "#e42536", "poisson": "#5790fc"}


def classify(g, counts, clamped, min_ok=200):
    conv = counts["n_ok"] / counts["ntoys"]
    if clamped or conv < 0.80 or counts["n_ok"] < min_ok \
            or (g is not None and g["r68"] < 0.50):
        return "BROKEN"
    if g is None:
        return "BROKEN"
    if conv >= 0.95 and g["jb_pvalue"] >= 0.01 and abs(g["r68"] - 1.0) <= 0.15:
        return "TRUSTED"
    return "MARGINAL"


def plot_overlay(nsps_by_stat, mWR, name, out, *, channel, topology, com, lumi,
                 hist_range, hist_bins):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    bins = np.linspace(-hist_range, hist_range, hist_bins + 1)
    for stat in STATS:
        v = nsps_by_stat.get(stat)
        if not v:
            continue
        clipped = np.clip(v, -hist_range, hist_range)
        ax.hist(clipped, bins=bins, histtype="step", lw=1.8,
                color=STAT_COLOR[stat],
                label=f"{stat} ({len(v)} ok)")
    ax.axvline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(r"fitted signal yield $\hat{N}_{\rm sig}$ [events]")
    ax.set_ylabel("toys")
    ax.text(0.03, 0.97,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            f"{name}\n" fr"$m_{{W_R}}={mWR:.0f}$ GeV",
            transform=ax.transAxes, va="top", fontsize=13)
    ax.legend(loc="upper right", fontsize=12)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_metrics(rows, name, out, *, channel, topology, com, lumi):
    """2x2 panels vs mass: conv, JB p-value, r68, skew -- chi2 vs poisson."""
    hep.style.use("CMS")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    panels = [("conv", "converged fraction", (0.0, 1.05), None),
              ("jb_pvalue", "Jarque-Bera p-value", (1e-12, 2.0), "log"),
              ("r68", r"$(q_{84}-q_{16})/2$ / RMS", (0.0, 1.6), None),
              ("skew", "skewness", (-2.0, 2.0), None)]
    for ax, (key, lab, ylim, yscale) in zip(axs.flat, panels):
        for stat in STATS:
            pts = [(r["mWR"], r[key]) for r in rows
                   if r["stat"] == stat and r["function"] == name
                   and r[key] != ""]
            if not pts:
                continue
            m, y = zip(*sorted(pts))
            ax.plot(m, y, "o-", ms=4, lw=1.2, color=STAT_COLOR[stat], label=stat)
        if key == "conv":
            ax.axhline(0.95, color="grey", lw=0.8, ls="--")
        if key == "jb_pvalue":
            ax.axhline(0.01, color="grey", lw=0.8, ls="--")
        if key == "r68":
            ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        if key == "skew":
            ax.axhline(0.0, color="grey", lw=0.8, ls="--")
        ax.set_ylabel(lab, fontsize=16)
        ax.set_ylim(*ylim)
        if yscale:
            ax.set_yscale(yscale)
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=12)
    for ax in axs[1]:
        ax.set_xlabel(r"$m_{W_R}$ [GeV]", fontsize=16)
    fig.suptitle(f"{CH_LAB[channel]} {TOPO_LAB[topology]}  --  {name}",
                 fontsize=17)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20260317_lo_dy")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--k", type=float, default=3.0)
    p.add_argument("--sigma-kind", default="median",
                   choices=["median", "conservative"])
    p.add_argument("--ntoys", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--masses", nargs="+", type=float,
                   default=[float(m) for m in range(1000, 5201, 200)])
    p.add_argument("--functions", nargs="+", default=["expo"],
                   choices=list(FUNCS))
    p.add_argument("--stats", nargs="+", default=STATS, choices=STATS)
    p.add_argument("--hist-range", type=float, default=60.0)
    p.add_argument("--hist-bins", type=int, default=48)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    channel, topology, k = args.channel, args.topology, args.k
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    tag = f"{channel}_{topology}"
    inp = Inputs(era=args.era, bkg_dir=args.dir, channel=channel,
                 topology=topology, bin_width=args.bin_width,
                 sigma_kind=args.sigma_kind)
    logger.info("Diagnostics: %s -- %d masses, stats %s, funcs %s, %d toys",
                tag, len(args.masses), args.stats, args.functions, args.ntoys)

    rows = []
    for mWR in args.masses:
        m_c, sigma_win, lo, hi = inp.fit_range(mWR, k, args.fit_min, args.fit_max)
        clamped = (lo > m_c - k * sigma_win + 1e-6) or (hi < m_c + k * sigma_win - 1e-6)
        b_win = inp.b_window(mWR, k)
        nsps_by_stat_fn = {}
        for name in args.functions:
            for si, stat in enumerate(args.stats):
                def fitter(data, _n=name, _s=stat):
                    return fit_splusb_v2(_n, inp.edges, data, lo, hi, m_c,
                                         sigma_win, m_c, sigma_win,
                                         inp.binwidth, stat=_s,
                                         s_mu=0.0, s_sigma=0.0)
                rng = make_rng(args.seed, mWR, 10 * args.functions.index(name) + si)
                recs, counts = run_toys(inp.mu_bkg, fitter, args.ntoys, rng)
                write_raw(args.output_dir / "raw" / tag
                          / f"{stat}_{name}_m{int(mWR)}.csv",
                          {"channel": channel, "topology": topology,
                           "function": name, "stat": stat, "mWR": mWR,
                           "m_c": round(m_c, 1), "sigma_win": round(sigma_win, 2),
                           "fit_lo": round(lo, 1), "fit_hi": round(hi, 1),
                           "B_window": round(b_win, 3)},
                          recs)
                nsps = [r["nsig"] for r in recs if r.get("acc")]
                nsps_by_stat_fn.setdefault(name, {})[stat] = nsps
                g = gaussianity(nsps)
                cls = classify(g, counts, clamped)
                conv = counts["n_ok"] / counts["ntoys"]
                row = {"channel": channel, "topology": topology,
                       "function": name, "stat": stat, "mWR": mWR,
                       "m_c": round(m_c, 1), "sigma_win": round(sigma_win, 2),
                       "B_window": round(b_win, 3), "clamped": int(clamped),
                       "ntoys": counts["ntoys"], "n_ok": counts["n_ok"],
                       "n_none": counts["n_none"],
                       "n_badstatus": counts["n_badstatus"],
                       "n_baderr": counts["n_baderr"],
                       "conv": round(conv, 4)}
                for key in ("mean", "rms", "q025", "q16", "q50", "q84", "q975",
                            "skew", "exkurt", "jb_pvalue", "r68", "r95"):
                    row[key] = round(g[key], 5) if g else ""
                row["class"] = cls
                rows.append(row)
                logger.info("  m=%.0f %-7s %-6s -> conv=%.2f jb_p=%s r68=%s  %s",
                            mWR, stat, name, conv,
                            f"{g['jb_pvalue']:.3f}" if g else "-",
                            f"{g['r68']:.2f}" if g else "-", cls)
        if not args.no_plots:
            for name in args.functions:
                plot_overlay(nsps_by_stat_fn.get(name, {}), mWR, name,
                             args.output_dir / "nsp_overlay" / tag / name
                             / f"m{int(mWR)}",
                             channel=channel, topology=topology, com=com,
                             lumi=lumi, hist_range=args.hist_range,
                             hist_bins=args.hist_bins)

    out_csv = args.output_dir / f"gaussianity_table_{tag}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s", out_csv)

    if not args.no_plots:
        for name in args.functions:
            plot_metrics(rows, name,
                         args.output_dir / "metrics_vs_mass" / tag / name,
                         channel=channel, topology=topology, com=com, lumi=lumi)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
