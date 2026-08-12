#!/usr/bin/env python3
"""Stage 10.2 -- deep scan of Gaussian priors on the S+B signal shape.

The Stage 5-8 S+B fit FIXES the signal Gaussian at the Stage-2 linear
(mu0, sigma0).  Stage 1's width-variation study showed the true width varies
~55-92 % with x = m_N/m_WR (sigma_true/sigma0 in [0.67, 1.77]), so a fixed
sigma under-covers the x-extremes; the alternative is to FLOAT (mu, sigma) with
Gaussian priors

    -2lnL += ((mu - mu0)/s_mu)^2 + ((sigma - sigma0)/s_sigma)^2
    s_mu = alpha_mu * sigma0,  s_sigma = alpha_sigma * sigma0

This script scans (alpha_mu, alpha_sigma) over a grid whose corners are the
limiting cases -- (0,0) = the fixed Stage 5-8 shape, (inf,inf) = bounded-free --
and, for every (mass, injection cell), runs PAIRED Poisson toys:

    mu[bin]  = bkg_MC[bin] + N_inj * shape[bin]
    each toy is drawn ONCE and fit by EVERY prior config

so config-to-config differences carry no toy noise.  N_inj = 0 is the null
(spurious) spread of each config; N_inj > 0 (default 9) is the recovery test
the user asked for (inject 9 -> recover 9 on average).  The injected shape is
the TRUE MC template at x = m_N/m_WR in --x-fracs (default min/0.5/0.9: the
U-shape extremes are exactly where the floating sigma should pay off).

Fit statistic: poisson (Stage-10 default; the chi2 estimator's multiplicative
under-recovery -- 0.53-0.84 of injected signal, Stage-8 tables -- is an
estimator pathology no prior can repair, so tuning priors on top of chi2 would
mis-attribute it).  --stat chi2 is kept for A/B.

The companion select_prior.py applies the documented gates + ranking to the
summary table and picks the recommended (alpha_mu, alpha_sigma).

Outputs (namespaced {ch}_{topo}):
  raw/{ch}_{topo}/m{mass}_N{n}_x{x}/{cfg}.csv    raw toys per config
  prior_scan_table_{ch}_{topo}.csv               summary per (cfg, mass, N, x)

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                          # prior_fit_lib, toy_engine
sys.path.insert(0, str(HERE.parents[2]))                      # repo root
sys.path.insert(0, str(HERE.parents[1] / "4_background_fits"))
sys.path.insert(0, str(HERE.parents[1] / "shared"))

from wrplotter.cli_utils import setup_logging                 # noqa: E402

import ROOT                                                   # noqa: E402
from bkg_fit_lib import FUNCS                                 # noqa: E402
from measure_fwhm import parse_masses                         # noqa: E402
from prior_fit_lib import fit_splusb_v2, gaussianity          # noqa: E402
from toy_engine import Inputs, accepted, write_raw            # noqa: E402

logger = logging.getLogger("prior_scan")

INF = "inf"          # sentinel: parameter floats without prior (bounded-free)


def a_label(v):
    return "inf" if v == INF else f"{float(v):g}".replace(".", "p")


def cfg_label(a_mu, a_sigma):
    return f"amu{a_label(a_mu)}_asig{a_label(a_sigma)}"


def a_to_width(a, sigma0):
    """Prior width in GeV from alpha ('inf' -> None = free, 0 -> 0.0 = fixed)."""
    return None if a == INF else float(a) * sigma0


def parse_alphas(vals):
    return [INF if str(v).lower() in ("inf", "free") else float(v) for v in vals]


def summarize_cell(recs, n_inj):
    """Summary metrics over one (config, cell) toy sample."""
    ok = [r for r in recs if r.get("acc")]
    out = {"n_ok": len(ok)}
    nsigs = [r["nsig"] for r in ok]
    g = gaussianity(nsigs)
    if not g:
        return out, None
    pulls = [(r["nsig"] - n_inj) / r["nsig_err"] for r in ok]
    gp = gaussianity(pulls)
    nrail_mu = sum(r.get("mu_railed", 0) for r in ok)
    nrail_sig = sum(r.get("sigma_railed", 0) for r in ok)
    nrail_nsig = sum(r.get("nsig_railed", 0) for r in ok)
    sig_fits = [r["sigma"] for r in ok if math.isfinite(r["sigma"])]
    mu_fits = [r["mu"] for r in ok if math.isfinite(r["mu"])]
    out.update({
        "mean_Nsig": round(g["mean"], 4), "median_Nsig": round(g["q50"], 4),
        "rms_Nsig": round(g["rms"], 4),
        "half68": round((g["q84"] - g["q16"]) / 2, 4),
        "bias_evt": round(g["mean"] - n_inj, 4),
        "bias_evt_err": round(g["rms"] / math.sqrt(g["n"]), 4),
        "median_bias_evt": round(g["q50"] - n_inj, 4),
        "jb_pvalue": round(g["jb_pvalue"], 5), "r68": round(g["r68"], 4),
        "pull_mean": round(gp["mean"], 4) if gp else "",
        "pull_width": round(gp["rms"], 4) if gp else "",
        "frac_mu_rail": round(nrail_mu / len(ok), 4),
        "frac_sigma_rail": round(nrail_sig / len(ok), 4),
        "frac_nsig_rail": round(nrail_nsig / len(ok), 4),
        "mean_mu_fit": round(float(np.mean(mu_fits)), 2) if mu_fits else "",
        "mean_sigma_fit": round(float(np.mean(sig_fits)), 2) if sig_fits else "",
        "rms_sigma_fit": round(float(np.std(sig_fits)), 2) if sig_fits else ""})
    return out, g


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20260317_lo_dy")
    p.add_argument("--signal-era", default="RunIISummer20UL18")
    p.add_argument("--signal-dir", default="20260624_signals",
                   help="canonical signal dir (has the WR1000-1800 points the "
                        "low-mass injections need; Stage 8's 20260406 does not)")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--k", type=float, default=3.0)
    p.add_argument("--sigma-kind", default="median",
                   choices=["median", "conservative"])
    p.add_argument("--stat", default="poisson", choices=["poisson", "chi2"])
    p.add_argument("--alpha-mu", nargs="+",
                   default=["0", "0.25", "0.5", "0.75", "1.0", "inf"])
    p.add_argument("--alpha-sigma", nargs="+",
                   default=["0", "0.1", "0.2", "0.3", "0.5", "inf"])
    p.add_argument("--inject", type=float, nargs="+", default=[0, 9])
    p.add_argument("--x-fracs", nargs="+", default=["min", "0.5", "0.9"],
                   help="injected x = m_N/m_WR targets ('min' = lowest grid "
                        "point); exercise the U-shape extremes")
    p.add_argument("--masses", nargs="+", type=float,
                   default=[1200, 1400, 2000, 2800, 3200, 3400, 4000],
                   help="trusted core {1400,2000,2800,3200} + edge diagnostics "
                        "{1200,3400,4000} (selection uses the core only)")
    p.add_argument("--functions", nargs="+", default=["expo"],
                   choices=list(FUNCS))
    p.add_argument("--ntoys", type=int, default=300,
                   help="paired toys per cell (each fit by every config)")
    p.add_argument("--seed", type=int, default=20260711)
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--no-raw", action="store_true")
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    channel, topology, k = args.channel, args.topology, args.k
    tag = f"{channel}_{topology}"
    inp = Inputs(era=args.era, bkg_dir=args.dir, signal_era=args.signal_era,
                 signal_dir=args.signal_dir, channel=channel,
                 topology=topology, bin_width=args.bin_width,
                 sigma_kind=args.sigma_kind)
    alphas_mu = parse_alphas(args.alpha_mu)
    alphas_sigma = parse_alphas(args.alpha_sigma)
    configs = [(a, b) for a in alphas_mu for b in alphas_sigma]
    logger.info("Prior scan %s: %d configs, masses %s, N %s, x %s, %d paired "
                "toys, stat=%s", tag, len(configs), args.masses, args.inject,
                args.x_fracs, args.ntoys, args.stat)

    rows = []
    for mWR in args.masses:
        m_c, sigma_win, lo, hi = inp.fit_range(mWR, k, args.fit_min, args.fit_max)

        # injection cells: null + one per (N>0, x-frac)
        cells = [(0.0, "", None, "")]                 # (N, xlabel, shape, tag)
        for n_inj in [n for n in args.inject if n > 0]:
            for xf in args.x_fracs:
                frac = 0.05 if xf == "min" else float(xf)
                stag = inp.signal_tag(mWR, frac)
                shape = inp.signal_shape(stag) if stag else None
                if shape is None:
                    logger.warning("  m=%.0f x=%s: no template (%s), skip",
                                   mWR, xf, stag)
                    continue
                cells.append((n_inj, xf, shape, stag))

        for n_inj, xlab, shape, stag in cells:
            mu_expect = inp.mu_bkg + (n_inj * shape if shape is not None else 0.0)
            mu_expect = np.where(np.isfinite(mu_expect) & (mu_expect > 0.0),
                                 mu_expect, 0.0)
            mN = parse_masses(stag)[1] if stag else ""
            x_true = (float(mN) / mWR) if mN != "" else ""
            for name in args.functions:
                # seed depends on the CELL only, never the config -> paired
                xcode = 0 if xlab == "" else int(round(100 * (0.05 if xlab == "min"
                                                              else float(xlab))))
                seed = (args.seed * 1_000_003 + int(mWR) * 1009
                        + int(round(n_inj)) * 101 + xcode * 13
                        + args.functions.index(name))
                rng = ROOT.TRandom3(seed)
                per_cfg = {cfg: [] for cfg in range(len(configs))}
                counts = {cfg: {"ntoys": args.ntoys, "n_ok": 0}
                          for cfg in range(len(configs))}
                for itoy in range(args.ntoys):
                    data = np.array([rng.Poisson(m) for m in mu_expect],
                                    dtype=float)
                    for icfg, (amu, asig) in enumerate(configs):
                        res = fit_splusb_v2(
                            name, inp.edges, data, lo, hi, m_c, sigma_win,
                            m_c, sigma_win, inp.binwidth, stat=args.stat,
                            s_mu=a_to_width(amu, sigma_win),
                            s_sigma=a_to_width(asig, sigma_win))
                        acc = accepted(res)
                        if acc:
                            counts[icfg]["n_ok"] += 1
                        rec = {"itoy": itoy, "acc": int(acc), "status": -1}
                        if res is not None:
                            rec.update({
                                "nsig": res["nsig"], "nsig_err": res["nsig_err"],
                                "mu": res["mu"], "sigma": res["sigma"],
                                "status": res["status"], "cov": res["cov"],
                                "passed": int(res["passed"]),
                                "mu_railed": int(res["mu_railed"]),
                                "sigma_railed": int(res["sigma_railed"]),
                                "nsig_railed": int(res["nsig_railed"])})
                        per_cfg[icfg].append(rec)

                for icfg, (amu, asig) in enumerate(configs):
                    cfg = cfg_label(amu, asig)
                    meta = {"channel": channel, "topology": topology,
                            "function": name, "stat": args.stat, "cfg": cfg,
                            "alpha_mu": amu, "alpha_sigma": asig,
                            "mWR": mWR, "N_inj": n_inj, "x_frac": xlab,
                            "x_true": round(x_true, 3) if x_true != "" else "",
                            "signal_tag": stag or "", "m_N": mN,
                            "m_c": round(m_c, 1),
                            "sigma_win": round(sigma_win, 2),
                            "fit_lo": round(lo, 1), "fit_hi": round(hi, 1)}
                    if not args.no_raw:
                        write_raw(args.output_dir / "raw" / tag
                                  / f"m{int(mWR)}_N{int(n_inj)}_x{xlab or 'na'}"
                                  / f"{cfg}.csv", meta, per_cfg[icfg])
                    summ, g = summarize_cell(per_cfg[icfg], n_inj)
                    conv = counts[icfg]["n_ok"] / args.ntoys
                    rows.append({**meta, "ntoys": args.ntoys,
                                 "conv": round(conv, 4), **summ})
                logger.info("  m=%.0f N=%-3g x=%-4s %s done (%d configs)",
                            mWR, n_inj, xlab or "-", name, len(configs))

    out_csv = args.output_dir / f"prior_scan_table_{tag}.csv"
    fields = []
    for r in rows:
        for k_ in r:
            if k_ not in fields:
                fields.append(k_)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", out_csv, len(rows))


if __name__ == "__main__":
    main()
