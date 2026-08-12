#!/usr/bin/env python3
"""Stage 7 -- expected-limit ("Brazil band") plots from the Stage-6 toys.

Turns the Stage-6 spurious-signal toy distributions into an expected 95% CL
upper-limit-on-the-signal-yield band vs m_WR. The Stage-6 toys are Poisson draws
of the background-only MC, each fit with bkg + fixed-shape Gaussian; the fitted
signal yield N_sp is the background-only distribution of the estimator. That
distribution IS the ingredient of an expected-limit Brazil band, so here we map
it into a limit.

Method -- CLs asymptotic (Cowan, Cranmer, Gross, Vitells 2011), evaluated with
the toy-measured moments.  For a Gaussian estimator N_hat ~ N(mu0, sigma) under
background-only, the one-sided CLs 95% CL upper limit on the signal yield is

    UL(N_hat) = N_hat + sigma * Phi^{-1}( 1 - alpha * Phi(N_hat/sigma) )

(alpha = 1 - CL = 0.05).  UL(.) is monotonic in N_hat, so the p-quantile of the
limit distribution is UL evaluated at the p-quantile of N_hat.  The +-N sigma
Brazil band is therefore, with mu0 = <N_sp> and sigma = RMS(N_sp) read straight
from the Stage-6 CSV,

    UL_N = mu0 + sigma*N + sigma * Phi^{-1}( 1 - alpha * Phi(N + mu0/sigma) )
                                                     for N in {-2,-1,0,+1,+2}

i.e. exactly "integrate the (per-toy) limit distribution from the left and read
the 2.5 / 16 / 50 / 84 / 97.5 % quantiles", in closed form.  sigma is the
*measured toy spread* (RMS), not the covariance-matrix fit error -- the toys'
pull width ~1.1 shows the fit errors slightly under-cover, so the RMS is the
honest sigma.  The CLs flavour keeps the -2 sigma edge positive.

  * median expected limit  = UL_0  ~ 1.96 sigma  (dashed line)
  * green  (+-1 sigma) band = [UL_-1, UL_+1]
  * yellow (+-2 sigma) band = [UL_-2, UL_+2]
  * the nominal-MC spurious yield N_sp (Stage-5/6 Asimov value) is overlaid so
    the fit bias can be read against the statistical sensitivity.

Input:  the Stage-6 summary CSV `spurious_toy_table_{ch}_{topo}.csv`
Outputs (namespaced by {channel}_{topology}):
  expected_limit/{ch}_{topo}/{fn}.*        Brazil band vs m_WR, one per function
  expected_limit_table_{ch}_{topo}.csv     the band (UL_-2..UL_+2) per (mass, fn)

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                        # repo root
sys.path.insert(0, str(HERE.parents[0] / "4_background_fits"))  # bkg_fit_lib

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402

import ROOT                                                             # noqa: E402
from bkg_fit_lib import FUNCS, CH_LAB, TOPO_LAB                         # noqa: E402

logger = logging.getLogger("expected_limit")

# +-N sigma quantiles of the standard normal, used to label the band.
BANDS = (-2, -1, 0, 1, 2)


# ---------------------------------------------------------------------------
# CLs asymptotic band
# ---------------------------------------------------------------------------

def _cls_upper_limit(n_hat, sigma, alpha):
    """One-sided CLs asymptotic (Gaussian) 95% CL upper limit for a single
    estimator N_hat with standard deviation sigma. See module docstring."""
    if not (sigma > 0.0 and math.isfinite(sigma) and math.isfinite(n_hat)):
        return float("nan")
    p = 1.0 - alpha * ROOT.TMath.Freq(n_hat / sigma)   # Freq = standard-normal CDF
    p = min(max(p, 1e-12), 1.0 - 1e-12)                # keep NormQuantile finite
    return n_hat + sigma * ROOT.TMath.NormQuantile(p)


def cls_band(mu0, sigma, alpha):
    """The +-2/-1/0/1/2 sigma expected-limit band as a dict {N: UL_N}.  The band
    is UL evaluated at the +-N sigma quantiles of the (Gaussian) toy N_sp
    distribution N(mu0, sigma) -- valid because UL(.) is monotonic in N_hat."""
    if not (sigma > 0.0 and math.isfinite(sigma)):
        return {N: float("nan") for N in BANDS}
    return {N: _cls_upper_limit(mu0 + sigma * N, sigma, alpha) for N in BANDS}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _f(x):
    """Parse a CSV cell to float, returning nan for the empty (skipped) cells."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_table(csv_path, channel, topology):
    """Read the Stage-6 summary CSV -> list of per-(mass, function) dicts with
    the fields this stage needs: mWR, mean/RMS(N_sp), and the nominal-MC N_sp."""
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if r.get("channel") != channel or r.get("topology") != topology:
            continue
        out.append({
            "function": r["function"],
            "mWR": _f(r["mWR"]),
            "mean_Nsp": _f(r.get("mean_Nsp")),
            "rms_Nsp": _f(r.get("rms_Nsp")),
            "nsp_asimov": _f(r.get("nsp_asimov")),
            "nsp_asimov_err": _f(r.get("nsp_asimov_err")),
        })
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _cmslabel(ax, com, lumi):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)


def plot_brazil(name, pts, out, *, channel, topology, com, lumi, cl, trust_max,
                center):
    """Expected 95% CL upper-limit band on the signal yield vs m_WR: yellow
    (+-2 sigma), green (+-1 sigma), dashed median, with the nominal-MC spurious
    yield overlaid."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    pts = sorted(pts, key=lambda p: p["mWR"])
    m = [p["mWR"] for p in pts]
    lo2 = [p["band"][-2] for p in pts]
    lo1 = [p["band"][-1] for p in pts]
    med = [p["band"][0] for p in pts]
    hi1 = [p["band"][1] for p in pts]
    hi2 = [p["band"][2] for p in pts]
    ax.fill_between(m, lo2, hi2, color="#f5d800", label=r"$\pm2\sigma$ expected")
    ax.fill_between(m, lo1, hi1, color="#00cc00", label=r"$\pm1\sigma$ expected")
    ax.plot(m, med, color="black", lw=1.6, ls="--",
            label=f"median expected {cl*100:.0f}% CL limit")
    asi = [(p["mWR"], p["nsp_asimov"]) for p in pts
           if math.isfinite(p["nsp_asimov"])]
    if asi:
        ma, ya = zip(*asi)
        ax.plot(ma, ya, "o", color="#e42536", ms=4,
                label=r"nominal-MC spurious yield $N_{\rm sp}$")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    if trust_max is not None:
        ax.axvline(trust_max, color="grey", lw=1.0, ls="-.")
        ax.text(trust_max, ax.get_ylim()[1], f"  trustworthy $\\lesssim${trust_max:.0f}",
                color="grey", fontsize=11, va="top", ha="left")
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(rf"expected {cl*100:.0f}% CL upper limit on $N_{{\rm sig}}$ [events]")
    ax.set_ylim(bottom=min(0.0, min(lo2) if lo2 else 0.0))
    ax.text(0.03, 0.97,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            f"{name}: {FUNCS[name][1]}\n"
            r"CL$_s$ asymptotic, $\sigma=$RMS$(N_{\rm sp})$" "\n"
            f"centre: {center}",
            transform=ax.transAxes, va="top", fontsize=13)
    ax.legend(loc="upper left", fontsize=11, bbox_to_anchor=(0.03, 0.80))
    _cmslabel(ax, com, lumi)
    _save(fig, out)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--table", type=Path, default=None,
                   help="Stage-6 summary CSV. Default: "
                        "6_spurious_signal_toys/spurious_toy_table_{ch}_{topo}.csv")
    p.add_argument("--functions", nargs="+", default=["expo", "powlaw"],
                   choices=list(FUNCS))
    p.add_argument("--cl", type=float, default=0.95, help="confidence level")
    p.add_argument("--center", default="zero", choices=["mean", "zero"],
                   help="band centre. 'zero' (default, the agreed convention -- "
                        "matches combine's expected band, Stage 9/10): the pure-"
                        "statistical expected limit, with the spurious bias "
                        "treated as a separate systematic. 'mean': the toy "
                        "<N_sp>, i.e. the band integrates the actual nsp_hist "
                        "(fit bias folded in).")
    p.add_argument("--min-mass", type=float, default=1200.0,
                   help="drop masses below this (the m_WR=1000 point has a "
                        "clamped window -> huge, non-physical RMS)")
    p.add_argument("--max-mass", type=float, default=None,
                   help="drop masses above this (e.g. 3200 to stay inside the "
                        "Gaussian-toy region; the >=3400 points are peaked / "
                        "low-n_ok, so mean+RMS no longer summarizes them)")
    p.add_argument("--trust-max", type=float, default=None,
                   help="draw a marker at this m_WR beyond which the ee-resolved "
                        "background closure is not trusted (e.g. 3400)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <script dir>/<run2|run3>, chosen by --era.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    channel, topology = args.channel, args.topology
    tag = f"{channel}_{topology}"
    alpha = 1.0 - args.cl
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    run_sub = {"RunII": "run2", "Run3": "run3"}[str(info["run"])]
    if args.output_dir is None:
        args.output_dir = HERE / run_sub

    table = args.table or (HERE.parents[0] / "6_spurious_signal_toys"
                           / run_sub / f"spurious_toy_table_{tag}.csv")
    if not table.exists():
        sys.exit(f"Stage-6 table not found: {table}")
    rows = read_table(table, channel, topology)
    logger.info("Read %d rows from %s", len(rows), table)

    funcs = [f for f in args.functions if f in FUNCS]
    per_fn, out_rows = {n: [] for n in funcs}, []
    for r in rows:
        if r["function"] not in funcs or r["mWR"] < args.min_mass:
            continue
        if args.max_mass is not None and r["mWR"] > args.max_mass:
            continue
        mu0 = r["mean_Nsp"] if args.center == "mean" else 0.0
        band = cls_band(mu0, r["rms_Nsp"], alpha)
        if not math.isfinite(band[0]):
            logger.info("  m=%.0f %-6s -> no valid RMS, skip", r["mWR"], r["function"])
            continue
        ul_obs = _cls_upper_limit(r["nsp_asimov"], r["nsp_asimov_err"], alpha)
        per_fn[r["function"]].append({**r, "band": band})
        out_rows.append({
            "channel": channel, "topology": topology, "function": r["function"],
            "mWR": r["mWR"],
            "mean_Nsp": round(r["mean_Nsp"], 4)
            if math.isfinite(r["mean_Nsp"]) else "",
            "rms_Nsp": round(r["rms_Nsp"], 4)
            if math.isfinite(r["rms_Nsp"]) else "",
            "nsp_asimov": round(r["nsp_asimov"], 4)
            if math.isfinite(r["nsp_asimov"]) else "",
            "ul_m2s": round(band[-2], 4), "ul_m1s": round(band[-1], 4),
            "ul_med": round(band[0], 4), "ul_p1s": round(band[1], 4),
            "ul_p2s": round(band[2], 4),
            "ul_obs": round(ul_obs, 4) if math.isfinite(ul_obs) else "",
        })
        logger.info("  m=%.0f %-6s -> median UL=%.1f  [%.1f, %.1f] (68%%)  "
                    "[%.1f, %.1f] (95%%)", r["mWR"], r["function"], band[0],
                    band[-1], band[1], band[-2], band[2])

    if not out_rows:
        logger.error("No band points produced (empty/invalid table?).")
        sys.exit(1)

    for name in funcs:
        if not per_fn[name]:
            continue
        plot_brazil(name, per_fn[name],
                    args.output_dir / "expected_limit" / tag / name,
                    channel=channel, topology=topology, com=com, lumi=lumi,
                    cl=args.cl, trust_max=args.trust_max, center=args.center)

    csv_path = args.output_dir / f"expected_limit_table_{tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    logger.info("  wrote %s", csv_path)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
