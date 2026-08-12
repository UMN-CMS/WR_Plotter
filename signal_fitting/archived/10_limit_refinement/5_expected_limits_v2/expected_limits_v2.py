#!/usr/bin/env python3
"""Stage 10.5 -- expected-limit band v2: empirical per-toy CLs quantiles plus
an exact counting band in the sparse regime.

Replaces the Stage-7 construction (Gaussian closed form fed with mean+RMS of
the converged-toy subset) with three regime-appropriate methods, per mass:

  FIT regime (healthy toys, from the 10.1/10.2 raw dumps):
      per-toy CLs upper limit  UL_i = N_i + c*s_i * z(N_i/(c*s_i)),
      z(t) = -Phi^-1(alpha * Phi(t))   [algebraically identical to Stage 7's
      formula, numerically robust on the downward side, UL > 0 always];
      s_i = the toy's own fit error, c = an optional coverage calibration
      (pull width; c=1 default -- the Poisson core's pulls are near-unit).
      Band = empirical 2.5/16/50/84/97.5 % quantiles of {UL_i}.  No Gaussian
      assumption, no homoskedasticity assumption; failed toys are REPORTED
      (acceptance column), and a band point is only quoted when the failed
      fraction is small compared to the quantile being quoted.

  COUNTING regime (envelope masses; window nearly empty so fits carry no
      information):  exact discrete construction, no toys needed --
      n ~ Poisson(b), b = envelope-predicted window background; for each n,
      UL(n) solves  CLs(s) = P(n' <= n | s*w + b) / P(n' <= n | b) = alpha
      by bisection (w = in-window Gaussian fraction erf(k/sqrt2)).  Band =
      quantiles of UL(n) under the discrete Poisson(b) weights.  The envelope
      sigma_model/sigma_theta enter as alternative-b curves (b widened /
      narrowed), reported as ul_med_bhi / ul_med_blo.

  CENTER convention: default 'zero' -- the band is the pure-statistical
      expected limit; the spurious signal (mu0, from the envelope realMC
      Asimov or the toy mean) can only WIDEN it: ul = max(ul(0), ul(mu0)) per
      quantile when --spurious widen (default).  '--center mean' reproduces
      the Stage-7 default for comparison.

Also overlays the Stage-7 closed-form band (recomputed from the same inputs)
so the change is visible, and writes a single table both regimes feed.

Inputs:  10.1 raw toys (default; or 10.2 winner-config raws via --raw-dir),
         10.4 nsp_prediction CSV.
Outputs: expected_limit_v2/{ch}_{topo}/expo.*   band vs mass
         expected_limit_v2_table_{ch}_{topo}.csv

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
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / "4_background_fits"))

from wrplotter.cli_utils import setup_logging                 # noqa: E402
from wrplotter.config import load_lumi                        # noqa: E402

import ROOT                                                   # noqa: E402
from bkg_fit_lib import CH_LAB, TOPO_LAB                      # noqa: E402
from toy_engine import read_raw                               # noqa: E402

logger = logging.getLogger("expected_limits_v2")

QUANTS = (0.025, 0.16, 0.50, 0.84, 0.975)
QKEYS = ("ul_m2s", "ul_m1s", "ul_med", "ul_p1s", "ul_p2s")


# ---------------------------------------------------------------------------
# CLs machinery
# ---------------------------------------------------------------------------

def ul_gauss(n_hat, sigma, alpha):
    """Per-toy asymptotic CLs UL, clamp-free form: UL = N - s*Phi^-1(a*Phi(N/s)).
    Identical to Stage 7's expression; stays positive for any downward N."""
    if not (sigma > 0 and math.isfinite(sigma) and math.isfinite(n_hat)):
        return float("nan")
    p = alpha * ROOT.TMath.Freq(n_hat / sigma)
    p = max(p, 1e-300)
    return n_hat - sigma * ROOT.TMath.NormQuantile(p)


def ul_counting(n, b, w, alpha):
    """Exact Poisson CLs UL on an observed window count n with background b and
    in-window signal fraction w: solve CLs(s)=alpha by bisection."""
    def cls(s):
        num = ROOT.Math.poisson_cdf(n, s * w + b)
        den = ROOT.Math.poisson_cdf(n, b)
        return num / den if den > 0 else 0.0
    lo_s, hi_s = 0.0, 10.0
    while cls(hi_s) > alpha and hi_s < 1e6:
        hi_s *= 2
    for _ in range(60):
        mid = 0.5 * (lo_s + hi_s)
        if cls(mid) > alpha:
            lo_s = mid
        else:
            hi_s = mid
    return 0.5 * (lo_s + hi_s)


def _poisson_quantile(q, b):
    """Smallest n with P(n' <= n | b) >= q (Gaussian jump start, exact walk)."""
    n = max(0, int(b + ROOT.TMath.NormQuantile(q) * math.sqrt(b + 1)) - 3)
    while ROOT.Math.poisson_cdf(n, b) < q:
        n += 1
    while n > 0 and ROOT.Math.poisson_cdf(n - 1, b) >= q:
        n -= 1
    return n


def counting_band(b, w, alpha):
    """Quantiles of UL(n) under n ~ Poisson(b) -- fully deterministic.  UL(n)
    is increasing in n, so only the five quantile counts need a bisection."""
    if not (b >= 0 and math.isfinite(b)):
        return None
    return {key: ul_counting(_poisson_quantile(q, b), b, w, alpha)
            for q, key in zip(QUANTS, QKEYS)}


def empirical_band(uls, n_failed, alpha):
    """Empirical quantiles of the per-toy ULs; a quantile is quotable only when
    the failed fraction is smaller than its distance to the edges."""
    vals = sorted(u for u in uls if math.isfinite(u))
    n = len(vals)
    if n < 50:
        return None
    ftot = n + n_failed
    ffail = n_failed / ftot
    out = {}
    for q, key in zip(QUANTS, QKEYS):
        if ffail >= min(q, 1 - q):
            out[key] = float("nan")          # quantile not quotable
            continue
        idx = min(int(q * n), n - 1)
        out[key] = vals[idx]
    return out


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
    p.add_argument("--raw-dir", type=Path,
                   default=HERE.parents[0] / "1_nsp_diagnostics" / "raw",
                   help="raw-toy dir; files {stat}_{fn}_m{mass}.csv "
                        "(10.1 layout). For the 10.2 winner config point this "
                        "at 2_prior_scan/raw and use --raw-pattern")
    p.add_argument("--raw-pattern", default="poisson_expo_m{mass}.csv")
    p.add_argument("--envelope", type=Path,
                   default=HERE.parents[0] / "4_bkg_envelope"
                   / "nsp_prediction_ee_resolved.csv")
    p.add_argument("--fit-masses", nargs="+", type=float,
                   default=[1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800,
                            3000, 3200],
                   help="masses quoted from per-toy fit ULs")
    p.add_argument("--counting-masses", nargs="+", type=float,
                   default=[1000, 1200, 3400, 3600, 3800, 4000, 4200, 4400,
                            4600, 4800, 5000],
                   help="masses quoted from the envelope counting band")
    p.add_argument("--cl", type=float, default=0.95)
    p.add_argument("--k", type=float, default=3.0)
    p.add_argument("--calib", type=float, default=1.0,
                   help="coverage calibration c on the per-toy sigma "
                        "(pull width; 1.0 for the near-unit Poisson core)")
    p.add_argument("--center", default="zero", choices=["zero", "mean"],
                   help="'zero' (default): pure-statistical band, spurious "
                        "handled by --spurious. 'mean': Stage-7 default.")
    p.add_argument("--spurious", default="widen", choices=["widen", "off"],
                   help="'widen': each quantile = max(band(0), band(mu0)) -- "
                        "bias can only weaken the limit")
    p.add_argument("--stage6-table", type=Path,
                   default=HERE.parents[1] / "6_spurious_signal_toys"
                   / "run3" / "spurious_toy_table_ee_resolved.csv",
                   help="for the Stage-7 closed-form overlay")
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def stage7_band(mu0, sigma, alpha):
    """The Stage-7 closed form (for the overlay)."""
    if not (sigma > 0 and math.isfinite(sigma)):
        return None
    out = {}
    for nq, key in zip((-2, -1, 0, 1, 2), QKEYS):
        out[key] = ul_gauss(mu0 + sigma * nq, sigma, alpha)
    return out


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    tag = f"{args.channel}_{args.topology}"
    alpha = 1.0 - args.cl
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    w_in = math.erf(args.k / math.sqrt(2.0))
    rows = []

    # --- FIT regime: per-toy UL quantiles from raw toys ----------------------
    for mWR in args.fit_masses:
        f = args.raw_dir / tag / args.raw_pattern.format(mass=int(mWR))
        if not f.exists():
            logger.warning("  m=%.0f: no raw file %s, skip", mWR, f)
            continue
        recs = read_raw(f, accepted_only=False)
        ok = [r for r in recs if r.get("acc") == "1"]
        n_failed = len(recs) - len(ok)
        mu0 = (sum(r["nsig"] for r in ok) / len(ok)) if ok else 0.0
        shift = mu0 if args.center == "mean" else 0.0
        uls = [ul_gauss(r["nsig"] - mu0 + shift, args.calib * r["nsig_err"],
                        alpha) for r in ok]
        band = empirical_band(uls, n_failed, alpha)
        if band is None:
            logger.warning("  m=%.0f: too few accepted toys", mWR)
            continue
        if args.spurious == "widen" and args.center == "zero":
            uls_b = [ul_gauss(r["nsig"], args.calib * r["nsig_err"], alpha)
                     for r in ok]
            band_b = empirical_band(uls_b, n_failed, alpha)
            if band_b:
                band = {k_: max(band[k_], band_b[k_])
                        if math.isfinite(band[k_]) and math.isfinite(band_b[k_])
                        else band[k_] for k_ in band}
        rows.append({"channel": args.channel, "topology": args.topology,
                     "function": "expo", "mWR": mWR, "regime": "fit",
                     "n_toys": len(recs), "n_ok": len(ok),
                     "mu0": round(mu0, 3), "b_window": "",
                     **{k_: round(v, 3) if math.isfinite(v) else ""
                        for k_, v in band.items()}})
        logger.info("  m=%.0f fit      -> med %.1f  [%.1f, %.1f]68  "
                    "[%.1f, %.1f]95", mWR, band["ul_med"], band["ul_m1s"],
                    band["ul_p1s"], band["ul_m2s"], band["ul_p2s"])

    # --- COUNTING regime: envelope-driven discrete band ----------------------
    env = {}
    if args.envelope.exists():
        with open(args.envelope, newline="") as fh:
            for r in csv.DictReader(fh):
                env[float(r["mWR"])] = r
    for mWR in args.counting_masses:
        r = env.get(mWR)
        if r is None:
            logger.warning("  m=%.0f: not in envelope table, skip", mWR)
            continue
        b = float(r["B_window_env"])
        band = counting_band(b, w_in, alpha)
        if band is None:
            continue
        # spurious widening: recompute with the window background shifted by
        # mu0 (the realMC-envelope difference, one-sided) and take the max
        mu0 = float(r["mu0_realmc"]) if r.get("mu0_realmc") else 0.0
        if args.spurious == "widen" and mu0 > 0:
            band_b = counting_band(b + mu0, w_in, alpha)
            if band_b:
                band = {k_: max(band[k_], band_b[k_]) for k_ in band}
        # envelope model/theta uncertainty as alternative-b medians
        smod = float(r["sigma_model"]) if r.get("sigma_model") else 0.0
        sth = float(r["sigma_theta"]) if r.get("sigma_theta") else 0.0
        db = math.hypot(smod, sth)
        bhi = counting_band(b + db, w_in, alpha)
        blo = counting_band(max(b - db, 0.0), w_in, alpha)
        rows.append({"channel": args.channel, "topology": args.topology,
                     "function": "expo", "mWR": mWR, "regime": "counting",
                     "n_toys": "", "n_ok": "",
                     "mu0": round(mu0, 3), "b_window": round(b, 3),
                     **{k_: round(v, 3) for k_, v in band.items()},
                     "ul_med_bhi": round(bhi["ul_med"], 3) if bhi else "",
                     "ul_med_blo": round(blo["ul_med"], 3) if blo else ""})
        logger.info("  m=%.0f counting -> med %.1f  [%.1f, %.1f]68  (b=%.2f)",
                    mWR, band["ul_med"], band["ul_m1s"], band["ul_p1s"], b)

    if not rows:
        sys.exit("no band points produced")

    # --- Stage-7 closed-form overlay -----------------------------------------
    s7 = []
    if args.stage6_table.exists():
        with open(args.stage6_table, newline="") as fh:
            for r in csv.DictReader(fh):
                if (r["function"] == "expo" and r.get("rms_Nsp")
                        and 1200 <= float(r["mWR"])):
                    mu0 = float(r["mean_Nsp"])
                    band = stage7_band(mu0, float(r["rms_Nsp"]), alpha)
                    if band:
                        s7.append((float(r["mWR"]), band["ul_med"]))

    # --- plot -----------------------------------------------------------------
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    pts = sorted(rows, key=lambda r: r["mWR"])
    m = [r["mWR"] for r in pts]
    med = [r["ul_med"] for r in pts]
    lo1 = [r["ul_m1s"] if r["ul_m1s"] != "" else float("nan") for r in pts]
    hi1 = [r["ul_p1s"] if r["ul_p1s"] != "" else float("nan") for r in pts]
    lo2 = [r["ul_m2s"] if r["ul_m2s"] != "" else float("nan") for r in pts]
    hi2 = [r["ul_p2s"] if r["ul_p2s"] != "" else float("nan") for r in pts]
    ax.fill_between(m, lo2, hi2, color="#f5d800", label=r"$\pm2\sigma$ expected")
    ax.fill_between(m, lo1, hi1, color="#00cc00", label=r"$\pm1\sigma$ expected")
    ax.plot(m, med, "k--", lw=1.6, label="median expected (v2)")
    fit_m = [r["mWR"] for r in pts if r["regime"] == "fit"]
    if fit_m:
        ax.axvspan(min(fit_m), max(fit_m), color="grey", alpha=0.06)
        ax.text(0.5 * (min(fit_m) + max(fit_m)), 0.02, "fit regime",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=10,
                color="grey")
    if s7:
        ax.plot(*zip(*sorted(s7)), "o-", ms=3, lw=1.0, color="#e42536",
                label="Stage-7 closed form (mean+RMS)")
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(rf"expected {args.cl*100:.0f}% CL UL on $N_{{\rm sig}}$ [events]")
    ax.set_yscale("log")
    ax.text(0.03, 0.97,
            f"{CH_LAB[args.channel]}\n{TOPO_LAB[args.topology]}\n"
            f"expo | per-toy CLs quantiles + counting\n"
            f"centre: {args.center}, spurious: {args.spurious}",
            transform=ax.transAxes, va="top", fontsize=12)
    ax.legend(loc="upper right", fontsize=11)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    out = args.output_dir / "expected_limit_v2" / tag / "expo"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    out_csv = args.output_dir / f"expected_limit_v2_table_{tag}.csv"
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
