#!/usr/bin/env python3
"""Stage 10.4 -- background envelope: predict N_sp at the untrusted masses from
the trusted range.

The direct in-window S+B fit is untrustworthy at m_WR >= ~3400 (window holds
3-11 events: underdetermined floating background, convergence collapse) and at
m_WR = 1000 (window clamped at the 800 GeV selection threshold: no left
sideband, N_sig collinear with the background normalization).  Both are the
SAME disease -- the in-window fit gets its background from too little local
information -- and the cure is the user's envelope idea: take the background
from the mass range where the fit is trusted, and carry a spread of
possibilities for what it could look like at the target mass.

Construction (per channel/topology, expo family):

  ANCHOR FITS   full-range weighted-chi2 fits of the summed background MC over
                the trusted spectrum (bkg_fit_lib.fit_model), refit per target
                mass with the pivot at that window's m_c so the parameters
                transport directly into the S+B fit basis.
  MEMBERS       central        (expo,   [1000, 3500])
                function vars  (expo2, powexp on [1000, 3500])
                range vars     (expo on [800,3500], [1400,3500], [1000,6000]
                                -- the last is tail-anchored: single-MC-event
                                tail bins pull the slope shallow)
                cov samples    N draws of the central fit's parameter
                               covariance (ROOT TDecompChol + TRandom3),
                               non-monotonic draws rejected
                realMC         the jagged MC expectation itself (Stage-6's)
  ESTIMATOR     at the target mass the S+B fit runs with the background
                ANCHORED (fixed) to the member parameters; only N_sig floats
                (optionally with the Stage-10.2 shape priors).  Every toy
                converges; no occupancy gate, no survivor bias.
  PREDICTION    per target mass:
                  sigma_stat   robust half-68 spread of anchored-fit N_sp over
                               Poisson toys of the CENTRAL smooth expectation
                  sigma_model  max over deterministic members g of
                               |N_sp(Asimov data = g, anchor = central)|
                  sigma_theta  RMS over cov-sample Asimovs (same construction)
                  mu0          N_sp(Asimov data = realMC, anchor = central)
                               = the MC-shape spurious signal (threshold shape
                               at low mass, single-event tail noise at high
                               mass -- quoted but, per review, used as the band
                               centre only at m <= mu0-max)
                  sigma_total  quadrature sum
  CLOSURE       at trusted masses the same machinery must reproduce the direct
                Stage-10.1 toys: smooth-central toys refit with the standard
                FLOATING-background fit, spread compared to the direct spread
                (acceptance |ratio - 1| <= 0.25), plus the RMS^2 = a*B + b
                scaling law fit and extrapolation as a fit-free cross-check.

Outputs (namespaced {ch}_{topo}):
  envelope_members_{ch}_{topo}.json      member parameters + fit quality
  envelope_table_{ch}_{topo}.csv         per (target mass, member) numbers
  nsp_prediction_{ch}_{topo}.csv         the combined per-mass prediction; has
                                         mean_Nsp / rms_Nsp columns so Stage-7
                                         expected_limit.py consumes it as-is
  raw/{ch}_{topo}/m{mass}_central.csv    raw anchored toys (for limits v2)
  spectrum_{ch}_{topo}.*                 spectrum + member extrapolation band
  closure_{ch}_{topo}.*                  predicted vs direct spread (trusted)

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
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                          # prior_fit_lib, toy_engine
sys.path.insert(0, str(HERE.parents[2]))                      # repo root
sys.path.insert(0, str(HERE.parents[1] / "4_background_fits"))

from wrplotter.cli_utils import setup_logging                 # noqa: E402
from wrplotter.config import load_lumi                        # noqa: E402

import ROOT                                                   # noqa: E402
from bkg_fit_lib import (                                     # noqa: E402
    FUNCS, CH_LAB, TOPO_LAB, MASS_LABEL, fit_model, predict,
)
from prior_fit_lib import fit_splusb_v2, summarize            # noqa: E402
from toy_engine import Inputs, accepted, write_raw, make_rng  # noqa: E402

logger = logging.getLogger("bkg_envelope")

# Deterministic member set (label, function, fit range).  The central member is
# first; expo2/expo3 are excluded from S+B duty but fine for background-only
# full-range fits (Stage-8 pathology was expo2/3 in the in-window S+B fit).
MEMBER_SPECS = [
    ("central", "expo", (1000.0, 3500.0)),
    ("expo2", "expo2", (1000.0, 3500.0)),
    ("powexp", "powexp", (1000.0, 3500.0)),
    ("lo800", "expo", (800.0, 3500.0)),
    ("lo1400", "expo", (1400.0, 3500.0)),
    ("tail", "expo", (1000.0, 6000.0)),
]


def fit_member(spec, inp, m0):
    """Anchor-fit one member with the pivot at m0; returns (params, cov, meta)
    or None if the fit fails its quality checks."""
    label, name, (flo, fhi) = spec
    r = fit_model(name, inp.edges, inp.values, inp.variances, flo, fhi, m0)
    if r is None:
        return None
    return {"label": label, "name": name, "range": [flo, fhi],
            "params": r.params, "cov": r.cov, "chi2": r.chi2, "ndf": r.ndf,
            "checks": r.checks, "passed": r.passed}


def member_expectation(member, inp, threshold=800.0):
    """Smooth per-bin expectation of a member over the full histogram, zeroed
    below the selection threshold."""
    mu = predict(member["name"], member["params"], inp.centers, member["_m0"])
    mu = np.where(np.isfinite(mu) & (mu > 0), mu, 0.0)
    mu[inp.centers < threshold] = 0.0
    return mu


def cov_samples(member, nsamples, rng, inp, lo, hi):
    """Cholesky draws of the member's parameter covariance (ROOT linear
    algebra + TRandom3); non-monotonic-on-[lo,hi] draws are rejected."""
    npar = len(member["params"])
    m = ROOT.TMatrixDSym(npar)
    for i in range(npar):
        for j in range(npar):
            m[i][j] = float(member["cov"][i][j])
    dec = ROOT.TDecompChol(m)
    if not dec.Decompose():
        logger.warning("Cholesky failed for %s -- no cov samples", member["label"])
        return []
    u = dec.GetU()          # upper triangular, cov = U^T U
    grid = np.linspace(lo, hi, 100)
    out, tries = [], 0
    while len(out) < nsamples and tries < 20 * nsamples:
        tries += 1
        z = [rng.Gaus(0.0, 1.0) for _ in range(npar)]
        step = [sum(u[j][i] * z[j] for j in range(i + 1)) for i in range(npar)]
        theta = np.array(member["params"]) + np.array(step)
        f = predict(member["name"], theta, grid, member["_m0"])
        if np.all(np.isfinite(f)) and np.all(f[1:] <= f[:-1] * (1 + 1e-6)):
            out.append(theta)
    return out


def anchored_fit(inp, name, theta, data, lo, hi, m_c, sigma_win,
                 s_mu, s_sigma, stat):
    """S+B fit with the background FIXED at theta; only N_sig (+ optional
    prior-constrained shape) floats."""
    return fit_splusb_v2(
        name, inp.edges, data, lo, hi, m_c, sigma_win, m_c, sigma_win,
        inp.binwidth, stat=stat, s_mu=s_mu, s_sigma=s_sigma,
        bkg_constraints=[(float(t), 0.0) for t in theta])


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
    p.add_argument("--stat", default="poisson", choices=["poisson", "chi2"])
    p.add_argument("--s-mu", type=float, default=0.0,
                   help="shape-prior width on mu as a fraction of sigma0 "
                        "(apply the 10.2 winner here; 0 = fixed shape)")
    p.add_argument("--s-sigma", type=float, default=0.0)
    p.add_argument("--masses", nargs="+", type=float,
                   default=[1000, 1200, 3400, 3600, 3800, 4000, 4200, 4400,
                            4600, 4800, 5000, 5200, 5400, 5600, 5800, 6000],
                   help="target (untrusted) masses")
    p.add_argument("--closure-masses", nargs="+", type=float,
                   default=[1600, 2000, 2400, 2800, 3200])
    p.add_argument("--diag-table", type=Path,
                   default=HERE.parents[0] / "1_nsp_diagnostics",
                   help="10.1 output dir (gaussianity table + raw toys) used "
                        "for the closure comparison and the scaling law")
    p.add_argument("--ntoys", type=int, default=1000,
                   help="anchored toys per target mass (central member)")
    p.add_argument("--ntoys-closure", type=int, default=500)
    p.add_argument("--ncov", type=int, default=30)
    p.add_argument("--mu0-max", type=float, default=3300.0,
                   help="use the realMC-anchored Asimov as the band centre "
                        "only below this mass (above, the MC tail is "
                        "single-event noise; mu0 is recorded as diagnostic)")
    p.add_argument("--seed", type=int, default=97531)
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def predict_at_mass(mWR, inp, args, tag, *, closure=False):
    """Run the full envelope machinery at one mass; returns (pred_row,
    member_rows, raw_records)."""
    k = args.k
    m_c, sigma_win, lo, hi = inp.fit_range(mWR, k, args.fit_min, args.fit_max)
    clamped = (lo > m_c - k * sigma_win + 1e-6) or (hi < m_c + k * sigma_win - 1e-6)
    s_mu = args.s_mu * sigma_win if args.s_mu else 0.0
    s_sigma = args.s_sigma * sigma_win if args.s_sigma else 0.0

    # anchor fits with the pivot at THIS window's centre
    members = []
    for spec in MEMBER_SPECS:
        m = fit_member(spec, inp, m_c)
        if m is None:
            logger.warning("  m=%.0f member %s failed, skipped", mWR, spec[0])
            continue
        m["_m0"] = m_c
        members.append(m)
    central = members[0]
    assert central["label"] == "central"

    mu_central = member_expectation(central, inp)
    win = (inp.centers >= m_c - k * sigma_win) & (inp.centers <= m_c + k * sigma_win)
    b_env = float(mu_central[win].sum())
    b_mc = float(inp.mu_bkg[win].sum())

    # sigma_stat: anchored toys of the central smooth expectation
    rng = make_rng(args.seed, mWR, 1)
    ntoys = args.ntoys_closure if closure else args.ntoys
    recs = []
    n_ok = 0
    for itoy in range(ntoys):
        data = np.array([rng.Poisson(v) for v in mu_central], dtype=float)
        r = anchored_fit(inp, central["name"], central["params"], data,
                         lo, hi, m_c, sigma_win, s_mu, s_sigma, args.stat)
        acc = accepted(r)
        n_ok += int(acc)
        rec = {"itoy": itoy, "acc": int(acc), "status": -1}
        if r is not None:
            rec.update({"nsig": r["nsig"], "nsig_err": r["nsig_err"],
                        "mu": r["mu"], "sigma": r["sigma"],
                        "status": r["status"], "cov": r["cov"],
                        "passed": int(r["passed"]),
                        "mu_railed": int(r["mu_railed"]),
                        "sigma_railed": int(r["sigma_railed"]),
                        "nsig_railed": int(r["nsig_railed"])})
        recs.append(rec)
    nsps = [r["nsig"] for r in recs if r.get("acc")]
    s = summarize(nsps) or {}
    sigma_stat = ((s["q84"] - s["q16"]) / 2.0) if s else float("nan")

    # sigma_model: Asimov(data = member g, anchor = central).  The weighted-chi2
    # tail member alone cannot represent "the MC tail is real" (the 50-100%
    # tail-bin errors make it collapse onto the central fit), so an explicit
    # expectation-level member "mcmax" = max(central smooth, MC) over the tail
    # is added -- it covers the real-tail hypothesis at m > mu0-max where the
    # realMC Asimov is not used as the band centre.
    member_rows = []
    devs = []
    expectations = [(g["label"], g["name"], g["range"], g["chi2"], g["ndf"],
                     g["passed"], member_expectation(g, inp)) for g in members]
    mu_mcmax = mu_central.copy()
    tail = inp.centers >= 3300.0
    mu_mcmax[tail] = np.maximum(mu_central[tail], inp.mu_bkg[tail])
    expectations.append(("mcmax", "expo", (0, 0), float("nan"), 0, 1, mu_mcmax))
    for label, gname, grange, gchi2, gndf, gpassed, mu_g in expectations:
        r = anchored_fit(inp, central["name"], central["params"], mu_g,
                         lo, hi, m_c, sigma_win, s_mu, s_sigma, args.stat)
        nsp_g = float(r["nsig"]) if r else float("nan")
        if label != "central" and math.isfinite(nsp_g):
            devs.append(abs(nsp_g))
        member_rows.append({
            "channel": inp.channel, "topology": inp.topology, "mWR": mWR,
            "member": label, "function": gname,
            "fit_range": f"[{grange[0]:.0f},{grange[1]:.0f}]",
            "chi2_ndf": round(gchi2 / max(gndf, 1), 3)
            if math.isfinite(gchi2) else "",
            "passed": int(gpassed),
            "B_window_member": round(float(mu_g[win].sum()), 3),
            "nsp_asimov_vs_central": round(nsp_g, 4)
            if math.isfinite(nsp_g) else ""})
    sigma_model = max(devs) if devs else float("nan")

    # sigma_theta: cov-sample Asimovs
    rng_t = make_rng(args.seed, mWR, 2)
    thetas = cov_samples(central, args.ncov, rng_t, inp, lo, hi)
    tdevs = []
    for th in thetas:
        mu_t = predict(central["name"], th, inp.centers, m_c)
        mu_t = np.where(np.isfinite(mu_t) & (mu_t > 0), mu_t, 0.0)
        mu_t[inp.centers < 800.0] = 0.0
        r = anchored_fit(inp, central["name"], central["params"], mu_t,
                         lo, hi, m_c, sigma_win, s_mu, s_sigma, args.stat)
        if r and math.isfinite(r["nsig"]):
            tdevs.append(r["nsig"])
    sigma_theta = (summarize(tdevs) or {}).get("rms", float("nan"))

    # mu0: Asimov(data = realMC, anchor = central)
    r = anchored_fit(inp, central["name"], central["params"], inp.mu_bkg,
                     lo, hi, m_c, sigma_win, s_mu, s_sigma, args.stat)
    mu0_realmc = float(r["nsig"]) if r else float("nan")
    use_mu0 = mWR <= args.mu0_max and math.isfinite(mu0_realmc)
    mu0_used = mu0_realmc if use_mu0 else 0.0

    parts = [x for x in (sigma_stat, sigma_model, sigma_theta)
             if math.isfinite(x)]
    sigma_total = math.sqrt(sum(x * x for x in parts)) if parts else float("nan")

    pred = {
        "channel": inp.channel, "topology": inp.topology, "function": "expo",
        "mWR": mWR, "m_c": round(m_c, 1), "sigma_win": round(sigma_win, 2),
        "fit_lo": round(lo, 1), "fit_hi": round(hi, 1),
        "window_clamped": int(clamped),
        "B_window_env": round(b_env, 3), "B_window_mc": round(b_mc, 3),
        "ntoys": ntoys, "n_ok": n_ok,
        "sigma_stat": round(sigma_stat, 4) if math.isfinite(sigma_stat) else "",
        "sigma_model": round(sigma_model, 4) if math.isfinite(sigma_model) else "",
        "sigma_theta": round(sigma_theta, 4) if math.isfinite(sigma_theta) else "",
        "sigma_total": round(sigma_total, 4) if math.isfinite(sigma_total) else "",
        "mu0_realmc": round(mu0_realmc, 4) if math.isfinite(mu0_realmc) else "",
        "mu0_used": round(mu0_used, 4),
        # Stage-7 compatibility columns.  rms_Nsp carries sigma_stat ONLY --
        # the CLs formula interprets it as the estimator's per-experiment
        # sampling sigma; sigma_model/sigma_theta are forecast uncertainties
        # and belong ON the band (separate columns), not inside it.
        # sigma_total is kept for a deliberately conservative variant.
        "mean_Nsp": round(mu0_used, 4),
        "rms_Nsp": round(sigma_stat, 4) if math.isfinite(sigma_stat) else "",
        "q16": round(s.get("q16", float("nan")), 4) if s else "",
        "q50": round(s.get("q50", float("nan")), 4) if s else "",
        "q84": round(s.get("q84", float("nan")), 4) if s else "",
    }
    logger.info("  m=%.0f -> stat=%.2f model=%.2f theta=%.2f mu0=%.2f%s "
                "total=%.2f [%d/%d toys]", mWR,
                sigma_stat if math.isfinite(sigma_stat) else -1,
                sigma_model if math.isfinite(sigma_model) else -1,
                sigma_theta if math.isfinite(sigma_theta) else -1,
                mu0_realmc if math.isfinite(mu0_realmc) else -1,
                "(used)" if use_mu0 else "(diag)",
                sigma_total if math.isfinite(sigma_total) else -1, n_ok, ntoys)
    return pred, member_rows, recs


def run_closure(inp, args, tag):
    """Smooth-central toys with the standard FLOATING fit vs the direct
    Stage-10.1 toys -- validates the smooth extrapolated background as the toy
    generator inside the trusted range."""
    diag_csv = args.diag_table / f"gaussianity_table_{tag}.csv"
    direct = {}
    if diag_csv.exists():
        with open(diag_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                if (r["stat"] == args.stat and r["function"] == "expo"
                        and r.get("rms")):
                    direct[float(r["mWR"])] = {
                        "rms": float(r["rms"]),
                        "half68": (float(r["q84"]) - float(r["q16"])) / 2.0,
                        "mean": float(r["mean"])}
    rows = []
    for mWR in args.closure_masses:
        m_c, sigma_win, lo, hi = inp.fit_range(mWR, args.k, args.fit_min,
                                               args.fit_max)
        m = fit_member(MEMBER_SPECS[0], inp, m_c)
        m["_m0"] = m_c
        mu_central = member_expectation(m, inp)
        rng = make_rng(args.seed, mWR, 3)
        nsps = []
        for _ in range(args.ntoys_closure):
            data = np.array([rng.Poisson(v) for v in mu_central], dtype=float)
            r = fit_splusb_v2("expo", inp.edges, data, lo, hi, m_c, sigma_win,
                              m_c, sigma_win, inp.binwidth, stat=args.stat,
                              s_mu=0.0, s_sigma=0.0)      # floating background
            if accepted(r):
                nsps.append(r["nsig"])
        s = summarize(nsps)
        d = direct.get(mWR)
        half68 = (s["q84"] - s["q16"]) / 2.0 if s else float("nan")
        ratio = (half68 / d["half68"]) if (s and d and d["half68"] > 0) else float("nan")
        rows.append({"mWR": mWR, "n_ok": len(nsps),
                     "pred_half68": round(half68, 3) if s else "",
                     "pred_mean": round(s["mean"], 3) if s else "",
                     "direct_half68": round(d["half68"], 3) if d else "",
                     "direct_mean": round(d["mean"], 3) if d else "",
                     "ratio": round(ratio, 3) if math.isfinite(ratio) else "",
                     "pass": int(math.isfinite(ratio) and abs(ratio - 1) <= 0.25)})
        logger.info("  closure m=%.0f: pred half68 %.2f vs direct %.2f "
                    "(ratio %.2f) %s", mWR, half68,
                    d["half68"] if d else float("nan"),
                    ratio, "PASS" if rows[-1]["pass"] else "FAIL")
    return rows


def plot_spectrum(inp, args, tag, com, lumi):
    """Spectrum with the member extrapolations overlaid (pivot 2000)."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    pos = inp.values > 0
    ax.errorbar(inp.centers[pos], inp.values[pos],
                yerr=np.sqrt(inp.variances[pos]), fmt="o", ms=3, lw=0.8,
                color="black", label="summed background MC")
    grid = np.linspace(850, 6000, 400)
    for spec, color in zip(MEMBER_SPECS,
                           ["#e42536", "#5790fc", "#964a8b", "#f89c20",
                            "#2ca02c", "#9c9ca1"]):
        m = fit_member(spec, inp, 2000.0)
        if m is None:
            continue
        m["_m0"] = 2000.0
        f = predict(m["name"], m["params"], grid, 2000.0)
        ax.plot(grid, f, lw=1.2, color=color,
                label=f"{m['label']} ({m['name']} "
                      f"[{m['range'][0]:.0f},{m['range'][1]:.0f}])")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 2 * inp.values.max())
    ax.set_xlabel(MASS_LABEL[inp.topology])
    ax.set_ylabel(f"events / {inp.binwidth:.0f} GeV")
    ax.legend(fontsize=10, loc="upper right")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    out = args.output_dir / f"spectrum_{tag}"
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    tag = f"{args.channel}_{args.topology}"
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    inp = Inputs(era=args.era, bkg_dir=args.dir, channel=args.channel,
                 topology=args.topology, bin_width=args.bin_width,
                 sigma_kind=args.sigma_kind)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # member bookkeeping at the reference pivot (documentation JSON)
    ref = []
    for spec in MEMBER_SPECS:
        m = fit_member(spec, inp, 2000.0)
        if m:
            ref.append({"label": m["label"], "function": m["name"],
                        "range": m["range"],
                        "params_pivot2000": [float(x) for x in m["params"]],
                        "chi2": m["chi2"], "ndf": m["ndf"],
                        "passed": bool(m["passed"]), "checks": m["checks"]})
    with open(args.output_dir / f"envelope_members_{tag}.json", "w") as fh:
        json.dump(ref, fh, indent=2)
    plot_spectrum(inp, args, tag, com, lumi)

    # tail ownership diagnostic (which sample owns the sparse tail bins)
    try:
        import uproot
        from bkg_fit_lib import BKG_SAMPLES
        region = f"wr_{args.channel}_{args.topology}_sr"
        from bkg_fit_lib import MASS_VAR
        key = f"{region}/{MASS_VAR[args.topology]}_{region}"
        logger.info("MC tail [3600,6000] ownership:")
        for d in inp.bkg_dirs:
            for smp in BKG_SAMPLES:
                f = d / f"WRAnalyzer_{smp}.root"
                if not f.exists():
                    continue
                h = uproot.open(f)[key]
                e, v = h.axes[0].edges(), h.values()
                c = 0.5 * (e[:-1] + e[1:])
                mask = (c >= 3600) & (c <= 6000)
                logger.info("    %-10s sum=%7.3f", smp, float(v[mask].sum()))
    except Exception as e:                                    # diagnostic only
        logger.warning("tail diagnostic failed: %s", e)

    preds, mrows = [], []
    logger.info("Predicting at %d target masses", len(args.masses))
    for mWR in args.masses:
        pred, mr, recs = predict_at_mass(mWR, inp, args, tag)
        preds.append(pred)
        mrows.extend(mr)
        write_raw(args.output_dir / "raw" / tag / f"m{int(mWR)}_central.csv",
                  {"channel": args.channel, "topology": args.topology,
                   "mWR": mWR, "member": "central", "stat": args.stat,
                   "s_mu": args.s_mu, "s_sigma": args.s_sigma}, recs)

    logger.info("Closure at trusted masses")
    closure = run_closure(inp, args, tag)

    def _write(path, rows):
        if not rows:
            return
        fields = []
        for r in rows:
            for k_ in r:
                if k_ not in fields:
                    fields.append(k_)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)
        logger.info("wrote %s", path)

    _write(args.output_dir / f"nsp_prediction_{tag}.csv", preds)
    _write(args.output_dir / f"envelope_table_{tag}.csv", mrows)
    _write(args.output_dir / f"closure_{tag}.csv", closure)

    npass = sum(r["pass"] for r in closure if r["pass"] != "")
    logger.info("Done. Closure: %d/%d pass |ratio-1|<=0.25", npass, len(closure))


if __name__ == "__main__":
    main()
