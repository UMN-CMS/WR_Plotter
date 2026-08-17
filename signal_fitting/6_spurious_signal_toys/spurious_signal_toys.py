#!/usr/bin/env python3
"""Stage 6 -- spurious-signal TOY study (Poisson-fluctuated background).

The toy generalization of the Stage-5 Asimov spurious signal. Stage 5 fits the
S+B model ONCE to the unfluctuated background-only MC and reads off a single fake
signal `N_sp` per window. Here we draw many Poisson toys of the background-only
expectation and fit each, so for every window we get the **distribution** of the
fake signal -- how big it typically is, how much it scatters, and whether that
scatter is consistent with the quoted `sigma(N_sig)`.

For each (channel, topology), grid mass m_WR, and background function, the
expected pseudo-data is the background-only MC

    mu[bin] = bkg_MC[bin]              (NO signal injected)

and each toy is a bin-wise Poisson draw of that expectation

    data_toy[bin] = Poisson(mu[bin])  (data statistics; the MC template's own
                                       stat error does NOT enter -- the Poisson
                                       draw already is the data fluctuation)

then we fit  background TF1 (Stage-4 recentered) + fixed-shape Gaussian(mu, sigma)
inside the window [m_c - k*sigma, m_c + k*sigma] only (the Stage-4 range), with
the background coefficients and `N_sig` floating. The fitted yield is the toy's
spurious signal N_sp; its pull is N_sp/sigma_fit (the injected signal is zero).

The window and the Gaussian shape are defined exactly as in Stage 5: the window
(and background recentering) from the **median** over m_N -- the Stage-2 linear
m_WR parameterization by default -- and the fixed Gaussian uses the SAME linear
(mu, sigma) (mu_sig = m_c, sigma_sig = sigma_win).

Per (mass, function), over the converged toys, we report
  - <N_sp> +- RMS                  the central fake signal (<N_sp> ~ the Stage-5
                                    Asimov value) and its toy spread (~ sigma(N_sig))
  - pull_mean = <N_sp/sigma_fit>    the typical fake signal in sigma  (bands +-0.2/0.5)
  - pull_width = RMS(N_sp/sigma)    coverage (~1 if the error bars are honest)
  - frac |pull|>0.5                 how often a single experiment fakes a notable signal
  - q95(|N_sp|)                     a conservative spurious magnitude

Outputs (namespaced by {channel}_{topology}, one plot per function):
  spurious_yield_vs_mass/{ch}_{topo}/{fn}.*  <N_sp> +- RMS vs m_WR, Asimov overlaid
  pull_mean_vs_mass/{ch}_{topo}/{fn}.*       toy pull mean vs m_WR (+-0.2/0.5 bands)
  pull_width_vs_mass/{ch}_{topo}/{fn}.*      coverage vs m_WR (line at 1.0)
  nsp_hist/{ch}_{topo}/{fn}/m{mWR}.*         per-mass N_sp distribution (--no-toy-plots skips)
  spurious_toy_table_{ch}_{topo}.csv         every (mass, function) summary row

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import array
import csv
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
sys.path.insert(0, str(HERE.parents[1]))                       # repo root
sys.path.insert(0, str(HERE.parents[0] / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(HERE.parents[0] / "shared"))           # sb_fit, loaders

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402

import ROOT                                                             # noqa: E402
from bkg_fit_lib import (                                               # noqa: E402
    FUNCS, MASS_VAR, CH_LAB, TOPO_LAB,
    load_grid_widths, grid_widths_from_params, load_summed_background,
)
from measure_fwhm import parse_masses                                   # noqa: E402
from shape_estimators import load_master_masses                         # noqa: E402
from sb_fit import fit_splusb, pick_signal_tag                          # noqa: E402

logger = logging.getLogger("spurious_signal_toys")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _leftbox(ax, channel, topology, name, k, ntoys):
    ax.text(0.03, 0.97,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            f"{name}: {FUNCS[name][1]}\n"
            fr"$m_{{c}}\pm{k:g}\sigma$ window" "\n"
            fr"{ntoys} Poisson toys",
            transform=ax.transAxes, va="top", fontsize=13)


def _cmslabel(ax, com, lumi):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)


def _quantile(nsps, p):
    """ROOT quantile (not numpy) of the toy N_sp list."""
    a = array.array("d", sorted(nsps))
    out, pr = array.array("d", [0.0]), array.array("d", [p])
    ROOT.TMath.Quantiles(len(a), 1, a, out, pr, True)
    return out[0]


def plot_nsp_hist(nsps, n_asimov, mWR, name, out, *, mean, rms, hist_range,
                  hist_bins, channel, topology, com, lumi, k, ntoys,
                  adaptive=False):
    """Per-mass distribution of the toy spurious yield N_sp, with the original-MC
    (unfluctuated) value and the toy mean overlaid. `mean`/`rms` are the
    ROOT-computed summary stats (TMath), passed in so the plot and the CSV agree.

    Range: by default a FIXED [-hist_range, +hist_range] so every mass shares
    identical axes (toys beyond it pile into the edge bins). With `adaptive=True`
    the range instead tracks the distribution's own 2.5/97.5% quantiles (padded),
    so a shifted mean (e.g. <N_sp> = -55) is centred rather than dumped into the
    overflow bin, and the 5/95% quantiles are always on-axis (drawn as markers)."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    m, s, n = mean, rms, len(nsps)
    hb = hist_bins
    q05 = q95 = None
    if adaptive:
        qlo, qhi = _quantile(nsps, 0.025), _quantile(nsps, 0.975)
        q05, q95 = _quantile(nsps, 0.05), _quantile(nsps, 0.95)
        pad = max(0.15 * (qhi - qlo), 1.0)             # breathing room; never zero
        lo, hi = qlo - pad, qhi + pad
    else:
        lo, hi = -hist_range, hist_range
    bins = [lo + (hi - lo) * i / hb for i in range(hb + 1)]
    clipped = [min(max(v, lo), hi) for v in nsps]    # overflow -> edge bins
    ax.hist(clipped, bins=bins, color="#5790fc", alpha=0.7, edgecolor="black",
            linewidth=0.5)
    ax.set_xlim(lo, hi)
    if q05 is not None:
        for qv, lab in ((q05, "5%"), (q95, "95%")):
            ax.axvline(qv, color="#555555", lw=1.2, ls=(0, (4, 3)))
            ax.text(qv, 0.02, f" {lab}", transform=ax.get_xaxis_transform(),
                    color="#555555", fontsize=11, va="bottom", ha="left")
    ax.axvline(0.0, color="grey", lw=0.8, ls=":")
    ax.axvline(m, color="#1f77b4", lw=1.6,
               label=fr"Mean over {n} toys $\langle \hat{{N}}_{{\rm sig}}\rangle={m:.1f}$")
    if math.isfinite(n_asimov):
        ax.axvline(n_asimov, color="#e42536", lw=1.6, ls="--",
                   label=fr"Original MC $\hat{{N}}_{{\rm sig}}={n_asimov:.1f}$")
    ax.set_xlabel(r"Fitted signal yield $\hat{N}_{\rm sig}$ [events]")
    ax.set_ylabel("Toys")
    ax.text(0.04, 0.95,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            fr"$m_{{W_R}}={mWR:.0f}$ GeV" "\n"
            fr"$\langle \hat{{N}}_{{\rm sig}}\rangle = {m:.1f}$" "\n"
            fr"RMS $= {s:.1f}$" "\n"
            fr"$N_{{\rm toys}}={n}$",
            transform=ax.transAxes, va="top", fontsize=15)
    ax.legend(loc="upper right", fontsize=14)
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_yield_vs_mass(name, pts, out, *, channel, topology, com, lumi, k, ntoys):
    """Toy <N_sp> +- RMS vs m_WR, with the Original-MC fit value +- its
    covariance-matrix error overlaid (x-offset for clarity) so the toy spread
    (RMS) and the fit's own error bar can be compared directly. The m_WR=1000 GeV
    point (clamped window) is dropped."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    pts = [p for p in pts if p["mWR"] > 1000]      # drop clamped-window 1000 GeV point
    ax.axhline(0.0, color="black", lw=1.0, ls="--", label="no fake signal")
    m = [p["mWR"] for p in pts]
    y = [p["mean_Nsp"] for p in pts]
    e = [p["rms_Nsp"] for p in pts]
    ax.errorbar(m, y, yerr=e, fmt="o-", color=FUNCS[name][0], ms=4, lw=1.3,
                elinewidth=0.8, capsize=1.5,
                label=r"toy $\langle \hat{N}_{\rm sig}\rangle \pm$ RMS")
    omc = [(p["mWR"] + 60.0, p["nsp_asimov"], p["nsp_asimov_err"]) for p in pts
           if math.isfinite(p["nsp_asimov"]) and math.isfinite(p["nsp_asimov_err"])]
    if omc:
        mo, yo, eo = zip(*omc)
        ax.errorbar(mo, yo, yerr=eo, fmt="s--", color="#e42536", ms=4, lw=1.0,
                    elinewidth=0.8, capsize=1.5,
                    label=r"Original MC $\hat{N}_{\rm sig}\pm$ cov. error")
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"fitted signal yield $\hat{N}_{\rm sig}$ [events]")
    _leftbox(ax, channel, topology, name, k, ntoys)
    ax.legend(fontsize=10, loc="upper right")
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_pull_mean_vs_mass(name, pts, out, *, channel, topology, com, lumi, k,
                           ntoys):
    """Toy pull mean (fake signal in sigma) vs m_WR, +-0.2 (tight) / 0.5 (loose)."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhspan(-0.2, 0.2, color="#2ca02c", alpha=0.15, zorder=0)
    ax.axhspan(0.2, 0.5, color="#ffcc00", alpha=0.15, zorder=0)
    ax.axhspan(-0.5, -0.2, color="#ffcc00", alpha=0.15, zorder=0)
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    m = [p["mWR"] for p in pts]
    y = [p["pull_mean"] for p in pts]
    e = [p["pull_mean_err"] for p in pts]
    ax.errorbar(m, y, yerr=e, fmt="o-", color=FUNCS[name][0], ms=4, lw=1.3,
                elinewidth=0.8, capsize=1.5, label=name)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"toy pull mean $\langle \hat{N}_{\rm sig}/\sigma\rangle$")
    _leftbox(ax, channel, topology, name, k, ntoys)
    ax.legend(fontsize=10, loc="upper right")
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_pull_width_vs_mass(name, pts, out, *, channel, topology, com, lumi, k,
                            ntoys):
    """Toy pull RMS vs m_WR -- the coverage check (1.0 = honest errors)."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhspan(0.9, 1.1, color="#2ca02c", alpha=0.15, zorder=0)
    ax.axhline(1.0, color="black", lw=0.9, ls="--", label="ideal coverage")
    m = [p["mWR"] for p in pts]
    y = [p["pull_width"] for p in pts]
    e = [p["pull_width_err"] for p in pts]
    ax.errorbar(m, y, yerr=e, fmt="o-", color=FUNCS[name][0], ms=4, lw=1.3,
                elinewidth=0.8, capsize=1.5, label=name)
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"toy pull width (RMS)")
    _leftbox(ax, channel, topology, name, k, ntoys)
    ax.legend(fontsize=10, loc="upper right")
    _cmslabel(ax, com, lumi)
    _save(fig, out)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20260317_lo_dy", help="background dir")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--mn-frac", type=float, default=0.5,
                   help="representative signal point with m_N closest to this*m_WR "
                        "(informational; the Gaussian shape uses the linear mu/sigma)")
    p.add_argument("--k", type=float, default=3.0,
                   help="window half-width in sigma; the S+B fit is INSIDE "
                        "[mu-k*sigma, mu+k*sigma] only (matches Stage 4/5)")
    p.add_argument("--window-source", default="param",
                   choices=["param", "measured"],
                   help="'param' (default): window mu/sigma from the Stage-2 "
                        "linear parameterization. 'measured': per-m_WR aggregate "
                        "of the Stage-1 widths.")
    p.add_argument("--sigma-kind", default="median",
                   choices=["median", "conservative"],
                   help="Which fitted sigma to use when --window-source param.")
    p.add_argument("--sigma-agg", default="median",
                   choices=["median", "mean", "max", "min"],
                   help="Aggregation over M_N when --window-source measured.")
    p.add_argument("--ntoys", type=int, default=1000,
                   help="Poisson toys per (mass, function)")
    p.add_argument("--min-toys", type=int, default=100,
                   help="need at least this many converged toys to summarize a point")
    p.add_argument("--seed", type=int, default=12345,
                   help="base RNG seed (per-point seeds derived deterministically)")
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--mass-min", type=float, default=1000.0)
    p.add_argument("--mass-max", type=float, default=6000.0)
    p.add_argument("--masses", nargs="+", type=float, default=None,
                   help="Explicit m_WR list (overrides the grid; may be OFF-GRID, "
                        "e.g. 2341). Window + signal shape come from the linear fit, "
                        "so no signal MC point is needed. Requires --window-source param.")
    p.add_argument("--functions", nargs="+", default=["expo", "powlaw"],
                   choices=list(FUNCS),
                   help="default = the Stage-4-validated functions; toys are "
                        "expensive (~30 ms/fit), add others explicitly if needed")
    p.add_argument("--width-csv", type=Path,
                   default=HERE.parents[0] / "1_signal_widths" / "gaussian"
                   / "gauss_fit_table.csv")
    p.add_argument("--window-params", type=Path,
                   default=HERE.parents[0] / "2_width_parameterization" / "wr"
                   / "window_params.json",
                   help="Stage-2 linear window parameterization (window_source param).")
    p.add_argument("--mass-csv", type=Path,
                   default=HERE.parents[0] / "master_masses.csv")
    p.add_argument("--no-toy-plots", action="store_true",
                   help="skip the per-(mass, function) N_sp histograms")
    p.add_argument("--hist-range", type=float, default=60.0,
                   help="nsp_hist x-axis half-range in events; FIXED across all "
                        "masses, so every histogram shares identical axes. Toys "
                        "beyond +-hist_range pile into the edge bins.")
    p.add_argument("--hist-bins", type=int, default=30,
                   help="number of bins across the nsp_hist range")
    p.add_argument("--hist-adaptive", action="store_true",
                   help="per-mass nsp_hist x-range from the toy 2.5/97.5%% "
                        "quantiles (padded) instead of fixed [-hist_range, "
                        "+hist_range]; centres a shifted mean and keeps the "
                        "5/95%% quantiles on-axis. Loses cross-mass axis "
                        "comparability but avoids overflow pile-up.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <script dir>/<run2|run3>, chosen by --era.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    channel, topology, k = args.channel, args.topology, args.k
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    if args.output_dir is None:
        args.output_dir = HERE / {"RunII": "run2", "Run3": "run3"}[str(info["run"])]
    funcs = [f for f in FUNCS if f in args.functions]

    measured = load_grid_widths(args.width_csv, channel, topology, args.sigma_agg)
    if args.masses:
        masses = sorted(float(m) for m in args.masses)
    else:
        masses = sorted(m for m in measured if args.mass_min <= m <= args.mass_max)
    if args.window_source == "param":
        grid = grid_widths_from_params(
            args.window_params, channel, topology, masses, args.sigma_kind)
        logger.info("Window source: Stage-2 linear fit (%s sigma) <- %s",
                    args.sigma_kind, args.window_params)
    else:
        if args.masses and any(m not in measured for m in masses):
            sys.exit("Off-grid --masses require --window-source param (the measured "
                     "table has no widths off the signal grid).")
        grid = measured
        logger.info("Window source: measured per-m_WR %s over M_N", args.sigma_agg)
    sig_tags = load_master_masses(args.mass_csv, topology)
    bkg_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    region = f"wr_{channel}_{topology}_sr"
    factor = max(1, round(args.bin_width / 10.0))
    edges, values, variances = load_summed_background(
        bkg_dirs, region, MASS_VAR[topology], factor)
    centers = 0.5 * (edges[:-1] + edges[1:])
    binwidth = float(edges[1] - edges[0])
    # background-only Poisson mean: clip negative-weight / non-finite MC bins to 0
    # (they sit in the sparse tail outside the fit window, so this never touches a fit)
    mu_bkg = [float(v) if (math.isfinite(v) and v > 0.0) else 0.0 for v in values]
    tag = f"{channel}_{topology}"
    logger.info("Spurious toys: %s %s -- %d masses, funcs %s, %d toys/point",
                channel, topology, len(masses), funcs, args.ntoys)

    rows, per_fn = [], {n: [] for n in funcs}
    for mWR in masses:
        m_c, sigma_win = grid[mWR]                  # linear fit -> window + recentering
        stag = pick_signal_tag(sig_tags, mWR, args.mn_frac)  # informational only
        if stag is None and not args.masses:
            logger.info("  m=%.0f: no %s signal point, skipping", mWR, topology)
            continue
        mN_i = parse_masses(stag)[1] if stag else ""
        mu_sig, sigma_sig = m_c, sigma_win          # same linear Gaussian as Stage 5
        lo = max(m_c - k * sigma_win, args.fit_min)
        hi = min(m_c + k * sigma_win, args.fit_max)
        win = (centers >= m_c - k * sigma_win) & (centers <= m_c + k * sigma_win)
        b_window = float(values[win].sum())

        for name in funcs:
            # Original-MC reference: single S+B fit to the unfluctuated background
            # (= Stage 5); keep its covariance-matrix error to compare with the toy RMS
            asi = fit_splusb(name, edges, values, lo, hi, m_c, sigma_win,
                             mu_sig, sigma_sig, binwidth, k)
            nsp_asimov = float(asi["nsig"]) if asi else float("nan")
            nsp_asimov_err = float(asi["nsig_err"]) if asi else float("nan")

            seed = args.seed * 1_000_003 + int(mWR) * 1009 + funcs.index(name)
            rng = ROOT.TRandom3(seed)               # ROOT RNG (not numpy)
            nsps, pulls = [], []
            for _ in range(args.ntoys):
                # per-bin ROOT Poisson draw of the background-only expectation;
                # the numpy array is only the container fit_splusb indexes
                data_toy = np.array([rng.Poisson(mu) for mu in mu_bkg], dtype=float)
                res = fit_splusb(name, edges, data_toy, lo, hi, m_c, sigma_win,
                                 mu_sig, sigma_sig, binwidth, k)
                if res is None or res["status"] != 0:
                    continue
                nf, ne = res["nsig"], res["nsig_err"]
                if not (ne > 0 and math.isfinite(ne) and math.isfinite(nf)):
                    continue
                nsps.append(nf)
                pulls.append(nf / ne)
            n_ok = len(nsps)
            base = {"channel": channel, "topology": topology, "function": name,
                    "mWR": mWR, "signal_tag": stag or "", "m_N": mN_i,
                    "m_c": round(m_c, 1), "sigma_win": round(sigma_win, 2),
                    "mu_sig": round(mu_sig, 1), "sigma_sig": round(sigma_sig, 2),
                    "fit_lo": round(lo, 1), "fit_hi": round(hi, 1),
                    "B_window": round(b_window, 3), "ntoys": args.ntoys,
                    "n_ok": n_ok,
                    "nsp_asimov": round(nsp_asimov, 4)
                    if math.isfinite(nsp_asimov) else "",
                    "nsp_asimov_err": round(nsp_asimov_err, 4)
                    if math.isfinite(nsp_asimov_err) else ""}
            if n_ok < args.min_toys:
                rows.append({**base, "mean_Nsp": "", "rms_Nsp": "",
                             "pull_mean": "", "pull_mean_err": "",
                             "pull_width": "", "pull_width_err": "",
                             "frac_pull_gt_0p5": "", "q95_abs_Nsp": ""})
                logger.info("  m=%.0f %-6s -> only %d/%d toys ok, skip",
                            mWR, name, n_ok, args.ntoys)
                continue
            # all summary statistics via ROOT (TMath.RMS is ROOT's RMS = the
            # standard deviation, dividing by N); nsps/pulls stay Python lists
            nsps_a = array.array("d", nsps)
            pulls_a = array.array("d", pulls)
            mean_nsp = float(ROOT.TMath.Mean(n_ok, nsps_a))
            rms_nsp = float(ROOT.TMath.RMS(n_ok, nsps_a))
            pmean = float(ROOT.TMath.Mean(n_ok, pulls_a))
            pwidth = float(ROOT.TMath.RMS(n_ok, pulls_a))
            frac_gt = sum(1 for p in pulls if abs(p) > 0.5) / n_ok
            absn = array.array("d", sorted(abs(v) for v in nsps))
            qval, qprob = array.array("d", [0.0]), array.array("d", [0.95])
            ROOT.TMath.Quantiles(n_ok, 1, absn, qval, qprob, True)
            q95 = float(qval[0])
            rec = {"mWR": mWR, "mean_Nsp": mean_nsp, "rms_Nsp": rms_nsp,
                   "pull_mean": pmean, "pull_mean_err": pwidth / math.sqrt(n_ok),
                   "pull_width": pwidth,
                   "pull_width_err": pwidth / math.sqrt(2 * (n_ok - 1)),
                   "nsp_asimov": nsp_asimov, "nsp_asimov_err": nsp_asimov_err}
            rows.append({**base,
                         "mean_Nsp": round(mean_nsp, 4), "rms_Nsp": round(rms_nsp, 4),
                         "pull_mean": round(pmean, 4),
                         "pull_mean_err": round(rec["pull_mean_err"], 4),
                         "pull_width": round(pwidth, 4),
                         "pull_width_err": round(rec["pull_width_err"], 4),
                         "frac_pull_gt_0p5": round(frac_gt, 4),
                         "q95_abs_Nsp": round(q95, 4)})
            per_fn[name].append(rec)
            logger.info("  m=%.0f %-6s -> <Nsp>=%.1f+/-%.1f (Asimov %.1f) "
                        "pull=%.2f+/-%.2f [%d toys]", mWR, name, mean_nsp, rms_nsp,
                        nsp_asimov, pmean, pwidth, n_ok)
            if not args.no_toy_plots:
                plot_nsp_hist(
                    nsps, nsp_asimov, mWR, name,
                    args.output_dir / "nsp_hist" / tag / name / f"m{int(mWR)}",
                    mean=mean_nsp, rms=rms_nsp,
                    hist_range=args.hist_range, hist_bins=args.hist_bins,
                    channel=channel, topology=topology, com=com, lumi=lumi, k=k,
                    ntoys=args.ntoys, adaptive=args.hist_adaptive)

    if not rows:
        logger.error("No results produced.")
        sys.exit(1)
    for name in funcs:
        if not per_fn[name]:
            continue
        for kind, fn in (("spurious_yield_vs_mass", plot_yield_vs_mass),
                         ("pull_mean_vs_mass", plot_pull_mean_vs_mass),
                         ("pull_width_vs_mass", plot_pull_width_vs_mass)):
            fn(name, per_fn[name], args.output_dir / kind / tag / name,
               channel=channel, topology=topology, com=com, lumi=lumi, k=k,
               ntoys=args.ntoys)
    csv_path = args.output_dir / f"spurious_toy_table_{tag}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s", csv_path)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
