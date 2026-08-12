#!/usr/bin/env python3
"""Stage 7 (toys) -- signal-injection TOY study (Poisson-fluctuated pseudo-data).

The Asimov counterpart (../asimov/signal_injection.py) injects the exact expected
signal and fits once -> central recovery bias. This script draws many Poisson
toys per (mass, N) and fits each -> the DISTRIBUTION of the recovered yield,
which gives both the mean bias AND the pull width (do the error bars cover?).

For each (channel, topology), grid mass m_WR, injection level N, and background
function, the expected pseudo-data is the FULL expectation

    mu[bin] = bkg_MC[bin]  +  N * signal_shape[bin]

and each toy is a bin-wise Poisson draw of that expectation

    data_toy[bin] = Poisson(mu[bin])              (background AND signal fluctuate)

Because Poisson(bkg) + Poisson(N*shape) = Poisson(bkg + N*shape), fluctuating the
total is identical to fluctuating background and signal independently -- the full
data fluctuation. The MC template's own statistical error does NOT enter the toy
generation (the Poisson draw already is the data's statistical fluctuation); the
template jaggedness contributes to the mean bias (the same caveat as Asimov) but
not to the pull width, so the COVERAGE result here is clean.

Per (mass, N, function), over the toys whose S+B fit converges, we report
  - recovered yield  <N_fit> +/- RMS            (bias_evt = <N_fit> - N)
  - pull             (N_fit - N)/sigma_fit  ->  pull_mean  (bias in sigma)
                                                pull_width (RMS; ~1 if honest)
The toy-based null/spurious check (N = 0) lives in Stage 6
(6_spurious_signal_toys); inject N >= 10 here. Pass --inject 0 only as a
cross-check.

Outputs (per category {ch}_{topo}):
  pull_mean_vs_mass/{ch}_{topo}/{fn}.*    bias-in-sigma vs m_WR, N overlaid, +/-0.2/0.5 bands
  pull_width_vs_mass/{ch}_{topo}/{fn}.*   coverage vs m_WR, N overlaid, line at 1.0
  bias_vs_mass/{ch}_{topo}/{fn}.*         <N_fit>-N (events) vs m_WR, N overlaid
  pull_hist/{ch}_{topo}/{fn}/m{mWR}_N{N}.*  per-point pull distribution (--no-toy-plots skips)
  toy_table_{ch}_{topo}.csv               every (mass, N, function) summary row

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
import numpy as np

HERE = Path(__file__).resolve().parent              # .../8_signal_injection_study/toys
SIGDIR = HERE.parents[1]                             # signal_fitting
sys.path.insert(0, str(HERE.parents[2]))            # repo root
sys.path.insert(0, str(SIGDIR / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(SIGDIR / "shared"))         # signal loaders

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402

import ROOT                                                             # noqa: E402,F401
from bkg_fit_lib import (                                               # noqa: E402
    FUNCS, MASS_VAR, CH_LAB, TOPO_LAB, load_grid_widths,
    load_summed_background,
)
from measure_fwhm import build_region_name, build_hist_key, parse_masses  # noqa: E402
from shape_estimators import load_master_masses                         # noqa: E402
from sb_fit import (                                                    # noqa: E402
    fit_splusb, load_point_widths, pick_signal_tag, load_signal_shape,
)

logger = logging.getLogger("signal_injection_toys")

N_PALETTE = ["#1f77b4", "#e42536", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]


def _ncolor(i):
    return N_PALETTE[i % len(N_PALETTE)]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _leftbox(ax, channel, topology, name, k):
    ax.text(0.03, 0.97,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            f"{name}: {FUNCS[name][1]}\n"
            fr"$m_{{c}}\pm{k:g}\sigma$ window",
            transform=ax.transAxes, va="top", fontsize=13)


def _cmslabel(ax, com, lumi):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)


def plot_pull_hist(pulls, mWR, n_inj, name, out, *, channel, topology, com, lumi):
    """Per-point pull distribution with the overlaid unit Gaussian."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    pulls = np.asarray(pulls)
    rng = max(3.0, float(np.abs(pulls).max()) * 1.1) if pulls.size else 3.0
    bins = np.linspace(-rng, rng, 31)
    ax.hist(pulls, bins=bins, color="#5790fc", alpha=0.7, edgecolor="black",
            linewidth=0.5)
    x = np.linspace(-rng, rng, 200)
    binw = bins[1] - bins[0]
    ax.plot(x, pulls.size * binw * np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi),
            color="black", lw=1.5, label=r"unit $\mathcal{N}(0,1)$")
    ax.axvline(0.0, color="grey", lw=0.8, ls=":")
    m = float(pulls.mean())
    s = float(pulls.std(ddof=1)) if pulls.size > 1 else float("nan")
    ax.set_xlabel(r"pull $(N_{\rm fit}-N_{\rm inj})/\sigma_{\rm fit}$")
    ax.set_ylabel("toys")
    ax.text(0.04, 0.95,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            fr"$m_{{W_R}}={mWR:.0f}$ GeV" "\n"
            fr"$N_{{\rm inj}}={n_inj:g}$" "\n"
            fr"mean $={m:.2f}$" "\n"
            fr"width $={s:.2f}$" "\n"
            fr"$N_{{\rm toys}}={pulls.size}$",
            transform=ax.transAxes, va="top", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_pull_mean_vs_mass(name, summ, inj_levels, out, *, channel, topology,
                           com, lumi, k):
    """Bias-in-sigma vs mass, N overlaid, with +/-0.2 (tight) / +/-0.5 (loose) bands."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhspan(-0.2, 0.2, color="#2ca02c", alpha=0.15, zorder=0)
    ax.axhspan(0.2, 0.5, color="#ffcc00", alpha=0.15, zorder=0)
    ax.axhspan(-0.5, -0.2, color="#ffcc00", alpha=0.15, zorder=0)
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    for i, n in enumerate(inj_levels):
        pts = summ.get(n)
        if not pts:
            continue
        m = [p["mWR"] for p in pts]
        y = [p["pull_mean"] for p in pts]
        e = [p["pull_mean_err"] for p in pts]
        ax.errorbar(m, y, yerr=e, fmt="o-", color=_ncolor(i), ms=4, lw=1.1,
                    elinewidth=0.8, capsize=1.5, label=fr"$N={n:g}$")
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"toy pull mean $\langle(N_{\rm fit}-N_{\rm inj})/\sigma\rangle$")
    _leftbox(ax, channel, topology, name, k)
    ax.legend(loc="upper right", fontsize=10, ncol=len(inj_levels))
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_pull_width_vs_mass(name, summ, inj_levels, out, *, channel, topology,
                            com, lumi, k):
    """Pull RMS vs mass, N overlaid -- the coverage check (1.0 = honest errors)."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhspan(0.9, 1.1, color="#2ca02c", alpha=0.15, zorder=0)
    ax.axhline(1.0, color="black", lw=0.9, ls="--", label="ideal coverage")
    for i, n in enumerate(inj_levels):
        pts = summ.get(n)
        if not pts:
            continue
        m = [p["mWR"] for p in pts]
        y = [p["pull_width"] for p in pts]
        e = [p["pull_width_err"] for p in pts]
        ax.errorbar(m, y, yerr=e, fmt="o-", color=_ncolor(i), ms=4, lw=1.1,
                    elinewidth=0.8, capsize=1.5, label=fr"$N={n:g}$")
    ax.set_ylim(0.0, 2.0)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"toy pull width (RMS)")
    _leftbox(ax, channel, topology, name, k)
    ax.legend(loc="upper right", fontsize=10, ncol=len(inj_levels) + 1)
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_bias_vs_mass(name, summ, inj_levels, out, *, channel, topology,
                      com, lumi, k):
    """<N_fit> - N (events) vs mass, N overlaid."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    for i, n in enumerate(inj_levels):
        pts = summ.get(n)
        if not pts:
            continue
        m = [p["mWR"] for p in pts]
        y = [p["bias_evt"] for p in pts]
        e = [p["bias_evt_err"] for p in pts]
        ax.errorbar(m, y, yerr=e, fmt="o-", color=_ncolor(i), ms=4, lw=1.1,
                    elinewidth=0.8, capsize=1.5, label=fr"$N={n:g}$")
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"toy bias $\langle N_{\rm fit}\rangle - N_{\rm inj}$ [events]")
    _leftbox(ax, channel, topology, name, k)
    ax.legend(loc="upper right", fontsize=10, ncol=len(inj_levels))
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
    p.add_argument("--signal-dir", default="20260406_signals")
    p.add_argument("--signal-era", default="RunIISummer20UL18",
                   help="era for the signal MC (as in 0_signal_samples)")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--mn-frac", type=float, default=0.5,
                   help="inject the signal point with m_N closest to this*m_WR")
    p.add_argument("--inject", type=float, nargs="+",
                   default=[5, 10, 15, 20, 30, 50],
                   help="injected signal yields (events). The N=0 null/spurious toy "
                        "lives in Stage 6 (6_spurious_signal_toys) now -- pass "
                        "--inject 0 here only as a cross-check. The extreme N=1000 "
                        "injection (bias-dominated, non-Gaussian) is deliberately "
                        "excluded here; that lesson lives in asimov/.")
    p.add_argument("--ntoys", type=int, default=1000,
                   help="Poisson toys per (mass, N, function)")
    p.add_argument("--min-toys", type=int, default=50,
                   help="need at least this many converged toys to summarize a point")
    p.add_argument("--seed", type=int, default=12345,
                   help="base RNG seed (per-point seeds derived deterministically)")
    p.add_argument("--k", type=float, default=3.0,
                   help="window half-width in sigma; S+B fit is INSIDE the window only")
    p.add_argument("--sigma-agg", default="median",
                   choices=["median", "mean", "max", "min"])
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--mass-min", type=float, default=2000.0)
    p.add_argument("--mass-max", type=float, default=6000.0)
    p.add_argument("--functions", nargs="+", default=["expo", "powlaw"],
                   choices=list(FUNCS),
                   help="default = the Stage-4-validated functions; toys are "
                        "expensive (~30 ms/fit), add others explicitly if needed")
    p.add_argument("--width-csv", type=Path,
                   default=SIGDIR / "1_signal_widths" / "gaussian"
                   / "gauss_fit_table.csv")
    p.add_argument("--mass-csv", type=Path, default=SIGDIR / "master_masses.csv")
    p.add_argument("--no-toy-plots", action="store_true",
                   help="skip the per-(mass, N, function) pull histograms")
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
    funcs = [f for f in FUNCS if f in args.functions]
    inj_levels = [float(n) for n in args.inject]

    grid = load_grid_widths(args.width_csv, channel, topology, args.sigma_agg)
    point_widths = load_point_widths(args.width_csv, channel, topology)
    masses = sorted(m for m in grid if args.mass_min <= m <= args.mass_max)
    sig_tags = load_master_masses(args.mass_csv, topology)
    bkg_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    sig_dirs, _ = input_dirs_for_era(args.signal_era, repo_root(), args.signal_dir)
    region = f"wr_{channel}_{topology}_sr"
    hist_key = build_hist_key(build_region_name(channel, topology),
                              MASS_VAR[topology])
    factor = max(1, round(args.bin_width / 10.0))
    edges, values, variances = load_summed_background(
        bkg_dirs, region, MASS_VAR[topology], factor)
    binwidth = float(edges[1] - edges[0])
    tag = f"{channel}_{topology}"
    logger.info("Category %s %s -- %d masses, funcs %s, N=%s, %d toys/point",
                channel, topology, len(masses), funcs, inj_levels, args.ntoys)

    rows = []
    summ = {name: {n: [] for n in inj_levels} for name in funcs}

    for mWR in masses:
        m_c, sigma_win = grid[mWR]                  # median -> window + recentering
        stag = pick_signal_tag(sig_tags, mWR, args.mn_frac)
        if stag is None:
            logger.info("  m=%.0f: no %s signal point, skipping", mWR, topology)
            continue
        try:
            shape = load_signal_shape(sig_dirs, hist_key, stag, edges)
        except Exception as e:
            logger.warning("  m=%.0f %s: signal load failed (%s), skipping",
                           mWR, stag, e)
            continue
        if shape is None:
            continue
        mWR_i, mN_i = parse_masses(stag)
        mu_sig, sigma_sig = point_widths.get((mWR_i, mN_i), (m_c, sigma_win))
        lo = max(m_c - k * sigma_win, args.fit_min)
        hi = min(m_c + k * sigma_win, args.fit_max)

        for n_inj in inj_levels:
            # full expectation; clip negative-weight / non-finite MC bins to 0 so
            # the Poisson mean is valid (those bins sit in the sparse tail, outside
            # the fit window, so clipping them does not touch the fit).
            mu_expect = values + n_inj * shape
            mu_expect = np.where(np.isfinite(mu_expect) & (mu_expect > 0.0),
                                 mu_expect, 0.0)
            for name in funcs:
                seed = (args.seed * 1_000_003 + int(mWR) * 1009
                        + int(round(n_inj)) * 101 + funcs.index(name))
                rng = np.random.default_rng(seed)
                nfits, pulls = [], []
                for _ in range(args.ntoys):
                    data_toy = rng.poisson(mu_expect).astype(float)
                    res = fit_splusb(name, edges, data_toy, lo, hi, m_c,
                                     sigma_win, mu_sig, sigma_sig, binwidth, k)
                    if res is None or res["status"] != 0:
                        continue
                    nf, ne = res["nsig"], res["nsig_err"]
                    if not (ne > 0 and np.isfinite(ne) and np.isfinite(nf)):
                        continue
                    nfits.append(nf)
                    pulls.append((nf - n_inj) / ne)
                n_ok = len(nfits)
                base = {"channel": channel, "topology": topology,
                        "function": name, "mWR": mWR, "signal_tag": stag,
                        "m_N": mN_i, "m_c": round(m_c, 1),
                        "sigma_win": round(sigma_win, 2),
                        "mu_sig": round(mu_sig, 1),
                        "sigma_sig": round(sigma_sig, 2), "N_inj": n_inj,
                        "fit_lo": round(lo, 1), "fit_hi": round(hi, 1),
                        "ntoys": args.ntoys, "n_ok": n_ok}
                if n_ok < args.min_toys:
                    rows.append({**base, "mean_Nsig": "", "rms_Nsig": "",
                                 "bias_evt": "", "bias_evt_err": "",
                                 "pull_mean": "", "pull_mean_err": "",
                                 "pull_width": "", "pull_width_err": ""})
                    logger.info("  m=%.0f N=%-5g %-6s -> only %d/%d toys ok, skip",
                                mWR, n_inj, name, n_ok, args.ntoys)
                    continue
                nfits = np.array(nfits)
                pulls = np.array(pulls)
                mean_nf = float(nfits.mean())
                rms_nf = float(nfits.std(ddof=1))
                pmean = float(pulls.mean())
                pwidth = float(pulls.std(ddof=1))
                bias_evt = mean_nf - n_inj
                rec = {"mWR": mWR,
                       "pull_mean": pmean, "pull_mean_err": pwidth / math.sqrt(n_ok),
                       "pull_width": pwidth,
                       "pull_width_err": pwidth / math.sqrt(2 * (n_ok - 1)),
                       "bias_evt": bias_evt, "bias_evt_err": rms_nf / math.sqrt(n_ok)}
                rows.append({**base,
                             "mean_Nsig": round(mean_nf, 4), "rms_Nsig": round(rms_nf, 4),
                             "bias_evt": round(bias_evt, 4),
                             "bias_evt_err": round(rec["bias_evt_err"], 4),
                             "pull_mean": round(pmean, 4),
                             "pull_mean_err": round(rec["pull_mean_err"], 4),
                             "pull_width": round(pwidth, 4),
                             "pull_width_err": round(rec["pull_width_err"], 4)})
                summ[name][n_inj].append(rec)
                logger.info("  m=%.0f N=%-5g %-6s -> <Nfit>=%.1f bias=%.2f "
                            "pull=%.2f+/-%.2f [%d toys]", mWR, n_inj, name,
                            mean_nf, bias_evt, pmean, pwidth, n_ok)
                if not args.no_toy_plots:
                    plot_pull_hist(
                        pulls, mWR, n_inj, name,
                        args.output_dir / "pull_hist" / tag / name
                        / f"m{int(mWR)}_N{int(n_inj)}",
                        channel=channel, topology=topology, com=com, lumi=lumi)

    for name in funcs:
        if not any(summ[name][n] for n in inj_levels):
            continue
        for kind, fn in (("pull_mean_vs_mass", plot_pull_mean_vs_mass),
                         ("pull_width_vs_mass", plot_pull_width_vs_mass),
                         ("bias_vs_mass", plot_bias_vs_mass)):
            fn(name, summ[name], inj_levels,
               args.output_dir / kind / tag / name,
               channel=channel, topology=topology, com=com, lumi=lumi, k=k)

    if rows:
        csv_path = args.output_dir / f"toy_table_{tag}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        logger.info("wrote %s", csv_path)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
