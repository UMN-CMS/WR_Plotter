#!/usr/bin/env python3
"""Stage 7b -- expected cross-section limits from the event-yield limits.

Converts the Stage-7 expected 95% CL upper limit on the signal yield N_sig
(built from the Stage-6 spurious-signal toys) into an upper limit on
sigma x BR(pp -> W_R -> lljj) via

    sigma_UL = N_UL / (1000 * L[fb^-1] * eff),
    eff = S_fit / (B_ch * genEventSumw)

where per mass point
  * S_fit        = raw signal-MC yield in the fit's window: the histogram is
                   rebinned to 100 GeV and the bins whose CENTRE lies in
                   [fit_lo, fit_hi] are summed -- exactly the bins fit_splusb
                   uses, so S_fit is measured in the same window as the N_sig
                   limit (fit_lo/fit_hi come from the Stage-6 CSV),
  * genEventSumw = the genWeight sum over ALL generated events, from the
                   analyzer config JSON,
  * B_ch         = the generated share of the plotted channel in the
                   flavor-mixed WRtoNLtoLLJJ scan samples (--channel-bfrac,
                   default 0.5: e:mu = 50:50 by genWeight, measured Jul 2026
                   from the GenModel N-flavor (9900012/9900014) shares in the
                   raw NanoAODs; no tau, no other decays).

With B_ch the efficiency is PER CHANNEL (eff_ee ~ 0.27-0.42 for ee resolved),
so sigma_UL and the theory curve (config xsec * B_ch) are both
sigma x BR(W_R -> l l q qbar') for the single plotted channel -- matching the
y-axis label and the published 2018 convention. mu = sigma/sigma_theory is
independent of B_ch. Set --channel-bfrac 1.0 for single-flavor samples.

The pairing is consistent because the signal histograms were produced with the
xsec*lumi/sumw scaling turned OFF (raw genWeight fills, verified Jul 2026):
the in-window integral divided by genEventSumw is exactly the per-event
efficiency x acceptance x window fraction. The conversion is a positive
per-mass scale factor, so dividing every band edge preserves the quantiles.

The RunII signal samples currently stand in for Run3: shapes/efficiencies and
the theory sigma come from the RunII config (13 TeV, inclusive LLJJ -- the
per-channel branching is absorbed by the selection), while L is the Run3
luminosity the background band lives at.

Input:  the Stage-6 summary CSV `spurious_toy_table_{ch}_{topo}.csv`
        (NOT the Stage-7 output CSV -- the band is recomputed here from the
        unrounded toy moments with the same code, and the Stage-6 table also
        carries the window and signal tag).
Outputs (namespaced by {channel}_{topology}):
  xsec_limit/{ch}_{topo}/{fn}.*        sigma x BR Brazil band + theory curve
  xsec_limit/{ch}_{topo}/{fn}_mu.*     the same band in signal strength mu
  xsec_limit_table_{ch}_{topo}.csv     efficiencies + sigma/mu bands per (mass, fn)

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
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                        # repo root
sys.path.insert(0, str(HERE.parents[0] / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(HERE.parents[0] / "shared"))             # loaders

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_lumi                                  # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter             # noqa: E402

from bkg_fit_lib import FUNCS, MASS_VAR, CH_LAB, TOPO_LAB               # noqa: E402
from measure_fwhm import (                                              # noqa: E402
    build_hist_key, build_region_name, load_and_combine_signal, parse_masses,
)
from expected_limit import BANDS, cls_band, _cls_upper_limit            # noqa: E402

logger = logging.getLogger("xsec_limit")


# ---------------------------------------------------------------------------
# Signal efficiency
# ---------------------------------------------------------------------------

def integral_in_range(edges, values, lo, hi):
    """Histogram integral over [lo, hi] with fractional edge bins."""
    left = np.maximum(edges[:-1], lo)
    right = np.minimum(edges[1:], hi)
    frac = np.clip((right - left) / np.diff(edges), 0.0, 1.0)
    return float((values * frac).sum())


def signal_efficiency(sig_dirs, hist_key, tag, fit_lo, fit_hi, sig_cfg,
                      bin_width=100.0, bfrac=0.5):
    """(eff, bookkeeping dict) for one signal point, or None if unavailable.

    eff = S_fit / (bfrac * genEventSumw), where S_fit is the raw genWeight
    signal yield in the fit's window: the native histogram is rebinned to
    `bin_width` (100 GeV, as in the S+B fit) and the bins whose CENTRE lies in
    [fit_lo, fit_hi] are summed -- identical to fit_splusb's bin selection, so
    the efficiency uses exactly the window the N_sig limit was measured in.
    Raw genWeight fills, so this is the PER-CHANNEL efficiency x acceptance x
    window-fraction in one number: bfrac removes the other-flavor share of the
    flavor-mixed sample from the denominator (and scales the reported theory
    xsec to the same single channel).
    """
    mwr, mn = parse_masses(tag)
    key = f"WRtoNLtoLLJJ_MWR{mwr}_MN{mn}"
    if key not in sig_cfg:
        logger.warning("  %s: no '%s' in the signal config, skipping", tag, key)
        return None
    try:
        edges, values, _ = load_and_combine_signal(sig_dirs, hist_key, tag)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("  %s: %s, skipping", tag, exc)
        return None
    sumw_gen = float(sig_cfg[key]["genEventSumw"])
    # rebin the native histogram to the fit's bin width, then keep the bins whose
    # centre falls in [fit_lo, fit_hi] -- exactly fit_splusb's selection.
    factor = max(1, round(bin_width / float(edges[1] - edges[0])))
    n = (len(values) // factor) * factor
    edges_r = edges[0:n + 1:factor]
    vals_r = values[:n].reshape(-1, factor).sum(axis=1)
    centers_r = 0.5 * (edges_r[:-1] + edges_r[1:])
    infit = (centers_r >= fit_lo) & (centers_r <= fit_hi)
    s_fit = float(vals_r[infit].sum())
    s_tot = float(values.sum())
    return {
        "m_N": mn,
        "xsec_pb": bfrac * float(sig_cfg[key]["xsec"]),
        "genEventSumw": sumw_gen,
        "s_fitrange": s_fit,
        "s_total": s_tot,
        "f_fitrange": s_fit / s_tot if s_tot > 0 else float("nan"),
        "eff": s_fit / (bfrac * sumw_gen),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_stage6_table(csv_path, channel, topology):
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if r.get("channel") != channel or r.get("topology") != topology:
            continue
        out.append({
            "function": r["function"],
            "mWR": _f(r["mWR"]),
            "signal_tag": r.get("signal_tag", ""),
            "fit_lo": _f(r.get("fit_lo")),
            "fit_hi": _f(r.get("fit_hi")),
            "mean_Nsp": _f(r.get("mean_Nsp")),
            "rms_Nsp": _f(r.get("rms_Nsp")),
            "nsp_asimov": _f(r.get("nsp_asimov")),
            "nsp_asimov_err": _f(r.get("nsp_asimov_err")),
        })
    return out


def default_signal_config(signal_era):
    info = load_lumi(signal_era)
    return (repo_root().parent / "data" / "configs" / str(info["run"])
            / str(info["year"]) / signal_era / f"{signal_era}_signal.json")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _cmslabel(ax, com, lumi, data=False):
    hep.cms.label(loc=0, ax=ax, data=data, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=17)


def plot_band(name, pts, out, *, ykey, obskey, ylabel, theory, channel,
              topology, com, lumi, cl, trust_max, center, scale=1.0,
              legend_loc="lower left", obs_label="Observed limit (MC Asimov)",
              data=False, xlim=(800.0, 6000.0), ylim=None, trust_range=None):
    """Limit plot vs m_WR [GeV], styled on the official 2018 WR plots: solid
    observed, dotted median expected, 68/95% bands, thin red theory curve
    (sigma plot) or mu=1 line (mu plot). Sigma plots get the official fixed
    axes (x 800-6000 GeV, y 1e-4..1e4 with '1'/'10' tick labels); the mu plot
    keeps an auto y-range. `scale` converts stored values (pb -> fb)."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(9, 8.5))
    pts = sorted(pts, key=lambda p: p["mWR"])
    m = [p["mWR"] for p in pts]
    band = {N: [p[ykey][N] * scale for p in pts] for N in BANDS}
    h95 = ax.fill_between(m, band[-2], band[2], color="#f5d800")
    h68 = ax.fill_between(m, band[-1], band[1], color="#00cc00")
    (hexp,) = ax.plot(m, band[0], color="black", lw=2, ls=":")
    obs = [(mm, p[obskey] * scale) for mm, p in zip(m, pts)
           if math.isfinite(p[obskey])]
    hobs = None
    if obs:
        mo, yo = zip(*obs)
        (hobs,) = ax.plot(mo, yo, color="black", lw=2, ls="-")
    if theory:
        thy = [p["xsec_pb"] * scale for p in pts]
        (hth,) = ax.plot(m, thy, color="#e42536", lw=1)
        th_label = r"Theory ($g_{R} = g_{L}$)"
    else:
        thy = [1.0]
        hth = ax.axhline(1.0, color="#e42536", lw=1)
        th_label = r"$\mu = 1$"
    ax.set_yscale("log")
    if ylim is None and theory:
        ylim = (1e-4, 1e4)                      # official fixed range
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        lo = min(v for v in band[-2] if v > 0)
        hi = max(max(band[2]), max(thy))
        ax.set_ylim(10.0 ** (math.floor(math.log10(lo)) - 1),
                    10.0 ** math.ceil(math.log10(hi)))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_xlim(*xlim)
    if trust_max is not None:
        ax.axvline(trust_max, color="grey", lw=0.8, ls="-.")
        ax.text(trust_max, ax.get_ylim()[0],
                f" trustworthy $\\lesssim${trust_max/1000.0:.1f}",
                color="grey", fontsize=10, va="bottom", ha="left")
    if trust_range is not None:
        lo_t, hi_t = trust_range
        for xv in (lo_t, hi_t):
            ax.axvline(xv, color="0.35", lw=1.4, ls=(0, (6, 3)))
        ax.text(0.5 * (lo_t + hi_t), 0.25 * ax.get_ylim()[1],
                "trusted region", color="0.35", fontsize=13,
                ha="center", va="top")
    ax.set_xlabel(r"$m_{W_R}$ (GeV)")
    ax.set_ylabel(ylabel)
    ax.text(0.95, 0.93, r"$\mathbf{m_{N} = m_{W_R}/2}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=19)
    ax.text(0.95, 0.87, f"{TOPO_LAB[topology]} {CH_LAB[channel]} channel",
            transform=ax.transAxes, ha="right", va="top", fontsize=19,
            weight="bold")
    fit_lbl = {"expo": "Exponential fit", "powlaw": "Power-law fit"}.get(name)
    if fit_lbl:
        ax.text(0.95, 0.81, fit_lbl, transform=ax.transAxes, ha="right",
                va="top", fontsize=16)
    handles = [h for h in (hobs, hexp, h68, h95, hth) if h is not None]
    labels = ([] if hobs is None else [obs_label]) + [
        "Expected limit", "68% expected", "95% expected", th_label]
    ax.legend(handles, labels, loc=legend_loc, fontsize=16, frameon=False)
    _cmslabel(ax, com, lumi, data=data)
    _save(fig, out)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24",
                   help="era whose luminosity the limit is computed at (must "
                        "match the background band)")
    p.add_argument("--signal-era", default="RunIISummer20UL18",
                   help="era of the (stand-in) signal samples")
    p.add_argument("--signal-dir", default="20260624_signals",
                   help="rootfiles subdir with the RAW (unscaled) signal histos")
    p.add_argument("--signal-config", type=Path, default=None,
                   help="analyzer config JSON with per-dataset xsec and "
                        "genEventSumw. Default: derived from --signal-era")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--channel-bfrac", type=float, default=0.5,
                   help="generated genWeight share of the plotted channel in "
                        "the flavor-mixed WRtoNLtoLLJJ samples (e:mu = 50:50, "
                        "measured from the GenModel N-flavor shares; no tau). "
                        "Divides the efficiency denominator and scales the "
                        "theory xsec, so sigma_UL is per-channel sigma x BR. "
                        "Use 1.0 for single-flavor samples.")
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--table", type=Path, default=None,
                   help="Stage-6 summary CSV. Default: "
                        "6_spurious_signal_toys/spurious_toy_table_{ch}_{topo}.csv")
    p.add_argument("--functions", nargs="+", default=["expo", "powlaw"],
                   choices=list(FUNCS))
    p.add_argument("--bin-width", type=float, default=100.0,
                   help="fit bin width [GeV]; S_fit is summed over the 100 GeV "
                        "bins whose centre is in the window, matching the S+B fit")
    p.add_argument("--cl", type=float, default=0.95, help="confidence level")
    p.add_argument("--center", default="zero", choices=["mean", "zero"],
                   help="band centre, as in expected_limit.py: 'zero' (default, "
                        "the agreed convention -- matches combine's expected "
                        "band) is the pure-statistical expected limit; 'mean' "
                        "folds the spurious-fit bias into the band")
    p.add_argument("--observed", action="store_true",
                   help="ALSO draw the Observed (MC Asimov) line on the sigma "
                        "and mu plots. Default: expected-only (the CSV always "
                        "keeps its obs columns).")
    p.add_argument("--trust-range", nargs=2, type=float, default=None,
                   metavar=("LO", "HI"),
                   help="draw two vertical dashed lines marking the trusted "
                        "mass range (e.g. 1400 3200)")
    p.add_argument("--min-mass", type=float, default=1200.0,
                   help="drop masses below this (the m_WR=1000 point has a "
                        "clamped window -> huge, non-physical RMS)")
    p.add_argument("--trust-max", type=float, default=None,
                   help="marker at the m_WR beyond which the background "
                        "closure is not trusted (e.g. 3400)")
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

    cfg_path = args.signal_config or default_signal_config(args.signal_era)
    with open(cfg_path) as fh:
        sig_cfg = {v["dataset"]: v for v in json.load(fh).values()}
    logger.info("Signal config: %s (%d datasets)", cfg_path, len(sig_cfg))

    sig_dirs, _ = input_dirs_for_era(args.signal_era, repo_root(), args.signal_dir)
    hist_key = build_hist_key(build_region_name(channel, topology),
                              MASS_VAR[topology])

    table = args.table or (HERE.parents[0] / "6_spurious_signal_toys"
                           / run_sub / f"spurious_toy_table_{tag}.csv")
    if not table.exists():
        sys.exit(f"Stage-6 table not found: {table}")
    rows = read_stage6_table(table, channel, topology)
    logger.info("Read %d rows from %s", len(rows), table)

    funcs = [f for f in args.functions if f in FUNCS]
    eff_cache = {}          # mWR -> efficiency dict (shared across functions)
    per_fn, out_rows = {n: [] for n in funcs}, []
    for r in rows:
        if r["function"] not in funcs or r["mWR"] < args.min_mass:
            continue
        mWR = r["mWR"]
        if mWR not in eff_cache:
            if not r["signal_tag"]:
                logger.info("  m=%.0f: no signal tag (off-grid mass?), skip", mWR)
                eff_cache[mWR] = None
            else:
                eff_cache[mWR] = signal_efficiency(
                    sig_dirs, hist_key, r["signal_tag"],
                    r["fit_lo"], r["fit_hi"], sig_cfg, args.bin_width,
                    bfrac=args.channel_bfrac)
                if eff_cache[mWR]:
                    e = eff_cache[mWR]
                    logger.info(
                        "  m=%.0f %s: eff=%.4f (S_fit=%.1f / sumw=%.0f, "
                        "f_win=%.3f)  sigma_th=%.4g pb", mWR, r["signal_tag"],
                        e["eff"], e["s_fitrange"], e["genEventSumw"],
                        e["f_fitrange"], e["xsec_pb"])
        eff = eff_cache[mWR]
        if eff is None or not (eff["eff"] > 0.0):
            continue

        mu0 = r["mean_Nsp"] if args.center == "mean" else 0.0
        band = cls_band(mu0, r["rms_Nsp"], alpha)
        if not math.isfinite(band[0]):
            logger.info("  m=%.0f %-6s -> no valid RMS, skip", mWR, r["function"])
            continue
        # "Observed" = the limit on the unfluctuated-MC (Asimov) N_sp, with the
        # toy-calibrated sigma -- NOT the per-fit covariance error, which
        # under-covers (pull width ~1.1) and collapses in the sparse high-mass
        # windows. Same sigma as the band, so obs and expected are comparable.
        ul_obs = _cls_upper_limit(r["nsp_asimov"], r["rms_Nsp"], alpha)
        denom = 1000.0 * lumi * eff["eff"]                 # events per pb
        sigma_band = {N: band[N] / denom for N in BANDS}
        sigma_obs = ul_obs / denom
        mu_band = {N: sigma_band[N] / eff["xsec_pb"] for N in BANDS}

        per_fn[r["function"]].append(
            {"mWR": mWR, "sigma": sigma_band, "mu": mu_band,
             "sigma_obs": sigma_obs, "mu_obs": sigma_obs / eff["xsec_pb"],
             "xsec_pb": eff["xsec_pb"]})
        out_rows.append({
            "channel": channel, "topology": topology, "function": r["function"],
            "mWR": mWR, "m_N": eff["m_N"], "signal_tag": r["signal_tag"],
            "xsec_pb": f'{eff["xsec_pb"]:.6g}',
            "genEventSumw": f'{eff["genEventSumw"]:.6g}',
            "s_fitrange": f'{eff["s_fitrange"]:.4f}',
            "s_total": f'{eff["s_total"]:.4f}',
            "f_fitrange": f'{eff["f_fitrange"]:.4f}',
            "eff": f'{eff["eff"]:.6g}',
            **{f"sigma_ul_{lbl}": f"{sigma_band[N]:.6g}"
               for N, lbl in zip(BANDS, ("m2s", "m1s", "med", "p1s", "p2s"))},
            "sigma_ul_obs": f"{sigma_obs:.6g}" if math.isfinite(sigma_obs) else "",
            "mu_ul_med": f"{mu_band[0]:.6g}",
            "mu_ul_obs": (f"{sigma_obs / eff['xsec_pb']:.6g}"
                          if math.isfinite(sigma_obs) else ""),
        })
        logger.info("  m=%.0f %-6s -> median sigma_UL=%.4g pb  (mu_UL=%.3g)",
                    mWR, r["function"], sigma_band[0], mu_band[0])

    if not out_rows:
        logger.error("No limit points produced (empty table / no signal files?).")
        sys.exit(1)

    for name in funcs:
        if not per_fn[name]:
            continue
        common = dict(channel=channel, topology=topology, com=com, lumi=lumi,
                      cl=args.cl, trust_max=args.trust_max, center=args.center,
                      trust_range=args.trust_range)
        ll = {"ee": "ee", "mumu": r"\mu\mu"}[channel]
        pts_plot = per_fn[name]
        if not args.observed:                   # expected-only by default
            pts_plot = [dict(p, sigma_obs=float("nan"), mu_obs=float("nan"))
                        for p in pts_plot]
        plot_band(name, pts_plot,
                  args.output_dir / "xsec_limit" / tag / name,
                  ykey="sigma", obskey="sigma_obs", theory=True, scale=1000.0,
                  ylabel=(rf"$\sigma(pp \to W_R)\,\mathcal{{B}}"
                          rf"(W_R \to {ll}q\bar{{q}}\,')$ (fb)"), **common)
        plot_band(name, pts_plot,
                  args.output_dir / "xsec_limit" / tag / f"{name}_mu",
                  ykey="mu", obskey="mu_obs", theory=False,
                  ylabel=(rf"{args.cl*100:.0f}% CL upper limit on "
                          r"$\mu = \sigma/\sigma_{\rm theory}$"), **common)

    csv_path = args.output_dir / f"xsec_limit_table_{tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    logger.info("  wrote %s", csv_path)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
