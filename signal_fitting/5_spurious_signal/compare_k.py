#!/usr/bin/env python3
"""Stage 5 -- overlay the spurious-signal results across window widths.

Reads the per-k spurious tables written by spurious_signal.py
(`k{K}/spurious_table_{ch}_{topo}.csv`) and overlays k=2,3,4,5 on one axis, for
each (channel, topology) and background function:

  k_comparison/{ch}_{topo}/pull_vs_mass_{fn}.*    pull = N_sp/sigma vs m_WR per k
  k_comparison/{ch}_{topo}/yield_vs_mass_{fn}.*   spurious N_sp +- sigma vs m_WR per k

ONLY fits that passed all four quality checks (valid_minimum / cov_ok /
no_param_at_limit / monotonic) are plotted; failed fits are dropped.

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
sys.path.insert(0, str(HERE.parents[1]))                        # repo root
sys.path.insert(0, str(HERE.parents[0] / "4_background_fits"))  # bkg_fit_lib

from wrplotter.cli_utils import setup_logging                  # noqa: E402
from wrplotter.config import load_lumi                         # noqa: E402
from bkg_fit_lib import FUNCS, CH_LAB, TOPO_LAB                 # noqa: E402

logger = logging.getLogger("compare_k")


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _load_k(output_dir, k, channel, topology):
    """{function: [rows]} of fit_ok points from one k folder, or {} if missing."""
    path = output_dir / f"k{k}" / f"spurious_table_{channel}_{topology}.csv"
    if not path.exists():
        logger.warning("missing %s", path)
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        if r["fit_ok"] != "True" or r["fit_passed"] != "True":  # passed only
            continue
        pt = {"mWR": float(r["mWR"]), "nsig": float(r["N_spur"]),
              "nsig_err": float(r["N_spur_err"]), "pull": float(r["pull"])}
        for nm in "abcd":                       # background-component coefficients
            v = r.get(f"coef_{nm}", "")
            if v not in ("", None):
                pt[f"coef_{nm}"] = float(v)
                pt[f"coef_{nm}_err"] = float(r[f"coef_{nm}_err"])
        out.setdefault(r["function"], []).append(pt)
    return out


def _k_colors(ks):
    cmap = matplotlib.colormaps["viridis"]
    n = max(len(ks) - 1, 1)
    return {k: cmap(i / n) for i, k in enumerate(sorted(ks))}


def _plot_split(ax, pts, key, color, label, yerr_key=None):
    """Line + markers through the (passed-only) points for one k."""
    pts = sorted(pts, key=lambda p: p["mWR"])
    m = [p["mWR"] for p in pts]
    y = [p[key] for p in pts]
    ye = [p[yerr_key] for p in pts] if yerr_key else None
    ax.errorbar(m, y, yerr=ye, fmt="o-", color=color, ms=4.5, lw=1.3,
                elinewidth=0.8, capsize=1.5, zorder=3, label=label)


def _robust_ylim(ax, vals, pad=0.30):
    """Non-symmetric robust y-limits from the 5-95 pct of central values, so a
    few high-mass outliers don't crush the scale."""
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        return
    lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    span = max(hi - lo, 1e-6)
    ax.set_ylim(lo - pad * span, hi + pad * span)


def plot_coef_vs_k(per_k, name, ks, out, *, channel, topology, com, lumi,
                   mass_max):
    """Background-component coefficients a (log-yield at mu) and b (slope) vs the
    window half-width k, one curve per m_WR (m_WR <= mass_max). m_WR is the
    colorbar z-axis. Shows how the fitted background shifts as the window opens
    up. Passed-only fits."""
    by_mass = {}                                 # mWR -> {k: point}
    for k in ks:
        for p in per_k.get(k, {}).get(name, []):
            if "coef_a" in p and p["mWR"] <= mass_max:
                by_mass.setdefault(p["mWR"], {})[k] = p
    masses = sorted(by_mass)
    if not masses:
        return
    vmin, vmax = masses[0], masses[-1]
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.cm.viridis
    hep.style.use("CMS")
    fig, (axa, axb) = plt.subplots(2, 1, sharex=True, figsize=(8.5, 9.0),
                                   constrained_layout=True)
    for which, ax in (("a", axa), ("b", axb)):
        allc = []
        for m in masses:
            d = by_mass[m]
            kk = sorted(d)
            y = [d[k][f"coef_{which}"] for k in kk]
            ye = [d[k][f"coef_{which}_err"] for k in kk]
            allc += y
            ax.errorbar(kk, y, yerr=ye, fmt="o-", color=cmap(norm(m)), ms=4,
                        lw=1.2, elinewidth=0.7, capsize=1.2, zorder=2)
        _robust_ylim(ax, allc)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=(axa, axb), pad=0.02)
    cbar.set_label(r"$m_{W_R}$ [GeV]", fontsize=16)
    axa.set_ylabel(r"$a$  ($\log$-yield at $m_{c}$)")
    axb.set_ylabel(r"$b$  (slope, /TeV)")
    axb.set_xlabel(r"window half-width $k$  [$\sigma$]")
    axb.set_xticks(sorted(ks))
    axa.text(0.03, 0.95,
             f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n{name}: {FUNCS[name][1]}"
             fr"  ($m_{{W_R}}\leq{mass_max:.0f}$)",
             transform=axa.transAxes, va="top", fontsize=12)
    hep.cms.label(loc=0, ax=axa, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    _save(fig, out)


def _annot(ax, channel, topology, name):
    ax.text(0.03, 0.96,
            f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n"
            f"{name}: {FUNCS[name][1]}",
            transform=ax.transAxes, va="top", fontsize=12)


def plot_pull(per_k, name, ks, out, *, channel, topology, com, lumi):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhspan(-0.5, 0.5, color="#f7d600", alpha=0.20, lw=0, zorder=0)
    ax.axhspan(-0.2, 0.2, color="#74c476", alpha=0.30, lw=0, zorder=0)
    ax.axhline(0.0, color="black", lw=0.8, zorder=1)
    colors = _k_colors(ks)
    for k in ks:
        pts = per_k.get(k, {}).get(name)
        if pts:
            _plot_split(ax, pts, "pull", colors[k], fr"$k={k}$")
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"spurious signal / $\sigma_{N_{\rm sp}}$  (pull)")
    ax.set_ylim(-1.5, 1.5)
    _annot(ax, channel, topology, name)
    ax.legend(fontsize=10, loc="upper right", ncol=2,
              title="window")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    _save(fig, out)


def plot_yield(per_k, name, ks, out, *, channel, topology, com, lumi):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ax.axhline(0.0, color="black", lw=1.0, ls="--", zorder=1)
    colors = _k_colors(ks)
    central = []
    for k in ks:
        pts = per_k.get(k, {}).get(name)
        if pts:
            _plot_split(ax, pts, "nsig", colors[k], fr"$k={k}$",
                        yerr_key="nsig_err")
            central += [p["nsig"] for p in pts]
    if central:
        v = np.array(central)
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        span = max(hi - lo, 1e-6)
        mlim = max(abs(lo - 0.25 * span), abs(hi + 0.25 * span))
        ax.set_ylim(-mlim, mlim)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"spurious yield $N_{\rm sp}$ [events]")
    _annot(ax, channel, topology, name)
    ax.legend(fontsize=10, loc="upper right", ncol=2,
              title="window")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    _save(fig, out)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4, 5])
    p.add_argument("--functions", nargs="+", default=["expo", "powlaw"],
                   choices=list(FUNCS))
    p.add_argument("--mass-max", type=float, default=3400.0,
                   help="coef_vs_k plot: only show m_WR <= this (trustworthy region)")
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    channel, topology = args.channel, args.topology
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)

    per_k = {k: _load_k(args.output_dir, k, channel, topology) for k in args.ks}
    ks = [k for k in args.ks if per_k[k]]
    if not ks:
        logger.error("No per-k tables found under %s", args.output_dir)
        sys.exit(1)

    outd = args.output_dir / "k_comparison" / f"{channel}_{topology}"
    for name in args.functions:
        if not any(name in per_k[k] for k in ks):
            continue
        plot_pull(per_k, name, ks, outd / f"pull_vs_mass_{name}",
                  channel=channel, topology=topology, com=com, lumi=lumi)
        plot_yield(per_k, name, ks, outd / f"yield_vs_mass_{name}",
                   channel=channel, topology=topology, com=com, lumi=lumi)
        plot_coef_vs_k(per_k, name, ks, outd / f"coef_vs_k_{name}",
                       channel=channel, topology=topology, com=com, lumi=lumi,
                       mass_max=args.mass_max)
        logger.info("  %s: wrote pull + yield + coef overlays (k=%s)", name, ks)
    logger.info("Done. Outputs in %s", outd)


if __name__ == "__main__":
    main()
