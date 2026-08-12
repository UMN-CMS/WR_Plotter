#!/usr/bin/env python3
"""Overlay the toy spurious-signal N_sp distributions across window widths k,
for a fixed mass -- the direct picture of "widening the window pins the
background shape": more sideband bins constrain the background, so the fitted
signal yield scatters less and the N_sp distribution NARROWS (smaller RMS =
smaller N_UL = tighter limit).

Redraws the SAME Stage-6 toys (identical TRandom3 seed scheme), so the per-k RMS
here matches the sweep tables. Density-normalized (area=1) on a COMMON x-axis, so
a narrower curve is literally a taller, tighter peak.

Watch for the flip side at high mass: past the point where the window outgrows
the well-modeled region the peak stops narrowing and the whole distribution
shifts off zero (the spurious BIAS) -- the wide window is then pinning a
mis-modeled shape. The mean is drawn as a tick under each curve.

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import array
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
sys.path.insert(0, str(HERE.parents[2]))                       # repo root
sys.path.insert(0, str(HERE.parents[1] / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(HERE.parents[1] / "shared"))           # loaders

from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root                # noqa: E402

import ROOT                                                              # noqa: E402
from bkg_fit_lib import MASS_VAR, CH_LAB, TOPO_LAB, grid_widths_from_params  # noqa: E402
from sb_fit import fit_splusb                                            # noqa: E402

logger = logging.getLogger("nsp_vs_k")
DEFAULT_KGRID = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def draw_nsps(edges, mu_bkg, m_c, sigma_win, k, binwidth, fit_min, fit_max,
              name, ntoys, seed):
    """Reproduce the Stage-6 toy loop for one (mass, k): list of converged N_sp."""
    lo, hi = max(m_c - k * sigma_win, fit_min), min(m_c + k * sigma_win, fit_max)
    rng = ROOT.TRandom3(seed)
    nsps = []
    for _ in range(ntoys):
        toy = np.array([rng.Poisson(mu) for mu in mu_bkg], dtype=float)
        r = fit_splusb(name, edges, toy, lo, hi, m_c, sigma_win, m_c, sigma_win,
                       binwidth, k)
        if r is None or r["status"] != 0:
            continue
        ne, nf = r["nsig_err"], r["nsig"]
        if ne > 0 and math.isfinite(ne) and math.isfinite(nf):
            nsps.append(nf)
    return nsps


def _stats(nsps):
    a = array.array("d", nsps)
    n = len(a)
    return float(ROOT.TMath.Mean(n, a)), float(ROOT.TMath.RMS(n, a))


def plot_overlay(mWR, per_k, out, *, channel, topology, com, lumi, name, ntoys):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    kmax_rms = max(s["rms"] for s in per_k.values())
    span = 3.5 * kmax_rms
    bins = np.linspace(-span, span, 46)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(per_k)))
    for (k, s), c in zip(sorted(per_k.items()), cmap):
        clipped = np.clip(s["nsps"], bins[0], bins[-1])
        ax.hist(clipped, bins=bins, density=True, histtype="step", color=c,
                lw=1.8, label=fr"$k={k:g}$  (RMS ${s['rms']:.1f}$, mean ${s['mean']:+.1f}$)")
        ax.plot([s["mean"]], [0], marker="v", color=c, ms=7, clip_on=False, zorder=5)
    ax.axvline(0.0, color="0.5", lw=1.0, ls=":")
    ax.set_xlim(-span, span)
    ax.set_xlabel(r"Fitted signal yield $\hat{N}_{\rm sig}$ [events]")
    ax.set_ylabel("Toy density [1/event]")
    ax.text(0.03, 0.97, f"{CH_LAB[channel]}  {TOPO_LAB[topology]}  ({name})\n"
            fr"$m_{{W_R}}={mWR:.0f}$ GeV,  {ntoys} toys/$k$" "\n"
            "narrower = background better pinned\n"
            r"$\blacktriangledown$ = distribution mean (spurious bias)",
            transform=ax.transAxes, fontsize=12, va="top")
    ax.legend(loc="upper right", fontsize=11, frameon=False)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=16)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260714_run2_bkgs", help="background dir")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved", choices=["resolved", "boosted"])
    p.add_argument("--masses", nargs="+", type=float, default=[2000, 3000])
    p.add_argument("--k-grid", nargs="+", type=float, default=DEFAULT_KGRID)
    p.add_argument("--function", default="expo")
    p.add_argument("--ntoys", type=int, default=1000)
    p.add_argument("--seed", type=int, default=12345,
                   help="Stage-6 base seed (keep default to match the sweep toys)")
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--sigma-kind", default="median", choices=["median", "conservative"])
    p.add_argument("--window-params", type=Path,
                   default=HERE.parents[1] / "2_width_parameterization" / "wr"
                   / "window_params.json")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    ch, topo, name = args.channel, args.topology, args.function
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    run_sub = {"RunII": "run2", "Run3": "run3"}[str(info["run"])]
    out_dir = args.output_dir or (HERE / run_sub / "nsp_hist_vs_k")
    kgrid = sorted(args.k_grid)

    bkg_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    region = f"wr_{ch}_{topo}_sr"
    factor = max(1, round(args.bin_width / 10.0))
    from bkg_fit_lib import load_summed_background
    edges, values, _ = load_summed_background(bkg_dirs, region, MASS_VAR[topo], factor)
    binwidth = float(edges[1] - edges[0])
    mu_bkg = [float(v) if (math.isfinite(v) and v > 0.0) else 0.0 for v in values]
    grid = grid_widths_from_params(args.window_params, ch, topo,
                                   [float(m) for m in args.masses], args.sigma_kind)
    tag = f"{ch}_{topo}"

    for mWR in args.masses:
        m_c, sw = grid[mWR]
        per_k = {}
        for k in kgrid:
            seed = args.seed * 1_000_003 + int(mWR) * 1009 + 0
            nsps = draw_nsps(edges, mu_bkg, m_c, sw, k, binwidth,
                             args.fit_min, args.fit_max, name, args.ntoys, seed)
            if len(nsps) < 50:
                logger.info("  m=%.0f k=%g: only %d toys, skip", mWR, k, len(nsps))
                continue
            mean, rms = _stats(nsps)
            per_k[k] = {"nsps": nsps, "mean": mean, "rms": rms}
            logger.info("  m=%.0f k=%g -> RMS=%.1f mean=%+.1f (%d toys)",
                        mWR, k, rms, mean, len(nsps))
        if per_k:
            plot_overlay(mWR, per_k, out_dir / tag / f"m{int(mWR)}_{name}",
                         channel=ch, topology=topo, com=com, lumi=lumi,
                         name=name, ntoys=args.ntoys)
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
