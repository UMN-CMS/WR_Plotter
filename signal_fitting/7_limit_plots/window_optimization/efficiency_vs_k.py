#!/usr/bin/env python3
"""Window-optimization, Part A -- signal efficiency vs window width k.

For each grid m_WR (m_N = m_WR/2 diagonal, as the limit uses) and a grid of
window half-widths k, compute:
  * f_win   = S_fit / S_tot            signal containment in the window
  * eff     = S_fit / (bfrac*sumw)     per-channel efficiency x acceptance x f_win

using EXACTLY the window the S+B limit uses: [m_c - k*sigma_win, m_c + k*sigma_win]
clamped to [fit_min, fit_max], with (m_c, sigma_win) the Stage-2 linear window
parameterization and S_fit summed over the 100-GeV bins whose CENTRE lies in the
window (identical to fit_splusb / xsec_limit.signal_efficiency).

Standalone and cheap -- histogram integration only, no fitting. Feeds Part B
(sensitivity_vs_k.py), which multiplies eff(k) by the Stage-6 N_UL(k) to get the
sensitivity sigma_UL(k).

Why the study is non-trivial: sigma_win is the MEDIAN over M_N, so it is narrower
than the real (tailed, M_N-dependent) signal -- containment at k=3 is only ~92%,
not the Gaussian 99.7%, so widening the window recovers real signal. At low mass
the window's lower edge freezes at fit_min (clamp), capping that gain.

Outputs (namespaced {ch}_{topo}):
  efficiency_vs_k_{ch}_{topo}.csv         per (mass,k): window, clamp, f_win, eff
  containment_vs_k/{ch}_{topo}.*          f_win vs k, one line per mass
  eff_vs_k/{ch}_{topo}.*                  eff (normalized to k=3) vs k, per mass
  eff_map/{ch}_{topo}.*                   2D heatmap of f_win(mass, k)
  mn_spread/{ch}_{topo}/m{mWR}.*          f_win vs k with the M_N-point spread band

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
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
sys.path.insert(0, str(HERE.parent))                          # xsec_limit

from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root                # noqa: E402

from bkg_fit_lib import MASS_VAR, CH_LAB, TOPO_LAB, grid_widths_from_params  # noqa: E402
from measure_fwhm import parse_masses, build_region_name, build_hist_key  # noqa: E402
from shape_estimators import load_master_masses                          # noqa: E402
from sb_fit import pick_signal_tag                                       # noqa: E402
from xsec_limit import signal_efficiency, default_signal_config          # noqa: E402

logger = logging.getLogger("eff_vs_k")

DEFAULT_KGRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
KREF = 3.0                       # reference window for the eff-ratio plot


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _cmslabel(ax, com, lumi):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=16)


def compute(sig_dirs, hist_key, tag, m_c, sigma_win, kgrid, sig_cfg,
            bin_width, bfrac, fit_min, fit_max):
    """List of per-k dicts for one signal point (skips k with no efficiency)."""
    out = []
    for k in kgrid:
        nom_lo = m_c - k * sigma_win
        lo, hi = max(nom_lo, fit_min), min(m_c + k * sigma_win, fit_max)
        e = signal_efficiency(sig_dirs, hist_key, tag, lo, hi, sig_cfg,
                              bin_width, bfrac)
        if e is None:
            continue
        out.append({"k": k, "win_lo": lo, "win_hi": hi,
                    "clamped": nom_lo < fit_min - 1e-6,
                    "f_win": e["f_fitrange"], "eff": e["eff"]})
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_containment_vs_k(per_mass, out, *, channel, topology, com, lumi):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(per_mass)))
    for (mWR, rows), c in zip(sorted(per_mass.items()), cmap):
        ks = [r["k"] for r in rows]
        ax.plot(ks, [100 * r["f_win"] for r in rows], "o-", color=c, ms=4,
                lw=1.4, label=f"{mWR/1000:.1f}")
        clamped = [(r["k"], 100 * r["f_win"]) for r in rows if r["clamped"]]
        if clamped:
            ax.plot(*zip(*clamped), "s", color=c, ms=7, mfc="none", mew=1.4)
    ax.set_xlabel(r"Window half-width $k$  ($\pm k\sigma$)")
    ax.set_ylabel(r"Signal containment $f_{\rm win}=S_{\rm win}/S_{\rm tot}$ [%]")
    ax.text(0.04, 0.30, f"{CH_LAB[channel]}  {TOPO_LAB[topology]}\n"
            r"$m_N=m_{W_R}/2$" "\nopen square = clamped window",
            transform=ax.transAxes, fontsize=13, va="top")
    leg = ax.legend(loc="lower right", fontsize=10, ncol=2, frameon=False,
                    title=r"$m_{W_R}$ [TeV]")
    leg.get_title().set_fontsize(11)
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_eff_ratio_vs_k(per_mass, out, *, channel, topology, com, lumi):
    """Efficiency vs k, each mass normalized to its own k=KREF -- the *relative*
    signal gain from widening, which is what multiplies the sensitivity."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(per_mass)))
    for (mWR, rows), c in zip(sorted(per_mass.items()), cmap):
        ref = next((r["eff"] for r in rows if abs(r["k"] - KREF) < 1e-6), None)
        if not ref:
            continue
        ax.plot([r["k"] for r in rows], [r["eff"] / ref for r in rows], "o-",
                color=c, ms=4, lw=1.4, label=f"{mWR/1000:.1f}")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"Window half-width $k$  ($\pm k\sigma$)")
    ax.set_ylabel(fr"$\mathrm{{eff}}(k)\,/\,\mathrm{{eff}}(k={KREF:g})$")
    ax.text(0.04, 0.95, f"{CH_LAB[channel]}  {TOPO_LAB[topology]}\n"
            r"$m_N=m_{W_R}/2$" "\nrelative signal gain from widening",
            transform=ax.transAxes, fontsize=13, va="top")
    leg = ax.legend(loc="lower right", fontsize=10, ncol=2, frameon=False,
                    title=r"$m_{W_R}$ [TeV]")
    leg.get_title().set_fontsize(11)
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_eff_map(per_mass, kgrid, out, *, channel, topology, com, lumi):
    hep.style.use("CMS")
    masses = sorted(per_mass)
    grid = np.full((len(masses), len(kgrid)), np.nan)
    for i, mWR in enumerate(masses):
        by_k = {r["k"]: r["f_win"] for r in per_mass[mWR]}
        for j, k in enumerate(kgrid):
            if k in by_k:
                grid[i, j] = 100 * by_k[k]
    fig, ax = plt.subplots()
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                   extent=[min(kgrid) - 0.25, max(kgrid) + 0.25,
                           masses[0] / 1000 - 0.1, masses[-1] / 1000 + 0.1])
    fig.colorbar(im, ax=ax, label=r"containment $f_{\rm win}$ [%]")
    ax.set_xlabel(r"Window half-width $k$")
    ax.set_ylabel(r"$m_{W_R}$ [TeV]")
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def plot_mn_spread(mWR, spread, diag, out, *, channel, topology, com, lumi):
    """f_win vs k for every M_N point at this m_WR (grey) + the diagonal (m_WR/2)
    used by the limit, showing how much the median-sigma window under/over-covers
    across M_N."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    for tag, rows in spread.items():
        ax.plot([r["k"] for r in rows], [100 * r["f_win"] for r in rows],
                "-", color="0.75", lw=1.0, zorder=1)
    if diag:
        ax.plot([r["k"] for r in diag], [100 * r["f_win"] for r in diag],
                "o-", color="#e42536", ms=5, lw=1.8, zorder=3,
                label=r"$m_N=m_{W_R}/2$ (limit)")
    ax.set_xlabel(r"Window half-width $k$  ($\pm k\sigma$)")
    ax.set_ylabel(r"Signal containment $f_{\rm win}$ [%]")
    ax.text(0.04, 0.30, f"{CH_LAB[channel]}  {TOPO_LAB[topology]}\n"
            fr"$m_{{W_R}}={mWR:.0f}$ GeV" "\ngrey = each $M_N$ point",
            transform=ax.transAxes, fontsize=13, va="top")
    ax.legend(loc="lower right", fontsize=12, frameon=False)
    _cmslabel(ax, com, lumi)
    _save(fig, out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260624_signals", help="signal MC dir")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved", choices=["resolved", "boosted"])
    p.add_argument("--k-grid", nargs="+", type=float, default=DEFAULT_KGRID)
    p.add_argument("--mass-min", type=float, default=1000.0)
    p.add_argument("--mass-max", type=float, default=4000.0)
    p.add_argument("--mn-frac", type=float, default=0.5,
                   help="M_N / M_WR of the signal point (0.5 = the limit diagonal)")
    p.add_argument("--window-params", type=Path,
                   default=HERE.parents[1] / "2_width_parameterization" / "wr"
                   / "window_params.json")
    p.add_argument("--sigma-kind", default="median",
                   choices=["median", "conservative"])
    p.add_argument("--mass-csv", type=Path,
                   default=HERE.parents[1] / "master_masses.csv")
    p.add_argument("--signal-config", type=Path, default=None)
    p.add_argument("--signal-era", default=None,
                   help="era whose config carries genEventSumw (default: --era)")
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=6000.0)
    p.add_argument("--channel-bfrac", type=float, default=0.5)
    p.add_argument("--no-mn-spread", action="store_true",
                   help="skip the per-mass M_N-spread plots (faster)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    ch, topo = args.channel, args.topology
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    run_sub = {"RunII": "run2", "Run3": "run3"}[str(info["run"])]
    out_dir = args.output_dir or (HERE / run_sub)
    kgrid = sorted(args.k_grid)

    cfg_path = args.signal_config or default_signal_config(args.signal_era or args.era)
    with open(cfg_path) as fh:
        sig_cfg = {v["dataset"]: v for v in json.load(fh).values()}
    sig_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    hist_key = build_hist_key(build_region_name(ch, topo), MASS_VAR[topo])
    all_tags = load_master_masses(args.mass_csv, topo)
    mwr_all = sorted({parse_masses(t)[0] for t in all_tags})
    masses = [m for m in mwr_all if args.mass_min <= m <= args.mass_max]
    grid = grid_widths_from_params(args.window_params, ch, topo,
                                   [float(m) for m in masses], args.sigma_kind)

    tag = f"{ch}_{topo}"
    logger.info("Efficiency vs k: %s %s, k=%s, %d masses [%g,%g]",
                ch, topo, kgrid, len(masses), args.mass_min, args.mass_max)

    per_mass, rows = {}, []
    for mWR in masses:
        stag = pick_signal_tag(all_tags, mWR, args.mn_frac)
        if stag is None:
            logger.info("  m=%.0f: no m_N=%.2f signal point, skip", mWR, args.mn_frac)
            continue
        m_c, sw = grid[mWR]
        recs = compute(sig_dirs, hist_key, stag, m_c, sw, kgrid, sig_cfg,
                       args.bin_width, args.channel_bfrac, args.fit_min, args.fit_max)
        if not recs:
            logger.info("  m=%.0f (%s): no efficiency, skip", mWR, stag)
            continue
        per_mass[mWR] = recs
        for r in recs:
            rows.append({"channel": ch, "topology": topo, "mWR": int(mWR),
                         "signal_tag": stag, "m_N": parse_masses(stag)[1],
                         "m_c": round(m_c, 1), "sigma_win": round(sw, 2),
                         "k": r["k"], "win_lo": round(r["win_lo"], 1),
                         "win_hi": round(r["win_hi"], 1), "clamped": int(r["clamped"]),
                         "f_win": round(r["f_win"], 4), "eff": round(r["eff"], 5)})
        f3 = next((r["f_win"] for r in recs if abs(r["k"] - KREF) < 1e-6), None)
        f5 = next((r["f_win"] for r in recs if abs(r["k"] - 5.0) < 1e-6), None)
        logger.info("  m=%.0f %-11s sigma=%3.0f  f_win: k3=%.3f k5=%.3f (gain %+.1f%%)",
                    mWR, stag, sw, f3 or 0, f5 or 0,
                    100 * (f5 / f3 - 1) if (f3 and f5) else 0)

    if not per_mass:
        logger.error("No efficiencies computed."); sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"efficiency_vs_k_{tag}.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info("  wrote %s", csv_path)

    plot_containment_vs_k(per_mass, out_dir / "containment_vs_k" / tag,
                          channel=ch, topology=topo, com=com, lumi=lumi)
    plot_eff_ratio_vs_k(per_mass, out_dir / "eff_vs_k" / tag,
                        channel=ch, topology=topo, com=com, lumi=lumi)
    plot_eff_map(per_mass, kgrid, out_dir / "eff_map" / tag,
                 channel=ch, topology=topo, com=com, lumi=lumi)

    if not args.no_mn_spread:
        for mWR in per_mass:
            tags_here = [t for t in all_tags if parse_masses(t)[0] == mWR]
            if len(tags_here) < 2:
                continue
            m_c, sw = grid[mWR]
            spread = {t: compute(sig_dirs, hist_key, t, m_c, sw, kgrid, sig_cfg,
                                 args.bin_width, args.channel_bfrac,
                                 args.fit_min, args.fit_max) for t in tags_here}
            spread = {t: r for t, r in spread.items() if r}
            plot_mn_spread(mWR, spread, per_mass[mWR],
                           out_dir / "mn_spread" / tag / f"m{int(mWR)}",
                           channel=ch, topology=topo, com=com, lumi=lumi)

    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
