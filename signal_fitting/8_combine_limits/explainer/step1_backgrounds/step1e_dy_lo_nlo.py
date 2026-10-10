#!/usr/bin/env python3
"""Step 1e -- two Drell-Yan simulations of the same region: LO (MadGraph MLM)
against NLO (aMC@NLO FxFx), both scaled to the era luminosity, with their
ratio. Seminar point: at high m_lljj the DY prediction depends on which
simulation you ask.

    WR_SEMINAR_ERA=2024 python step1e_dy_lo_nlo.py [--region wr_mumu_resolved_sr]

Reads WRAnalyzer_DYJets.root from the era's lo/ and nlo/ sets (symlinks to
the dated inclusive-DY runs). Made 2026-10-03 for the HEP seminar.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C
from wrplotter.paths import input_dirs_for_era, repo_root

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")


def load(subdir, region, rebin):
    dirs, _ = input_dirs_for_era(C.ERA, repo_root(), subdir)
    edges = vals = w2 = None
    for d in dirs:
        h = uproot.open(d / "WRAnalyzer_DYJets.root")[f"{region}/mass_fourobject_{region}"]
        e, v, var = h.axes[0].edges(), h.values(), h.variances()
        edges = e
        vals = v.copy() if vals is None else vals + v
        w2 = var.copy() if w2 is None else w2 + var
    n = (len(vals) // rebin) * rebin
    return (edges[0:n + 1:rebin], vals[:n].reshape(-1, rebin).sum(axis=1),
            np.sqrt(w2[:n].reshape(-1, rebin).sum(axis=1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--region", default="wr_mumu_resolved_sr")
    ap.add_argument("--rebin", type=int, default=20, help="x 10 GeV native bins")
    ap.add_argument("--xmax", type=float, default=6000.0)
    args = ap.parse_args()
    e, lo, lo_e = load("lo", args.region, args.rebin)
    _, nlo, nlo_e = load("nlo", args.region, args.rebin)
    c = 0.5 * (e[:-1] + e[1:])
    bw = e[1] - e[0]
    chan = "ee" if "_ee_" in args.region else r"$\mu\mu$"

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
    for y, ye, col, lab in ((lo, lo_e, "#5790fc", "DY+jets, LO (MadGraph MLM)"),
                            (nlo, nlo_e, "#e42536", "DY+jets, NLO (aMC@NLO FxFx)")):
        ax.stairs(np.clip(y, 1e-4, None), e, color=col, lw=2.2, label=lab)
        ax.errorbar(c, np.clip(y, 1e-4, None), yerr=np.abs(np.nan_to_num(ye)), fmt="none",
                    ecolor=col, elinewidth=1.0, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlim(800, args.xmax)
    ax.set_ylim(1e-2, 2e3)
    C.log_yaxis_one_ten(ax)
    ax.set_ylabel(f"Events / {bw:.0f} GeV")
    ax.legend(loc="upper right", fontsize=15, frameon=False)
    ax.text(0.03, 0.96, f"{chan}\nResolved SR\n{C.MC_LABEL}\nDY+jets only",
            transform=ax.transAxes, fontsize=16, va="top")
    C.cms_label(ax)
    good = lo > 0
    ratio = np.where(good, nlo / np.where(good, lo, 1.0), np.nan)
    rerr = np.where(good, ratio * np.sqrt((nlo_e / np.where(nlo > 0, nlo, 1)) ** 2
                                          + (lo_e / np.where(good, lo, 1)) ** 2), np.nan)
    rax.errorbar(c, ratio, yerr=np.abs(np.nan_to_num(rerr)), fmt="ko", ms=4)
    rax.axhline(1.0, color="gray", ls="--", lw=1.2)
    rax.set_ylim(0, 2.5)
    rax.set_ylabel("NLO / LO")
    rax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
    C.savefig(fig, HERE / f"step1e_dy_lo_nlo_{args.region.split('_')[1]}")
    for lo_m in (1000, 2000, 3000, 4000):
        sel = c >= lo_m
        print(f"m > {lo_m}: LO {lo[sel].sum():7.1f}  NLO {nlo[sel].sum():7.1f}  "
              f"ratio {nlo[sel].sum() / max(lo[sel].sum(), 1e-9):.2f}")


if __name__ == "__main__":
    main()
