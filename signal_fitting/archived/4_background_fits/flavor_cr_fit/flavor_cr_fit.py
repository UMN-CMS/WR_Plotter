#!/usr/bin/env python3
"""Stage 4 — how well do the background functions fit the DATA in the flavor CR?

The flavor (eμ) control region is orthogonal to the ee/μμ signal region by
construction, so fitting *data* there never touches the SR — no unblinding. It is
tt/tW-dominated and signal-free, so it is the right place to ask whether the
candidate background shapes describe the real data (not just MC).

For one topology (default resolved) and one data stream (EGamma or Muon — the eμ
events are nearly the same in both, so use ONE, do not sum):

  1. Load the four-object mass in the flavor CR
     (resolved → wr_resolved_flavor_cr/mass_fourobject;
      boosted  → wr_{emu,mue}_boosted_flavor_cr/mass_twoobject, summed).
  2. Fit each candidate function to the data over [fit-min, fit-max] (weighted
     log least squares; Poisson per-bin variance = N).
  3. Report χ²/ndf and per-bin pulls — how well each function fits the data.

NO signal-region data is used anywhere.

Outputs (co-located):
  fits/{data}_{topology}.{png,pdf}   data + all fits (log y) + pull panel
  flavor_cr_fit_table.csv            per-function χ²/ndf and fitted coefficients

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
Usage:
  python flavor_cr_fit.py -v
  python flavor_cr_fit.py --data Muon --fit-max 3500 -v
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
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))                  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                  # bkg_fit_lib

from wrplotter.cli_utils import setup_logging                                  # noqa: E402
from wrplotter.config import load_lumi                                         # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root                      # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter                    # noqa: E402

from bkg_fit_lib import (                                                       # noqa: E402
    COEF_SYMS, FUNCS, MASS_VAR, MASS_LABEL, TOPO_LAB,
    func_label, fit_model, predict, chi2_ndf,
)

logger = logging.getLogger(__name__)

# Flavor-CR region(s) per topology (eμ; resolved is one region, boosted two).
FLAVOR_REGIONS = {
    "resolved": ["wr_resolved_flavor_cr"],
    "boosted": ["wr_emu_boosted_flavor_cr", "wr_mue_boosted_flavor_cr"],
}


def load_cr_data(input_dirs, data, regions, variable, factor):
    """Flavor-CR data histogram, summed over regions and sub-era dirs; rebinned.

    Returns (edges, counts). The per-bin variance is Poisson (= counts), set by
    the caller — data is unweighted."""
    edges = values = None
    for d in input_dirs:
        fpath = d / f"WRAnalyzer_{data}.root"
        if not fpath.exists():
            logger.warning("Missing data file: %s", fpath)
            continue
        f = uproot.open(fpath)
        for region in regions:
            key = f"{region}/{variable}_{region}"
            h = f[key]
            e, v = h.axes[0].edges(), h.values()
            if edges is None:
                edges, values = e, v.copy()
            else:
                values += v
    if edges is None:
        raise RuntimeError("No flavor-CR data histograms found")
    if factor > 1:
        n = (len(values) // factor) * factor
        values = values[:n].reshape(-1, factor).sum(axis=1)
        edges = edges[::factor][: len(values) + 1]
    return edges, values


def plot_fits(edges, values, fits, out_path, *, data, topology, region_lab,
              fit_lo, fit_hi, com, lumi, era, bin_w):
    """Data spectrum + every fit (log y), with a per-bin pull panel below."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    pos = values > 0
    hep.style.use("CMS")
    fig, (ax, axp) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 8.6),
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))

    ax.errorbar(centers[pos], values[pos], yerr=np.sqrt(values[pos]), fmt="o",
                color="black", markersize=4, elinewidth=0.9, capsize=1.5,
                zorder=5, label="data")
    grid = np.linspace(fit_lo, fit_hi, 500)
    for name, (color, _, _) in FUNCS.items():
        d = fits.get(name)
        if d is None:
            continue
        ax.plot(grid, predict(name, d["params"], grid), color=color, lw=1.8,
                zorder=3, label=f"{func_label(name)}   "
                rf"$\chi^2/\mathrm{{ndf}}={d['chi2_ndf']:.2f}$")
        # pull = (data - fit)/sqrt(N) on the fitted bins
        fb = d["fit_bins"]
        pull = (values[fb] - predict(name, d["params"], centers[fb])) / np.sqrt(values[fb])
        axp.plot(centers[fb], pull, marker="o", markersize=3, lw=1.0, color=color)

    ax.axvspan(fit_lo, fit_hi, color="#5790fc", alpha=0.06, zorder=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    pp = values[pos]
    ax.set_ylim(0.5, pp.max() * 5.0)
    ax.set_ylabel(f"Events / {bin_w:.0f} GeV")
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=True, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=17)
    ax.text(0.05, 0.94, f"{region_lab}\n{era}\nfit data: {data}",
            transform=ax.transAxes, fontsize=13, va="top")
    ax.legend(loc="upper right", fontsize=10.5, title=r"fit $f(m)$", ncol=1)

    for y in (0, 2, -2):
        axp.axhline(y, color="gray", lw=0.8,
                    ls="-" if y == 0 else ":")
    axp.set_ylim(-5, 5)
    axp.set_ylabel(r"$\frac{\mathrm{data}-\mathrm{fit}}{\sqrt{N}}$", fontsize=15)
    axp.set_xlabel(MASS_LABEL[topology])
    axp.set_xlim(edges[0], fit_hi + 2 * bin_w)
    axp.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("  wrote %s", out_path.with_suffix(".png"))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20260317_lo_dy")
    p.add_argument("--data", default="EGamma", choices=["EGamma", "Muon"],
                   help="Data stream to fit (eμ events are ~the same in both; "
                        "use one, do not sum).")
    p.add_argument("--topology", default="resolved", choices=["resolved", "boosted"])
    p.add_argument("--bin-width", type=float, default=100.0)
    p.add_argument("--fit-min", type=float, default=800.0)
    p.add_argument("--fit-max", type=float, default=3000.0,
                   help="Upper fit edge (GeV); the data thins out above ~3 TeV.")
    p.add_argument("--functions", nargs="+", default=list(FUNCS),
                   choices=list(FUNCS))
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    era, topology = args.era, args.topology
    info = load_lumi(era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    funcs = [f for f in FUNCS if f in args.functions]

    input_dirs, _ = input_dirs_for_era(era, repo_root(), args.dir)
    regions = FLAVOR_REGIONS[topology]
    variable = MASS_VAR[topology]
    factor = max(1, round(args.bin_width / 10.0))
    edges, values = load_cr_data(input_dirs, args.data, regions, variable, factor)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fit_bins = ((centers >= args.fit_min) & (centers <= args.fit_max)
                & (values > 0))
    m, n = centers[fit_bins], values[fit_bins]
    var = n.copy()                                  # Poisson: Var(N) = N
    logger.info("%s flavor CR, %s: %d events, %d populated fit bins in [%g, %g]",
                topology, args.data, int(values.sum()), int(fit_bins.sum()),
                args.fit_min, args.fit_max)

    region_lab = (rf"$e\mu$ flavor CR ({TOPO_LAB[topology]})")
    rows, fits = [], {}
    print(f"\n{'function':8} {'ndf':>4} {'chi2':>8} {'chi2/ndf':>9}")
    print("-" * 33)
    for name in funcs:
        res = fit_model(name, m, n, var)
        if res is None:
            logger.warning("  %s: fit failed", name)
            continue
        params, cov = res
        c2, ndf = chi2_ndf(name, params, m, n, var)
        cn = c2 / ndf if ndf > 0 else float("nan")
        fits[name] = {"params": params, "cov": cov, "chi2_ndf": cn,
                      "fit_bins": fit_bins}
        print(f"{name:8} {ndf:>4} {c2:>8.1f} {cn:>9.2f}")
        cerr = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        npar = FUNCS[name][2]
        row = {"data": args.data, "topology": topology, "region": "+".join(regions),
               "function": name, "n_fit_bins": int(fit_bins.sum()), "ndf": ndf,
               "chi2": round(c2, 3), "chi2_ndf": round(cn, 4) if ndf > 0 else ""}
        for i in range(4):
            row[f"coef_{COEF_SYMS[i]}"] = round(float(params[i]), 6) if i < npar else ""
            row[f"coef_{COEF_SYMS[i]}_err"] = (
                round(float(cerr[i]), 6) if i < npar and np.isfinite(cerr[i]) else
                ("inf" if i < npar else ""))
        rows.append(row)

    if not fits:
        logger.error("No successful fits.")
        sys.exit(1)

    best = min(fits, key=lambda k: fits[k]["chi2_ndf"])
    print(f"\nbest χ²/ndf: {best} ({fits[best]['chi2_ndf']:.2f})")

    plot_fits(edges, values, fits,
              args.output_dir / "fits" / f"{args.data}_{topology}",
              data=args.data, topology=topology, region_lab=region_lab,
              fit_lo=args.fit_min, fit_hi=args.fit_max, com=com, lumi=lumi,
              era=era, bin_w=args.bin_width)

    csv_path = args.output_dir / "flavor_cr_fit_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s", csv_path)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
