#!/usr/bin/env python3
"""Plot the 1D MC mass distribution for every signal point in the master grid.

Reads `master_masses.csv` and plots each point in its tagged topology — resolved
points use the resolved SR (mass_fourobject), boosted points use the boosted SR
(mass_twoobject). For each (channel, mass), produces one plot of the MC histogram
(the signal sample shape), with the curve labeled by its (M_WR, M_N).

Layout:
  signal_samples/1d_histograms/{mass_lower}/{channel}.{pdf,png}

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

try:
    import ROOT
except ImportError:
    sys.exit("ERROR: PyROOT unavailable. Source LCG_106 first.")
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

from wrplotter.cli_utils import setup_logging
from wrplotter.paths import input_dirs_for_era, repo_root
from measure_fwhm import (
    build_hist_key, build_region_name,
    load_and_combine_signal, parse_masses, rebin_histogram,
)
from shape_estimators import load_master_masses

logger = logging.getLogger(__name__)

MASS_VAR = {"resolved": "mass_fourobject", "boosted": "mass_twoobject"}
# Resolved is the four-object mass (ℓℓjj); boosted is the two-object mass —
# one lepton + one large-R jet (ℓJ).
MASS_LABEL = {
    "resolved": r"$m_{\ell\ell jj}$ [GeV]",
    "boosted": r"$m_{\ell J}$ [GeV]",
}

CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260624_signals")
    p.add_argument("--channels", nargs="+", default=["ee", "mumu"])
    p.add_argument("--topologies", nargs="+", default=["resolved", "boosted"],
                   help="Which topologies to plot. Each mass is drawn in its "
                        "CSV-tagged topology.")
    p.add_argument("--mass-csv", type=Path,
                   default=Path(__file__).resolve().parents[1] / "master_masses.csv")
    p.add_argument("--masses", nargs="+", default=None,
                   help="Optional subset of mass tags to plot. Default: every "
                        "point in the master CSV.")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "1d_histograms")
    p.add_argument("--bin-width", type=float, default=80.0,
                   help="Display bin width in GeV (rebinned from native). "
                        "Default 80.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def plot_one(channel, mass, topology, edges, vals, out_path, era):
    M_WR, M_N = parse_masses(mass)

    hep.style.use("CMS")
    fig, ax = plt.subplots()

    # MC histogram (stairs), labeled by its (M_WR, M_N).
    ax.stairs(vals, edges, color="#1f77b4", linewidth=1.5,
              label=rf"$M_{{W_R}}={int(M_WR)}$, $M_N={int(M_N)}$ GeV")

    ax.set_xlim(0, 10000)
    _, y_hi = ax.get_ylim()
    ax.set_ylim(0, y_hi * 1.30)

    bin_width = float(edges[1] - edges[0])
    ax.set_xlabel(MASS_LABEL[topology])
    ax.set_ylabel(f"Events / {bin_width:.0f} GeV")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=18)

    # Info text (no box), upper-left.
    ax.text(
        0.05, 0.96,
        f"{CH_LAB[channel]}\n{topology.capitalize()} SR\n{era}",
        transform=ax.transAxes, fontsize=14, verticalalignment="top",
    )

    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.98),
              fontsize=13, framealpha=0.85)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    grouped = load_master_masses(args.mass_csv)
    wanted = set(args.masses) if args.masses else None

    n_made = 0
    for topology in args.topologies:
        mass_var = MASS_VAR[topology]
        masses = grouped.get(topology, [])
        if wanted is not None:
            masses = [m for m in masses if m in wanted]
        for mass in masses:
            for channel in args.channels:
                region = build_region_name(channel, topology)
                hist_key = build_hist_key(region, mass_var)
                try:
                    edges_n, vals_n, var_n = load_and_combine_signal(
                        input_dirs, hist_key, mass)
                except Exception as e:
                    logger.warning("Load fail %s/%s/%s: %s",
                                   channel, topology, mass, e); continue
                # Rebin native (10 GeV) bins up to the requested display width.
                native = float(edges_n[1] - edges_n[0])
                factor = max(1, round(args.bin_width / native))
                if factor > 1:
                    edges, vals, _ = rebin_histogram(edges_n, vals_n, var_n, factor)
                else:
                    edges, vals = edges_n, vals_n
                out_path = args.output_dir / mass.lower() / f"{channel}.png"
                plot_one(channel, mass, topology, edges, vals, out_path, args.era)
                n_made += 1
                logger.info("  %s/%s/%s → %s", channel, topology, mass, out_path)
    logger.info("signal_samples: wrote %d figures", n_made)


if __name__ == "__main__":
    main()
