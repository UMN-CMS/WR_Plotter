#!/usr/bin/env python3
"""Pedagogy plot: 100 toy fit-vs-truth offsets from ONE pull-study cell.

Reads results.csv from the pull study, picks one (channel, mass, n_events,
model, config) cell, and plots the 100 per-toy offsets (fit - truth) for a
chosen parameter as a histogram in physical units (GeV).

A reference Gaussian is overlaid whose width is the average fit-reported
error in that cell — i.e. the scatter the fit's own error bars *predict*
the toys should have. Comparing the observed histogram to that prediction
is the calibration check:

  • histogram width ~ reference width   → reported errors are honest
  • histogram narrower than reference   → reported errors are too big
                                           (often: prior is pinning the fit)
  • histogram wider than reference      → reported errors are too small

The center of the histogram (median offset) is the bias.

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Usage:
  python signal_fitting/pull_cell_demo.py
  python signal_fitting/pull_cell_demo.py --mass WR3000_N1500 --n-events 10 --param width
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrplotter.cli_utils import setup_logging
from wrplotter.paths import repo_root

logger = logging.getLogger(__name__)

PARAMS = {
    # plabel: short label (math content only, no $$ wrappers — caller adds them)
    "mu":    ("mu_fit",    "mu_err",    "mu_truth",    r"\mu"),
    "width": ("width_fit", "width_err", "width_truth", r"\mathrm{width}"),
    "delta": ("delta_fit", "delta_err", "delta_truth", r"\Delta"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--csv", default=None,
                   help="Override path to results.csv.")
    p.add_argument("--channel", choices=["ee", "mumu"], default="ee")
    p.add_argument("--mass", default="WR4000_N2000",
                   help="Signal tag, e.g. WR4000_N2000.")
    p.add_argument("--n-events", type=int, default=20)
    p.add_argument("--model", choices=["gauss", "bifur"], default="bifur")
    p.add_argument("--config",
                   choices=["no_priors", "mu_only", "width_only", "both"],
                   default="both")
    p.add_argument("--param", choices=list(PARAMS), default="mu")
    p.add_argument("--output-dir", default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    csv_path = Path(args.csv or
                    repo_root() / "signal_fitting" / "outputs" / args.era /
                    "pull_study" / "results.csv")
    fit_k, err_k, truth_k, plabel = PARAMS[args.param]

    pulls = []
    n_total = 0
    n_unconverged = 0
    n_fixed = 0
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if (r["channel"] != args.channel or r["mass"] != args.mass or
                    int(r["n_events"]) != args.n_events or
                    r["model"] != args.model or r["config"] != args.config):
                continue
            n_total += 1
            if int(r["status"]) != 0 or int(r["covqual"]) < 3:
                n_unconverged += 1
                continue
            try:
                err = float(r[err_k])
                if err <= 0 or not np.isfinite(err):
                    n_fixed += 1
                    continue
                fit = float(r[fit_k]); truth = float(r[truth_k])
                if not (np.isfinite(fit) and np.isfinite(truth)):
                    continue
            except (KeyError, ValueError):
                continue
            pulls.append((fit - truth) / err)

    if not pulls:
        logger.error("No usable toys in cell. n_total=%d, "
                     "unconverged=%d, fixed=%d.", n_total, n_unconverged, n_fixed)
        sys.exit(1)
    pulls = np.asarray(pulls)
    median = float(np.median(pulls))
    p16, p84 = np.percentile(pulls, [16, 84])
    half68 = 0.5 * (p84 - p16)
    logger.info("Cell: %s/%s/%s/n=%d/%s/%s   N_used=%d/%d  median=%+.3f  half-68%%=%.3f",
                args.channel, args.mass, args.model, args.n_events,
                args.config, args.param, len(pulls), n_total, median, half68)

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 7))

    bins = np.linspace(-4, 4, 33)
    counts, edges = np.histogram(pulls, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw = edges[1] - edges[0]
    ax.bar(centers, counts, width=bw, color="#1f77b4", alpha=0.85,
           edgecolor="white", linewidth=0.6, label=f"{len(pulls)} toys")

    # Reference: what an unbiased, well-calibrated fit would produce.
    xs = np.linspace(-4, 4, 400)
    ref = (np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)) * len(pulls) * bw
    ax.plot(xs, ref, color="black", linewidth=1.8, linestyle="--",
            label=r"reference (unbiased, calibrated fit)")

    ymax_data = float(max(counts.max() if counts.size else 1.0, ref.max()))

    ax.axvline(median, color="red", linewidth=2.2,
               label=rf"observed median (bias) $= {median:+.2f}$")

    ax.set_xlabel(
        rf"pull on ${plabel}$  $= ({plabel}_{{\mathrm{{fit}}}} - "
        rf"{plabel}_{{\mathrm{{truth}}}})/\sigma_{{{plabel}, \mathrm{{fit}}}}$",
        fontsize=18,
    )
    ax.set_ylabel("toys / 0.25", fontsize=18)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, ymax_data * 1.55)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)

    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=18)

    ch_lab = {"ee": "ee", "mumu": r"$\mu\mu$"}[args.channel]
    pdf_name = "Gaussian" if args.model == "gauss" else "Bifurcated Gaussian"
    cfg_lab = {"no_priors": "No Priors", "mu_only": r"$\mu$ Constrained",
               "width_only": "Width Constrained",
               "both": "Both Constrained"}[args.config]
    ax.text(
        0.04, 0.96,
        f"{ch_lab}\nResolved SR\n{args.era}\n"
        f"{args.mass.replace('_', ', ')}\n"
        f"{pdf_name} / {cfg_lab}\n"
        rf"$N_{{\rm events}} = {args.n_events}$",
        transform=ax.transAxes, fontsize=12, verticalalignment="top",
    )

    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98),
              fontsize=11, framealpha=0.90)

    out_dir = Path(args.output_dir or
                   repo_root() / "signal_fitting" / "outputs" / args.era /
                   "pull_demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = (out_dir /
           f"cell_offsets_{args.param}_{args.mass}_{args.channel}_"
           f"{args.model}_{args.config}_n{args.n_events}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
