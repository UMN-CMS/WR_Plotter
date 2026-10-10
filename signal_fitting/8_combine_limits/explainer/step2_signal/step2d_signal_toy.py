#!/usr/bin/env python3
"""Step 2d -- the step-2a signal peak at data-like statistics.

The raw-genWeight m_WR = 2000 signal histogram (50 GeV bins) is scaled to a
chosen number of events and Poisson-fluctuated once; the step-2a Gaussian
(mu, sigma from the Stage-2 straight lines, normalized to the in-window count)
is drawn on top, and a Poisson chi2/ndf inside the k=5 window is quoted.
Default yields: the LO theory expectation for 59.8 fb^-1 (xsec x lumi x
sum(w)/genEventSumw, ~2650 events) and 22 events (about the expected limit).

    python step2d_signal_toy.py [--nsig N ...] [--seed S]

Made 2026-10-03 for the HEP-seminar signal-model slides
(research-slides/talks/20261007_hep_seminar).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")
TAG, MWR, K = "WR2000_N1000", 2000.0, 5.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nsig", type=float, nargs="*", default=None,
                    help="event yields to draw (default: theory expectation and 22)")
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()

    edges, raw = C.load_signal(TAG, rebin=5)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cfg = json.load(open(C.SIGFIT.parents[1] / "data" / "configs" / "RunII" / "2018"
                         / C.ERA / f"{C.ERA}_signal.json"))["WRtoNLtoLLJJ_MWR2000_MN1000"]
    n_theory = cfg["xsec"] * 1000.0 * C.load_meta()["lumi"] / cfg["genEventSumw"] * raw.sum()
    yields = args.nsig if args.nsig else [n_theory, 22.0]

    p = json.load(open(C.SIGFIT / "2_width_parameterization" / "wr" / "window_params.json"))["ee"]["resolved"]
    b_mu, a_mu = p["mu"]
    b_sig, a_sig = p["sigma_median"]
    mu, sigma = a_mu + b_mu * MWR, a_sig + b_sig * MWR
    lo, hi = mu - K * sigma, mu + K * sigma
    inwin = (centers >= lo) & (centers <= hi)

    rng = np.random.default_rng(args.seed)
    for n in yields:
        shape = np.clip(raw, 0.0, None)          # a few bins carry negative weights
        exp_counts = shape / shape.sum() * n
        toy = rng.poisson(exp_counts)
        # Gaussian normalized to the toy's in-window count, like step 2a
        gauss_norm = 0.5 * (math.erf((hi - mu) / (sigma * math.sqrt(2)))
                            - math.erf((lo - mu) / (sigma * math.sqrt(2))))
        bw = edges[1] - edges[0]
        g_bins = (toy[inwin].sum() / gauss_norm) * bw * np.exp(-0.5 * ((centers - mu) / sigma) ** 2) \
            / (sigma * math.sqrt(2 * math.pi))
        # Baker-Cousins Poisson chi2 over the window bins
        o, e = toy[inwin].astype(float), np.clip(g_bins[inwin], 1e-9, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = 2 * (e - o + np.where(o > 0, o * np.log(o / e), 0.0))
        chi2, ndf = float(terms.sum()), int(inwin.sum()) - 1
        print(f"N = {n:7.1f}: in-window toy {toy[inwin].sum():.0f}, chi2/ndf = {chi2:.1f}/{ndf} = {chi2 / ndf:.2f}")

        fig, ax = plt.subplots(figsize=(9, 7.5))
        ax.axvspan(lo, hi, color="tab:red", alpha=0.12)
        nz = toy > 0
        ax.errorbar(centers[nz], toy[nz], yerr=np.sqrt(toy[nz]), fmt="o", color="black", markersize=4,
                    label=f"pseudo-data, {n:.0f} signal events")
        xs = np.linspace(lo, hi, 400)
        ax.plot(xs, (toy[inwin].sum() / gauss_norm) * bw * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
                / (sigma * math.sqrt(2 * math.pi)), color="#c0392b", linewidth=2.2,
                label="Gaussian core (step 2a)")
        ax.set_xlim(0, 2800)
        ax.set_ylim(0, 1.35 * max(toy.max(), 1))
        ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
        ax.set_ylabel("Events / 50 GeV")
        ax.legend(loc="upper right", fontsize=15, frameon=False)
        ax.text(0.03, 0.96,
                "ee  Resolved SR\nRunIISummer20UL18\n"
                rf"$M_{{W_R}}=2000,\ M_N=1000$ GeV" "\n"
                rf"$\mu={mu:.0f}$ GeV,  $\sigma={sigma:.0f}$ GeV" "\n"
                rf"$\chi^2/\mathrm{{ndf}}={chi2:.0f}/{ndf}$ in the window",
                transform=ax.transAxes, fontsize=14, va="top")
        hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", com=13, fontsize=15)
        C.savefig(fig, HERE / f"step2d_signal_toy_n{n:.0f}")
        plt.close(fig)


if __name__ == "__main__":
    main()
