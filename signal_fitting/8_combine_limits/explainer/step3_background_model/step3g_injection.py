#!/usr/bin/env python3
"""Step 3g -- signal injection: does the window fit recover what is put in?

For each injected yield r_inj: N_toys Poisson draws of (summed 2018 background MC
+ r_inj x the m_WR = 2000 signal shape) in the 31 window bins, each refit with the
S+B model (r, N, b free; mu, sigma fixed to the step-2 values). Plotted: the mean
fitted r with the RMS as error bar, against r_inj, with the diagonal.

    python step3g_injection.py [--ntoys 2000] [--inject 0 10 25 50 100 200]

Made 2026-10-03 for the HEP seminar (research-slides/talks/20261007_hep_seminar).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")
M_C, SIGMA, LO, HI = C.window_2000()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ntoys", type=int, default=2000)
    ap.add_argument("--inject", type=float, nargs="+", default=[0, 10, 25, 50, 100, 200])
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()

    comps = C.load_bkg_components(rebin=5)
    edges = comps["DYJets"][0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = np.maximum(sum(comps[s][1] for s in C.BKG_SAMPLES), 0.0)
    sel = (centers >= LO) & (centers <= HI)
    c, lo_e, hi_e, bkg = centers[sel], edges[:-1][sel], edges[1:][sel], total[sel]
    cdf = lambda x: 0.5 * (1 + erf((x - M_C) / (SIGMA * np.sqrt(2))))
    S = cdf(hi_e) - cdf(lo_e); S /= S.sum()
    dm = (c - M_C) / 1000.0

    def fit(data):
        def nll(par):
            r, N, b = par
            e = np.exp(b * dm); e /= e.sum()
            nu = np.clip(r * S + N * e, 1e-9, None)
            return float((nu - data * np.log(nu)).sum())
        res = minimize(nll, x0=[data.sum() - bkg.sum(), bkg.sum(), -2.85], method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 4000})
        return res.x[0]

    rng = np.random.default_rng(args.seed)
    means, rmss = [], []
    for rinj in args.inject:
        rhat = np.array([fit(rng.poisson(bkg + rinj * S).astype(float)) for _ in range(args.ntoys)])
        means.append(rhat.mean()); rmss.append(rhat.std())
        print(f"injected {rinj:6.1f}: mean fitted {rhat.mean():6.1f}  rms {rhat.std():5.1f}  ({args.ntoys} toys)")

    fig, ax = plt.subplots(figsize=(9, 7.5))
    xmax = 1.15 * max(args.inject)
    ax.plot([0, xmax], [0, xmax], color="gray", lw=1.4, ls="--", label="fitted = injected")
    ax.errorbar(args.inject, means, yerr=rmss, fmt="o", color="black", markersize=6, capsize=4,
                label=rf"mean fitted $r$, $\pm$ RMS over {args.ntoys} toys")
    ax.set_xlim(-5, xmax); ax.set_ylim(-20, xmax)
    ax.set_xlabel(r"injected signal $r_{\mathrm{inj}}$ [events]")
    ax.set_ylabel(r"fitted signal $\hat r$ [events]")
    ax.legend(loc="upper left", fontsize=15, frameon=False)
    ax.text(0.97, 0.05, "ee  Resolved SR\n" + rf"$m_{{W_R}}=2000$ GeV window" + f"\nbackground: summed {C.MC_LABEL.split()[-2]} MC",
            transform=ax.transAxes, fontsize=15, va="bottom", ha="right")
    C.cms_label(ax)
    C.savefig(fig, HERE / "step3g_injection")


if __name__ == "__main__":
    main()
