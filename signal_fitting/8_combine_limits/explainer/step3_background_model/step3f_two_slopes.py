#!/usr/bin/env python3
"""Step 3f -- one curve, two backgrounds.

Inside the m_WR = 2000 window the summed background is tt+tW plus the rest
(DY, nonprompt, other). Each component is fit with its own exponential (Poisson
ML, 50 GeV bins), and the single exponential of the total is drawn over them:
tt falls faster than the rest, so the mixture shifts across the window and the
single curve is an average of two slopes.

Made 2026-10-03 for the HEP seminar (research-slides/talks/20261007_hep_seminar).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")
M_C, LO, HI = 1927.7, 1150.0, 2700.0

comps = C.load_bkg_components(rebin=5)
edges = comps["DYJets"][0]
centers = 0.5 * (edges[:-1] + edges[1:])
sel = (centers >= LO) & (centers <= HI)
c = centers[sel]
tt = np.maximum(comps["tt_tW"][1], 0.0)
rest = np.maximum(comps["DYJets"][1] + comps["Nonprompt"][1] + comps["Other"][1], 0.0)
total = tt + rest


def fit_expo(y):
    """Poisson-ML exponential N e^{b (m - m_c)/1000} to the window bins of y."""
    def shape(b):
        e = np.exp(b * (c - M_C) / 1000.0); return e / e.sum()
    def nll(par):
        N, b = par
        nu = np.clip(N * shape(b), 1e-9, None)
        return float((nu - y[sel] * np.log(nu)).sum())
    res = minimize(nll, x0=[y[sel].sum(), -3.0], method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 20000})
    N, b = res.x
    mfine = np.linspace(LO, HI, 400)
    return N, b, mfine, N * np.exp(b * (mfine - M_C) / 1000.0) / np.exp(b * (c - M_C) / 1000.0).sum()


fits = {k: fit_expo(y) for k, y in [("tt", tt), ("rest", rest), ("total", total)]}
for k, (N, b, _, _) in fits.items():
    print(f"{k:6s}: N = {N:6.1f} events in window, slope b = {b:.2f}/TeV")
frac = tt[sel] / total[sel]
print(f"tt share of the background: {frac[0]:.0%} at {c[0]:.0f} GeV -> {frac[-1]:.0%} at {c[-1]:.0f} GeV")

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.axvspan(LO, HI, color="tab:red", alpha=0.10, zorder=0)
ax.stairs(total, edges, color="black", linewidth=1.4, label="all backgrounds (summed 2018 MC)")
ax.stairs(tt, edges, color="#f89c20", linewidth=1.4, label=r"$t\bar{t}$+tW")
ax.stairs(rest, edges, color="#5790fc", linewidth=1.4, label="rest: DY + nonprompt + other")
for k, col, lab in [("total", "black", "one exponential for the total"),
                    ("tt", "#c05a00", r"$t\bar{t}$+tW alone"),
                    ("rest", "#1f4fb5", "rest alone")]:
    N, b, m, y = fits[k]
    ax.plot(m, y, color=col, linewidth=2.6, linestyle="-" if k == "total" else "--", zorder=6,
            label=f"{lab}: slope ${b:.2f}$/TeV")
ax.set_yscale("log"); ax.set_xlim(800, 3200); ax.set_ylim(1e-1, 3e2)
C.log_yaxis_one_ten(ax)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)"); ax.set_ylabel("Events / 50 GeV")
ax.legend(loc="upper right", fontsize=13, frameon=False)
ax.text(0.03, 0.05, "ee\nResolved SR\nRunII 2018 MC\n" + rf"window $[{LO:.0f}, {HI:.0f}]$ GeV",
        transform=ax.transAxes, fontsize=15, va="bottom")
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step3f_two_slopes")
