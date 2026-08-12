#!/usr/bin/env python3
"""Step 6e -- what "forcing r" actually looks like.

Three fits to the SAME data. In each, r is held fixed at a chosen value and the
other four parameters are re-fitted as well as they can be. The resulting -2lnL
is printed on each panel:

    r = -0.5   ->  -2lnL = 125.5   (the model needs a DIP at the peak)
    r = +0.088 ->  -2lnL = 123.0   (best fit -- lowest)
    r = +1.5   ->  -2lnL = 131.6   (the model needs a BUMP the data lacks)

Reading those three numbers off against r is exactly the -2lnL curve of 6e.
Card: `card_float_m2000.txt`.

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _common as C
from combine_from_scratch import CardModel

hep.style.use("CMS")

m = CardModel(mass=2000)
n = m.n
rate = json.load(open(HERE.parent / "reference_fitdiag.json"))["prod_m2000"]["rate_per_fb"]
B0, SB = -2.9435, 0.12435
MU0, SMU = 1927.7, 47.331
SG0, SSG = 157.77, 47.331
SQ2 = math.sqrt(2.0)
CONST = 2 * np.sum([math.lgamma(v + 1) for v in n])


def curve(r, b, Bn, mu, sg, x, width):
    k = b / 1000.0
    bn = (math.exp(k * (m.hi - m.m_c)) - math.exp(k * (m.lo - m.m_c))) / k
    bkg = Bn * np.exp(k * (x - m.m_c)) * width / bn
    gn = 0.5 * (math.erf((m.hi - mu) / (SQ2 * sg))
                - math.erf((m.lo - mu) / (SQ2 * sg)))
    sig = r * rate * (np.exp(-0.5 * ((x - mu) / sg) ** 2)
                      / (sg * math.sqrt(2 * math.pi))) * width / gn
    return bkg + sig


def m2(r, b, Bn, mu, sg):
    nu = np.clip(curve(r, b, Bn, mu, sg, m.centers, m.width), 1e-12, None)
    return (2 * np.sum(nu - n * np.log(nu)) + ((b - B0) / SB) ** 2
            + ((mu - MU0) / SMU) ** 2 + ((sg - SG0) / SSG) ** 2)


def fit_at(r):
    f = lambda q: m2(r, q[0], np.exp(q[1]), q[2], np.exp(q[3]))
    res = minimize(f, [-2.92, np.log(361.0), 1927.7, np.log(158.0)],
                   method="Nelder-Mead",
                   options=dict(xatol=1e-9, fatol=1e-11, maxiter=40000))
    return res.x, res.fun + CONST


PICKS = [(-0.5, "#2ca02c"), (0.0885, C.BLUE), (1.5, "#e42536")]
xs = np.linspace(m.lo, m.hi, 500)

fig, axes = plt.subplots(1, 3, figsize=(19, 6.8), sharey=True)
for ax, (r, col) in zip(axes, PICKS):
    q, val = fit_at(r)
    b, Bn, mu, sg = q[0], np.exp(q[1]), q[2], np.exp(q[3])
    ax.errorbar(m.centers, n, yerr=n ** 0.5, fmt="ko", ms=5, zorder=5,
                label="Data")
    ax.plot(xs, curve(r, b, Bn, mu, sg, xs, m.width), color=col, lw=2.8,
            label="fit with $r$ held fixed")
    ax.set_yscale("log")
    ax.set_xlim(m.lo - 30, m.hi + 30)
    ax.set_ylim(0.3, 300)
    ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
    ax.grid(alpha=0.3)
    best = "  (best fit)" if abs(r - 0.0885) < 1e-6 else ""
    ax.set_title(rf"$r = {r:+.3f}$ fb{best}", fontsize=17, color=col)
    ax.text(0.05, 0.06, rf"$-2\ln L = {val:.2f}$", transform=ax.transAxes,
            fontsize=17, va="bottom", color=col)
    ax.text(0.05, 0.955,
            rf"$N_{{\rm sig}} = {r*rate:+.1f}$ events",
            transform=ax.transAxes, fontsize=14, va="top")
    print(f"r = {r:+.4f}:  -2lnL = {val:8.2f}   N_sig = {r*rate:+7.2f} events")

axes[0].set_ylabel(f"Events / {m.width:.0f} GeV")
axes[0].legend(loc="upper right", fontsize=13, frameon=False)
C.log_yaxis_one_ten(axes[0])
hep.cms.label(loc=0, ax=axes[0], data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=14)
fig.tight_layout()
C.savefig(fig, HERE / "step6e_forced_r")
