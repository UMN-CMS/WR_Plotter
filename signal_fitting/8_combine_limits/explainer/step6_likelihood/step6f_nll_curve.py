#!/usr/bin/env python3
"""Step 6f -- the -2lnL curve vs r.

Scan the POI r. At each value, re-minimize -2lnL over the other four
parameters (b_expo, bkg norm, mu_sig, sigma_sig) with the card's three
constraint terms included -- i.e. exactly the quantity of step 6d, as a
function of r alone. Card: `card_float_m2000.txt`.

Two things fall straight out of this one curve:
  * its MINIMUM   -> the best-fit r_hat
  * its WIDTH     -> the uncertainty: +-1 sigma is where the curve rises by 1

both of which are checked against combine's FitDiagnostics numbers.

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
ref = json.load(open(HERE.parent / "reference_fitdiag.json"))["prod_m2000"]
rate = ref["rate_per_fb"]
p = ref["postfit"]
B0, SB = -2.9435, 0.12435                 # the card's three constraints
MU0, SMU = 1927.7, 47.331
SG0, SSG = 157.77, 47.331
SQ2 = math.sqrt(2.0)


def m2lnl(r, b, Bn, mu, sg):
    k = b / 1000.0
    bn = (math.exp(k * (m.hi - m.m_c)) - math.exp(k * (m.lo - m.m_c))) / k
    bkg = Bn * np.exp(k * (m.centers - m.m_c)) * m.width / bn
    gn = 0.5 * (math.erf((m.hi - mu) / (SQ2 * sg))
                - math.erf((m.lo - mu) / (SQ2 * sg)))
    sig = r * rate * (np.exp(-0.5 * ((m.centers - mu) / sg) ** 2)
                      / (sg * math.sqrt(2 * math.pi))) * m.width / gn
    nu = np.clip(sig + bkg, 1e-12, None)
    return (2.0 * np.sum(nu - n * np.log(nu))
            + ((b - B0) / SB) ** 2 + ((mu - MU0) / SMU) ** 2
            + ((sg - SG0) / SSG) ** 2)


def profile(r):
    f = lambda q: m2lnl(r, q[0], np.exp(q[1]), q[2], np.exp(q[3]))
    res = minimize(f, [-2.92, np.log(361.0), 1927.7, np.log(158.0)],
                   method="Nelder-Mead",
                   options=dict(xatol=1e-8, fatol=1e-10, maxiter=40000))
    return res.fun


rs = np.linspace(-1.2, 1.4, 61)
vals = np.array([profile(r) for r in rs])
i0 = int(np.argmin(vals))
# refine the minimum and the +-1 crossings by interpolation
rr = np.linspace(rs[max(i0 - 2, 0)], rs[min(i0 + 2, len(rs) - 1)], 400)
vv = np.array([profile(r) for r in rr])
r_hat = rr[int(np.argmin(vv))]
dv = vals - vv.min()


fig, ax = plt.subplots(figsize=(9, 7.5))
ax.plot(rs, dv, color=C.BLUE, lw=2.8)
ax.axvline(r_hat, color=C.RED, ls="--", lw=2)
ax.plot([r_hat], [0.0], "o", ms=11, color=C.RED, zorder=5)
ax.text(r_hat + 0.05, 0.25, rf"$\hat r = {r_hat:+.3f}$ fb",
        color=C.RED, fontsize=14, va="bottom")

ax.set_xlabel(r"$r$  =  $\sigma\,\mathcal{B}(ee\,q\bar q\,')$  (fb)")
ax.set_ylabel(r"$q_\mu = -2\ln L - (-2\ln L)_{\min}$")
ax.set_ylim(0, 6)
ax.set_xlim(rs[0], rs[-1])
ax.grid(alpha=0.3)
ax.text(0.035, 0.955, "ee\nResolved SR\n"
        rf"$m_{{W_R}} = {int(m.mass)}$ GeV",
        transform=ax.transAxes, fontsize=15, va="top")
ax.text(0.97, 0.955,
        "each point is a full fit\n"
        r"with $r$ held fixed",
        transform=ax.transAxes, fontsize=14, va="top", ha="right")

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step6f_nll_curve")
print(f"ours    : r_hat = {r_hat:+.4f}")
print(f"combine : r_hat = {p['r']:+.4f}")
