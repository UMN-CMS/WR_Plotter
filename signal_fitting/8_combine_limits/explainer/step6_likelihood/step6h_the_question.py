#!/usr/bin/env python3
"""Step 6h -- the question step 7 has to answer.

The same q_mu curve as 6f, with one candidate value of r picked out:
r = 0.911 fb, where q_mu = 3.54. Is 3.54 large enough to call that value of r
excluded? Nothing so far can answer that: a single q_mu means nothing until we
know what values of q_mu would be TYPICAL if that r were true.

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

R_MARK = 0.9111
m = CardModel(mass=2000)
n = m.n
rate = json.load(open(HERE.parent / "reference_fitdiag.json"))["prod_m2000"]["rate_per_fb"]
B0, SB = -2.9435, 0.12435
MU0, SMU = 1927.7, 47.331
SG0, SSG = 157.77, 47.331
SQ2 = math.sqrt(2.0)


def m2(r, b, Bn, mu, sg):
    k = b / 1000.0
    bn = (math.exp(k * (m.hi - m.m_c)) - math.exp(k * (m.lo - m.m_c))) / k
    bkg = Bn * np.exp(k * (m.centers - m.m_c)) * m.width / bn
    gn = 0.5 * (math.erf((m.hi - mu) / (SQ2 * sg))
                - math.erf((m.lo - mu) / (SQ2 * sg)))
    sig = r * rate * (np.exp(-0.5 * ((m.centers - mu) / sg) ** 2)
                      / (sg * math.sqrt(2 * math.pi))) * m.width / gn
    v = np.clip(sig + bkg, 1e-12, None)
    return (2 * np.sum(v - n * np.log(v)) + ((b - B0) / SB) ** 2
            + ((mu - MU0) / SMU) ** 2 + ((sg - SG0) / SSG) ** 2)


def profile(r):
    f = lambda q: m2(r, q[0], np.exp(q[1]), q[2], np.exp(q[3]))
    return minimize(f, [-2.92, np.log(361.0), 1927.7, np.log(158.0)],
                    method="Powell",
                    options=dict(xtol=1e-7, ftol=1e-9, maxiter=40000)).fun


R_HAT = 0.0883                                   # combine's best fit
rs = np.linspace(-1.2, 1.4, 81)
qmin = profile(R_HAT)
dv = np.array([profile(r) for r in rs]) - qmin
q_mark = profile(R_MARK) - qmin

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.plot(rs, dv, color=C.BLUE, lw=2.8)
ax.plot([R_HAT], [0.0], "o", ms=10, color=C.RED, zorder=5)
ax.text(R_HAT - 0.06, 0.22, rf"$\hat r = {R_HAT:+.3f}$ fb", color=C.RED,
        fontsize=13, va="bottom", ha="right")

# the candidate value under test
ax.plot([R_MARK, R_MARK], [0, q_mark], color="black", ls="--", lw=2)
ax.plot([rs[0], R_MARK], [q_mark, q_mark], color="black", ls="--", lw=2)
ax.plot([R_MARK], [q_mark], "o", ms=12, color="black", zorder=6)
ax.text(rs[0] + 0.05, q_mark + 0.15, rf"$q_\mu = {q_mark:.2f}$",
        fontsize=15, ha="left", va="bottom")
ax.text(R_MARK + 0.03, 0.15, rf"$r = {R_MARK:.3f}$", fontsize=14,
        ha="left", va="bottom")

ax.text(0.50, 0.72,
        "is 3.54 big enough\n"
        r"to exclude $r=0.911$?",
        transform=ax.transAxes, fontsize=17, ha="center", va="center",
        bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.95))

ax.set_xlabel(r"$r$  =  $\sigma\,\mathcal{B}(ee\,q\bar q\,')$  (fb)")
ax.set_ylabel(r"$q_\mu = -2\ln L - (-2\ln L)_{\min}$")
ax.set_ylim(0, 6)
ax.set_xlim(rs[0], rs[-1])
ax.grid(alpha=0.3)
ax.text(0.035, 0.955, "ee\nResolved SR\n"
        rf"$m_{{W_R}} = {int(m.mass)}$ GeV",
        transform=ax.transAxes, fontsize=15, va="top")

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step6h_the_question")
print(f"r_hat  = {R_HAT:+.4f} fb")
print(f"at r = {R_MARK}:  q_mu = {q_mark:.4f}")
