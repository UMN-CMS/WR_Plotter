#!/usr/bin/env python3
"""Step 6g -- what "testing r = 0.911" actually looks like.

The fit answers "what is r?" (0.088 fb). The limit asks a different question:
"how large could r be?" To answer it we take candidate values one at a time and
ask how well each describes the data.

Here is one candidate, r = 0.911 fb, drawn against the best fit. The signal it
requires is a visible bump at the peak that the data does not show. How badly
it misses is measured by q_mu = 3.54. Card: `card_float_m2000.txt`.

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

R_HAT, R_TEST = 0.0883, 0.9111
m = CardModel(mass=2000)
n = m.n
rate = json.load(open(HERE.parent / "reference_fitdiag.json"))["prod_m2000"]["rate_per_fb"]
B0, SB = -2.9435, 0.12435
MU0, SMU = 1927.7, 47.331
SG0, SSG = 157.77, 47.331
SQ2 = math.sqrt(2.0)


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
    v = np.clip(curve(r, b, Bn, mu, sg, m.centers, m.width), 1e-12, None)
    return (2 * np.sum(v - n * np.log(v)) + ((b - B0) / SB) ** 2
            + ((mu - MU0) / SMU) ** 2 + ((sg - SG0) / SSG) ** 2)


def fit_at(r):
    f = lambda q: m2(r, q[0], np.exp(q[1]), q[2], np.exp(q[3]))
    res = minimize(f, [-2.92, np.log(361.0), 1927.7, np.log(158.0)],
                   method="Powell", options=dict(xtol=1e-7, ftol=1e-9))
    return res.x, res.fun


p_hat, v_hat = fit_at(R_HAT)
p_tst, v_tst = fit_at(R_TEST)
q_mu = v_tst - v_hat
xs = np.linspace(m.lo, m.hi, 600)

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.errorbar(m.centers, n, yerr=n ** 0.5, fmt="ko", ms=6, zorder=5,
            label="Data")
ax.plot(xs, curve(R_HAT, p_hat[0], np.exp(p_hat[1]), p_hat[2], np.exp(p_hat[3]),
                  xs, m.width), color=C.BLUE, lw=2.6,
        label=rf"best fit:  $r = {R_HAT:.3f}$ fb")
ax.plot(xs, curve(R_TEST, p_tst[0], np.exp(p_tst[1]), p_tst[2], np.exp(p_tst[3]),
                  xs, m.width), color=C.RED, lw=2.6,
        label=rf"tested:   $r = {R_TEST:.3f}$ fb")

ax.set_yscale("log")
ax.set_xlim(m.lo - 40, m.hi + 40)
ax.set_ylim(0.3, 300)
C.log_yaxis_one_ten(ax)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel(f"Events / {m.width:.0f} GeV")
ax.legend(loc="upper right", fontsize=14, frameon=False)
ax.grid(alpha=0.3)
ax.text(0.035, 0.955, "ee\nResolved SR\n"
        rf"$m_{{W_R}} = {int(m.mass)}$ GeV",
        transform=ax.transAxes, fontsize=15, va="top")
ax.text(0.035, 0.06,
        rf"$r = {R_TEST:.3f}$ needs "
        rf"${R_TEST*rate:.0f}$ signal events" "\n"
        rf"the data does not show them: $q_\mu = {q_mu:.2f}$",
        transform=ax.transAxes, fontsize=13, va="bottom")

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step6g_testing_a_value")
print(f"best fit r = {R_HAT}:  N_sig = {R_HAT*rate:5.2f} events, -2lnL = {v_hat:.3f}")
print(f"tested   r = {R_TEST}:  N_sig = {R_TEST*rate:5.2f} events, -2lnL = {v_tst:.3f}")
print(f"q_mu = {q_mu:.4f}")
