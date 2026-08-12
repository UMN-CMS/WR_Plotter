#!/usr/bin/env python3
"""Step 7f -- the Asimov dataset: the data with the noise taken out.

Combine's AsymptoticLimits throws no toys. Instead it builds ONE artificial
dataset -- the Asimov dataset -- in which every bin is set to exactly its
expected value under the background-only fit:

    n_i^Asimov = nu_i(r = 0, theta-hat)

No Poisson fluctuation, by construction. Fitting this perfect dataset tells
combine how sharply the likelihood curves, which is the width that a toy
ensemble would have measured. Card: `card_float_m2000.txt`.

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
B0, SB = -2.9435, 0.12435
MU0, SMU = 1927.7, 47.331
SG0, SSG = 157.77, 47.331
SQ2 = math.sqrt(2.0)


def model(r, b, Bn, mu, sg):
    k = b / 1000.0
    bn = (math.exp(k * (m.hi - m.m_c)) - math.exp(k * (m.lo - m.m_c))) / k
    bkg = Bn * np.exp(k * (m.centers - m.m_c)) * m.width / bn
    gn = 0.5 * (math.erf((m.hi - mu) / (SQ2 * sg))
                - math.erf((m.lo - mu) / (SQ2 * sg)))
    sig = r * rate * (np.exp(-0.5 * ((m.centers - mu) / sg) ** 2)
                      / (sg * math.sqrt(2 * math.pi))) * m.width / gn
    return np.clip(sig + bkg, 1e-12, None)


def m2lnl(r, b, Bn, mu, sg, data):
    nu = model(r, b, Bn, mu, sg)
    return (2.0 * np.sum(nu - data * np.log(nu))
            + ((b - B0) / SB) ** 2 + ((mu - MU0) / SMU) ** 2
            + ((sg - SG0) / SSG) ** 2)


# background-only fit to the real data -> the Asimov dataset
f = lambda q: m2lnl(0.0, q[0], np.exp(q[1]), q[2], np.exp(q[3]), n)
res = minimize(f, [-2.92, np.log(361.0), 1927.7, np.log(158.0)],
               method="Nelder-Mead",
               options=dict(xatol=1e-8, fatol=1e-10, maxiter=40000))
b_h, B_h, mu_h, sg_h = res.x[0], np.exp(res.x[1]), res.x[2], np.exp(res.x[3])
asimov = model(0.0, b_h, B_h, mu_h, sg_h)

fig, ax = plt.subplots(figsize=(9, 7.5))
edges = np.append(m.centers - m.width / 2, m.centers[-1] + m.width / 2)
ax.stairs(asimov, edges, color=C.BLUE, lw=2.6, fill=True, alpha=0.25)
ax.stairs(asimov, edges, color=C.BLUE, lw=2.6,
          label="Asimov dataset (no fluctuation)")
ax.errorbar(m.centers, n, yerr=n ** 0.5, fmt="ko", ms=6, zorder=5,
            label="the real data")

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
        r"$n_i^{\rm Asimov} = \nu_i(r=0)$  exactly" "\n"
        "every bin sits on the fit, with zero noise",
        transform=ax.transAxes, fontsize=13, va="bottom")

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step7f_asimov")
print(f"{'bin':>4} {'real data':>10} {'Asimov':>9}")
for i in (0, 5, 15, 25, 30):
    print(f"{i+1:>4} {n[i]:10.1f} {asimov[i]:9.2f}")
print(f"sum: data = {n.sum():.1f},  Asimov = {asimov.sum():.1f}")
