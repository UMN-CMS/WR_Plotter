#!/usr/bin/env python3
"""Step 6c -- from 31 bins to one number.

Each bin contributes a Poisson probability of what it saw given what the model
predicts,

    P_i = P(n_i | nu_i) = nu_i^n_i e^-nu_i / n_i!

and the likelihood is their product. Working with logs turns the product into
a sum, so each bin contributes an additive amount to -2lnL:

    -2 ln L = sum_i ( -2 ln P_i )  + constraint terms

This plot shows those per-bin contributions as bars: they literally add up to
the number combine minimizes. Card: `card_float_m2000.txt`, at combine's
post-fit values.

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _common as C
from combine_from_scratch import CardModel

hep.style.use("CMS")

m = CardModel(mass=2000)
n = m.n
ref = json.load(open(HERE.parent / "reference_fitdiag.json"))["prod_m2000"]
p, rate = ref["postfit"], ref["rate_per_fb"]
mu, sg, k = p["mu_sig"], p["sigma_sig"], p["b_expo"] / 1000.0
SQ2 = math.sqrt(2.0)

bn = (math.exp(k * (m.hi - m.m_c)) - math.exp(k * (m.lo - m.m_c))) / k
b_i = p["bkg_norm"] * np.exp(k * (m.centers - m.m_c)) * m.width / bn
gn = 0.5 * (math.erf((m.hi - mu) / (SQ2 * sg)) - math.erf((m.lo - mu) / (SQ2 * sg)))
s_i = rate * (np.exp(-0.5 * ((m.centers - mu) / sg) ** 2)
              / (sg * math.sqrt(2 * math.pi))) * m.width / gn
nu = p["r"] * s_i + b_i

lnP = n * np.log(nu) - nu - np.array([math.lgamma(v + 1) for v in n])
P = np.exp(lnP)
contrib = -2.0 * lnP                                     # per-bin piece of -2lnL
pen = (((p["b_expo"] + 2.9435) / 0.12435) ** 2
       + ((mu - 1927.7) / 47.331) ** 2 + ((sg - 157.77) / 47.331) ** 2)

fig, ax = plt.subplots(figsize=(9.5, 7.5))
ax.bar(m.centers, contrib, width=m.width * 0.92, color=C.BLUE, alpha=0.75,
       edgecolor="black", linewidth=0.7,
       label=r"each bin's share of $-2\ln L$")
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel(r"$-2\ln P_i$")
ax.set_xlim(m.lo - 40, m.hi + 40)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=14, frameon=False)

ax.text(0.035, 0.955, "ee\nResolved SR\n"
        rf"$m_{{W_R}} = {int(m.mass)}$ GeV",
        transform=ax.transAxes, fontsize=15, va="top")
ax.text(0.35, 0.955,
        rf"$\sum_i$ (bars) $= {contrib.sum():.2f}$" "\n"
        rf"$+$ constraints $= {pen:.2f}$" "\n"
        rf"$\Rightarrow -2\ln L = {contrib.sum()+pen:.2f}$",
        transform=ax.transAxes, fontsize=15, va="top")

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step6c_likelihood")

print(f"{'bin':>4} {'n_i':>7} {'nu_i':>7} {'P_i':>8} {'-2 ln P_i':>10}")
for i in (0, 10, 15, 30):
    print(f"{i+1:>4} {n[i]:7.1f} {nu[i]:7.2f} {P[i]:8.4f} {contrib[i]:10.3f}")
print(f"\nsum of bars      = {contrib.sum():.2f}")
print(f"+ constraints    = {pen:.2f}")
print(f"= -2 lnL         = {contrib.sum()+pen:.2f}")
