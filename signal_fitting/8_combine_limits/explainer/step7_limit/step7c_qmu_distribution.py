#!/usr/bin/env python3
"""Step 7c -- what "asymptotic" means: the distribution of q_mu.

A single q_mu value means nothing on its own. To judge it we need to know what
values of q_mu would be TYPICAL if the tested r were the true one. Because of
Poisson noise, repeating the experiment gives a different dataset, a different
curve, and a different q_mu.

Here we generate that distribution the honest way: many pseudo-datasets drawn
under the hypothesis r = 0.911 fb (the value that turns out to be the limit),
computing q_mu for each. Overlaid is the ASYMPTOTIC prediction, which needs no
toys at all:

    q_mu ~ 1/2 delta(0)  +  1/2 chi2(1)        =>   p_mu = 1 - Phi(sqrt(q_mu))

"Asymptotic regime" means the counts are large enough for that limiting form to
hold, which is the case here (363 events). It fails when bins hold ~1 event,
which is why the sparse high-mass windows need HybridNew toys instead.

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
import ROOT
from scipy.optimize import minimize
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _common as C
from combine_from_scratch import CardModel

ROOT.gROOT.SetBatch(True)
hep.style.use("CMS")

R_TEST = 0.9111                      # the 95% limit value
NTOYS = 3000
m = CardModel(mass=2000)
rate = json.load(open(HERE.parent / "reference_fitdiag.json"))["prod_m2000"]["rate_per_fb"]
B0, SB = -2.9435, 0.12435
MU0, SMU = 1927.7, 47.331
SG0, SSG = 157.77, 47.331
SQ2 = math.sqrt(2.0)
Phi = NormalDist().cdf


def nus(r, b, Bn, mu, sg):
    k = b / 1000.0
    bn = (math.exp(k * (m.hi - m.m_c)) - math.exp(k * (m.lo - m.m_c))) / k
    bkg = Bn * np.exp(k * (m.centers - m.m_c)) * m.width / bn
    gn = 0.5 * (math.erf((m.hi - mu) / (SQ2 * sg))
                - math.erf((m.lo - mu) / (SQ2 * sg)))
    sig = r * rate * (np.exp(-0.5 * ((m.centers - mu) / sg) ** 2)
                      / (sg * math.sqrt(2 * math.pi))) * m.width / gn
    return np.clip(sig + bkg, 1e-12, None)


def m2(p, d):
    v = nus(*p)
    return (2 * np.sum(v - d * np.log(v)) + ((p[1] - B0) / SB) ** 2
            + ((p[3] - MU0) / SMU) ** 2 + ((p[4] - SG0) / SSG) ** 2)


OPT = dict(xtol=1e-6, ftol=1e-8, maxiter=20000)


def cond(r, d):
    return minimize(lambda q: m2([r, q[0], np.exp(q[1]), q[2], np.exp(q[3])], d),
                    [-2.92, np.log(361.0), 1927.7, np.log(158.0)],
                    method="Powell", options=OPT)


def qmu(r, d):
    g = minimize(lambda q: m2([q[0], q[1], np.exp(q[2]), q[3], np.exp(q[4])], d),
                 [0.09, -2.92, np.log(361.0), 1927.7, np.log(158.0)],
                 method="Powell", options=OPT)
    if g.x[0] > r:
        return 0.0
    return max(cond(r, d).fun - g.fun, 0.0)


q_obs = qmu(R_TEST, m.n)

# toys thrown under the tested hypothesis, nuisances at their conditional MLE
c = cond(R_TEST, m.n).x
nu_gen = nus(R_TEST, c[0], np.exp(c[1]), c[2], np.exp(c[3]))
rng = ROOT.TRandom3(20260729)
qs = np.array([qmu(R_TEST, np.array([rng.Poisson(v) for v in nu_gen], float))
               for _ in range(NTOYS)])

frac0 = float(np.mean(qs < 1e-9))
p_toys = float(np.mean(qs >= q_obs))
p_asym = 1 - Phi(math.sqrt(q_obs))

fig, ax = plt.subplots(figsize=(9.5, 7.5))
bins = np.linspace(0, 12, 49)
w = bins[1] - bins[0]
ax.hist(qs, bins=bins, color=C.BLUE, alpha=0.7, edgecolor="black",
        linewidth=0.6, label=f"{NTOYS} toys thrown at $r={R_TEST:.3f}$")
qq = np.linspace(0.02, 12, 400)
ax.plot(qq, NTOYS * 0.5 * np.exp(-qq / 2) / np.sqrt(2 * np.pi * qq) * w,
        color=C.RED, lw=2.8,
        label=r"asymptotic: $\frac{1}{2}\chi^2_1$ (no toys)")
ax.axvline(q_obs, color="black", lw=2.4, ls="--",
           label=fr"observed $q_\mu = {q_obs:.2f}$")

ax.set_yscale("log")
ax.set_ylim(0.5, NTOYS)
ax.set_xlim(0, 12)
ax.set_xlabel(r"$q_\mu$")
ax.set_ylabel("Toys")
ax.legend(loc="upper right", fontsize=13, frameon=False)
ax.grid(alpha=0.3)
ax.text(0.035, 0.955, "ee\nResolved SR\n"
        rf"$m_{{W_R}} = {int(m.mass)}$ GeV",
        transform=ax.transAxes, fontsize=15, va="top")
ax.text(0.035, 0.06,
        rf"fraction with $q_\mu = 0$:  {frac0:.2f}" "\n"
        rf"$p$ from toys:      {p_toys:.4f}" "\n"
        rf"$p$ from formula:  {p_asym:.4f}",
        transform=ax.transAxes, fontsize=13, va="bottom")

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step7c_qmu_distribution")
print(f"observed q_mu at r={R_TEST}: {q_obs:.4f}")
print(f"fraction of toys with q_mu = 0 : {frac0:.4f}   (asymptotic says 0.50)")
print(f"p-value from {NTOYS} toys        : {p_toys:.4f}")
print(f"p-value from 1-Phi(sqrt q)     : {p_asym:.4f}")
