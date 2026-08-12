#!/usr/bin/env python3
"""Step 7a -- what one pseudo-dataset actually is.

To build the distribution of q_mu we need many alternative versions of the same
experiment. Each is generated from the model at the tested hypothesis
(r = 0.911 fb, nuisances at their conditional best-fit values), by drawing a
Poisson random number in every bin:

    n_i^toy = Poisson( nu_i(r = 0.911) )

The result is a complete 31-bin fake histogram: what we might have measured if
that value of r were true. Three of them are drawn here beside the real data.

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _common as C
from combine_from_scratch import CardModel

ROOT.gROOT.SetBatch(True)
hep.style.use("CMS")

R_TEST = 0.9111
m = CardModel(mass=2000)
rate = json.load(open(HERE.parent / "reference_fitdiag.json"))["prod_m2000"]["rate_per_fb"]
B0, SB = -2.9435, 0.12435
MU0, SMU = 1927.7, 47.331
SG0, SSG = 157.77, 47.331
SQ2 = math.sqrt(2.0)


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


c = minimize(lambda q: m2([R_TEST, q[0], np.exp(q[1]), q[2], np.exp(q[3])], m.n),
             [-2.92, np.log(361.0), 1927.7, np.log(158.0)],
             method="Powell", options=dict(xtol=1e-6, ftol=1e-8)).x
nu_gen = nus(R_TEST, c[0], np.exp(c[1]), c[2], np.exp(c[3]))

rng = ROOT.TRandom3(4321)
toys = [np.array([rng.Poisson(v) for v in nu_gen], float) for _ in range(3)]

# each dataset, fitted, returns ONE q_mu -- the link to 7b
OPT = dict(xtol=1e-6, ftol=1e-8)


def _glob(d):
    return minimize(lambda q: m2([q[0], q[1], np.exp(q[2]), q[3], np.exp(q[4])], d),
                    [0.09, -2.92, np.log(361.0), 1927.7, np.log(158.0)],
                    method="Powell", options=OPT)


def _cond(r, d):
    return minimize(lambda q: m2([r, q[0], np.exp(q[1]), q[2], np.exp(q[3])], d),
                    [-2.92, np.log(361.0), 1927.7, np.log(158.0)],
                    method="Powell", options=OPT)


def qmu(r, d):
    g = _glob(d)
    return 0.0 if g.x[0] > r else max(_cond(r, d).fun - g.fun, 0.0)


q_data = qmu(R_TEST, m.n)
q_toys = [qmu(R_TEST, t) for t in toys]

edges = np.append(m.centers - m.width / 2, m.centers[-1] + m.width / 2)
fig, axes = plt.subplots(1, 4, figsize=(21, 6.2), sharey=True)

axes[0].stairs(nu_gen, edges, color=C.RED, lw=2.6,
               label="generating model, $r=0.911$")
axes[0].errorbar(m.centers, m.n, yerr=m.n ** 0.5, fmt="ko", ms=5,
                 zorder=5, label="the real data")
axes[0].text(0.05, 0.90, "what we actually measured", transform=axes[0].transAxes,
             fontsize=15, va="top")
axes[0].legend(fontsize=12, frameon=False, loc="lower left")
axes[0].set_ylabel(f"Events / {m.width:.0f} GeV")

for ax, t, i in zip(axes[1:], toys, range(1, 4)):
    ax.errorbar(m.centers, t, yerr=np.sqrt(np.maximum(t, 0)), fmt="o",
                color=C.BLUE, ms=5, zorder=5)
    ax.set_title(f"pseudo-dataset {i}", fontsize=16)

for ax in axes:
    ax.set_yscale("log")
    ax.set_xlim(m.lo - 30, m.hi + 30)
    ax.set_ylim(0.3, 300)
    ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
    ax.grid(alpha=0.3)
C.log_yaxis_one_ten(axes[0])
axes[1].text(0.05, 0.06,
             r"$n_i^{\rm toy} = {\rm Poisson}(\nu_i)$" "\n"
             "one draw per bin",
             transform=axes[1].transAxes, fontsize=13, va="bottom")
hep.cms.label(loc=0, ax=axes[0], data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=14)
fig.tight_layout()
C.savefig(fig, HERE / "step7a_what_is_a_toy")

print(f"model at r={R_TEST}: {nu_gen.sum():.1f} events expected in total")
print(f"real data           : {m.n.sum():.1f} events")
for i, t in enumerate(toys, 1):
    print(f"pseudo-dataset {i}    : {t.sum():.0f} events")
