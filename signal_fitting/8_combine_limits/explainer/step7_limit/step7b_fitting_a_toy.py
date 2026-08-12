#!/usr/bin/env python3
"""Step 7b -- the two fits behind one q_mu.

The red generating curve of 7a is not what gets fitted. Once a pseudo-dataset
exists it is treated exactly like real data: fitted twice, and the DIFFERENCE
of those two fits is its q_mu.

  global fit      all five parameters free      ->  that dataset's own r_hat
  conditional fit r fixed at 0.911, rest free   ->  the height at 0.911
  q_mu = (conditional) - (global)

Left: the real data, whose best fit (r = 0.088) is far from 0.911, so forcing
0.911 hurts a lot and q_mu is large. Right: pseudo-dataset 1, which happens to
prefer r = 0.74, close to 0.911, so forcing it barely hurts and q_mu is small.

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
OPT = dict(xtol=1e-7, ftol=1e-9)


def curve(r, b, Bn, mu, sg, x, width):
    k = b / 1000.0
    bn = (math.exp(k * (m.hi - m.m_c)) - math.exp(k * (m.lo - m.m_c))) / k
    bkg = Bn * np.exp(k * (x - m.m_c)) * width / bn
    gn = 0.5 * (math.erf((m.hi - mu) / (SQ2 * sg))
                - math.erf((m.lo - mu) / (SQ2 * sg)))
    sig = r * rate * (np.exp(-0.5 * ((x - mu) / sg) ** 2)
                      / (sg * math.sqrt(2 * math.pi))) * width / gn
    return bkg + sig


def m2(p, d):
    v = np.clip(curve(p[0], p[1], p[2], p[3], p[4], m.centers, m.width), 1e-12, None)
    return (2 * np.sum(v - d * np.log(v)) + ((p[1] - B0) / SB) ** 2
            + ((p[3] - MU0) / SMU) ** 2 + ((p[4] - SG0) / SSG) ** 2)


def glob(d):
    r = minimize(lambda q: m2([q[0], q[1], np.exp(q[2]), q[3], np.exp(q[4])], d),
                 [0.4, -2.92, np.log(361.0), 1927.7, np.log(158.0)],
                 method="Powell", options=OPT)
    return [r.x[0], r.x[1], np.exp(r.x[2]), r.x[3], np.exp(r.x[4])], r.fun


def cond(rv, d):
    r = minimize(lambda q: m2([rv, q[0], np.exp(q[1]), q[2], np.exp(q[3])], d),
                 [-2.92, np.log(361.0), 1927.7, np.log(158.0)],
                 method="Powell", options=OPT)
    return [rv, r.x[0], np.exp(r.x[1]), r.x[2], np.exp(r.x[3])], r.fun


# regenerate pseudo-dataset 1 exactly as in 7a
cg, _ = cond(R_TEST, m.n)
nu_gen = curve(*cg, m.centers, m.width)
rng = ROOT.TRandom3(4321)
toy1 = np.array([rng.Poisson(v) for v in nu_gen], float)

xs = np.linspace(m.lo, m.hi, 600)
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

for ax, d, name, col in ((axes[0], m.n, "the real data", "black"),
                         (axes[1], toy1, "pseudo-dataset 1", C.BLUE)):
    pg, vg = glob(d)
    pc, vc = cond(R_TEST, d)
    q = max(vc - vg, 0.0) if pg[0] <= R_TEST else 0.0
    ax.errorbar(m.centers, d, yerr=np.sqrt(np.maximum(d, 0)), fmt="o",
                color=col, ms=5.5, zorder=5, label=name)
    ax.plot(xs, curve(*pg, xs, m.width), color="#2ca02c", lw=2.6,
            label=rf"global fit: $\hat r = {pg[0]:.3f}$")
    ax.plot(xs, curve(*pc, xs, m.width), color=C.RED, lw=2.6, ls="--",
            label=rf"forced $r = {R_TEST:.3f}$")
    ax.set_yscale("log")
    ax.set_xlim(m.lo - 30, m.hi + 30)
    ax.set_ylim(0.3, 300)
    ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=13, frameon=False)
    ax.text(0.04, 0.06, rf"$q_\mu = {q:.2f}$", transform=ax.transAxes,
            fontsize=20, va="bottom")
    print(f"{name:18}: r_hat = {pg[0]:6.3f},  q_mu = {q:.3f}")

axes[0].set_ylabel(f"Events / {m.width:.0f} GeV")
C.log_yaxis_one_ten(axes[0])
hep.cms.label(loc=0, ax=axes[0], data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=14)
fig.tight_layout()
C.savefig(fig, HERE / "step7b_fitting_a_toy")
