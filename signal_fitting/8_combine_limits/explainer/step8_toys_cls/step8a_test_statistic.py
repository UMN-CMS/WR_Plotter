#!/usr/bin/env python3
"""Step 8a -- the test statistic q~_mu: one number per (hypothesis, dataset).

To exclude a signal size mu, combine needs to compare "how signal-like did the
data come out" between datasets.  The comparison currency is the LHC test
statistic (CCGV eq. 16), built purely from Delta(-2lnL):

              /  0                                    if  mu-hat > mu
    q~_mu  = <   -2ln[ L(mu) / L(mu-hat) ]            if  0 <= mu-hat <= mu
              \  -2ln[ L(mu) / L(0) ]                 if  mu-hat < 0

(everything profiled).  Reading the three branches:
  * mu-hat > mu: the data saw MORE signal than the hypothesis -- that can
    never be evidence AGAINST mu for an upper limit, so the case is defined
    away to zero (this one-sidedness is why limits use q~ and not the plain
    likelihood ratio);
  * middle: the ordinary likelihood-ratio distance between the hypothesis and
    the best fit;
  * mu-hat < 0: unphysical downward fluctuations are measured from mu-hat = 0
    instead, so they cannot fake arbitrarily strong exclusion.

Large q~_mu  =  the data is far (on the low side) from what signal mu
predicts  =  mu is in trouble.

Figure: q~_mu(mu) for our observed dataset (the MC window), with the three
branch regions annotated; plus q~ for two artificial datasets (data shifted
up / down by 1 sigma of fitted signal) to show how the curve responds.

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python step8a_test_statistic.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from combine_from_scratch import CardModel

m = CardModel(mass=2000)
print(__doc__.split("\n\n")[0])

(r_hat, _, _), errs, _ = m.fit_global()
sig = errs[0]
mus = np.linspace(0.5, 50, 60)

datasets = [
    (None, "#e42536", fr"observed (MC window): $\hat\mu={r_hat:.1f}$"),
    (np.clip(m.n + sig * m.S, 0, None), "#2ca02c",
     r"data shifted UP by $1\sigma$ of signal"),
    (np.clip(m.n - 1.5 * sig * m.S, 0, None), "#5790fc",
     r"data shifted DOWN by $1.5\sigma$ (\hat\mu < 0 branch)"),
]

fig, ax = plt.subplots(figsize=(10, 6.5))
for data, col, lab in datasets:
    q = [m.qmu_tilde(mu, data)[0] for mu in mus]
    muh = m.fit_global(data)[0][0]
    ax.plot(mus, q, color=col, lw=2.2, label=lab + fr"  ($\hat\mu={muh:.1f}$)")
    ax.axvline(max(muh, 0.0), color=col, lw=0.9, ls=":")
q_obs_30 = m.qmu_tilde(30.0)[0]
print(f"\n  example: q~_mu at mu=30 for the observed dataset = {q_obs_30:.2f}")
print(f"  global best fit: mu-hat = {r_hat:.2f} +- {sig:.2f}")

ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel(r"tested signal yield $\mu$ [events]")
ax.set_ylabel(r"$\tilde{q}_\mu$")
ax.set_ylim(-0.3, 12)
ax.set_title(r"$\tilde q_\mu$: zero below $\hat\mu$ (one-sided), then grows "
             "as the hypothesis outruns the data")
ax.text(0.02, 0.97,
        "dotted line = each dataset's $\\hat\\mu$\n"
        "left of it: $\\tilde q_\\mu = 0$ (more signal seen than tested)\n"
        "right of it: hypothesis increasingly disfavoured",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", fc="#f0f0f0", ec="grey"))
ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(0.02, 0.72))

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"step8a_test_statistic.{ext}", dpi=150)
print(f"\nwrote {HERE}/step8a_test_statistic.png")
print("NEXT (8b): q~_mu is only useful once we know its DISTRIBUTION -- "
      "throw toys under s+b and b-only and compare tails -> CLs")
