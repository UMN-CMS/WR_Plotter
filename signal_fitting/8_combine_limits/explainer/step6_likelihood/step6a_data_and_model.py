#!/usr/bin/env python3
"""Step 6a -- what we have, and what we are about to fit.

The starting point, before any fitting: 31 measured counts, and a model with
five UNKNOWN parameters,

    nu(m) = N_sig * Gauss(m; mu, sigma)  +  N_bkg * expo(m; b)

The rest of step 6 is the procedure that picks those five numbers. To show the
model is not yet determined, three arbitrary trial parameter sets are drawn --
none of them fitted. Card: `card_float_m2000.txt`.

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
rate = json.load(open(HERE.parent / "reference_fitdiag.json"))["prod_m2000"]["rate_per_fb"]
SQ2 = math.sqrt(2.0)
xs = np.linspace(m.lo, m.hi, 500)


def curve(r, b, Bn, mu=1927.7, sg=157.77):
    k = b / 1000.0
    bn = (math.exp(k * (m.hi - m.m_c)) - math.exp(k * (m.lo - m.m_c))) / k
    bkg = Bn * np.exp(k * (xs - m.m_c)) * m.width / bn
    gn = 0.5 * (math.erf((m.hi - mu) / (SQ2 * sg))
                - math.erf((m.lo - mu) / (SQ2 * sg)))
    sig = r * rate * (np.exp(-0.5 * ((xs - mu) / sg) ** 2)
                      / (sg * math.sqrt(2 * math.pi))) * m.width / gn
    return bkg + sig


# three arbitrary guesses -- deliberately NOT the fit
TRIALS = [(0.0, -2.2, 300.0, "#2ca02c", "guess 1"),
          (1.2, -3.4, 420.0, "#f89c20", "guess 2"),
          (0.4, -2.8, 340.0, "#964a8b", "guess 3")]

fig, ax = plt.subplots(figsize=(9, 7.5))
for r, b, Bn, col, lab in TRIALS:
    ax.plot(xs, curve(r, b, Bn), color=col, lw=2.0, ls="--", alpha=0.9,
            label=lab)
ax.errorbar(m.centers, n, yerr=n ** 0.5, fmt="ko", ms=6, zorder=5,
            label="Data (31 counts)")

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
        r"$\nu(m) = r\cdot{\rm rate}\;{\rm Gauss}(m;\mu,\sigma)"
        r" + N_{\rm bkg}\,{\rm expo}(m;b)$" "\n"
        r"five unknown parameters: which values are right?",
        transform=ax.transAxes, fontsize=13, va="bottom")

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step6a_data_and_model")
print(f"data: {m.nbins} bins of {m.width:.0f} GeV, {n.sum():.1f} events total")
print("three arbitrary trial parameter sets drawn (none of them fitted)")
