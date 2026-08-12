#!/usr/bin/env python3
"""Step 6d -- the answer: the fitted model.

The card is `card_float_m2000.txt` (production: k5 window [1150, 2700], 50 GeV
bins, slope constrained, mu/sigma priors). The curve drawn here is COMBINE's
own fit -- the post-fit parameter values from

    combine -M FitDiagnostics card_float_m2000.txt

read from reference_fitdiag.json key `prod_m2000`. Nothing is re-fitted here.

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

# geometry + data: identical window/binning to the production card
m = CardModel(mass=2000)
ref = json.load(open(HERE.parent / "reference_fitdiag.json"))["prod_m2000"]
p, rate = ref["postfit"], ref["rate_per_fb"]

# combine's post-fit model, as a continuous curve (events per bin width)
xs = np.linspace(m.lo, m.hi, 600)
k = p["b_expo"] / 1000.0
bnorm = (math.exp(k * (m.hi - m.m_c)) - math.exp(k * (m.lo - m.m_c))) / k
bkg = p["bkg_norm"] * np.exp(k * (xs - m.m_c)) * m.width / bnorm
mu, sg = p["mu_sig"], p["sigma_sig"]
gnorm = 0.5 * (math.erf((m.hi - mu) / (math.sqrt(2) * sg))
               - math.erf((m.lo - mu) / (math.sqrt(2) * sg)))
sig = (p["r"] * rate) * (np.exp(-0.5 * ((xs - mu) / sg) ** 2)
                         / (sg * math.sqrt(2 * math.pi))) * m.width / gnorm

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.errorbar(m.centers, m.n, yerr=m.n ** 0.5, fmt="ko", ms=6, zorder=5,
            label="Data")
ax.plot(xs, bkg + sig, color=C.BLUE, lw=2.8, label="S+B fit")

ax.set_yscale("log")
ax.set_xlim(m.lo - 40, m.hi + 40)
ax.set_ylim(0.3, 300)
C.log_yaxis_one_ten(ax)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel(f"Events / {m.width:.0f} GeV")
ax.legend(loc="upper right", fontsize=15, frameon=False)
ax.grid(alpha=0.3)
ax.text(0.035, 0.955, "ee\nResolved SR\n"
        rf"$m_{{W_R}} = {int(m.mass)}$ GeV",
        transform=ax.transAxes, fontsize=15, va="top")

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step6d_fitted_model")
print("combine's post-fit values (card_float_m2000.txt):")
for key in ("r", "b_expo", "bkg_norm", "mu_sig", "sigma_sig"):
    print(f"   {key:10s} = {p[key]:10.4f} +/- {p[key+'_err']:8.4f}")
