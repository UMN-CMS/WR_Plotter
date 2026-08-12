#!/usr/bin/env python3
"""Step 5 (intro) -- where the b = -2.94 anchor slope comes from.

The single central anchor fit, focused (step5a adds the other members for the
model spread). The card's `b_expo param -2.9435 ...` is the slope of a binned
Poisson-ML exponential fit of the SUMMED MC over the TRUSTED SPECTRUM
[1000, 3500] GeV (pivot 2000): 25 well-populated 100-GeV bins that pin the
falling shape far better than the ~30 bins of one narrow window (and far
better than the 1-3 sparse bins of a high-mass window). Beyond ~4.5 TeV the MC
is single-large-weight jagged and sits above every smooth fit -- excluded from
the trusted range on purpose.

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")

WINDOW = (1150.0, 2700.0)          # the m=2000 card window (step 3), for contrast
comps = C.load_bkg_components(rebin=5)             # 50 GeV bins
edges = comps["DYJets"][0]
centers = 0.5 * (edges[:-1] + edges[1:])
total = np.maximum(sum(comps[s][1] for s in C.BKG_SAMPLES), 0.0)

with open(C.ANCHORS) as fh:
    central = json.load(fh)["central"]
A0, b = central["params"]                           # A per 100 GeV (pivot 2000), slope b
lo, hi = central["range"]
chi2, ndf = central["chi2"], central["ndf"]

PIVOT = 2250.0                                       # fit-range centre; b is pivot-independent
A = A0 * np.exp(b * (PIVOT - 2000.0) / 1000.0)       # same curve, re-expressed at PIVOT

# expo drawn at 50 GeV binning: A is per-100 GeV, so halve it
def expo(m):
    return 0.5 * A * np.exp(b * (m - PIVOT) / 1000.0)

mfit = np.linspace(lo, hi, 300)
mext = np.linspace(hi, 6000, 300)

fig, ax = plt.subplots(figsize=(9.5, 7.5))
ax.stairs(total, edges, color="black", linewidth=1.5,
          label="summed background MC")
ax.axvspan(lo, hi, color="grey", alpha=0.13,
           label=f"trusted spectrum [{lo:.0f}, {hi:.0f}] (fit range)")
ax.plot(mfit, expo(mfit), color=C.RED, lw=2.8,
        label=rf"expo fit to spectrum,  $b={b:.2f}$/TeV")
ax.plot(mext, expo(mext), color=C.RED, lw=2.0, ls="--",
        label="extrapolation (used by other windows)")

# the narrow m=2000 window, for contrast (dotted verticals)
for xw in WINDOW:
    ax.axvline(xw, color=C.BLUE, ls=":", lw=1.3)
ax.text(0.5 * sum(WINDOW), 2.2e3, "one card\nwindow", color=C.BLUE,
        fontsize=10.5, ha="center", va="top")
ax.annotate("", xy=(WINDOW[0], 1.4e3), xytext=(WINDOW[1], 1.4e3),
            arrowprops=dict(arrowstyle="<->", color=C.BLUE, lw=1.2))

ax.text(4700, 6e-2, "jagged tail\n(not trusted)", color="#555555",
        fontsize=10.5, ha="center")

ax.set_yscale("log")
ax.set_xlim(800, 6000)
ax.set_ylim(1e-3, 1e4)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("Events / 50 GeV")
C.log_yaxis_one_ten(ax)
ax.legend(loc="upper right", fontsize=12.5, frameon=False)
ax.grid(alpha=0.3)

info = "\n".join([
    r"slope fit:  $A\,e^{\,b(m-2250)/1000}$",
    r"binned Poisson-ML over [1000, 3500]",
    rf"$\chi^2/$ndf $={chi2/max(ndf,1):.2f}$   $\Rightarrow$   $b={b:.2f}$/TeV",
    r"this $b$ constrains every window's slope",
    r"(robust: [1000,3200] gives $-2.95$)",
])
ax.text(0.035, 0.045, info, transform=ax.transAxes, fontsize=12, va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec=C.RED, alpha=0.92))

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step3b_constrained_slope")
print(f"central anchor: A={A:.4g} (per 100 GeV), b={b:.4f}/TeV, "
      f"range [{lo:.0f},{hi:.0f}], chi2/ndf={chi2/max(ndf,1):.2f}")
