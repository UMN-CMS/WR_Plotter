#!/usr/bin/env python3
"""Step 3 -- the background MODEL (bkg_pdf), the piece between signal and card.

Step 1 gave data_obs (the summed-MC histogram); step 2 gave sig_pdf. This step
introduces the third card object -- the background pdf that combine FITS to
data_obs:

    bkg_pdf(m) = N * exp[ b * (m - m_c) / 1000 ]

a single smooth falling curve with TWO free parameters:
  * b   (`b_expo`)        the slope -- in the production card constrained to
                          the trusted-spectrum anchor fit (b = -2.94 /TeV,
                          step 5); on the minimal float card it floats free.
  * N   (`bkg_pdf_norm`)  the normalization -- floats freely, pinned by this
                          window's own sidebands (= the window integral B).

This is the bump-hunt background estimate: an analytic curve fit to data_obs,
NOT the MC composition. A signal (step 2) would appear as an excess on top of
it. On a log axis the exponential is a straight line -- the black MC follows it.

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
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

MWR, M_C, SIGMA = 2000, 1927.7, 157.77
LO, HI = 1150.0, 2700.0
B_SLOPE, B_SLOPE_ERR = -2.9435, 0.1244             # constrained slope +- its error (3b)
B_FREE, B_FREE_ERR = -2.8504, 0.1719               # same window, slope left FREE

comps = C.load_bkg_components(rebin=5)             # 50 GeV bins
edges = comps["DYJets"][0]
centers = 0.5 * (edges[:-1] + edges[1:])
total = np.maximum(sum(comps[s][1] for s in C.BKG_SAMPLES), 0.0)
sel = (centers >= LO) & (centers <= HI)
wedges = np.append(edges[:-1][sel], edges[1:][sel][-1])
B = float(total[sel].sum())

# bkg_pdf = expo, normalized so the bin-sum over the window equals B
shape = np.exp(B_SLOPE * (centers[sel] - M_C) / 1000.0)
norm_const = B / shape.sum()
mfine = np.linspace(LO, HI, 400)
curve = norm_const * np.exp(B_SLOPE * (mfine - M_C) / 1000.0)

fig, ax = plt.subplots(figsize=(9, 7.5))
# full spectrum (same object + axes as step1b) ...
ax.stairs(total, edges, color="black", linewidth=1.6,
          label="data_obs = summed MC (step 1)")
# ... with the m=2000 fit window highlighted and its bkg_pdf drawn there
ax.axvspan(LO, HI, color=C.BLUE, alpha=0.08)
ax.plot(mfine, curve, color=C.BLUE, lw=2.8,
        label=r"bkg_pdf $= N\,e^{\,b\,(m-m_c)/1000}$  (in window)")
ax.axvline(M_C, color="black", ls=":", lw=1.1)

ax.set_yscale("log")
ax.set_xlim(800, 5000)
ax.set_ylim(1e-3, 1e4)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("Events / 50 GeV")
C.log_yaxis_one_ten(ax)
ax.text(0.035, 0.95, rf"$m_{{W_R}}={MWR}$" "\n"
        rf"k5 window $[{LO:.0f},{HI:.0f}]$", transform=ax.transAxes,
        fontsize=14, va="top")
ax.legend(loc="upper right", fontsize=13, frameon=False)
ax.grid(alpha=0.3)

info = "\n".join([
    r"$\bf{Slope\ constrained}$ (the card):",
    rf"   $b = {B_SLOPE:.2f} \pm {B_SLOPE_ERR:.2f}$ /TeV",
    rf"   $N = {B:.0f} \pm {np.sqrt(B):.0f}$ events",
    "",
    r"$\bf{Slope\ free}$:",
    rf"   $b = {B_FREE:.2f} \pm {B_FREE_ERR:.2f}$ /TeV",
    rf"   $N = {B:.0f} \pm {np.sqrt(B):.0f}$ events",
    rf"   $\Rightarrow$ {B_FREE_ERR/B_SLOPE_ERR:.1f}$\times$ larger error on $b$",
])
ax.text(0.97, 0.76, info, transform=ax.transAxes, fontsize=12.5, va="top",
        ha="right", bbox=dict(boxstyle="round", fc="white", ec=C.BLUE,
                              alpha=0.92))

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step3a_background_pdf")
print(f"m_WR={MWR}: window [{LO:.0f},{HI:.0f}], B={B:.1f} ev, slope b={B_SLOPE}")
print("next: step4 -- the datacard that assembles data_obs + bkg_pdf + sig_pdf + rate")
