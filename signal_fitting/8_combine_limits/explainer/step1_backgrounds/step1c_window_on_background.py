#!/usr/bin/env python3
"""Step 1c -- the m_WR = 2000 card's window drawn on the summed background.

Ties step 1 (the summed MC) to step 2 (the window): the black curve is the
same data_obs of step 1b, and the shaded band is the slice the m_WR = 2000
card actually fits. The centre m_c and width sigma are NOT per-mass Gaussian
fits -- they are the Stage-2 LINEAR parameterizations evaluated at 2000 GeV:

    m_c   = a_mu + b_mu * m_WR             (2_width_parameterization/wr/fit/ee_resolved_mu)
    sigma = a_sig + b_sig * m_WR           (2_width_parameterization/wr/fit/ee_resolved_sigma)

window = [m_c - k*sigma, m_c + k*sigma] with k=5 (the optimized float-card
geometry), then snapped to the 50 GeV bin grid the way TH1::Fit selects bins
-- exactly the bins in the k5_bw50 workspace.
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

MWR, K = 2000.0, 5.0
# repo_root/signal_fitting/8_combine_limits/explainer/step1_backgrounds -> up 3
PARAM = (HERE.parents[2] / "2_width_parameterization" / "wr"
         / "window_params.json")

# --- the parameterization, evaluated at m_WR = 2000 (what the card freezes) ---
p = json.load(open(PARAM))["ee"]["resolved"]
b_mu, a_mu = p["mu"]
b_sig, a_sig = p["sigma_median"]
m_c = a_mu + b_mu * MWR
sigma = a_sig + b_sig * MWR
lo, hi = m_c - K * sigma, m_c + K * sigma

# --- summed background MC (same object as step 1b) ---
comps = C.load_bkg_components(rebin=5)
edges = comps["DYJets"][0]
total = np.maximum(sum(comps[s][1] for s in C.BKG_SAMPLES), 0.0)

# --- snapped window: bins whose CENTRE lies in [lo, hi] (TH1::Fit selection) ---
centers = 0.5 * (edges[:-1] + edges[1:])
insel = (centers >= lo) & (centers <= hi)
slo = edges[:-1][insel][0]
shi = edges[1:][insel][-1]
nbins = int(insel.sum())

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.stairs(total, edges, color="black", linewidth=1.8,
          label="summed background MC (data_obs)")

# raw parameterized window m_c +- 3 sigma
ax.axvspan(lo, hi, color="tab:blue", alpha=0.12,
           label=fr"window $m_c\pm{K:g}\sigma$ = [{lo:.0f}, {hi:.0f}]")
# snapped fit range (the 9 bins actually in the workspace)
ax.axvline(slo, color="tab:blue", ls=":", lw=1.4)
ax.axvline(shi, color="tab:blue", ls=":", lw=1.4,
           label=f"snapped fit range [{slo:.0f}, {shi:.0f}] ({nbins} bins)")
# the centre
ax.axvline(m_c, color="tab:red", ls="--", lw=1.8,
           label=fr"$m_c={m_c:.1f}$ GeV")

ax.set_yscale("log")
ax.set_xlim(800, 6000)
ax.set_ylim(1e-4, 2e3)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("events / 50 GeV")
ax.legend(loc="upper right", fontsize=13, frameon=False)
ax.grid(alpha=0.3)

txt = (r"from the linear parameterizations ($m_{W_R}=2000$):" "\n"
       fr"$m_c=a_\mu+b_\mu m_{{W_R}}={m_c:.1f}$ GeV" "\n"
       fr"$\sigma=a_\sigma+b_\sigma m_{{W_R}}={sigma:.1f}$ GeV" "\n"
       fr"window $=m_c\pm{K:g}\sigma=[{lo:.0f},\,{hi:.0f}]$")
ax.text(0.03, 0.03, txt, transform=ax.transAxes, fontsize=13, va="bottom",
        ha="left", bbox=dict(boxstyle="round", fc="white", ec="tab:blue",
                             alpha=0.9))

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step1c_window_on_background")

print(f"m_WR={MWR:.0f}: m_c={m_c:.1f}, sigma={sigma:.2f}, "
      f"window [{lo:.1f}, {hi:.1f}] -> snapped [{slo:.0f}, {shi:.0f}] ({nbins} bins)")
print(f"B in snapped window = {total[insel].sum():.1f} events")
