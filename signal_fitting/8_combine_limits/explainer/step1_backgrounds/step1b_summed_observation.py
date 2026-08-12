#!/usr/bin/env python3
"""Step 1b -- the summed background IS the observation (the no-data setting).

Everything downstream sees only this one histogram: it is the shape the
background model is fit to, AND it sits in the `data_obs` slot of every card
(the "MC Asimov" observation). The real EGamma data (1095 SR events) exists
in the converted files but is deliberately not used -- the expected band then
reflects the background MODEL, not one dataset's fluctuations (quantified in
step 7a: up to ~35% at 3-5 TeV, where the 2018 data sits above this MC).
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

comps = C.load_bkg_components(rebin=5)
edges = comps["DYJets"][0]
total = sum(comps[s][1] for s in C.BKG_SAMPLES)

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.stairs(np.maximum(total, 0.0), edges, color="black", linewidth=1.8,
          label="summed background MC\n= fit input = data_obs (no data)")
ax.set_yscale("log")
ax.set_xlim(800, 6000)
ax.set_ylim(1e-3, 1e4)
C.log_yaxis_one_ten(ax)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("Events / 50 GeV")
ax.legend(loc="upper right", fontsize=16, frameon=False)
ax.text(0.03, 0.96, "ee\nResolved SR\nRunII 2018 MC",
        transform=ax.transAxes, fontsize=16, va="top")
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step1b_summed_observation")

print(f"  summed background, full spectrum: {float(total.sum()):.1f} events")
print("next: step2 -- the signal shape and the windows")
