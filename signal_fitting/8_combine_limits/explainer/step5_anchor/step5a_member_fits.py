#!/usr/bin/env python3
"""Step 5a -- the anchor member fits on the trusted spectrum.

Four binned Poisson-ML fits of the summed MC (pivot 2000 GeV), done once in
the Stage-10.8 workspace builder and reused by every anchored card:

  central   expo  on [1000, 3500]   the card background (b = -2.94/TeV)
  tail      expo  on [1000, 6000]   'is the tail still expo?'   (b = -2.92)
  expo2     + curvature             'does curvature matter?'    (c = +0.22)
  powexp    power law x expo        a different falling family

The grey band marks the anchor range: 25 populated bins determine the
family well -- which is exactly why importing the background beats fitting
1-3 sparse bins in place. Beyond ~4.5 TeV the jagged single-large-weight MC
events sit above every smooth member (the 'mcmax' caveat, README).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")
comps = C.load_bkg_components()
edges = comps["DYJets"][0]
centers = 0.5 * (edges[:-1] + edges[1:])
total = sum(comps[s][1] for s in C.BKG_SAMPLES)
with open(C.ANCHORS) as fh:
    anchors = json.load(fh)


def member_f(name, x):
    p = anchors[name]["params"]
    t = (x - 2000.0) / 1000.0
    if name in ("central", "tail"):
        return p[0] * np.exp(p[1] * t)
    if name == "expo2":
        return p[0] * np.exp(p[1] * t + p[2] * t * t)
    return p[0] * np.power(x / 2000.0, p[1]) * np.exp(p[2] * t)


STYLES = {"central": (C.RED, "-", 2.5), "tail": (C.BLUE, "--", 2),
          "expo2": ("#f89c20", "-.", 2), "powexp": ("#964a8b", ":", 2.2)}

fig, ax = plt.subplots(figsize=(10, 7.5))
sel = centers >= 950
ax.stairs(np.maximum(total[sel], 0.0),
          np.append(edges[:-1][sel], edges[1:][sel][-1]),
          color="black", linewidth=1.5, label="summed background MC")
for name, (c, ls, lw) in STYLES.items():
    lo, hi = anchors[name]["range"]
    ax.plot(centers[sel], member_f(name, centers[sel]), color=c, ls=ls, lw=lw,
            label=f"{name}  [{lo:.0f},{hi:.0f}]")
ax.axvspan(1000, 3500, color="grey", alpha=0.10)
ax.text(1900, 3e-4, "anchor range (trusted spectrum)", fontsize=12,
        color="grey")
ax.set_yscale("log")
ax.set_ylim(1e-4, 3e3)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("events / 100 GeV")
ax.legend(fontsize=13, frameon=False)
ax.grid(alpha=0.3)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step5a_member_fits")

for name, a in anchors.items():
    print(f"  {name:<8} params={['%.4g' % v for v in a['params']]} "
          f"chi2/ndf={a['chi2']/max(a['ndf'],1):.2f}")
print("next: step5b -- how much the members disagree (the model spread)")
