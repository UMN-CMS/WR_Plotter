#!/usr/bin/env python3
"""Step 5c -- transporting the anchor into each window: B_env.

B_env = sum of the member function over the window's bin centres -- the fixed
background normalization of each anchored card. Compared per mass against
the windowed raw MC: at the low edge they agree (812 vs 837 at m=1000 --
the anchor range covers the window); in the deep tail the smooth central
member falls below the jagged MC (0.37 vs 1.05 at 4600) -- Stage 10.4 judged
that tail single-large-weight noise ('mcmax' caveat in the README; adding an
mcmax member is one entry in the 10.8 MEMBERS dict if wanted).
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
with open(C.ANCHORS) as fh:
    p = json.load(fh)["central"]["params"]

wins = [w for w in C.load_stage6_windows() if C.regime(w["mWR"]) != "float"]
m, b_env, b_mc = [], [], []
for w in wins:
    s = (centers >= w["fit_lo"]) & (centers <= w["fit_hi"])
    f = p[0] * np.exp(p[1] * (centers[s] - 2000.0) / 1000.0)
    m.append(w["mWR"])
    b_env.append(float(f.sum()))
    b_mc.append(w["B_window"])

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(m, b_mc, "o", color="black", ms=7, label="windowed raw MC")
ax.plot(m, b_env, "s", color=C.RED, ms=7,
        label="B_env (central anchor, the card constant)")
ax.set_yscale("log")
ax.set_xlabel(r"$m_{W_R}$ (GeV)")
ax.set_ylabel("background events in window")
ax.legend(fontsize=13, frameon=False)
ax.grid(alpha=0.3)
ax.text(0.35, 0.55, "agree at the low edge;\nsmooth anchor < jagged MC tail\n"
        "('mcmax' caveat)", transform=ax.transAxes, fontsize=13)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step5c_transport")

print(f"{'m':>6}{'B_env':>10}{'windowed MC':>13}")
for mm, be, bm in zip(m, b_env, b_mc):
    print(f"{mm:>6.0f}{be:>10.3f}{bm:>13.3f}")
print("next: step6 -- inside combine: the likelihood the card defines, one bin at a time")
