#!/usr/bin/env python3
"""Step 1a -- the background components.

Four converted run2 MC samples (rootfiles/.../20260714_run2_bkgs), already
scaled to 59.8 fb^-1, stacked in the ee resolved SR at 50 GeV binning (the
k5_bw50 card binning): DY+jets (LO MG-HT, ReweightedQCDErrorEWCorr_Reshaped),
tt+tW, Nonprompt, Other. tt+tW dominates below ~1.5 TeV, DY takes over in the
tail.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")

comps = C.load_bkg_components(rebin=5)
edges = comps["DYJets"][0]
centers = 0.5 * (edges[:-1] + edges[1:])
colors = {"Other": "#94a4a2", "Nonprompt": "#964a8b", "tt_tW": "#f89c20",
          "DYJets": "#5790fc"}
order = ["Other", "Nonprompt", "tt_tW", "DYJets"]

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.stackplot(centers, *[comps[s][1] for s in order], step="mid",
             colors=[colors[s] for s in order],
             labels=[C.BKG_LABELS[s] for s in order])

# combined MC statistical uncertainty as a hatched band on the stack top
_, total, stat = C.load_bkg_total(rebin=5)
lo = np.clip(total - stat, 1e-4, None)
ax.fill_between(centers, lo, total + stat, step="mid", facecolor="none",
                edgecolor="#555555", hatch="/////", linewidth=0.0, zorder=5)
stat_proxy = Patch(facecolor="none", edgecolor="#555555", hatch="/////",
                   label="MC stat. unc.")

ax.set_yscale("log")
ax.set_xlim(800, 6000)
ax.set_ylim(1e-3, 1e4)
C.log_yaxis_one_ten(ax)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("Events / 50 GeV")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1] + [stat_proxy], labels[::-1] + ["MC stat. unc."],
          loc="upper right", fontsize=16, frameon=False)
ax.text(0.03, 0.96, "ee\nResolved SR\nRunII 2018 MC",
        transform=ax.transAxes, fontsize=16, va="top")
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step1a_component_stack")

for s in C.BKG_SAMPLES:
    print(f"  {C.BKG_LABELS[s]:<28} {float(comps[s][1].sum()):>8.1f} events")
print("next: step1b -- the sum is the fit input AND the observation")
