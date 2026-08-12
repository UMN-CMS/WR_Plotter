#!/usr/bin/env python3
"""Step 11b -- setting: the background at the edges (anchored vs floating).

The bottom-line gain of the regime split: refined (anchored + HybridNew)
expected medians over the Stage-9 floating baseline at the edge masses.
x0.38 at 1000 (the collinear left-clamped window), x0.45-0.6 at 3400-4600
(no more free-norm runaway), rising through 1 at 5600-6000 -- that last part
is NOT a loss: the floating baseline's asymptotics under-covered there
(step 6a), so the refined number is the honest one.

Caveat: this comparison bundles the observation (7a) and method (6a) choices
at the sparse masses -- it is the total effect of adopting the optimized
recipe, while 7a/6a/7d isolate the individual settings.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")
refined = {r["mWR"]: r for r in C.load_refined()}
base = C.load_stage9()

mm = [m for m in sorted(refined) if refined[m]["regime"] != "float" and m in base]
ratio = [float(refined[m]["fb_med"]) / float(base[m]["comb_fb_med"]) for m in mm]

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(mm, ratio, "o-", color=C.RED, lw=2, ms=6)
ax.axhline(1, color="grey", ls=":")
ax.set_xlabel(r"$m_{W_R}$ (GeV)")
ax.set_ylabel("expected median:  anchored+HybridNew / floating baseline")
ax.set_title("the edge regimes: the refinement's bottom line", fontsize=15)
ax.grid(alpha=0.3)
ax.annotate("collinear window\nfixed: x0.38", (1000, ratio[0]),
            xytext=(1300, 0.9), fontsize=12,
            arrowprops=dict(arrowstyle="->", color="0.3"))
ax.text(0.55, 0.08, ">1 at 5800-6000 = the baseline\nunder-covered, not a loss",
        transform=ax.transAxes, fontsize=12, va="bottom")
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step11b_edges")
print("next: step11c -- the method choice, isolated")
