#!/usr/bin/env python3
"""Step 10b -- the expected band collapses when b-only toys are all n = 0.

Relative 68% band width ((+1s - -1s)/median) of the HybridNew expected band
vs B_window. The expected band exists because different background-only toy
outcomes ask for different limits; once B <~ 0.5 almost every toy observes
n = 0, all quantiles pose the same question, and the band width goes to zero.
That is genuine Poisson discreteness, not a failure -- and it is invisible to
AsymptoticLimits, whose Gaussian band keeps a fixed +-1sigma shape. The
residual wiggles at 3400-4200 are 500-toys-per-point noise.
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
wins = {w["mWR"]: w for w in C.load_stage6_windows()}
rows = [r for r in C.load_refined() if r["regime"] == "anch_sparse"]
b = [wins[r["mWR"]]["B_window"] for r in rows]
rel = [(float(r["fb_p1s"]) - float(r["fb_m1s"])) / float(r["fb_med"])
       for r in rows]

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.plot(b, rel, "s", color=C.BLUE, ms=8)
for r, bb, rr in zip(rows, b, rel):
    ax.annotate(f"{r['mWR']:.0f}", (bb, rr), fontsize=9,
                xytext=(4, 4), textcoords="offset points")
ax.axhline(0, color="grey", ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"$B_{\rm window}$ (events)")
ax.set_ylabel(r"($+1\sigma$ $-$ $-1\sigma$) / median   (HybridNew)")
ax.text(0.04, 0.90, "b-only toys nearly all $n=0$\n"
        r"$\Rightarrow$ quantiles coincide $\Rightarrow$ band width $\to$ 0",
        transform=ax.transAxes, fontsize=13, va="top")
ax.grid(alpha=0.3)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step10b_band_collapse")
print("next: step11 -- each optimized setting vs its alternative")
