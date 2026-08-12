#!/usr/bin/env python3
"""Step 3 -- the regime map: one model does NOT fit all masses.

B_window (the summed MC in each window) spans 837 -> 0.04 events from 1000 to
6000. Three regimes, each with a different failure mode of the naive
"floating background + AsymptoticLimits" recipe (Stage-10 findings, verified
on this background):

  anch_low (1000-1200)   window clamped at the 800 GeV floor -> no left
                         sideband; the floating background is collinear with
                         the signal (baseline 13.0 fb at m=1000 vs 4.96
                         anchored).
  float (1400-3200)      B = 512 -> 12 in the k3 window shown here: sidebands
                         on both sides, enough events for asymptotics. These
                         masses take the Stage-10.9 OPTIMIZED card (k5 window,
                         50 GeV bins, anchor-constrained slope; shape priors
                         <= 2600) -- see steps 4a and 7e.
  anch_sparse (>= 3400)  B < 7: free norm runs away on near-empty windows AND
                         AsymptoticLimits under-covers (10.6: 2.3 vs
                         HybridNew 3.25 events at 4600).

The B plotted is the Stage-6 (k3) window content -- the geometry the regimes
were classified on; the optimized float cards then widen their own windows
to k5 (B 37-1036). The boundaries are properties of THIS background
(59.8/fb run2 MC).
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
wins = C.load_stage6_windows()
m = [w["mWR"] for w in wins]
b = [w["B_window"] for w in wins]

fig, ax = plt.subplots(figsize=(10, 7))
ax.axvspan(900, C.FLOAT_MIN - 100, color="#e42536", alpha=0.12)
ax.axvspan(C.FLOAT_MIN - 100, C.FLOAT_MAX + 100, color="#00cc00", alpha=0.10)
ax.axvspan(C.FLOAT_MAX + 100, 6100, color="#f89c20", alpha=0.15)
ax.plot(m, b, "o-", color="black", lw=2, ms=5,
        label=r"$B_{\rm window}$ (summed MC in the $\mu\pm3\sigma$ window)")
ax.axhline(5, color="grey", ls="--", lw=1.5)
ax.text(1050, 6, "asymptotics unreliable below ~5 events (10.6)",
        fontsize=12, color="grey")
ax.set_yscale("log")
ax.set_xlim(900, 6100)
ax.set_xlabel(r"$m_{W_R}$ (GeV)")
ax.set_ylabel("background events in window")
ax.text(0.035, 0.80, "anch_low\nno left\nsideband", fontsize=13,
        color="#e42536", weight="bold", transform=ax.transAxes)
ax.text(0.25, 0.86, "float\noptimized card (k5, slope constr.)", fontsize=13,
        color="#007700", weight="bold", transform=ax.transAxes)
ax.text(0.62, 0.86, "anch_sparse\nanchored bkg + HybridNew", fontsize=13,
        color="#b06000", weight="bold", transform=ax.transAxes)
ax.legend(loc="lower left", fontsize=13, frameon=False)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step3_regime_map")

for w in wins:
    print(f"  m={w['mWR']:>6.0f}  B_window={w['B_window']:>8.2f}  -> {C.regime(w['mWR'])}")
print("next: step4 -- the datacard anatomy per regime")
