#!/usr/bin/env python3
"""Step 11c -- setting: the statistical method (asymptotic vs HybridNew).

The pure method effect on identical anchored cards, as a function of mass
(step 6a showed it against B_window). AsymptoticLimits is kept for the float
and anch_low regimes where it is valid (B >= 12); HybridNew (500 toys/point,
5 quantiles) replaces it for B < 7. The choice is per-regime, not global:
toys everywhere would cost hours for no change above B ~ 10.
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
rows = [r for r in C.load_refined() if r["regime"] == "anch_sparse"]
mm = [r["mWR"] for r in rows]
ratio = [float(r["asym_med_fb"]) / float(r["fb_med"]) for r in rows]

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(mm, ratio, "o-", color="#f89c20", lw=2, ms=6)
ax.axhline(1, color="grey", ls=":")
ax.set_xlabel(r"$m_{W_R}$ (GeV)")
ax.set_ylabel("expected median:  asymptotic / HybridNew")
ax.set_title("same anchored cards; only the method differs", fontsize=15)
ax.grid(alpha=0.3)
ax.text(0.04, 0.08, "the shortcut is anti-conservative\nexactly where the "
        "limit flattens;\nquote HybridNew below B ~ 5",
        transform=ax.transAxes, fontsize=13, va="bottom")
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step11c_method")
print("next: step11d -- the signal-shape choice")
