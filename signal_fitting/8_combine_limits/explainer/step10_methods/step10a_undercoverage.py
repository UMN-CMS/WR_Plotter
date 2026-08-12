#!/usr/bin/env python3
"""Step 10a -- the asymptotic shortcut under-covers at a few events.

Same anchored cards, two methods: AsymptoticLimits (the Asimov/Gaussian
closed forms -- derived from scratch in step9a) vs HybridNew (LHC-style CLs with actual q_mu
toys). The ratio of their expected medians against B_window: agreement needs
the profile likelihood to look parabolic, which fails at a few counts -- the
asymptotic limit comes out up to ~40% LOW (anti-conservative) below B ~ 5.
That is why the sparse regime quotes HybridNew.
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
ratio = [float(r["asym_med_fb"]) / float(r["fb_med"]) for r in rows]

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.plot(b, ratio, "o", color=C.RED, ms=8)
for r, bb, rr in zip(rows, b, ratio):
    ax.annotate(f"{r['mWR']:.0f}", (bb, rr), fontsize=9,
                xytext=(4, 4), textcoords="offset points")
ax.axhline(1.0, color="grey", ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"$B_{\rm window}$ (events)")
ax.set_ylabel("asymptotic median / HybridNew median")
ax.axvspan(ax.get_xlim()[0], 5, color="#f89c20", alpha=0.12)
ax.text(0.04, 0.06, "shaded: B < 5 --\nasymptotics not to be quoted (10.6)",
        transform=ax.transAxes, fontsize=13, va="bottom")
ax.grid(alpha=0.3)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step10a_undercoverage")

print(f"{'m':>6}{'B_win':>8}{'asymptotic':>12}{'HybridNew':>11}{'ratio':>7}")
for r, bb in zip(rows, b):
    a, h = float(r["asym_med_fb"]), float(r["fb_med"])
    print(f"{r['mWR']:>6.0f}{bb:>8.2f}{a:>12.3f}{h:>11.3f}{a/h:>7.2f}")
print("next: step10b -- the band collapse (Poisson discreteness)")
