#!/usr/bin/env python3
"""Step 5b -- the member-to-member spread (the background-model systematic).

Ratio of each anchor member to the central expo. Inside the anchor range the
members agree to a few percent; extrapolating, the curved families (expo2,
powexp) flare upward -- that flare, propagated through the anchored cards as
the max |median shift| across members, is the quoted model systematic
(8-25% of the median; grey bars in the 10.8 comparison plot). It is drawn ON
the band, not folded into the CLs sigma -- the Stage-10.4 rule: forecast
uncertainties belong next to the result, not inside the test statistic.
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


STYLES = {"tail": (C.BLUE, "--", 2), "expo2": ("#f89c20", "-.", 2),
          "powexp": ("#964a8b", ":", 2.2)}
x = np.linspace(1000, 6000, 500)

fig, ax = plt.subplots(figsize=(10, 7))
for name, (c, ls, lw) in STYLES.items():
    ax.plot(x, member_f(name, x) / member_f("central", x), color=c, ls=ls,
            lw=lw, label=f"{name} / central")
ax.axhline(1, color=C.RED, lw=2)
ax.axvspan(1000, 3500, color="grey", alpha=0.10)
ax.set_yscale("log")
ax.set_ylim(0.5, 30)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("member / central")
ax.text(1500, 12, "anchor range:\nmembers agree to a few %", fontsize=13)
ax.text(4300, 3, "extrapolation:\nthe spread IS the\nmodel systematic",
        fontsize=13)
ax.legend(fontsize=13, frameon=False, loc="upper left")
ax.grid(alpha=0.3)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step5b_member_ratio")
print("next: step5c -- transporting the members into the windows (B_env)")
