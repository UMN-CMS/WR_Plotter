#!/usr/bin/env python3
"""How the tail-vs-s curve slides as the observation n-hat changes. The curve
keeps its shape and just translates: its 50% point always sits at s = n-hat.
Descriptive -- no thresholds / limits. CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

SIG = 13.4
LUMI, COM = 109.8, 13.6

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# a few observed values, low -> high
NHATS = [(-8.0, "#1f77b4"),
         (4.0,  "#2ca02c"),
         (16.0, "#ff7f0e"),
         (28.0, "#e42536")]

hep.style.use("CMS")
s = np.linspace(-30, 60, 900)
fig, ax = plt.subplots()

# faint 50% guide
ax.axhline(50, color="grey", lw=1.0, ls=":", zorder=1)
ax.text(-29, 52, "50%", color="grey", fontsize=11, va="bottom")

for nh, c in NHATS:
    tail = np.array([Phi((nh - si) / SIG) for si in s]) * 100
    ax.plot(s, tail, color=c, lw=2.6, zorder=2,
            label=fr"$\hat{{n}}_{{\rm sig}}={nh:g}$")
    # mark the 50% point, which sits at s = n-hat
    ax.plot([nh], [50], "o", color=c, ms=9, zorder=3)

ax.annotate(r"each curve's 50% point sits at $s=\hat{n}_{\rm sig}$",
            xy=(4, 50), xytext=(18, 63), fontsize=13,
            arrowprops=dict(arrowstyle="->", color="black"))

ax.set_xlabel(r"hypothesized signal  $s$  [events]")
ax.set_ylabel(r"tail  P($\hat{N}_{\rm sig}\leq\hat{n}_{\rm sig}\,|\,s$)  [%]")
ax.set_xlim(-30, 60)
ax.set_ylim(0, 100)
ax.text(0.03, 0.30,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV" "\n"
        fr"$\sigma={SIG}$ (fixed)",
        transform=ax.transAxes, va="top", fontsize=13)
ax.legend(loc="upper right", fontsize=13, title=r"observation $\hat{n}_{\rm sig}$")
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step2_explainer/step2_tail_vs_s_slide")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print("wrote step2_tail_vs_s_slide.{png,pdf}")
