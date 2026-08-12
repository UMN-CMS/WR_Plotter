#!/usr/bin/env python3
"""tail-vs-s for the n-hat=4 example, with a horizontal threshold line drawn in
and the crossover marked. CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

SIG = 13.4
LUMI, COM = 109.8, 13.6
NHAT = 1.4          # the nominal (Asimov) fit value from the diagnostic plot
THRESH = 5.0        # percent

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# crossover: Phi((nhat - s)/sig) = THRESH/100  ->  s = nhat + z*sig
Z_5PCT = 1.6449          # z with Phi(-z) = 0.05
s_cross = NHAT + Z_5PCT * SIG

hep.style.use("CMS")
s = np.linspace(-5, 60, 900)
tail = np.array([Phi((NHAT - si) / SIG) for si in s]) * 100

fig, ax = plt.subplots()
ax.plot(s, tail, color="#333333", lw=2.6, zorder=2)

# horizontal threshold
ax.axhline(THRESH, color="#e42536", lw=2.0, ls="--", zorder=1)
ax.text(-4, THRESH + 1.5, f"threshold = {THRESH:g}%", color="#e42536",
        fontsize=14, va="bottom")

# crossover
ax.plot([s_cross], [THRESH], "o", color="black", ms=11, zorder=3)
ax.plot([s_cross, s_cross], [0, THRESH], color="black", lw=1.2, ls=":", zorder=1)
ax.annotate(fr"crossover  $\rightarrow$  upper limit" "\n"
            fr"on $N_{{\rm sig}}\approx{s_cross:.0f}$ events",
            xy=(s_cross, THRESH), xytext=(s_cross + 3, 18), fontsize=14,
            arrowprops=dict(arrowstyle="->", color="black"))

ax.set_xlabel(r"hypothesized signal  $s$  [events]")
ax.set_ylabel(r"tail  P($\hat{N}_{\rm sig}\leq\hat{n}_{\rm sig}\,|\,s$)  [%]")
ax.set_xlim(-5, 60)
ax.set_ylim(0, 72)
ax.text(0.62, 0.96,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV" "\n"
        fr"observed $\hat{{n}}_{{\rm sig}}={NHAT:g}$" "\n"
        fr"$\sigma={SIG}$",
        transform=ax.transAxes, va="top", fontsize=14)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step2_explainer/step2_threshold")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step2_threshold.{{png,pdf}}  (crossover s = {s_cross:.2f})")
