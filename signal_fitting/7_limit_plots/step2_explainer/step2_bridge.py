#!/usr/bin/env python3
"""Bridge between the tail-vs-s threshold plot and the distribution-of-limits
plot. Three possible background-only observations n-hat (-1sigma, typical,
+1sigma of the nsp_hist). Each has its own tail-vs-s curve; each crosses the
SAME 5% threshold at its own upper limit on N_sig. Those three ULs are exactly
the -1sigma / median / +1sigma edges of the limit distribution. CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

MU0, SIG = 0.1, 13.4
Z = 1.6449
THRESH = 5.0
LUMI, COM = 109.8, 13.6

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# three representative observations from the background-only n-hat distribution
OBS = [(MU0 - SIG, "#1f77b4", fr"low  ($\hat{{n}}_{{\rm sig}}=\mu_0-\sigma={MU0-SIG:.1f}$)"),
       (MU0,       "#333333", fr"typical  ($\hat{{n}}_{{\rm sig}}=\mu_0={MU0:g}$)"),
       (MU0 + SIG, "#e42536", fr"high  ($\hat{{n}}_{{\rm sig}}=\mu_0+\sigma={MU0+SIG:.1f}$)")]

hep.style.use("CMS")
s = np.linspace(-20, 55, 900)
fig, ax = plt.subplots()

# 5% threshold
ax.axhline(THRESH, color="black", lw=1.6, ls="--", zorder=1)
ax.text(-19, THRESH + 1.5, "threshold = 5%", fontsize=13, va="bottom")

for nh, c, lab in OBS:
    tail = np.array([Phi((nh - si) / SIG) for si in s]) * 100
    ax.plot(s, tail, color=c, lw=2.6, zorder=2, label=lab)
    ul = nh + Z * SIG                       # crossover = this obs's upper limit
    ax.plot([ul], [THRESH], "o", color=c, ms=10, zorder=3)
    ax.plot([ul, ul], [0, THRESH], color=c, lw=1.2, ls=":", zorder=1)
    ax.annotate(fr"UL$={ul:.0f}$", xy=(ul, THRESH), xytext=(ul - 1, 11),
                color=c, fontsize=14, fontweight="bold", ha="center")

ax.set_xlabel(r"hypothesized signal $s$  =  upper limit on $N_{\rm sig}$  [events]")
ax.set_ylabel(r"tail  P($\hat{N}_{\rm sig}\leq\hat{n}_{\rm sig}\,|\,s$)  [%]")
ax.set_xlim(-20, 55)
ax.set_ylim(0, 62)
ax.text(0.63, 0.66,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV" "\n"
        fr"$\mu_0={MU0:g}$,  $\sigma={SIG}$  (bkg-only)" "\n"
        r"each crossover is that obs's" "\n"
        r"upper limit on $N_{\rm sig}$",
        transform=ax.transAxes, va="top", fontsize=13)
ax.legend(loc="upper right", fontsize=12, title=r"observation $\hat{n}_{\rm sig}$")
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step2_explainer/step2_bridge")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step2_bridge.{{png,pdf}}  ULs = "
      f"{MU0-SIG+Z*SIG:.1f}, {MU0+Z*SIG:.1f}, {MU0+SIG+Z*SIG:.1f}")
