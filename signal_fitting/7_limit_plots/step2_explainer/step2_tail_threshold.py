#!/usr/bin/env python3
"""Merged tail-vs-s + threshold: the tail P(N_sig-hat <= n_sig-hat | s) curve with
the sampled hypotheses marked, PLUS the 5% threshold line and the crossover that
defines the observed upper limit on N_sig. CMS style. (Replaces the separate
step2_tail_vs_s and step2_threshold plots.)"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

SIG = 13.4
LUMI, COM = 109.8, 13.6
NHAT = 1.4          # nominal (Asimov) fit value from the diagnostic plot
THRESH = 5.0
Z = 1.6449
s_cross = NHAT + Z * SIG            # crossover = observed upper limit ~ 23.4

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# sampled hypotheses (same family as the other observed-thread plots)
PTS = [(0.0,  "#1f77b4"),
       (9.0,  "#2ca02c"),
       (23.0, "#ff7f0e"),
       (36.0, "#e42536")]

hep.style.use("CMS")
s = np.linspace(-5, 60, 900)
tail = np.array([Phi((NHAT - si) / SIG) for si in s]) * 100

fig, ax = plt.subplots()
ax.plot(s, tail, color="#333333", lw=2.6, zorder=2)

# 5% threshold line
ax.axhline(THRESH, color="#e42536", lw=2.0, ls="--", zorder=1)
ax.text(-4, THRESH + 1.4, "threshold = 5%", color="#e42536", fontsize=13,
        va="bottom")

# sampled hypotheses
for sv, c in PTS:
    tv = Phi((NHAT - sv) / SIG) * 100
    ax.plot([sv], [tv], "o", color=c, ms=11, zorder=3)
    txt = (f"{tv:.0f}%" if tv >= 1 else f"{tv:.1f}%")
    ax.annotate(fr"$s={sv:g}$  $\rightarrow$  {txt}", xy=(sv, tv),
                xytext=(sv - 0.5, tv + 2.6), color=c, fontsize=16,
                fontweight="bold")

# crossover -> observed upper limit (the orange s=23 point sits on the 5% line)
ax.plot([s_cross, s_cross], [0, THRESH], color="black", lw=1.2, ls=":", zorder=1)
ax.annotate("curve crosses 5% here\n"
            fr"$\rightarrow$ upper limit on $N_{{\rm sig}}\approx{s_cross:.0f}$ events",
            xy=(s_cross, THRESH), xytext=(s_cross + 4, 20), fontsize=13,
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
        "7_limit_plots/step2_explainer/step2_tail_threshold")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step2_tail_threshold.{{png,pdf}}  (crossover / UL = {s_cross:.2f})")
