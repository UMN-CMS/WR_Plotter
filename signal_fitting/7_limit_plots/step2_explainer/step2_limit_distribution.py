#!/usr/bin/env python3
"""Push the background-only n-hat distribution (nsp_hist, N(0.1,13.4)) through
UL(n-hat) = n-hat + 1.645*sigma to get the DISTRIBUTION of limits, with median
and +/-1sigma (green) / +/-2sigma (yellow) Brazil bands. Single mass 2341.
Naive (CLs+b) version -- the -2sigma edge going negative is the pathology
Step 3 (CLs) will fix. CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

MU0, SIG = 0.1, 13.4
Z = 1.6449
SHIFT = Z * SIG                 # 1.645 sigma ~ 22.0
UL_MED = MU0 + SHIFT           # median limit
LUMI, COM = 109.8, 13.6

def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

hep.style.use("CMS")
x = np.linspace(-25, 65, 1800)
yUL = gauss(x, UL_MED, SIG)     # distribution of the limit
yN = gauss(x, MU0, SIG)         # distribution of the observation (nsp_hist)

fig, ax = plt.subplots()

# +/-2 sigma (yellow) then +/-1 sigma (green) quantile bands under the UL pdf
m2 = (x >= UL_MED - 2 * SIG) & (x <= UL_MED + 2 * SIG)
ax.fill_between(x[m2], 0, yUL[m2], color="#f5d800", alpha=0.9,
                label=r"$\pm2\sigma$ expected", zorder=1)
m1 = (x >= UL_MED - SIG) & (x <= UL_MED + SIG)
ax.fill_between(x[m1], 0, yUL[m1], color="#00b050", alpha=0.9,
                label=r"$\pm1\sigma$ expected", zorder=2)
ax.plot(x, yUL, color="#333333", lw=2.4, zorder=3)
ax.axvline(UL_MED, color="black", lw=2.0, ls="--", zorder=4,
           label=fr"median UL $={UL_MED:.0f}$")

# the observation distribution (nsp_hist) faint
ax.plot(x, yN, color="grey", lw=2.0, ls="-", zorder=2)
ax.fill_between(x, 0, yN, color="grey", alpha=0.12, zorder=0)
ax.text(-17, gauss(MU0, MU0, SIG) - 0.001,
        r"$\hat{n}_{\rm sig}$ dist" "\n(bkg-only)",
        ha="center", color="grey", fontsize=12)

# the SAME three bridge observations, each shifted to its UL
BRIDGE = [(MU0 - SIG, "#1f77b4", r"UL$=9$" "\n" r"($-1\sigma$)"),
          (MU0,       "#333333", r"UL$=22$" "\n" r"(median)"),
          (MU0 + SIG, "#e42536", r"UL$=36$" "\n" r"($+1\sigma$)")]
for nh, c, lab in BRIDGE:
    yv = gauss(nh, MU0, SIG)
    ul = nh + SHIFT
    ax.annotate("", xy=(ul, yv), xytext=(nh, yv),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.8), zorder=5)
    ax.plot([nh], [yv], "o", color=c, ms=7, zorder=6)      # obs on grey curve
    ax.plot([ul], [yv], "o", color=c, ms=7, zorder=6)      # UL on limit curve
    if lab:
        ax.text(ul, yv - 0.0016, lab, color=c, fontsize=12, va="top",
                ha="center")

ax.set_xlabel(r"upper limit on $N_{\rm sig}$  [events]")
ax.set_ylabel("Probability density")
ax.set_xlim(-25, 65)
ax.set_ylim(0, 0.034)
ax.text(0.03, 0.97,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV",
        transform=ax.transAxes, va="top", fontsize=14)
ax.legend(loc="upper right", fontsize=13)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step2_explainer/step2_limit_distribution")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step2_limit_distribution.{{png,pdf}}  median={UL_MED:.2f} "
      f"1sig=[{UL_MED-SIG:.1f},{UL_MED+SIG:.1f}] "
      f"2sig=[{UL_MED-2*SIG:.1f},{UL_MED+2*SIG:.1f}]")
