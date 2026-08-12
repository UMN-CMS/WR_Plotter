#!/usr/bin/env python3
"""Where 1.645 sigma comes from: the upper-limit hypothesis is the bell centered
1.645 sigma ABOVE the observation, so exactly 5% of its outcomes fall at/below
n_sig-hat. UL = n_sig-hat + 1.645 sigma. CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

SIG = 13.4
LUMI, COM = 109.8, 13.6
NHAT = 1.4
Z = 1.6449
UL = NHAT + Z * SIG                 # 23.44

def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

hep.style.use("CMS")
x = np.linspace(-25, 70, 1600)
y = gauss(x, UL, SIG)

fig, ax = plt.subplots()
ax.plot(x, y, color="#ff7f0e", lw=2.6,
        label=fr"limit hypothesis  $s=\mathrm{{UL}}={UL:.0f}$")

# 5% lower tail (at/below the observation)
m = x <= NHAT
ax.fill_between(x[m], 0, y[m], color="#ff7f0e", alpha=0.35)
ax.annotate("5% of the area\n"
            r"(P($\hat{N}_{\rm sig}\leq\hat{n}_{\rm sig}\,|\,$UL) = 5%)",
            xy=(NHAT - 6, gauss(NHAT - 6, UL, SIG)), xytext=(-23, 0.010),
            fontsize=13, arrowprops=dict(arrowstyle="->", color="black"))

# observation and centre
ax.axvline(NHAT, color="black", lw=2.0, ls="--")
ax.text(NHAT + 1, 0.0305, fr"observed $\hat{{n}}_{{\rm sig}}={NHAT:g}$",
        fontsize=13, ha="left")
ax.axvline(UL, color="#ff7f0e", lw=1.6, ls=":")
ax.text(UL + 1, 0.0305, fr"centre $=$ UL $={UL:.0f}$", color="#d2691e",
        fontsize=13, ha="left")

# the 1.645 sigma gap
yb = gauss(UL, UL, SIG) * 0.45
ax.annotate("", xy=(NHAT, yb), xytext=(UL, yb),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.6))
ax.text((NHAT + UL) / 2, yb + 0.0009,
        fr"$1.645\,\sigma = {Z*SIG:.0f}$ events", ha="center", fontsize=13)

ax.set_xlabel(r"Fitted signal yield  $\hat{N}_{\rm sig}$  [events]")
ax.set_ylabel("Probability density")
ax.set_ylim(0, 0.034)
ax.set_xlim(-25, 70)
ax.text(0.03, 0.97,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV" "\n"
        r"UL $=\hat{n}_{\rm sig}+1.645\,\sigma$",
        transform=ax.transAxes, va="top", fontsize=14)
ax.legend(loc="upper right", fontsize=12)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step2_explainer/step2_why_1645")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step2_why_1645.{{png,pdf}}  UL={UL:.2f}  1.645sig={Z*SIG:.2f}")
