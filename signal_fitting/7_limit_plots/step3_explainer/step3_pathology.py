#!/usr/bin/env python3
"""Step 3 motivation: the naive rule on a downward-fluctuated observation.
At the -2sigma observation n_sig-hat = mu0 - 2sigma = -26.7, EVERY hypothesis --
including background-only s=0 -- has a tail below 5%, so the naive rule wrongly
excludes s=0 and returns a negative limit (UL = -4.7). CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

MU0, SIG = 0.1, 13.4
LUMI, COM = 109.8, 13.6
NHAT = MU0 - 2 * SIG                 # -2 sigma downward fluctuation = -26.7
Z = 1.6449
UL_naive = NHAT + Z * SIG            # -4.66

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

HYPS = [(0.0,  "#1f77b4"),
        (9.0,  "#2ca02c"),
        (23.0, "#ff7f0e"),
        (36.0, "#e42536")]

hep.style.use("CMS")
x = np.linspace(-55, 90, 2200)
fig, ax = plt.subplots()

for s, c in HYPS:
    y = gauss(x, s, SIG)
    ax.plot(x, y, color=c, lw=2.4)
    m = x <= NHAT
    ax.fill_between(x[m], 0, y[m], color=c, alpha=0.45)

ax.set_ylim(0, 0.040)
ax.axvline(NHAT, color="black", lw=2.2, ls="--")
ax.text(NHAT + 3, 0.038,
        fr"observed $\hat{{n}}_{{\rm sig}}={NHAT:.0f}$" "\n"
        r"($-2\sigma$ down)", fontsize=13, ha="left", va="top")

# point at the s=0 tail
ax.annotate("even $s=0$ has only\n2.3% below $\\hat{n}_{\\rm sig}$",
            xy=(NHAT - 4, gauss(NHAT - 4, 0.0, SIG)), xytext=(-73, 0.023),
            fontsize=12, arrowprops=dict(arrowstyle="->", color="black"))

# tail list (far-right clear column)
ax.text(0.72, 0.86, r"tails  P($\hat{N}_{\rm sig}\leq\hat{n}_{\rm sig}|s$):",
        transform=ax.transAxes, fontsize=12, va="top")
for i, (s, c) in enumerate(HYPS):
    tail = Phi((NHAT - s) / SIG) * 100
    txt = (f"{tail:.1f}%" if tail >= 0.1 else f"{tail:.2f}%")
    ax.text(0.72, 0.79 - 0.06 * i, fr"$s={s:g}$:  {txt}",
            transform=ax.transAxes, fontsize=12, color=c, va="top")

ax.text(0.72, 0.48,
        r"$\Rightarrow$ all tails $<5\%$:" "\n"
        r"every $s$ excluded," "\n"
        r"even $s=0$" "\n"
        fr"naive UL $={UL_naive:.1f}$ !",
        transform=ax.transAxes, va="top", fontsize=12,
        bbox=dict(boxstyle="round", fc="#fff3f3", ec="#d62536"))

ax.set_xlabel(r"Fitted signal yield  $\hat{N}_{\rm sig}$  [events]")
ax.set_ylabel("Probability density")
ax.set_xlim(-75, 145)
ax.text(0.03, 0.97,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV",
        transform=ax.transAxes, va="top", fontsize=14)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step3_explainer/step3_pathology")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step3_pathology.{{png,pdf}}  nhat={NHAT:.1f}  UL_naive={UL_naive:.2f}"
      f"  P(s=0)={Phi(NHAT/SIG)*100:.2f}%")
