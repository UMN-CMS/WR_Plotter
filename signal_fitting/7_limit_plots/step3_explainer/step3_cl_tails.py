#!/usr/bin/env python3
"""The pathology picture, relabeled with CL_b and CL_{s+b}. For the -2sigma
observation n_sig-hat=-27, each shaded tail-below-the-observation is a CL:
  s=0 tail  -> CL_b       (background-only)  = 2.3%
  s>0 tails -> CL_{s+b}   (signal hypothesis) -- smaller as s grows
Same recipe (area left of n_sig-hat), different bell -> different value.
CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

MU0, SIG = 0.1, 13.4
LUMI, COM = 109.8, 13.6
NHAT = MU0 - 2 * SIG                 # -26.7

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

# (s, color) -- s=0 is CL_b, the rest are CL_{s+b}
HYPS = [(0.0,  "#1f77b4"),
        (9.0,  "#2ca02c"),
        (23.0, "#ff7f0e"),
        (36.0, "#e42536")]

hep.style.use("CMS")
x = np.linspace(-75, 145, 2400)
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

# the s=0 shaded tail IS CL_b
ax.annotate(r"$\mathrm{CL}_b = 2.3\%$" "\n(the $s=0$ tail)",
            xy=(NHAT - 4, gauss(NHAT - 4, 0.0, SIG)), xytext=(-73, 0.023),
            fontsize=13, color="#1f77b4", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"))

# right-hand list, grouped by CL type
ax.text(0.70, 0.90, r"tails below $\hat{n}_{\rm sig}$:", transform=ax.transAxes,
        fontsize=13, va="top")
ax.text(0.70, 0.83, r"$\mathrm{CL}_b$  ($s=0$):  2.3%", transform=ax.transAxes,
        fontsize=13, color="#1f77b4", va="top")
ax.text(0.70, 0.74, r"$\mathrm{CL}_{s+b}$  (signal $s$):", transform=ax.transAxes,
        fontsize=13, va="top")
sig_vals = [(9.0, "#2ca02c"), (23.0, "#ff7f0e"), (36.0, "#e42536")]
for i, (s, c) in enumerate(sig_vals):
    tail = Phi((NHAT - s) / SIG) * 100
    txt = (f"{tail:.1f}%" if tail >= 0.1 else f"{tail:.2f}%")
    ax.text(0.73, 0.67 - 0.06 * i, fr"$s={s:g}$:  {txt}", transform=ax.transAxes,
            fontsize=13, color=c, va="top")

ax.text(0.62, 0.40,
        r"same recipe (area left of $\hat{n}_{\rm sig}$)," "\n"
        r"different bell $\to$ different value" "\n"
        r"($\mathrm{CL}_{s+b}$ shrinks as $s$ grows)",
        transform=ax.transAxes, va="top", fontsize=12,
        bbox=dict(boxstyle="round", fc="#f4f4f4", ec="grey"))

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
        "7_limit_plots/step3_explainer/step3_cl_tails")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step3_cl_tails.{{png,pdf}}  CL_b={Phi(NHAT/SIG)*100:.2f}%")
