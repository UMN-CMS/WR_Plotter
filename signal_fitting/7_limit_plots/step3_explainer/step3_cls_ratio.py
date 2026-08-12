#!/usr/bin/env python3
"""CLs fixes the pathology. For the -2sigma observation n_sig-hat=-27:
  naive  uses CL_{s+b}          -> below 5% for ALL s>=0 (even s=0), UL=-4.7
  CLs    uses CL_{s+b}/CL_b     -> =100% at s=0, crosses 5% at s=+14 (positive)
Dividing by CL_b cancels the shared downward-fluctuation effect. CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

MU0, SIG = 0.1, 13.4
LUMI, COM = 109.8, 13.6
NHAT = MU0 - 2 * SIG                 # -26.7
ALPHA = 0.05

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

CLb = Phi(NHAT / SIG)                                  # background-only tail
s = np.linspace(-10, 45, 2400)
clsb = np.array([Phi((NHAT - si) / SIG) for si in s])  # CL_{s+b}
cls = np.minimum(clsb / CLb, 1.0)                       # CL_s (capped at 1)

# crossovers with the 5% line (both curves decreasing -> interp on reversed)
UL_naive = float(np.interp(ALPHA, clsb[::-1], s[::-1]))
UL_cls = float(np.interp(ALPHA, cls[::-1], s[::-1]))

hep.style.use("CMS")
fig, ax = plt.subplots()
ax.plot(s, clsb * 100, color="#888888", lw=2.4, ls="--",
        label=r"naive:  $\mathrm{CL}_{s+b}$")
ax.plot(s, cls * 100, color="#7a1fa2", lw=2.8,
        label=r"CL$_s = \mathrm{CL}_{s+b}/\mathrm{CL}_b$")
ax.axhline(ALPHA * 100, color="#e42536", lw=1.8, ls="--")
ax.text(-9, 7, "threshold = 5%", color="#e42536", fontsize=12, va="bottom")

# crossovers
ax.plot([UL_naive], [5], "o", color="#888888", ms=10, zorder=4)
ax.plot([UL_cls], [5], "o", color="#7a1fa2", ms=11, zorder=4)
ax.plot([UL_cls, UL_cls], [0, 5], color="#7a1fa2", lw=1.2, ls=":", zorder=1)
ax.annotate(fr"naive UL $={UL_naive:.1f}$" "\n(negative!)",
            xy=(UL_naive, 5), xytext=(-8.5, 34), color="#888888",
            fontsize=13, ha="left", arrowprops=dict(arrowstyle="->", color="#888888"))
ax.annotate(fr"CL$_s$ UL $=+{UL_cls:.0f}$" "\n(positive)",
            xy=(UL_cls, 5), xytext=(UL_cls + 6, 22), color="#7a1fa2",
            fontsize=13, arrowprops=dict(arrowstyle="->", color="#7a1fa2"))

ax.text(1.8, 90,
        r"at $s=0$:  $\mathrm{CL}_{s+b}=\mathrm{CL}_b$" "\n"
        r"$\Rightarrow \mathrm{CL}_s=100\%$ (never excluded)",
        fontsize=12.5, color="#7a1fa2")

ax.set_xlabel(r"hypothesized signal  $s$  [events]")
ax.set_ylabel(r"exclusion probability  [%]")
ax.set_xlim(-10, 45)
ax.set_ylim(0, 105)
ax.text(0.64, 0.72,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV" "\n"
        fr"observed $\hat{{n}}_{{\rm sig}}={NHAT:.0f}$ ($-2\sigma$)" "\n"
        fr"$\mathrm{{CL}}_b={CLb*100:.1f}\%$",
        transform=ax.transAxes, va="top", fontsize=13)
ax.legend(loc="upper right", fontsize=13)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step3_explainer/step3_cls_ratio")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step3_cls_ratio.{{png,pdf}}  CLb={CLb*100:.2f}%  "
      f"UL_naive={UL_naive:.2f}  UL_cls={UL_cls:.2f}")
