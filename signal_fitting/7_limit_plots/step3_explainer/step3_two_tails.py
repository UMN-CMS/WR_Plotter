#!/usr/bin/env python3
"""Just the two tails CLs compares, for the -2sigma observation n_sig-hat=-27:
  s=0 bell  -> CL_b     = 2.3%   (background-only)
  s=9 bell  -> CL_{s+b} = 0.4%   (signal hypothesis)
Both are small; CL_b measures how much of that smallness is 'just the low
background'. CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

MU0, SIG = 0.1, 13.4
LUMI, COM = 109.8, 13.6
NHAT = MU0 - 2 * SIG                 # -26.7
S_SIG = 9.0

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

CLb = Phi((NHAT - 0.0) / SIG) * 100
CLsb = Phi((NHAT - S_SIG) / SIG) * 100

hep.style.use("CMS")
x = np.linspace(-52, 52, 1800)
y0 = gauss(x, 0.0, SIG)
y9 = gauss(x, S_SIG, SIG)

fig, ax = plt.subplots()
ax.plot(x, y0, color="#1f77b4", lw=2.6, label=r"$s=0$  (background-only)")
ax.plot(x, y9, color="#2ca02c", lw=2.6, label=fr"$s={S_SIG:g}$  (signal)")

m = x <= NHAT
ax.fill_between(x[m], 0, y0[m], color="#1f77b4", alpha=0.35)
ax.fill_between(x[m], 0, y9[m], color="#2ca02c", alpha=0.55)

ax.set_ylim(0, 0.036)
ax.axvline(NHAT, color="black", lw=2.2, ls="--")
ax.text(NHAT + 2, 0.034, fr"observed $\hat{{n}}_{{\rm sig}}={NHAT:.0f}$"
        "\n" r"($-2\sigma$ down)", fontsize=13, ha="left", va="top")

# arrows to the two tails
ax.annotate(fr"$\mathrm{{CL}}_b = {CLb:.1f}\%$",
            xy=(NHAT - 3, gauss(NHAT - 3, 0.0, SIG)), xytext=(-50, 0.020),
            fontsize=14, color="#1f77b4", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"))
ax.annotate(fr"$\mathrm{{CL}}_{{s+b}} = {CLsb:.1f}\%$",
            xy=(NHAT - 6, gauss(NHAT - 6, S_SIG, SIG)), xytext=(-50, 0.012),
            fontsize=14, color="#2ca02c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ca02c"))

ax.text(0.62, 0.60,
        r"both tails are small." "\n"
        r"how much of $\mathrm{CL}_{s+b}$ is" "\n"
        r"just the low background" "\n"
        r"(which $\mathrm{CL}_b$ measures)?",
        transform=ax.transAxes, va="top", fontsize=13,
        bbox=dict(boxstyle="round", fc="#f4f4f4", ec="grey"))

ax.set_xlabel(r"Fitted signal yield  $\hat{N}_{\rm sig}$  [events]")
ax.set_ylabel("Probability density")
ax.set_xlim(-52, 52)
ax.text(0.03, 0.97,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV",
        transform=ax.transAxes, va="top", fontsize=14)
ax.legend(loc="upper right", fontsize=12)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step3_explainer/step3_two_tails")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step3_two_tails.{{png,pdf}}  CL_b={CLb:.2f}%  CL_sb={CLsb:.2f}%")
