#!/usr/bin/env python3
"""CL_b on its own: the background-only tail P(N_sig-hat <= n_sig-hat | s=0).
It is just the CDF of the background-only distribution (the nsp_hist, centred at
mu0) evaluated at the observation. For a typical observation it's ~50%; for a
downward fluctuation it's small (2.3% here), which is exactly the signal that
the background fluctuated down. CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

MU0, SIG = 0.1, 13.4
LUMI, COM = 109.8, 13.6
NHAT = MU0 - 2 * SIG                 # -2 sigma downward fluctuation = -26.7

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

CLb = Phi((NHAT - MU0) / SIG) * 100          # 2.3 %
CLb_typ = Phi((1.4 - MU0) / SIG) * 100        # ~54 % for the nominal obs

hep.style.use("CMS")
x = np.linspace(-55, 55, 1600)
y = gauss(x, MU0, SIG)

fig, ax = plt.subplots()
ax.plot(x, y, color="#1f77b4", lw=2.6,
        label=r"background-only $s=0$  (the nsp_hist)")

# CL_b = shaded lower tail below the observation
m = x <= NHAT
ax.fill_between(x[m], 0, y[m], color="#1f77b4", alpha=0.45)
ax.axvline(NHAT, color="black", lw=2.2, ls="--")
ax.text(NHAT + 2, 0.033, fr"observed $\hat{{n}}_{{\rm sig}}={NHAT:.0f}$"
        "\n" r"($-2\sigma$ down)", fontsize=13, ha="left", va="top")

ax.annotate(fr"$\mathrm{{CL}}_b = {CLb:.1f}\%$",
            xy=(NHAT - 5, gauss(NHAT - 5, MU0, SIG)), xytext=(-52, 0.016),
            fontsize=15, color="#1f77b4", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"))

ax.text(0.60, 0.94,
        r"$\mathrm{CL}_b = \mathrm{P}(\hat{N}_{\rm sig}\leq"
        r"\hat{n}_{\rm sig}\,|\,s=0)$" "\n"
        r"$=$ background-only tail" "\n"
        r"$=$ CDF of the nsp_hist at $\hat{n}_{\rm sig}$",
        transform=ax.transAxes, va="top", fontsize=13)
ax.text(0.60, 0.66,
        r"typical obs ($\hat{n}_{\rm sig}\approx0$):" "\n"
        fr"   $\mathrm{{CL}}_b \approx {CLb_typ:.0f}\%$" "\n"
        r"here (bkg fluctuated down):" "\n"
        fr"   $\mathrm{{CL}}_b = {CLb:.1f}\%$ (small)",
        transform=ax.transAxes, va="top", fontsize=13,
        bbox=dict(boxstyle="round", fc="#eef4ff", ec="#1f77b4"))

ax.set_xlabel(r"Fitted signal yield  $\hat{N}_{\rm sig}$  [events]")
ax.set_ylabel("Probability density")
ax.set_ylim(0, 0.036)
ax.set_xlim(-55, 55)
ax.text(0.03, 0.97,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV",
        transform=ax.transAxes, va="top", fontsize=14)
ax.legend(loc="lower right", fontsize=12)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step3_explainer/step3_clb")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step3_clb.{{png,pdf}}  CLb={CLb:.2f}%  CLb_typical={CLb_typ:.1f}%")
