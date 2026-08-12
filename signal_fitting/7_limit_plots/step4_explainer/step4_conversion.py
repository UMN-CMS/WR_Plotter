#!/usr/bin/env python3
"""Step 4c -- the conversion flow: N_UL (events) -> sigma_UL, for WR2400_N1200.
sigma_UL = N_UL / (1000 * L * eff). CMS style schematic."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.special import ndtri

MU0, SIG = 0.1434, 11.5615        # WR2400_N1200 nsp_hist moments
EFF = 0.2013                       # S_fit/genEventSumw (fit-consistent)
LUMI = 109.8                       # fb^-1
ALPHA = 0.05

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

N_UL = MU0 + SIG * ndtri(1.0 - ALPHA * Phi(MU0 / SIG))    # CLs median ~ 22.8
sig_pb = N_UL / (1000.0 * LUMI * EFF)
sig_fb = sig_pb * 1000.0
print(f"N_UL={N_UL:.1f}  sigma_UL={sig_pb:.4g} pb = {sig_fb:.2f} fb")

hep.style.use("CMS")
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

boxes = [
    (0.24, "#7a1fa2", fr"$N_{{\rm UL}} = {N_UL:.0f}$" "\n" "events",
     "CLs median\n(WR2400)"),
    (0.74, "#1f77b4", fr"$\sigma_{{\rm UL}} \approx {sig_fb:.1f}$" "\n" "fb",
     fr"$= {sig_pb*1000:.2f}\times10^{{-3}}$ pb"),
]
for x, c, big, sub in boxes:
    ax.text(x, 0.58, big, ha="center", va="center", fontsize=20,
            fontweight="bold", color=c,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=c, lw=2.5))
    ax.text(x, 0.30, sub, ha="center", va="top", fontsize=13, color=c)

# arrow with the operation
ax.annotate("", xy=(0.61, 0.58), xytext=(0.37, 0.58),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.4))
ax.text(0.49, 0.80, r"$\div\ (1000 \cdot L \cdot \mathrm{eff})$", ha="center",
        fontsize=15)
ax.text(0.49, 0.70, fr"$L={LUMI}$ fb$^{{-1}}$,  eff$={EFF:.2f}$", ha="center",
        fontsize=12, color="grey")

ax.text(0.5, 0.03,
        r"$\sigma_{\rm UL} = \dfrac{N_{\rm UL}}{1000 \cdot L \cdot \mathrm{eff}}$"
        r"$= \dfrac{%.0f}{1000 \cdot %.1f \cdot %.2f} \approx %.2f$ fb"
        % (N_UL, LUMI, EFF, sig_fb),
        ha="center", va="bottom", fontsize=15)

ax.text(0.01, 0.97, "ee  Resolved   WR2400_N1200\n"
        r"$m_{W_R}=2400$, $m_N=1200$", ha="left", va="top", fontsize=12)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=13.6, fontsize=14)
base = str(__import__("pathlib").Path(__file__).resolve().parent / "step4_conversion")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print("wrote step4_conversion.{png,pdf}")
