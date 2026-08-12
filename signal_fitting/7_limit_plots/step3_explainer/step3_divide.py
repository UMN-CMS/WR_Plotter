#!/usr/bin/env python3
"""The single division, for s=9 at the -2sigma observation:
  CL_s = CL_{s+b} / CL_b = 0.4% / 2.3% = 17%.
Naive compares CL_{s+b}=0.4% to 5% -> excluded. CLs compares CL_s=17% to 5%
-> NOT excluded. Dividing by CL_b lifts the tail above threshold. CMS style."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

MU0, SIG = 0.1, 13.4
LUMI, COM = 109.8, 13.6
NHAT = MU0 - 2 * SIG
S_SIG = 9.0

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

CLsb = Phi((NHAT - S_SIG) / SIG) * 100      # 0.39
CLb = Phi((NHAT - 0.0) / SIG) * 100         # 2.32
CLs = CLsb / CLb * 100                       # 16.7

hep.style.use("CMS")
fig, ax = plt.subplots()
xs = [0.0, 1.0, 2.6]
heights = [CLsb, CLb, CLs]
colors = ["#2ca02c", "#1f77b4", "#7a1fa2"]
ax.bar(xs, heights, width=0.62, color=colors, alpha=0.85, edgecolor="black",
       linewidth=0.6)
for xi, h in zip(xs, heights):
    ax.text(xi, h + 0.4, f"{h:.1f}%" if h < 10 else f"{h:.0f}%",
            ha="center", fontsize=15, fontweight="bold")

# 5% threshold
ax.axhline(5, color="#e42536", lw=2.0, ls="--")
ax.text(3.15, 5.2, "threshold = 5%", color="#e42536", fontsize=13, ha="right")

# the division
ax.text(1.8, 12.5, r"$\div$", fontsize=30, ha="center", va="center")
ax.annotate("", xy=(2.28, CLs * 0.6), xytext=(1.3, CLb + 1.5),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
ax.text(2.6, CLs + 2.2,
        r"$\mathrm{CL}_s=\dfrac{\mathrm{CL}_{s+b}}{\mathrm{CL}_b}"
        fr"=\dfrac{{{CLsb:.1f}}}{{{CLb:.1f}}}={CLs:.0f}\%$",
        ha="center", fontsize=14)

# verdicts
ax.text(0.0, -2.6, r"naive uses this" "\n" r"$0.4\% < 5\%$" "\n"
        r"$\Rightarrow$ excluded", ha="center", va="top", fontsize=12,
        color="#2ca02c")
ax.text(2.6, -2.6, r"CLs uses this" "\n" fr"${CLs:.0f}\% > 5\%$" "\n"
        r"$\Rightarrow$ NOT excluded", ha="center", va="top", fontsize=12,
        color="#7a1fa2")

ax.set_xticks(xs)
ax.set_xticklabels([r"$\mathrm{CL}_{s+b}$", r"$\mathrm{CL}_b$",
                    r"$\mathrm{CL}_s$"], fontsize=15)
ax.set_ylabel("probability  [%]")
ax.set_ylim(0, 22)
ax.set_xlim(-0.6, 3.3)
ax.text(0.03, 0.96,
        "ee   Resolved   "
        fr"$m_{{W_R}}=2341$ GeV,  $s=9$,  $\hat{{n}}_{{\rm sig}}=-27$",
        transform=ax.transAxes, va="top", fontsize=12)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step3_explainer/step3_divide")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"wrote step3_divide.{{png,pdf}}  CLsb={CLsb:.2f}  CLb={CLb:.2f}  CLs={CLs:.1f}")
