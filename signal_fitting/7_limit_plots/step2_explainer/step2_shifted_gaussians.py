#!/usr/bin/env python3
"""The 2341 estimator distribution N-hat ~ Normal(s, sigma) for several signal
hypotheses s: same width sigma=13.4, center slid from 0 to s. CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

SIG = 13.4
LUMI = 109.8
COM = 13.6
HYPS = [(0.0,  "#1f77b4", "background only"),
        (9.0,  "#2ca02c", None),
        (23.0, "#ff7f0e", None),
        (36.0, "#e42536", None)]

def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

hep.style.use("CMS")
x = np.linspace(-45, 85, 1800)
fig, ax = plt.subplots()

for s, c, extra in HYPS:
    y = gauss(x, s, SIG)
    lab = fr"$s={s:g}$" + (f"  ({extra})" if extra else "")
    ax.plot(x, y, color=c, lw=2.4, label=lab)
    ax.fill_between(x, 0, y, color=c, alpha=0.12)
    ax.axvline(s, color=c, lw=1.4, ls="--")
    ax.text(s, -0.0013, fr"$s={s:g}$", color=c, fontsize=11, ha="center",
            va="top")

# a double-headed arrow showing the common width on the s=0 curve
y0 = gauss(0.0, 0.0, SIG)
ax.annotate("", xy=(-SIG, y0 * 0.6), xytext=(SIG, y0 * 0.6),
            arrowprops=dict(arrowstyle="<->", color="grey", lw=1.3))
ax.text(0.0, y0 * 0.52, r"same $\sigma=13.4$ for every $s$",
        ha="center", color="grey", fontsize=11)

ax.set_xlabel(r"Fitted signal yield  $\hat{N}_{\rm sig}$  [events]")
ax.set_ylabel("Probability density")
ax.set_ylim(bottom=0)
ax.set_xlim(-45, 85)
ax.text(0.03, 0.97,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV" "\n"
        r"$\hat{N}_{\rm sig}\sim\mathcal{N}(s,\sigma)$" "\n"
        r"width fixed, center $=s$",
        transform=ax.transAxes, va="top", fontsize=14)
ax.legend(loc="upper right", fontsize=13)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step2_explainer/step2_shifted_gaussians")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print("wrote step2_shifted_gaussians.{png,pdf}")
