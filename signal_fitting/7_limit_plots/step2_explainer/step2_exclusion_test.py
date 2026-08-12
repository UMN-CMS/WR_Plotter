#!/usr/bin/env python3
"""One observed n-hat laid across the 2341 family: for each hypothesis s, shade
the lower tail P(N-hat <= n-hat | s). Small tail -> s excluded. CMS style."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

SIG = 13.4
LUMI, COM = 109.8, 13.6
NHAT = 1.4          # the nominal (Asimov) fit value from the diagnostic plot
ALPHA = 0.05

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

# (s, color, verdict) -- limit member = NHAT + 1.645*SIG = 23.4 for n-hat=1.4
HYPS = [(0.0,  "#1f77b4", "allowed"),
        (9.0,  "#2ca02c", "allowed"),
        (23.0, "#ff7f0e", "limit (5%)"),
        (36.0, "#e42536", "excluded")]

hep.style.use("CMS")
x = np.linspace(-45, 120, 2200)
fig, ax = plt.subplots()

for s, c, verdict in HYPS:
    y = gauss(x, s, SIG)
    ax.plot(x, y, color=c, lw=2.4)
    m = x <= NHAT
    ax.fill_between(x[m], 0, y[m], color=c, alpha=0.30)

# observed n-hat
ax.set_ylim(0, 0.038)
ax.axvline(NHAT, color="black", lw=2.2, ls="--")
ax.text(NHAT + 1.5, 0.0315, fr"observed $\hat{{n}}_{{\rm sig}}={NHAT:g}$"
        "\n(nominal fit)", fontsize=13, ha="left")

# color-coded tail summary
lines = []
for s, c, verdict in HYPS:
    tail = Phi((NHAT - s) / SIG) * 100
    lines.append((s, c, tail, verdict))
ax.text(0.71, 0.80, r"tail $=$ P($\hat{N}_{\rm sig}\leq\hat{n}_{\rm sig}\,|\,s$):",
        transform=ax.transAxes, fontsize=13, va="top")
for i, (s, c, tail, verdict) in enumerate(lines):
    txt = (f"{tail:.0f}%" if tail >= 1 else f"{tail:.1f}%")
    ax.text(0.71, 0.73 - 0.07 * i,
            fr"$s={s:g}$:  {txt}   {verdict}",
            transform=ax.transAxes, fontsize=13, color=c, va="top")

ax.set_xlabel(r"Fitted signal yield  $\hat{N}_{\rm sig}$  [events]")
ax.set_ylabel("Probability density")
ax.set_ylim(bottom=0)
ax.set_xlim(-45, 100)
ax.text(0.03, 0.97,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV" "\n"
        r"exclude $s$ if tail $\leq5\%$" "\n"
        r"UL $=$ largest allowed $s$",
        transform=ax.transAxes, va="top", fontsize=14)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step2_explainer/step2_exclusion_test")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print("wrote step2_exclusion_test.{png,pdf}")
