#!/usr/bin/env python3
"""Intermediate step: the 2341 N_sp distribution redrawn as its Gaussian model,
in CMS style to match the nsp_hist plots.
Mean mu0 = 0.1, RMS sigma = 13.4 (ee-resolved, expo, off-grid m_WR=2341)."""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

MU0 = 0.1
SIG = 13.4
LUMI = 109.8
COM = 13.6
NTOYS = 1000
BINW = 4.0          # nsp_hist bin width: 2*hist_range/hist_bins = 2*60/30
TOY_SCALE = NTOYS * BINW   # density -> toy counts

def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))

hep.style.use("CMS")
x = np.linspace(-60, 60, 1600)
y = gauss(x, MU0, SIG)

fig, ax = plt.subplots()
ax.plot(x, y, color="#5790fc", lw=2.6,
        label=r"model  $\hat{N}_{\rm sig}\sim\mathcal{N}(\mu_0,\sigma)$")

# +-2 sigma (lighter) then +-1 sigma (darker) shaded bands
m2 = (x >= MU0 - 2 * SIG) & (x <= MU0 + 2 * SIG)
ax.fill_between(x[m2], 0, y[m2], color="#5790fc", alpha=0.20,
                label=r"$\pm2\sigma$  (95%)")
m1 = (x >= MU0 - SIG) & (x <= MU0 + SIG)
ax.fill_between(x[m1], 0, y[m1], color="#5790fc", alpha=0.45,
                label=r"$\pm1\sigma$  (68%)")

ax.axvline(MU0, color="#e42536", lw=2.0, ls="--",
           label=fr"$\mu_0=\langle \hat{{N}}_{{\rm sig}}\rangle={MU0}$")

ax.set_xlabel(r"Fitted signal yield  $\hat{N}_{\rm sig}$  [events]")
ax.set_ylabel("Probability density")
ax.set_ylim(bottom=0)
ax.set_xlim(-60, 60)
ax.text(0.04, 0.95,
        "ee\nResolved\n"
        fr"$m_{{W_R}}=2341$ GeV" "\n"
        fr"$\langle \hat{{N}}_{{\rm sig}}\rangle = {MU0}$" "\n"
        fr"RMS $= {SIG}$" "\n"
        r"$N_{\rm toys}=1000$",
        transform=ax.transAxes, va="top", fontsize=15)
ax.legend(loc="upper right", fontsize=14)

# right-hand axis in toy counts: Toys = density * NTOYS * BINW (exact, linear)
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim()[0] * TOY_SCALE, ax.get_ylim()[1] * TOY_SCALE)
ax2.set_ylabel(fr"Toys / {BINW:g} events  (= density $\times$ {int(TOY_SCALE)})")

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
fig.savefig("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
            "7_limit_plots/step2_explainer/step1p5_nsp_gaussian.png",
            dpi=150, bbox_inches="tight")
fig.savefig("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
            "7_limit_plots/step2_explainer/step1p5_nsp_gaussian.pdf",
            bbox_inches="tight")
print("wrote step1p5_nsp_gaussian.{png,pdf}")
