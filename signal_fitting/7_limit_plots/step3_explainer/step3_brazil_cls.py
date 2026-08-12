#!/usr/bin/env python3
"""Single-mass Brazil band (m_WR=2341), naive vs CLs. The band quantiles are the
limit UL evaluated at the +-N sigma quantiles of the background-only n_sig-hat
distribution (the nsp_hist, mu0=0.1, sigma=13.4).
  naive:  UL = n_sig-hat + 1.645 sigma          (linear shift)
  CLs:    UL = n_sig-hat + sigma*Phi^-1(1 - alpha*Phi(n_sig-hat/sigma))
The naive -2sigma edge goes negative (-4.7); CLs keeps it positive (+14). CMS."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.special import ndtri

MU0, SIG = 0.1, 13.4            # off-grid m_WR=2341 (ee, resolved, expo)
LUMI, COM = 109.8, 13.6
ALPHA = 0.05
Z = 1.6449

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def ul_naive(nh):
    return nh + Z * SIG

def ul_cls(nh):
    return nh + SIG * ndtri(1.0 - ALPHA * Phi(nh / SIG))

Ns = [-2, -1, 0, 1, 2]
nq = {N: MU0 + N * SIG for N in Ns}           # +-N sigma observations (centre=mean)
naive = {N: ul_naive(nq[N]) for N in Ns}
cls = {N: ul_cls(nq[N]) for N in Ns}

hep.style.use("CMS")
fig, ax = plt.subplots()
w = 0.30
cols = [(0.0, naive, "naive\n" r"(CL$_{s+b}$)"), (1.0, cls, "CLs")]
for xi, band, _ in cols:
    ax.fill_between([xi - w, xi + w], band[-2], band[2], color="#f5d800",
                    zorder=1, label=r"$\pm2\sigma$" if xi == 0 else None)
    ax.fill_between([xi - w, xi + w], band[-1], band[1], color="#00b050",
                    zorder=2, label=r"$\pm1\sigma$" if xi == 0 else None)
    ax.plot([xi - w, xi + w], [band[0], band[0]], color="black", lw=2.5,
            ls="--", zorder=3, label="median" if xi == 0 else None)
    # edge value labels
    ax.text(xi, band[0] + 1.2, f"med {band[0]:.0f}", ha="center", fontsize=12,
            zorder=4)
    ax.text(xi + w + 0.03, band[-2], fr"$-2\sigma={band[-2]:+.0f}$", ha="left",
            va="center", fontsize=12, zorder=4,
            color=("#e42536" if band[-2] < 0 else "#006400"), fontweight="bold")

# the physical floor
ax.axhline(0, color="#e42536", lw=1.8, ls=":")
ax.text(1.46, 1.0, r"$N_{\rm sig}\geq0$ (physical)", color="#e42536",
        fontsize=12, ha="right", va="bottom")

ax.annotate("naive dips\nbelow zero", xy=(0.0, naive[-2]), xytext=(0.0, -10),
            ha="center", fontsize=12, color="#e42536",
            arrowprops=dict(arrowstyle="->", color="#e42536"))

ax.set_xticks([0.0, 1.0])
ax.set_xticklabels(["naive\n" r"(CL$_{s+b}$)", "CLs"], fontsize=15)
ax.set_ylabel(r"upper limit on $N_{\rm sig}$  [events]")
ax.set_xlim(-0.6, 1.9)
ax.set_ylim(-14, 60)
ax.text(0.03, 0.97,
        "ee   Resolved   "
        fr"$m_{{W_R}}=2341$ GeV" "\n"
        r"band $=$ UL at the $\pm N\sigma$ of the nsp_hist",
        transform=ax.transAxes, va="top", fontsize=12)
ax.legend(loc="upper right", fontsize=12)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=15)
base = ("/uscms_data/d3/bjackson/WrCoffea/WR_Plotter/signal_fitting/"
        "7_limit_plots/step3_explainer/step3_brazil_cls")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print("naive:", {k: round(v, 1) for k, v in naive.items()})
print("cls:  ", {k: round(v, 1) for k, v in cls.items()})
print(f"wrote step3_brazil_cls.{{png,pdf}}")
