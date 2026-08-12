#!/usr/bin/env python3
"""Step 4d -- the cross-section Brazil band at a SINGLE mass (WR2400_N1200).
Each CLs event-band edge N is converted to sigma x BR via
    sigma[fb] = N / (L[fb^-1] * eff),
and the theory sigma is overlaid. The expected limit (~1 fb) sits far below the
theory (~77 fb), so W_R at 2.4 TeV is excluded. CMS style."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.special import ndtri

MU0, SIG = 0.1434, 11.5615        # WR2400_N1200 nsp_hist moments
EFF = 0.2013                       # S_fit/genEventSumw
LUMI, COM = 109.8, 13.6            # fb^-1
XSEC_FB = 0.0768873 * 1000.0       # theory sigma x BR = 76.9 fb
NSP_ASIMOV = 1.8088                # nominal-MC (Asimov) fitted yield
ALPHA = 0.05
MWR = 2.4                          # TeV

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def ul_cls(nh):
    return nh + SIG * ndtri(1.0 - ALPHA * Phi(nh / SIG))

Ns = [-2, -1, 0, 1, 2]
band_ev = {N: ul_cls(MU0 + N * SIG) for N in Ns}       # CLs band in events
denom = LUMI * EFF                                      # N -> sigma[fb]
band = {N: band_ev[N] / denom for N in Ns}             # sigma x BR [fb]
obs = ul_cls(NSP_ASIMOV) / denom
print({N: round(v, 3) for N, v in band.items()}, "obs", round(obs, 3),
      "theory", round(XSEC_FB, 1))

hep.style.use("CMS")
fig, ax = plt.subplots()
w = 0.10
x0, x1 = MWR - w, MWR + w
ax.fill_between([x0, x1], band[-2], band[2], color="#f5d800", label="95% expected")
ax.fill_between([x0, x1], band[-1], band[1], color="#00cc00", label="68% expected")
ax.plot([x0, x1], [band[0]] * 2, color="black", lw=2.5, ls=":",
        label="Expected limit")
ax.plot([x0, x1], [XSEC_FB] * 2, color="#e42536", lw=2.5,
        label=r"Theory ($g_R=g_L$)")

ax.annotate(fr"median exp. $\approx {band[0]:.1f}$ fb", xy=(MWR, band[0]),
            xytext=(MWR + 0.13, band[0]), fontsize=13, va="center")
ax.annotate(fr"theory $\approx {XSEC_FB:.0f}$ fb", xy=(MWR, XSEC_FB),
            xytext=(MWR + 0.13, XSEC_FB), fontsize=13, va="center", color="#e42536")
ax.set_yscale("log")
ax.set_xlim(1.9, 3.1)
ax.set_ylim(0.3, 300)
ax.set_xlabel(r"$m_{W_R}$ (TeV)")
ax.set_ylabel(r"$\sigma(pp\to W_R)\,\mathcal{B}(W_R\to eeqq')$  (fb)")
ax.text(0.03, 0.90,
        "ee  Resolved\n"
        r"$m_N = m_{W_R}/2$   (WR2400_N1200)" "\n"
        r"CL$_s$,  eff $=0.20$",
        transform=ax.transAxes, va="top", fontsize=12)
ax.legend(loc="upper right", fontsize=12)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=14)
base = str(__import__("pathlib").Path(__file__).resolve().parent / "step4_xsec_band")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print("wrote step4_xsec_band.{png,pdf}")
