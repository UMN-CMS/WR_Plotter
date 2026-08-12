#!/usr/bin/env python3
"""Step 4d' -- what the single-mass cross-section plot looks like if a REAL
signal is present. The expected band (background-only toys) is UNCHANGED; only
the OBSERVED limit moves: a real signal makes the data fit return a large yield,
pulling the observed limit far above the expected band and above theory --- a
large excess, not an exclusion. Here a signal of 2x the theory rate is injected.
CMS style."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.special import ndtri

MU0, SIG = 0.1434, 11.5615
EFF = 0.2013
LUMI, COM = 109.8, 13.6
XSEC_FB = 0.0768873 * 1000.0       # theory ~ 76.9 fb
ALPHA = 0.05
MWR = 2.4
MU_INJ = 2.0                        # injected signal strength (x theory)

def Phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def ul_cls(nh):
    return nh + SIG * ndtri(1.0 - ALPHA * Phi(nh / SIG))

Ns = [-2, -1, 0, 1, 2]
band_ev = {N: ul_cls(MU0 + N * SIG) for N in Ns}       # unchanged expected band
denom = LUMI * EFF
band = {N: band_ev[N] / denom for N in Ns}

# a real signal: data yield ~ injected events; observed limit is pulled up
n_inj = MU_INJ * XSEC_FB * LUMI * EFF                   # events from the signal
n_obs = n_inj + MU0
obs_fb = ul_cls(n_obs) / denom
print(f"injected sigma={MU_INJ*XSEC_FB:.0f} fb ({n_inj:.0f} events)  "
      f"observed UL={obs_fb:.0f} fb")

hep.style.use("CMS")
fig, ax = plt.subplots()
w = 0.10
x0, x1 = MWR - w, MWR + w
ax.fill_between([x0, x1], band[-2], band[2], color="#f5d800", label="95% expected")
ax.fill_between([x0, x1], band[-1], band[1], color="#00cc00", label="68% expected")
ax.plot([x0, x1], [band[0]] * 2, color="black", lw=2.5, ls=":",
        label="Expected limit")
ax.plot([x0, x1], [obs_fb] * 2, color="black", lw=3.0, ls="-",
        label="Observed (signal present)")
ax.plot([x0, x1], [XSEC_FB] * 2, color="#e42536", lw=2.5,
        label=r"Theory ($g_R=g_L$)")

ax.annotate(fr"observed $\approx {obs_fb:.0f}$ fb", xy=(MWR, obs_fb),
            xytext=(MWR + 0.13, obs_fb), fontsize=13, va="center",
            fontweight="bold")
ax.annotate(fr"theory $\approx {XSEC_FB:.0f}$ fb", xy=(MWR, XSEC_FB),
            xytext=(MWR + 0.13, XSEC_FB), fontsize=13, va="center", color="#e42536")
ax.annotate(fr"expected $\approx {band[0]:.1f}$ fb", xy=(MWR, band[0]),
            xytext=(MWR + 0.13, band[0]), fontsize=13, va="center")
ax.annotate("", xy=(MWR - 0.16, obs_fb), xytext=(MWR - 0.16, band[2]),
            arrowprops=dict(arrowstyle="<->", color="#7a1fa2", lw=1.8))
ax.text(MWR - 0.20, math.sqrt(obs_fb * band[2]), "excess", rotation=90,
        ha="center", va="center", color="#7a1fa2", fontsize=13, fontweight="bold")

ax.set_yscale("log")
ax.set_xlim(1.9, 3.1)
ax.set_ylim(0.3, 600)
ax.set_xlabel(r"$m_{W_R}$ (TeV)")
ax.set_ylabel(r"$\sigma(pp\to W_R)\,\mathcal{B}(W_R\to eeqq')$  (fb)")
ax.text(0.03, 0.90,
        "ee  Resolved\n"
        r"$m_N = m_{W_R}/2$   (WR2400_N1200)" "\n"
        fr"signal injected at ${MU_INJ:g}\times$ theory",
        transform=ax.transAxes, va="top", fontsize=12)
ax.legend(loc="upper right", fontsize=12)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=14)
base = str(__import__("pathlib").Path(__file__).resolve().parent
           / "step4_xsec_band_signal")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print("wrote step4_xsec_band_signal.{png,pdf}")
