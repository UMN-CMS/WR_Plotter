#!/usr/bin/env python3
"""Step 4 closer -- the same single-mass limit as a SIGNAL STRENGTH band.
mu = sigma / sigma_theory: the cross section in units of the prediction, so
theory sits at mu = 1. The expected band at mu ~ 0.013 means the analysis is
expected to reach signals down to ~1.3% of the predicted rate. Same limit as
23 events / 1.0 fb, just rescaled. CMS style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

XSEC_FB = 0.0768873 * 1000.0       # theory ~ 76.9 fb
LUMI, COM = 109.8, 13.6
MWR = 2.4

# expected sigma band (fb), from step4_xsec_band (CLs, center-mean)
BAND_FB = {-2: 0.552, -1: 0.741, 0: 1.03, 1: 1.432, 2: 1.919}
BAND = {N: v / XSEC_FB for N, v in BAND_FB.items()}      # -> mu = sigma/sigma_th

hep.style.use("CMS")
fig, ax = plt.subplots()
w = 0.10
x0, x1 = MWR - w, MWR + w
ax.fill_between([x0, x1], BAND[-2], BAND[2], color="#f5d800", label="95% expected")
ax.fill_between([x0, x1], BAND[-1], BAND[1], color="#00cc00", label="68% expected")
ax.plot([x0, x1], [BAND[0]] * 2, color="black", lw=2.5, ls=":",
        label="Expected limit")
ax.axhline(1.0, color="#e42536", lw=2.0, label=r"$\mu = 1$ (theory)")

ax.annotate(fr"median exp. $\mu \approx {BAND[0]:.3f}$", xy=(MWR, BAND[0]),
            xytext=(MWR + 0.13, BAND[0]), fontsize=13, va="center")
ax.annotate(r"$\mu = 1$: predicted rate", xy=(MWR, 1.0),
            xytext=(MWR + 0.13, 1.0), fontsize=13, va="center", color="#e42536")

ax.set_yscale("log")
ax.set_xlim(1.9, 3.1)
ax.set_ylim(0.004, 3)
ax.set_xlabel(r"$m_{W_R}$ (TeV)")
ax.set_ylabel(r"95% CL upper limit on $\mu = \sigma/\sigma_{\rm theory}$")
ax.text(0.03, 0.60,
        "ee  Resolved   WR2400_N1200\n"
        r"$\mu = \sigma/\sigma_{\rm theory} = N/N_{\rm theory}$" "\n"
        r"same limit: 23 events $=$ 1.0 fb $=$ $\mu\,0.013$",
        transform=ax.transAxes, va="top", fontsize=12)
ax.legend(loc="upper right", fontsize=12)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=14)
base = str(__import__("pathlib").Path(__file__).resolve().parent / "step4_mu_band")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print("mu band:", {k: round(v, 4) for k, v in BAND.items()})
print("wrote step4_mu_band.{png,pdf}")
