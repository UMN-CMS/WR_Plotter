#!/usr/bin/env python3
"""Step 4 aside -- cross section and event count are the SAME axis. A signal of
N events sits at sigma = N / (L * eff); with L=109.8 fb^-1 and eff=0.20 that is
1 fb <-> 22 events. So '77 fb' is ~1700 events (fixed) -- you cannot have 10 or
1000 events at 77 fb. Right axis shows the event equivalent of the left (sigma)
axis. CMS style."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

EFF = 0.2013
LUMI, COM = 109.8, 13.6
XSEC_FB = 0.0768873 * 1000.0       # theory ~ 76.9 fb
MWR = 2.4
DENOM = LUMI * EFF                  # N = sigma[fb] * DENOM   (~22.1)

# expected band (fb) -- carried over from step4_xsec_band (CLs, center-mean)
BAND = {-2: 0.552, -1: 0.741, 0: 1.03, 1: 1.432, 2: 1.919}

hep.style.use("CMS")
fig, ax = plt.subplots()
w = 0.10
x0, x1 = MWR - w, MWR + w
ax.fill_between([x0, x1], BAND[-2], BAND[2], color="#f5d800", label="95% expected")
ax.fill_between([x0, x1], BAND[-1], BAND[1], color="#00cc00", label="68% expected")
ax.plot([x0, x1], [BAND[0]] * 2, color="black", lw=2.5, ls=":",
        label="Expected limit")
ax.plot([x0, x1], [XSEC_FB] * 2, color="#e42536", lw=2.5,
        label=r"Theory ($g_R=g_L$)")
ax.annotate(fr"theory $=77$ fb $= {XSEC_FB*DENOM:.0f}$ events", xy=(MWR, XSEC_FB),
            xytext=(MWR + 0.13, XSEC_FB), fontsize=12, va="center", color="#e42536")

# event-count reference lines: each N is a DIFFERENT cross section
for n, lab in [(10, "10 events"), (100, "100 events"), (1000, "1000 events")]:
    s = n / DENOM
    ax.axhline(s, color="grey", lw=1.0, ls="--")
    ax.text(3.05, s, f"{n} events\n= {s:.2g} fb", color="grey", fontsize=11,
            ha="right", va="center")

ax.set_yscale("log")
ax.set_xlim(1.9, 3.15)
ax.set_ylim(0.3, 300)
ax.set_xlabel(r"$m_{W_R}$ (TeV)")
ax.set_ylabel(r"$\sigma(pp\to W_R)\,\mathcal{B}(W_R\to eeqq')$  (fb)")

# right axis: the SAME axis, in event units
ax2 = ax.twinx()
ax2.set_yscale("log")
ax2.set_ylim(0.3 * DENOM, 300 * DENOM)
ax2.set_ylabel(r"signal events in window  ($N = \sigma\cdot L\cdot\mathrm{eff}$)")

ax.text(0.03, 0.90,
        "ee  Resolved   WR2400_N1200\n"
        fr"$L={LUMI}$ fb$^{{-1}}$,  eff $=0.20$" "\n"
        r"$\Rightarrow$ 1 fb $\leftrightarrow$ 22 events",
        transform=ax.transAxes, va="top", fontsize=12)
ax.legend(loc="upper right", fontsize=11)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=14)
base = str(__import__("pathlib").Path(__file__).resolve().parent
           / "step4_events_vs_xsec")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"1 fb = {DENOM:.1f} events; theory 77 fb = {XSEC_FB*DENOM:.0f} events")
print("wrote step4_events_vs_xsec.{png,pdf}")
