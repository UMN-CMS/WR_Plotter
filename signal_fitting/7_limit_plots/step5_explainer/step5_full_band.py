#!/usr/bin/env python3
"""Step 5 -- the expected 95% CL cross-section limit for ALL masses.
Each grid mass gets its own CLs event-band (center 0, expo background), converted
to sigma x BR via sigma = N / (1000 * L * eff), then strung together vs m_WR to
make the Brazil band. The theory sigma x BR is overlaid. Dashed markers at 1.8
and 3.2 TeV bound the window where the nsp_hist is a clean Gaussian; outside it
(shaded) the band is not trustworthy. Expected/stat-only, no observed (blind).
Reads the xsec_limit table CSV. CMS style."""
import csv
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

HERE = pathlib.Path(__file__).resolve().parent
CSV = HERE.parent / "xsec_limit_table_ee_resolved.csv"
LUMI, COM = 109.8, 13.6
TRUST_LO, TRUST_HI = 1.8, 3.2       # TeV -- Gaussian-trustworthy window

# --- read the expo band (pb -> fb) ---
rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        if r["function"] != "expo":
            continue
        rows.append(r)
rows.sort(key=lambda r: float(r["mWR"]))
m   = [float(r["mWR"]) / 1000.0 for r in rows]
th  = [float(r["xsec_pb"]) * 1000.0 for r in rows]
m2s = [float(r["sigma_ul_m2s"]) * 1000.0 for r in rows]
m1s = [float(r["sigma_ul_m1s"]) * 1000.0 for r in rows]
med = [float(r["sigma_ul_med"]) * 1000.0 for r in rows]
p1s = [float(r["sigma_ul_p1s"]) * 1000.0 for r in rows]
p2s = [float(r["sigma_ul_p2s"]) * 1000.0 for r in rows]

hep.style.use("CMS")
fig, ax = plt.subplots()

ax.fill_between(m, m2s, p2s, color="#f5d800", label="95% expected")
ax.fill_between(m, m1s, p1s, color="#00cc00", label="68% expected")
ax.plot(m, med, color="black", lw=2.2, ls=":", label="Median expected")
ax.plot(m, th, color="#e42536", lw=2.4, marker="o", ms=4,
        label=r"Theory ($g_R=g_L$)")

# --- trustworthy window: dashed bounds + shade the untrustworthy ends ---
xlo, xhi = 0.9, 5.1
ylo, yhi = 0.15, 3.0e4
ax.axvspan(xlo, TRUST_LO, color="0.5", alpha=0.13, lw=0)
ax.axvspan(TRUST_HI, xhi, color="0.5", alpha=0.13, lw=0)
for xv in (TRUST_LO, TRUST_HI):
    ax.axvline(xv, color="0.35", lw=1.6, ls="--")
ax.text(0.5 * (TRUST_LO + TRUST_HI), 1.2e4, "Gaussian-\ntrustworthy\nwindow",
        ha="center", va="top", fontsize=11, color="0.25")
ax.text(0.5 * (xlo + TRUST_LO), 1.2e4, "not\ntrustworthy", ha="center",
        va="top", fontsize=10, color="0.45", style="italic")
ax.text(0.5 * (TRUST_HI + xhi), 1.2e4, "not trustworthy", ha="center",
        va="top", fontsize=10, color="0.45", style="italic")

ax.set_yscale("log")
ax.set_xlim(xlo, xhi)
ax.set_ylim(ylo, yhi)
ax.set_xlabel(r"$m_{W_R}$ (TeV)")
ax.set_ylabel(r"95% CL $\sigma(pp\to W_R)\,\mathcal{B}(W_R\to eeqq')$  (fb)")
ax.text(0.03, 0.32,
        "ee  Resolved\n"
        r"expo bkg,  CL$_s$,  centre 0" "\n"
        "expected / stat-only",
        transform=ax.transAxes, va="top", fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
ax.legend(loc="lower left", fontsize=12, framealpha=0.9)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=14)
base = str(HERE / "step5_full_band")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"masses {m[0]:.1f}-{m[-1]:.1f} TeV, {len(m)} points")
print("wrote step5_full_band.{png,pdf}")
