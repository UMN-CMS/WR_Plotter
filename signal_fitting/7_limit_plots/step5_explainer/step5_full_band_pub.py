#!/usr/bin/env python3
"""Step 5 (publication style) -- expected 95% CL sigma x BR limit vs m_WR,
formatted to mirror the CMS Full Run 2 W_R -> eejj result: x-axis in GeV,
y 1e-4..1e4, thin red theory, Brazil band, legend lower-left. Expected-only
(blind, no observed) and our real header. Reads the xsec_limit table CSV."""
import csv
import math
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep

HERE = pathlib.Path(__file__).resolve().parent
CSV = HERE.parent / "xsec_limit_table_ee_resolved.csv"
LUMI, COM = 109.8, 13.6

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        if r["function"] == "expo":
            rows.append(r)
rows.sort(key=lambda r: float(r["mWR"]))
m   = [float(r["mWR"]) for r in rows]                       # GeV
th  = [float(r["xsec_pb"]) * 1000.0 for r in rows]
m2s = [float(r["sigma_ul_m2s"]) * 1000.0 for r in rows]
m1s = [float(r["sigma_ul_m1s"]) * 1000.0 for r in rows]
med = [float(r["sigma_ul_med"]) * 1000.0 for r in rows]
p1s = [float(r["sigma_ul_p1s"]) * 1000.0 for r in rows]
p2s = [float(r["sigma_ul_p2s"]) * 1000.0 for r in rows]

hep.style.use("CMS")
fig, ax = plt.subplots(figsize=(8, 8))

ax.fill_between(m, m2s, p2s, color="#f5c518", label="95% expected")
ax.fill_between(m, m1s, p1s, color="#00b050", label="68% expected")
ax.plot(m, med, color="black", lw=2.4, ls=":", label="Expected limit")
ax.plot(m, th, color="red", lw=1.4, label=r"Theory ($g_R=g_L$)")

ax.set_yscale("log")
ax.set_xlim(800, 6000)
ax.set_ylim(1e-4, 1e4)
ax.set_xlabel(r"$m_{W_R}$ (GeV)")
ax.set_ylabel(
    r"$\sigma(pp\to W_R)\,\mathcal{B}(W_R\to ee\,q\bar{q}')$  (fb)")
ax.xaxis.set_major_locator(mticker.MultipleLocator(1000))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(200))

def _ylab(y, _):
    if y == 1.0:
        return "1"
    if y == 10.0:
        return "10"
    return r"$10^{%d}$" % int(round(math.log10(y)))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(_ylab))

ax.text(0.60, 0.90,
        r"$m_N = m_{W_R}/2$" "\n" "ee channel (resolved)",
        transform=ax.transAxes, va="top", ha="left", fontsize=17,
        fontweight="bold")

h, l = ax.get_legend_handles_labels()
order = [l.index(x) for x in ["Expected limit", "68% expected",
                              "95% expected", r"Theory ($g_R=g_L$)"]]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower left",
          fontsize=15, frameon=False, handlelength=1.6, labelspacing=0.35)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=f"{LUMI:.1f}", com=COM, fontsize=16)
base = str(HERE / "step5_full_band_pub")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print(f"masses {m[0]:.0f}-{m[-1]:.0f} GeV, {len(m)} points")
print("wrote step5_full_band_pub.{png,pdf}")
