#!/usr/bin/env python3
"""Spurious signal versus mass from a card-model toy table (Stage-6 schema).

For each mass: the mean fitted signal yield over the toys with its RMS as the
error bar, and the fit on the unfluctuated pseudo-data (Asimov) as an open marker.

    python plot_card_spurious_vs_mass.py run2/card_fcrd/card_toy_table_ee_resolved.csv

Made 2026-10-03 for the HEP seminar (research-slides/talks/20261007_hep_seminar).
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")
table = Path(sys.argv[1])
rows = sorted((r for r in csv.DictReader(open(table))), key=lambda r: float(r["mWR"]))
m = [float(r["mWR"]) for r in rows]
mean = [float(r["mean_Nsp"]) for r in rows]
rms = [float(r["rms_Nsp"]) for r in rows]
asimov = [float(r["nsp_asimov"]) for r in rows]
label = rows[0]["function"]

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.axhline(0, color="gray", lw=1, ls="--")
ax.errorbar(m, mean, yerr=rms, fmt="o", color="black", markersize=6, capsize=4,
            label=r"mean of $\hat r$ over 1000 toys, $\pm$ RMS")
ax.plot(m, asimov, "s", mfc="none", mec="#c0392b", mew=2, markersize=9,
        label=r"$\hat r$ on the unfluctuated pseudo-data")
ax.set_xlabel(r"$m_{W_R}$ [GeV]")
ax.set_ylabel(r"fitted signal $\hat r$ [events]")
ax.set_ylim(-45, 45)
ax.legend(loc="upper right", fontsize=15, frameon=False)
ax.text(0.03, 0.96, "ee  Resolved\nsignal-free pseudo-data\n" + rf"$t\bar{{t}}$ anchored by $e\mu$ data",
        transform=ax.transAxes, fontsize=15, va="top")
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", lumi="59.8", com=13, fontsize=15)
out = table.parent / "spurious_yield_vs_mass_ee_resolved"
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
for r, a, s in zip(m, mean, rms):
    print(f"{r:5.0f}: mean {a:6.1f}  rms {s:5.1f}")
print("wrote", out)
