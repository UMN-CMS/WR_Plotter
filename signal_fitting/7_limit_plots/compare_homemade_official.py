#!/usr/bin/env python3
"""Homemade expected limit (config A) vs the OFFICIAL 2018 ee result + ratio.

Top: the homemade expected band (expo) for the chosen run, with the official
2018 ee-combined expected median (59.74 fb^-1, 13 TeV) overlaid from the
digitized reference (digitize_official2018.py, verified overlay). Dashed
verticals mark the run's trusted mass range; outside it the homemade band is
survivor-biased (high edge) or structurally broken (low edge).

Bottom: our median / official median, drawn ONLY inside the trusted region.

Trusted ranges (from the Stage-6 toy diagnostics of each run's table):
  run2  [1400, 3200]  1000 clamped+biased (pull_mean +1.09), 1200 clamped;
                      >=3400: convergence 78%->12% (survivor bias), B_win < 7
  run3  [1400, 3200]  same geometry at the low edge; >=3400 marginal
                      (Gaussianity), >=3800 survivor-biased (10.1 audit)

run2 vs the official is nearly apples-to-apples (13 TeV, 59.8 vs 59.74
fb^-1); the remaining gaps are resolved-only vs combined, our MC (LO-HT
reshaped DY) background, stat-only, and the toy-RMS band convention.
run3 (13.6 TeV, 109 fb^-1) is a gauge, not a comparison.

  python compare_homemade_official.py --run run2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from wrplotter.plotting_helpers import custom_log_formatter  # noqa: E402

FIT_LABEL = {"expo": "Exponential fit", "powlaw": "Power-law fit"}

CFG = {
    "run2": {"lumi": "59.8", "com": 13, "trust": (1400.0, 3200.0),
             "rlim": (0.8, 2.4)},
    "run3": {"lumi": "109.1", "com": 13.6, "trust": (1400.0, 3200.0),
             "rlim": (0.4, 1.6)},
}

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run", default="run3", choices=["run2", "run3"])
p.add_argument("--trust-range", nargs=2, type=float, default=None)
p.add_argument("--function", default="expo", choices=["expo", "powlaw"])
p.add_argument("--table", type=Path, default=None,
               help="xsec_limit table override (default <run>/xsec_limit_table_ee_resolved.csv)")
p.add_argument("--output-dir", type=Path, default=None,
               help="output dir override (default <run>/comparison)")
p.add_argument("--max-mass", type=float, default=None,
               help="drop masses above this from the plot (the band is "
                    "untrustworthy and can explode past the trusted top edge)")
p.add_argument("--k", type=float, default=None,
               help="window half-width in sigma; if set, annotate the plot "
                    "with the +/-k sigma window label")
args = p.parse_args()
cfg = CFG[args.run]
trust = tuple(args.trust_range) if args.trust_range else cfg["trust"]

table = args.table or (HERE / args.run / "xsec_limit_table_ee_resolved.csv")
rows = [r for r in csv.DictReader(open(table))
        if r["function"] == args.function]
rows.sort(key=lambda r: float(r["mWR"]))
m = np.array([float(r["mWR"]) for r in rows])
fb = {k: np.array([1000.0 * float(r[f"sigma_ul_{k}"]) for r in rows])
      for k in ("m2s", "m1s", "med", "p1s", "p2s")}
if args.max_mass is not None:
    keep = m <= args.max_mass
    m = m[keep]
    fb = {k: v[keep] for k, v in fb.items()}

od = list(csv.DictReader(open(HERE / "official2018_expected_digitized.csv")))
om = np.array([float(r["mass_GeV"]) for r in od])
omed = np.array([float(r["med_fb"]) for r in od])
off_at = lambda x: np.exp(np.interp(x, om, np.log(omed)))

hep.style.use("CMS")
fig, (ax, axr) = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1],
                              gridspec_kw={"hspace": 0.06}, figsize=(10, 11))
ax.fill_between(m, fb["m2s"], fb["p2s"], color="#f5d800", label="95% expected")
ax.fill_between(m, fb["m1s"], fb["p1s"], color="#00cc00", label="68% expected")
ax.plot(m, fb["med"], "k:", lw=2, label=f"Expected limit ({args.run}, this work, {args.function})")
ax.plot(om, omed, color="#e42536", lw=2,
        label="CMS 2018 ee expected, combined")
for xv in trust:
    ax.axvline(xv, color="0.35", lw=1.4, ls=(0, (6, 3)))
    axr.axvline(xv, color="0.35", lw=1.4, ls=(0, (6, 3)))
ax.text(0.5 * sum(trust), 2.5e3, "trusted region", color="0.35",
        fontsize=13, ha="center")
ax.set_yscale("log")
ax.set_xlim(800, 6000)
ax.set_ylim(1e-4, 1e4)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
ax.set_ylabel(r"$\sigma(pp \to W_R)\,\mathcal{B}(W_R \to eeq\bar{q}\,')$ (fb)")
ax.text(0.95, 0.93, r"$\mathbf{m_{N} = m_{W_R}/2}$", transform=ax.transAxes,
        ha="right", va="top", fontsize=19)
ax.text(0.95, 0.87, "Resolved ee channel", transform=ax.transAxes,
        ha="right", va="top", fontsize=19, weight="bold")
ax.text(0.95, 0.81, FIT_LABEL[args.function], transform=ax.transAxes,
        ha="right", va="top", fontsize=16)
if args.k is not None:
    ax.text(0.95, 0.75, rf"$\pm{args.k:g}\sigma$ window ($k={args.k:g}$)",
            transform=ax.transAxes, ha="right", va="top", fontsize=16)
ax.legend(loc="lower left", fontsize=14, frameon=False)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi=cfg["lumi"], com=cfg["com"], fontsize=17)

sel = (m >= trust[0]) & (m <= trust[1])
axr.plot(m[sel], fb["med"][sel] / off_at(m[sel]), "o-", color="#5790fc",
         lw=2, ms=6)
axr.axhline(1, color="grey", ls=":")
axr.set_ylabel("this work /\nCMS 2018", fontsize=14)
axr.set_xlabel(r"$m_{W_R}$ (GeV)")
axr.set_ylim(*cfg["rlim"])
axr.grid(alpha=0.3)

out = args.output_dir or (HERE / args.run / "comparison")
out.mkdir(parents=True, exist_ok=True)
stem = out / (f"{args.run}_vs_official2018" + ("" if args.function == "expo" else f"_{args.function}"))
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"wrote {stem}.pdf/.png")
for mm in m[(m >= trust[0]) & (m <= trust[1])]:
    i = np.argmin(np.abs(m - mm))
    print(f"  m={mm:.0f}: ours={fb['med'][i]:.3f} fb  official={off_at(mm):.3f} fb"
          f"  ratio={fb['med'][i]/off_at(mm):.2f}")
