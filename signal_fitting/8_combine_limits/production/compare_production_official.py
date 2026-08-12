#!/usr/bin/env python3
"""Production combine limit (config C) vs the OFFICIAL 2018 ee result + ratio.

Top: the refined run2 expected band (regime-split optimized cards: float
1400-3200 = k5 window / 50 GeV bins / anchor-constrained slope / mu-sigma
priors <= 2600; anchored background at 1000-1200 and >= 3400 with HybridNew
quantiles in the sparse tail; MC-Asimov observation everywhere) with the
official 2018 ee-combined expected median overlaid from the digitized
reference (7_limit_plots/digitize_official2018.py, verified overlay).

Bottom: our median / official median inside the production trusted region
[1000, 4400]: valid from 1000 (anchored background + asymptotics at B ~ 900,
no left-sideband problem by construction), and up to 4400, beyond which the
HybridNew expected band collapses (B_window <= 1.2: nearly every b-only toy
observes n = 0, so all quantiles coincide -- genuine Poisson discreteness,
compounded by 500-toys-per-point quantile noise at 3400-4200) and the
anchored background enters the mcmax deep-tail ambiguity (raw MC ~ 2-3x
above the smooth anchor members). Band edges are SORTED per mass for the
fill: HybridNew toy noise occasionally inverts neighbouring quantiles
(e.g. p1s < med), which is statistical nonsense to draw literally; sorting
displays the same numbers without impossible ordering. Residual caveats:
stat-only, resolved-only vs their combined, LO-HT MC background.

  python compare_production_official.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent                  # production
SIGFIT = HERE.parents[1]                                # signal_fitting
sys.path.insert(0, str(SIGFIT.parent))
from wrplotter.plotting_helpers import custom_log_formatter  # noqa: E402

rows = [r for r in csv.DictReader(
            open(HERE / "refined_limit_table_ee_resolved.csv"))
        if r.get("fb_med")]
rows.sort(key=lambda r: float(r["mWR"]))
m = np.array([float(r["mWR"]) for r in rows])
fb = {k: np.array([float(r[f"fb_{k}"]) for r in rows])
      for k in ("m2s", "m1s", "med", "p1s", "p2s")}
# sort band edges per mass: HybridNew toy noise can invert neighbouring
# quantiles in the collapsed regime; drawing them literally makes impossible
# (crossing) bands. Same numbers, valid ordering. The median is left as-is.
q = np.sort(np.vstack([fb[k] for k in ("m2s", "m1s", "med", "p1s", "p2s")]),
            axis=0)
fb["m2s"], fb["m1s"], fb["p1s"], fb["p2s"] = q[0], q[1], q[3], q[4]

TRUST = (1400.0, 3200.0)

od = list(csv.DictReader(
    open(SIGFIT / "7_limit_plots" / "official2018_expected_digitized.csv")))
om = np.array([float(r["mass_GeV"]) for r in od])
omed = np.array([float(r["med_fb"]) for r in od])
off_at = lambda x: np.exp(np.interp(x, om, np.log(omed)))

hep.style.use("CMS")
fig, (ax, axr) = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1],
                              gridspec_kw={"hspace": 0.06}, figsize=(10, 11))
ax.fill_between(m, fb["m2s"], fb["p2s"], color="#f5d800", label="95% expected")
ax.fill_between(m, fb["m1s"], fb["p1s"], color="#00cc00", label="68% expected")
ax.plot(m, fb["med"], "k:", lw=2,
        label="Expected limit (run2, this work, combine)")
ax.plot(om, omed, color="#e42536", lw=2,
        label="CMS 2018 ee expected, combined")
for xv in TRUST:
    ax.axvline(xv, color="0.35", lw=1.4, ls=(0, (6, 3)))
ax.text(0.5 * sum(TRUST), 2.5e3, "trusted region", color="0.35",
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
ax.legend(loc="lower left", fontsize=14, frameon=False)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=17)

for xv in TRUST:
    axr.axvline(xv, color="0.35", lw=1.4, ls=(0, (6, 3)))
sel = (m >= TRUST[0]) & (m <= TRUST[1])
axr.plot(m[sel], fb["med"][sel] / off_at(m[sel]), "o-", color="#5790fc",
         lw=2, ms=6)
axr.axhline(1, color="grey", ls=":")
axr.set_ylabel("this work /\nCMS 2018", fontsize=14)
axr.set_xlabel(r"$m_{W_R}$ (GeV)")
axr.set_ylim(0.4, 1.4)
axr.grid(alpha=0.3)

out = HERE / "plots" / "ee_resolved"
out.mkdir(parents=True, exist_ok=True)
stem = out / "production_vs_official2018"
fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=150)
print(f"wrote {stem}.pdf/.png")
for mm in (1000, 1400, 2000, 2600, 3200, 4000, 5000):
    i = np.argmin(np.abs(m - mm))
    print(f"  m={mm}: ours={fb['med'][i]:.3f} fb  official={off_at(mm):.3f} fb"
          f"  ratio={fb['med'][i]/off_at(mm):.2f}")
