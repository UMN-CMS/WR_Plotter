#!/usr/bin/env python3
"""Step 2c -- the per-channel efficiency and the card rate (k=5, all masses).

For every grid mass, the k5 window ([m_c - 5 sigma, m_c + 5 sigma], the
optimized-card geometry of 2a/2b) holds a fraction of the raw-genWeight signal:

    eff = S_window / (0.5 x genEventSumw)

the in-window signal over HALF the generated sum -- the 0.5 (channel-bfrac)
removes the muon share of the flavor-mixed WRtoNLtoLLJJ samples (measured
50/50 e:mu from the GenModel N-flavor genWeight shares; no tau). This makes
eff a genuine per-channel efficiency (0.20 -> 0.45 plateau, matching the 2018
analysis), and the card

    rate = lumi x eff   (events per fb of sigma x BR)

turns combine's POI into  r == sigma x BR(eeqq') in fb  -- the official
y-axis. Without the 0.5 the same numbers would be a limit on the TOTAL
sigma x BR(lljj, e+mu): a factor-2 mislabel (mu is unaffected).

At m=2000: eff=0.407, rate=24.3 ev/fb -- the numbers on the 2a card.

Efficiencies are read from `k5_efficiency_ee_resolved.csv` (this directory),
produced by
  7_limit_plots/window_optimization/efficiency_vs_k.py \
      --k-grid 5.0 --mass-min 1000 --mass-max 6000 --bin-width 50 --no-mn-spread
i.e. the same signal_efficiency the cards use, at the k5/50 GeV geometry.

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")

rows = sorted((r for r in csv.DictReader(open(HERE / "k5_efficiency_ee_resolved.csv"))),
              key=lambda r: float(r["mWR"]))
masses = [float(r["mWR"]) for r in rows]
eff = [float(r["eff"]) for r in rows]
lumi = C.load_meta()["lumi"]
rate = [lumi * e for e in eff]

fig, ax = plt.subplots(figsize=(9, 7.5))
i2000 = masses.index(2000.0)
ax.plot(masses, eff, "^-", color="#f89c20", lw=2, ms=6,
        label=r"$\varepsilon_{ee}$  ($k=5$ window, 50 GeV)" "\n"
              r"$M_N = M_{W_R}/2$ (half-N diagonal)")
ax.plot([2000], [eff[i2000]], "o", ms=11, mfc="none", mec="black", mew=1.6)
ax.set_xlabel(r"$m_{W_R}$ (GeV)")
ax.set_ylabel("Signal Efficiency")
ax.set_xlim(900, 6100)
ax.set_ylim(0, 0.8)
ax.grid(alpha=0.3)

formula = "\n".join([
    r"$\varepsilon = S_{\mathrm{window}}\,/\,(0.5\,\Sigma w_{\mathrm{gen}})$",
    rf"rate $= L\,\varepsilon = 59.8\times{eff[i2000]:.3f} = {rate[i2000]:.1f}$ ev/fb",
])
ax.text(0.03, 0.96, formula, transform=ax.transAxes, fontsize=14, va="top")
ax.legend(loc="upper right", fontsize=14, frameon=False)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step2c_efficiency")
print(f"m=2000: eff={eff[i2000]:.4f}, rate={rate[i2000]:.2f} ev/fb")
print("next: step3 -- the regime map")
