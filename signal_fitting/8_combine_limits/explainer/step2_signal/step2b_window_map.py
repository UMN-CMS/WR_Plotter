#!/usr/bin/env python3
"""Step 2b -- the window map: mu(m_WR) +- k sigma(m_WR) across the grid.

The Stage-2 LINEAR parameterization gives (mu, sigma) at ANY mass -- the same
recipe for every card, no per-point fits. Two window widths coexist:

  k = 3 (blue)   the Stage-6 geometry: anchored cards (1000-1200, >= 3400)
                 and all the toy/spurious machinery upstream.
  k = 5 (orange) the OPTIMIZED float cards (1400-3200, Stage 10.9): the wider
                 sidebands are one of the two ~27% sensitivity levers (the
                 other is the anchor-constrained slope). expo tolerates the
                 widening (k-sweep: powlaw does not -- its mismodeling
                 spurious grows).

The clamps are visible: the 800 GeV selection floor swallows the left
sideband of m = 1000-1200 (-> the anch_low regime) and bites the k5 windows
up to ~1800; the 6000 GeV edge truncates the last windows.
"""
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
wins = C.load_stage6_windows()
m = [w["mWR"] for w in wins]

opt = C.load_opt_inputs()
mf = sorted(int(k) for k in opt if C.FLOAT_MIN <= float(k) <= C.FLOAT_MAX)
k5lo = [opt[str(k)]["vars"][C.FLOAT_VAR]["fit_lo"] for k in mf]
k5hi = [opt[str(k)]["vars"][C.FLOAT_VAR]["fit_hi"] for k in mf]

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.fill_between(m, [w["fit_lo"] for w in wins], [w["fit_hi"] for w in wins],
                color=C.BLUE, alpha=0.25,
                label=r"$\mu \pm 3\sigma$ (Stage-6 geometry / anchored cards)")
ax.fill_between(mf, k5lo, k5hi, color="#f89c20", alpha=0.20,
                label=r"$\mu \pm 5\sigma$ (optimized float cards, 10.9)")
ax.plot(m, [w["m_c"] for w in wins], color=C.BLUE, lw=2,
        label=r"$\mu(m_{W_R})$ (Stage-2 linear)")
ax.plot(m, m, color="grey", ls="--", lw=1, label=r"$m_{W_R}$ diagonal")
for y, lab in ((800, "800 GeV selection floor"), (6000, "6000 GeV edge")):
    ax.axhline(y, color=C.RED, ls=(0, (6, 3)), lw=2)
ax.text(3500, 900, "clamp: m=1000-1200 lose their left sideband",
        fontsize=12, color=C.RED)
ax.set_xlabel(r"$m_{W_R}$ (GeV)")
ax.set_ylabel("window edge (GeV)")
ax.set_xlim(900, 6100)
ax.legend(fontsize=13, frameon=False, loc="upper left")
ax.grid(alpha=0.3)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              com=13, fontsize=15)
C.savefig(fig, HERE / "step2b_window_map")
print("next: step2c -- the rate that makes r a cross section")
