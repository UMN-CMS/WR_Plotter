#!/usr/bin/env python3
"""Step 11e -- setting: the float-card configuration (the Stage-10.9 scan).

The 2-3.2 TeV expected limit vs the old k3_bw100_float baseline, decomposed
into the scan's individual levers (each an expected-median ratio, identical
otherwise):

  k5 window alone          ~0.74   more sideband lever arm for the slope+norm
  slope constraint alone   ~0.72   the anchor measurement imported (`param`)
  50/20 GeV binning alone  ~1.02   nothing -- 100 GeV already resolves the peak
  k5 + 50 GeV + slope constraint          ~0.60  (the two levers compound)
  + mu,sigma priors ('both030')           ~0.62  (+3.5%, step 7d;
                                                  ADOPTED at <= 2600 only)
  fully anchored bound (norm fixed too)   ~0.59  (upper bound on this road)

The production card sits essentially at the background-knowledge bound while
keeping the norm locally measured; the shape uncertainty is in-model at
1400-2600 and an offline systematic at 2800-3200 (the both030 toy
re-validation collapsed convergence there: 84/51/32% vs 99/88/66% fixed).
Validation of the slope-constrained core (10.9 FitDiagnostics toys): null
spurious < 2% of the medians, injection at the sensitivity edge recovered
to 1-5%, pull RMS ~ 1 (soft spots: 1.17 at 2800; 66% convergence at 3200).
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
opt = C.load_opt_table()
base = opt["k3_bw100_float"]

SHOW = [("k3_bw50_float", "#94a4a2", ":", "50 GeV bins alone"),
        ("k5_bw100_float", "#5790fc", "--", "k5 window alone"),
        ("k3_bw100_bconstr", "#f89c20", "-.", "slope constraint alone"),
        ("k5_bw50_bconstr", "#e42536", "-", "k5 + 50 GeV + slope constraint"),
        ("k5_bw50_bconstr_both030", "#00aa00", "-",
         r"+ $\mu,\sigma$ priors (adopted $\leq$2600)"),
        ("k3_bw100_anch", "black", ":", "fully anchored (bound)")]

fig, ax = plt.subplots(figsize=(9.5, 7.5))
for key, color, ls, lab in SHOW:
    v = opt[key]
    mm = sorted(m for m in v if m in base)
    ax.plot(mm, [v[m] / base[m] for m in mm], marker="o", ms=4, lw=2,
            ls=ls, color=color, label=lab)
ax.axhline(1, color="grey", ls=":")
ax.set_xlabel(r"$m_{W_R}$ (GeV)")
ax.set_ylabel("expected median / old float baseline")
ax.set_ylim(0.4, 1.15)
ax.grid(alpha=0.3)
ax.legend(fontsize=12, frameon=False, loc="lower left", ncol=2)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step11e_float_optimization")
print("next: step12 -- assemble the final no-data limit")
