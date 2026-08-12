#!/usr/bin/env python3
"""Step 11d -- setting: the signal shape (fixed vs param-constrained).

TWO measurements on the run2 background (Stage-10.9 scan), same masses, same
prior widths -- only the background treatment differs:

  * on the FLOATING-background card (k3_bw100_float), adding the 0.3*sigma0
    sigma prior costs ~+22% (geometric mean; the run2 reproduction of the
    Stage-10.6 run3 result, +35% at 2 TeV falling to +7% at 3.2 TeV);
  * on the SLOPE-CONSTRAINED winner (k5_bw50_bconstr), constraining BOTH mu
    and sigma at 0.3*sigma0 costs only ~+3.5%.

Same prior, factor ~6 different price: the old cost was never the prior --
it was the signal<->background DEGENERACY. A floating background can trade
yield with a stretchable Gaussian; once the slope carries the anchor
constraint there is nothing to trade, and profiling the shape is nearly
free. Hence the adopted float cards at 1400-2600 ('both030') carry the M_N
width variation and the peak-position uncertainty as in-model nuisances.
Float cards at 2800-3200 keep the shape fixed -- the toy re-validation
showed the shape nuisances collapse convergence there (84/51/32% vs
99/88/66%) while costing limit -- as do the anchored cards (untested +
HybridNew toys pay per nuisance).
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

fig, ax = plt.subplots(figsize=(9, 7))
base, sig = opt["k3_bw100_float"], opt["k3_bw100_float_sig030"]
mm = sorted(m for m in sig if m in base)
ax.plot(mm, [sig[m] / base[m] for m in mm], "o-", color="#964a8b", lw=2, ms=6,
        label=r"floating bkg: $\sigma$-prior / fixed")
win, both = opt["k5_bw50_bconstr"], opt["k5_bw50_bconstr_both030"]
mm = sorted(m for m in both if m in win)
ax.plot(mm, [both[m] / win[m] for m in mm], "s-", color="#00cc00", lw=2, ms=6,
        label=r"slope-constrained bkg: $\mu$+$\sigma$-prior / fixed")
ax.axhline(1, color="grey", ls=":")
ax.set_xlabel(r"$m_{W_R}$ (GeV)")
ax.set_ylabel("expected median ratio (constrained / fixed shape)")
ax.set_title("the shape-prior price is the background degeneracy", fontsize=15)
ax.set_ylim(0.9, 1.5)
ax.grid(alpha=0.3)
ax.text(0.04, 0.95, "same $0.3\\sigma_0$ prior, same masses;\n"
        "only the background treatment differs",
        transform=ax.transAxes, fontsize=13, va="top")
ax.legend(fontsize=13, frameon=False, loc="center right")
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step11d_signal_shape")
print("next: step11e -- the float-region optimization that changed the card")
