#!/usr/bin/env python3
"""Step 11a -- setting: the observation (no-data vs data-seeded).

Identical Stage-9-style float cards (k3/100 GeV/floating), the ONLY
difference being what sits in data_obs: the summed MC (the 10.9 scan
baseline) vs the real EGamma data (the Stage-9 data-observed table).
AsymptoticLimits seeds its expected-band Asimov from the background-only fit
to the observation -- the one door through which data can enter an expected
limit. The 2018 ee data sits above this MC in the 3-5 TeV sidebands, so the
data-seeded expected is up to ~35% weaker there; at ~2 TeV (data/MC ~ 1) the
two agree.

(The comparison uses the OLD card so that only the observation differs; the
adopted optimized float card is data-free by the same mechanism, plus its
slope constraint further reduces what the observation could do. Anchored
cards have no background freedom at all -- observation-independent by
construction.)
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
mc = C.load_opt_table()["k3_bw100_float"]        # MC-Asimov, old float card
base = C.load_stage9()                           # data-observed, same card

mm = [m for m in sorted(mc) if m in base]
ratio = [mc[m] / float(base[m]["comb_fb_med"]) for m in mm]

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(mm, ratio, "o-", color=C.BLUE, lw=2, ms=6)
ax.axhline(1, color="grey", ls=":")
ax.set_xlabel(r"$m_{W_R}$ (GeV)")
ax.set_ylabel("expected median:  no-data / data-seeded")
ax.set_title("same float cards; only data_obs differs", fontsize=15)
ax.grid(alpha=0.3)
ax.text(0.04, 0.08, "data/MC $\\approx$ 1 below 2.2 TeV;\n"
        "data excess in 3-5 TeV sidebands\nweakens the data-seeded expected",
        transform=ax.transAxes, fontsize=13, va="bottom")
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step11a_observation")
print("next: step11b -- floating vs anchored background at the edges")
