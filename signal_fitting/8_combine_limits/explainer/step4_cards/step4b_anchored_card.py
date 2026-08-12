#!/usr/bin/env python3
"""Step 4b -- the anchored card (m_WR = 4600, B_window = 1.0).

Same three shapes, but the background has NO freedom: the expo slope and the
normalization are CONSTANTS imported from the trusted-spectrum anchor fit
(step 5), evaluated inside this window (B_env). The sidebands here are too
empty to measure anything -- 1-3 populated bins -- so the float card's
freedom would only let the fit trade background against signal (the
Stage-10.1 runaway). Only r floats; the card has no flatParam lines at all.

Note B_env (0.37) vs the windowed raw MC (1.05): the smooth anchor sits
below the jagged single-large-weight MC tail -- the "mcmax" caveat
(README): judged weight noise by Stage 10.4, but the first thing to check
before quoting 5-6 TeV.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
MASS = 4600

hep.style.use("CMS")
comps = C.load_bkg_components()
edges = comps["DYJets"][0]
centers = 0.5 * (edges[:-1] + edges[1:])
total = sum(comps[s][1] for s in C.BKG_SAMPLES)
w = next(x for x in C.load_stage6_windows() if x["mWR"] == float(MASS))
with open(C.ANCHORS) as fh:
    A, b = json.load(fh)["central"]["params"]

sel = (centers >= w["fit_lo"]) & (centers <= w["fit_hi"])
fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
ax.axvspan(w["fit_lo"], w["fit_hi"], color=C.RED, alpha=0.06, zorder=0)
ax.stairs(total[sel], np.append(edges[:-1][sel], edges[1:][sel][-1]),
          color="black", linewidth=2, label="data_obs = summed MC (no data)")
f_anchor = A * np.exp(b * (centers[sel] - 2000.0) / 1000.0)
ax.plot(centers[sel], f_anchor, color=C.RED, lw=2.4,
        label=f"bkg_pdf: expo FIXED from the anchor\n"
              f"(B_env = {f_anchor.sum():.2f}; windowed MC = {w['B_window']:.2f})")
g = np.exp(-0.5 * ((centers[sel] - w["m_c"]) / w["sigma_win"]) ** 2)
ax.plot(centers[sel], g * max(total[sel].max(), f_anchor.max()) * 0.8,
        color="#00cc00", lw=2.2,
        label=r"sig_pdf: Gaussian($m_c$, $\sigma$) fixed; $r$ floats")
ax.axvline(w["m_c"], color="black", linestyle=":", linewidth=1.3)
ax.set_yscale("log")
ax.set_ylim(1e-4, None)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("events / 100 GeV")
ax.set_title(f"anchored card, $m_{{W_R}}$={MASS} "
             f"(B_window={w['B_window']:.1f})", fontsize=15)
ax.legend(fontsize=12, frameon=False, loc="lower left")
ax.grid(alpha=0.3)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step4b_anchored_card")

card = C.REFINED / "cards" / C.TAG / f"card_anchored_m{MASS}.txt"
print(f"\n===== {card.name} =====\n{card.read_text()}")
print("next: step5 -- where the anchor comes from")
