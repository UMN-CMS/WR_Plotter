#!/usr/bin/env python3
"""Step 3d -- the step-1a component stack with the m_WR = 2000 window shaded and
the fitted exponential drawn inside it. Seminar version of step 3a: no text about
slope constraints or the card, just stack + window + fit.

The exponential uses the window's own free fit (slope B_FREE from step 3a) and is
normalized to the summed-MC count in the window, like step 3a.

Made 2026-10-03 for the HEP seminar (research-slides/talks/20261007_hep_seminar).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
hep.style.use("CMS")

M_C, _, LO, HI = C.window_2000()

comps = C.load_bkg_components(rebin=5)
edges = comps["DYJets"][0]
centers = 0.5 * (edges[:-1] + edges[1:])
colors = {"Other": "#94a4a2", "Nonprompt": "#964a8b", "tt_tW": "#f89c20", "DYJets": "#5790fc"}
order = ["Other", "Nonprompt", "tt_tW", "DYJets"]
total = np.maximum(sum(comps[s][1] for s in C.BKG_SAMPLES), 0.0)
sel = (centers >= LO) & (centers <= HI)
B = float(total[sel].sum())
# the window's own free-slope fit: binned Poisson likelihood of a single
# exponential (normalized to B) to the summed MC in the window
from scipy.optimize import minimize_scalar


def _nll(b):
    e = np.exp(b * (centers[sel] - M_C) / 1000.0)
    nu = np.clip(B * e / e.sum(), 1e-9, None)
    return float((nu - total[sel] * np.log(nu)).sum())


B_FREE = float(minimize_scalar(_nll, bounds=(-15.0, 0.0), method="bounded").x)
shape = np.exp(B_FREE * (centers[sel] - M_C) / 1000.0)
mfine = np.linspace(LO, HI, 400)
curve = B / shape.sum() * np.exp(B_FREE * (mfine - M_C) / 1000.0)

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.stackplot(centers, *[comps[s][1] for s in order], step="mid",
             colors=[colors[s] for s in order], labels=[C.BKG_LABELS[s] for s in order])
handles, labels = ax.get_legend_handles_labels()
ax.axvspan(LO, HI, color="tab:red", alpha=0.12, zorder=4)
win_proxy = Patch(facecolor="tab:red", alpha=0.12, label=rf"window $[{LO:.0f}, {HI:.0f}]$ GeV")
(fit_line,) = ax.plot(mfine, curve, color="black", linewidth=2.6, zorder=6,
                      label="exponential fit in the window")
_, _tot, b_up, b_dn, has_syst = C.load_bkg_band(rebin=5)
ax.fill_between(centers, np.clip(_tot - b_dn, 1e-4, None), _tot + b_up, step="mid",
                facecolor="none", edgecolor="#555555", hatch="/////", linewidth=0.0, zorder=5)
band_proxy = Patch(facecolor="none", edgecolor="#555555", hatch="/////",
                   label=(r"MC stat $\oplus$ syst" if has_syst else "MC stat. unc."))
ax.set_yscale("log")
ax.set_xlim(800, 6000)
ax.set_ylim(1e-3, 1e4)
C.log_yaxis_one_ten(ax)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("Events / 50 GeV")
ax.legend(handles[::-1] + [band_proxy, win_proxy, fit_line],
          labels[::-1] + [band_proxy.get_label(), win_proxy.get_label(), fit_line.get_label()],
          loc="upper right", fontsize=15, frameon=False)
ax.text(0.03, 0.96, f"ee\nResolved SR\n{C.MC_LABEL}", transform=ax.transAxes, fontsize=16, va="top")
C.cms_label(ax)
C.savefig(fig, HERE / "step3d_window_fit_on_stack")
print(f"window [{LO:.0f},{HI:.0f}]: B = {B:.1f} events, slope {B_FREE}/TeV")
