#!/usr/bin/env python3
"""Step 4a -- the OPTIMIZED float card (m_WR = 2000, k5 window, 50 GeV bins).

The card is three shapes + one number:
  data_obs   the summed MC in the mu +- 5 sigma window at 50 GeV binning
             (blue stairs region -- no data)
  bkg_pdf    exp(b*(m-m_c)/1000): the norm is FREE (flatParam, measured by
             this window's own sidebands) but the slope b is `param`-
             CONSTRAINED to the trusted-spectrum anchor fit (b = -2.94 +-
             fit error) -- one of the two ~27% sensitivity levers (10.9)
  sig_pdf    Gaussian(m_c, sigma); at THIS mass (<= 2600) mu and sigma are
             `param`-constrained at 0.3*sigma0 ('both030') -- nearly free
             (+3.5%) now that the slope carries a constraint, so the M_N
             width variation lives inside the model. Float cards at
             2800-3200 keep the shape FIXED (the toy re-validation showed
             the shape nuisances collapse convergence in those sparser
             windows -- see the README)
  rate       lumi x eff (recomputed for THIS window/binning) -> r is
             sigma x BR(eeqq') in fb

Combine profiles (r, b, norm, mu_sig, sigma_sig) with the three Gaussian
penalty terms from the `param` lines. The printed card below is the real
generated file from the Stage-10.8 refined chain.
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
MASS = 2000

hep.style.use("CMS")
comps = C.load_bkg_components(rebin=5)          # 50 GeV bins
edges = comps["DYJets"][0]
centers = 0.5 * (edges[:-1] + edges[1:])
total = sum(comps[s][1] for s in C.BKG_SAMPLES)
opt = C.load_opt_inputs()[str(MASS)]
v = opt["vars"][C.FLOAT_VAR]
m_c, sigma = opt["m_c"], opt["sigma"]
with open(C.ANCHORS) as fh:
    anc = json.load(fh)["central"]
A, b = anc["params"]

sel = (centers >= v["fit_lo"]) & (centers <= v["fit_hi"])
fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
ax.axvspan(v["fit_lo"], v["fit_hi"], color=C.RED, alpha=0.06, zorder=0)
ax.stairs(total[sel], np.append(edges[:-1][sel], edges[1:][sel][-1]),
          color="black", linewidth=1.6, label="data_obs = summed MC (no data)")
f_expo = A * np.exp(b * (centers[sel] - 2000.0) / 1000.0) * 0.5   # per 50 GeV
ax.plot(centers[sel], f_expo, color=C.BLUE, ls="--", lw=2.2,
        label=("bkg_pdf: expo, norm FREE,\n"
               r"slope $b$ param-constrained to the anchor"))
g = np.exp(-0.5 * ((centers[sel] - m_c) / sigma) ** 2)
ax.plot(centers[sel], g * total[sel].max() * 0.8, color=C.RED, lw=2.4,
        label=(r"sig_pdf: Gaussian($\mu$, $\sigma$),"
               "\n" r"both param-constrained at $0.3\sigma_0$; $r$ floats"))
ax.axvline(m_c, color="black", linestyle=":", linewidth=1.3)
ax.set_yscale("log")
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("events / 50 GeV")
ax.set_title(f"optimized float card, $m_{{W_R}}$={MASS} "
             f"($k$=5 window, B={v['B_window']:.0f})", fontsize=15)
ax.legend(fontsize=11, frameon=False, loc="lower left")
ax.grid(alpha=0.3)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step4a_float_card")

card = C.REFINED / "cards" / C.TAG / f"card_float_m{MASS}.txt"
print(f"\n===== {card.name} =====\n{card.read_text()}")
print("next: step4b -- the anchored card, where nothing but r floats")
