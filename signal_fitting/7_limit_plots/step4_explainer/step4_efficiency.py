#!/usr/bin/env python3
"""Step 4a -- the signal efficiency for WR2400_N1200 (ee, resolved), shown on
the SAME 100 GeV binning the S+B fit uses.

The fit rebins to 100 GeV and keeps bins whose CENTRE lies in [fit_lo, fit_hi]
= [1775.5, 2851.4], i.e. the effective window is the 11 bins spanning
[1800, 2900]. The efficiency uses the signal yield in exactly those bins:

    eff = S_fit / genEventSumw

(For reference, integrating the native 10 GeV histogram over the exact
[1775.5, 2851.4] gives 8784.5 vs 8754.2 here -- a 0.3% window-snapping effect.)

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent            # step4_explainer
SIGF = HERE.parents[1]                             # signal_fitting
sys.path.insert(0, str(HERE.parents[2]))           # repo root
sys.path.insert(0, str(SIGF / "4_background_fits"))
sys.path.insert(0, str(SIGF / "shared"))
sys.path.insert(0, str(HERE.parents[0]))           # 7_limit_plots (xsec_limit)

from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root                # noqa: E402
from bkg_fit_lib import MASS_VAR, load_summed_background                 # noqa: E402
from measure_fwhm import (                                               # noqa: E402
    build_hist_key, build_region_name, load_and_combine_signal, parse_masses)
from xsec_limit import integral_in_range, default_signal_config          # noqa: E402

CH, TOPO = "ee", "resolved"
TAG = "WR2400_N1200"
FIT_LO, FIT_HI = 1775.5, 2851.4                    # Stage-6 window (m_c +- 3sig)
SIG_ERA, SIG_DIR = "RunIISummer20UL18", "20260624_signals"
BKG_ERA, BKG_DIR = "RunIII2024Summer24", "20260317_lo_dy"

info = load_lumi(BKG_ERA)
LUMI, COM = info["lumi"], info.get("com", 13.6)

# native (10 GeV) signal
sig_dirs, _ = input_dirs_for_era(SIG_ERA, repo_root(), SIG_DIR)
hist_key = build_hist_key(build_region_name(CH, TOPO), MASS_VAR[TOPO])
se, sv, _ = load_and_combine_signal(sig_dirs, hist_key, TAG)
sc = 0.5 * (se[:-1] + se[1:])

# the fit's 100 GeV edges, and the signal rebinned onto them
bkg_dirs, _ = input_dirs_for_era(BKG_ERA, repo_root(), BKG_DIR)
be, _, _ = load_summed_background(bkg_dirs, f"wr_{CH}_{TOPO}_sr", MASS_VAR[TOPO], 10)
on100, _ = np.histogram(sc, bins=be, weights=sv)
bc = 0.5 * (be[:-1] + be[1:])
infit = (bc >= FIT_LO) & (bc <= FIT_HI)            # fit's bin selection (by centre)
win_lo, win_hi = be[:-1][infit][0], be[1:][infit][-1]   # 1800, 2900

# config
with open(default_signal_config(SIG_ERA)) as fh:
    sig_cfg = {v["dataset"]: v for v in json.load(fh).values()}
mwr, mn = parse_masses(TAG)
cfg = sig_cfg[f"WRtoNLtoLLJJ_MWR{mwr}_MN{mn}"]
sumw, xsec = float(cfg["genEventSumw"]), float(cfg["xsec"])

s_fit = float(on100[infit].sum())                  # fit-consistent S_fit
s_fine = integral_in_range(se, sv, FIT_LO, FIT_HI)  # exact-window reference
eff = s_fit / sumw
print(f"S_fit(100GeV, [{win_lo:.0f},{win_hi:.0f}])={s_fit:.1f}  "
      f"S_fine(exact)={s_fine:.1f}  eff={eff:.4f}")

# ---- plot -----------------------------------------------------------------
hep.style.use("CMS")
fig, ax = plt.subplots()
ax.step(be, np.append(on100, on100[-1]), where="post", color="#7a1fa2", lw=2.0)
ax.bar(bc[infit], on100[infit], width=100, color="#7a1fa2", alpha=0.35,
       label=fr"in window: $S_{{\rm fit}}={s_fit:.0f}$")
for xv in (win_lo, win_hi):
    ax.axvline(xv, color="grey", lw=1.4, ls="--")

ax.set_xlabel(r"$m_{\ell\ell jj}$  [GeV]  (100 GeV bins, as in the fit)")
ax.set_ylabel("signal MC (raw genWeight) / 100 GeV")
ax.set_xlim(1000, 3400)
ax.set_ylim(0, max(on100) * 1.28)
ax.text(0.03, 0.97,
        "ee  Resolved\n"
        r"WR2400_N1200  ($m_{W_R}=2400$, $m_N=1200$)" "\n"
        fr"fit window (100 GeV): $[{win_lo:.0f},\,{win_hi:.0f}]$" "\n"
        fr"$S_{{\rm fit}}={s_fit:.0f}$,  genEventSumw $={sumw:.0f}$" "\n"
        fr"$\mathbf{{eff = S_{{fit}}/genEventSumw = {eff:.3f}}}$",
        transform=ax.transAxes, va="top", fontsize=12)
ax.legend(loc="upper right", fontsize=12)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              com=COM, fontsize=15)
base = str(HERE / "step4_efficiency")
fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
print("wrote step4_efficiency.{png,pdf}")
