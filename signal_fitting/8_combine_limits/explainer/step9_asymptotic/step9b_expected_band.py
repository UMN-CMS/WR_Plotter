#!/usr/bin/env python3
"""Step 9b -- the expected (Brazil) band, and where the shortcut breaks.

The expected band answers: "if there is NO signal, what limits would this
experiment typically set?"  Concretely: imagine many background-only outcomes,
compute each one's limit, and report the 2.5/16/50/84/97.5 % quantiles of
those limits.  Step 9a gave the shortcut: each background-only quantile N of
mu-hat maps through the asymptotic formulas to one band edge -- that is
combine -M AsymptoticLimits' "Expected" block, which this step reproduces at
m = 2000 (all five numbers).

The same construction explains BOTH earlier methods:
  * Stage 7's closed-form band = this, with sigma taken from the toy RMS
    instead of sigma_A;
  * Stage 10.5's per-toy quantile band = the toy version of the same map
    (each toy = one background-only outcome, each UL_i = its limit).

WHERE IT BREAKS: the asymptotic map assumes mu-hat is Gaussian.  At the
sparse masses the window holds a handful of events, mu-hat is quasi-discrete,
and the 10.6 anchored-card comparison measured the failure directly:
AsymptoticLimits 2.28 vs HybridNew (toys) 3.25 at m = 4600 -- 30% too
aggressive.  That is why Stage 10.5 switches to the exact counting band and
why combine's own reference there is HybridNew.

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python step9b_expected_band.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from combine_from_scratch import CardModel, load_reference, check

m = CardModel(mass=2000)
ref = load_reference(2000).get("asymptotic", {})
print(__doc__.split("\n\n")[0])

quants = [(0.025, "ul_m2s"), (0.16, "ul_m1s"), (0.5, "ul_med"),
          (0.84, "ul_p1s"), (0.975, "ul_p2s")]
print(f"\n  {'quantile':>9} {'ours':>8} {'combine':>9}")
ours = {}
for q, key in quants:
    ours[key] = m.asymptotic_limit(quantile=q)
    print(f"  {q:9g} {ours[key]:8.2f} {ref.get(key, float('nan')):9.2f}")
for q, key in quants:
    check(f"expected {q:g} vs combine", ours[key], ref.get(key), tol=0.07)
print("  (a uniform ~+1-2% offset = the sigma_A evaluation-convention "
      "residual from steps 6d/9a;\n   the band SHAPE -- every quantile ratio "
      "to the median -- matches combine exactly)")
print(f"  shape check: ours m2s/med={ours['ul_m2s']/ours['ul_med']:.3f} "
      f"p2s/med={ours['ul_p2s']/ours['ul_med']:.3f}  |  combine "
      f"{ref['ul_m2s']/ref['ul_med']:.3f} / {ref['ul_p2s']/ref['ul_med']:.3f}")

# figure: the five band points as the familiar Brazil strip at one mass,
# combine's numbers overlaid; inset = the sparse-mass breakdown from 10.6
fig, ax = plt.subplots(figsize=(9, 6))
x0 = 0.0
ax.bar([x0], [ours["ul_p2s"] - ours["ul_m2s"]], bottom=[ours["ul_m2s"]],
       width=0.5, color="#f5d800", label=r"$\pm2\sigma$ expected")
ax.bar([x0], [ours["ul_p1s"] - ours["ul_m1s"]], bottom=[ours["ul_m1s"]],
       width=0.5, color="#00cc00", label=r"$\pm1\sigma$ expected")
ax.plot([x0 - 0.25, x0 + 0.25], [ours["ul_med"]] * 2, "k--", lw=2,
        label="median expected (from scratch)")
for key, qlab in (("ul_m2s", "2.5%"), ("ul_m1s", "16%"), ("ul_med", "50%"),
                  ("ul_p1s", "84%"), ("ul_p2s", "97.5%")):
    if key in ref:
        ax.plot([x0 + 0.32], [ref[key]], "o", ms=7, color="#e42536")
        ax.text(x0 + 0.36, ref[key], f"combine {qlab}: {ref[key]:.1f}",
                fontsize=9, va="center")
ax.set_xlim(-0.6, 1.4)
ax.set_xticks([])
ax.set_ylabel(r"95% CL UL on $N_{\rm sig}$ [events]")
ax.set_title("one mass point of the Brazil band, rebuilt from scratch "
             "(m$_{W_R}$ = 2000)")
ax.legend(fontsize=10, loc="upper left")
ax.text(0.02, 0.02,
        "where this breaks: sparse masses.\n"
        "10.6 anchored card, m=4600:  asymptotic 2.28  vs  HybridNew toys "
        "3.25\n-> at low counts, only toys (or the exact counting band) are "
        "valid.",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", fc="#fff4f4", ec="#e42536"))

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"step9b_expected_band.{ext}", dpi=150)
print(f"\nwrote {HERE}/step9b_expected_band.png")
print("\nDONE with the internals. The chain: bins+shapes (6a) -> per-bin "
      "Poisson (6b) -> product (6c) -> likelihood curve (6d) -> profiling (7) "
      "-> q~_mu (8a) -> toy CLs (8b) -> limit (8c) -> Asimov shortcut (9a) "
      "-> band (9b).")
print("NEXT (10): back to the analysis -- where the asymptotic shortcut "
      "breaks on the real cards, and why the sparse masses use toys")
