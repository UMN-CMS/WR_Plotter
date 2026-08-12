#!/usr/bin/env python3
"""Step 7 -- profiling: how the background gets its say.

Step 6d quietly re-fit the background (b, B) at every tested signal yield r.
This step shows what that does and why it matters.  Compare two curves:

  FIXED     -2lnL(r, b*, B*) - min      background frozen at its global
                                        best fit -- the naive 1-D slice
  PROFILED  -2lnL(r, b-hat(r), B-hat(r)) - min
                                        background re-optimized at each r --
                                        what combine (and step 1d) does

The profiled curve is WIDER: when you ask for more signal, the fit is allowed
to bend the background down to partially compensate, so the data objects less.
That extra width IS the background systematic, automatically included -- this
is how ANY nuisance parameter enters combine: not as an error bar added in
quadrature afterwards, but as a re-minimization at every hypothesis.

The inset shows the compensation directly: the best-fit background yield
B-hat(r) falls as r rises (the signal eats background events).

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python step7_profiling.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from combine_from_scratch import CardModel

m = CardModel(mass=2000)
print(__doc__.split("\n\n")[0])

(r_hat, b_star, B_star), _, nll_min = m.fit_global()
rs = np.linspace(r_hat - 45, r_hat + 60, 90)
fixed = np.array([m.nll2(r, b_star, B_star) - nll_min for r in rs])
prof_pts = [m.fit_profiled(r) for r in rs]
prof = np.array([p[2] - nll_min for p in prof_pts])
B_of_r = np.array([p[0][1] for p in prof_pts])

def width(d):
    l = float(np.interp(1.0, d[rs < r_hat][::-1], rs[rs < r_hat][::-1]))
    h = float(np.interp(1.0, d[rs > r_hat], rs[rs > r_hat]))
    return 0.5 * (h - l)

print(f"\n  half-width at Delta=1:  fixed bkg = {width(fixed):.2f} events, "
      f"profiled = {width(prof):.2f} events")
print(f"  ratio = {width(prof)/width(fixed):.3f} -- the background freedom "
      "widens the signal error; that IS the background systematic")

fig, ax = plt.subplots(figsize=(9.5, 6.5))
ax.plot(rs, fixed, color="#9c9ca1", lw=2, ls="--",
        label=fr"background FROZEN at global best fit "
              fr"($\pm1\sigma$ = {width(fixed):.1f})")
ax.plot(rs, prof, color="#e42536", lw=2.2,
        label=fr"background RE-FIT at each $r$ (profiled, "
              fr"$\pm1\sigma$ = {width(prof):.1f})")
ax.axhline(1.0, color="grey", lw=1.0, ls="--")
ax.axvline(r_hat, color="black", lw=1.0, ls=":")
ax.set_xlabel(r"signal yield hypothesis $r$ [events]")
ax.set_ylabel(r"$\Delta(-2\ln L)$")
ax.set_ylim(0, 8)
ax.set_title("profiling = letting every nuisance re-optimize at each "
             "hypothesis")
ax.legend(fontsize=11, loc="upper center")

ins = ax.inset_axes([0.62, 0.30, 0.35, 0.30])
ins.plot(rs, B_of_r, color="#5790fc", lw=2)
ins.set_xlabel("$r$", fontsize=9)
ins.set_ylabel(r"$\hat{B}(r)$", fontsize=9)
ins.tick_params(labelsize=8)
ins.set_title("background compensates", fontsize=9)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"step7_profiling.{ext}", dpi=150)
print(f"\nwrote {HERE}/step7_profiling.png")
print("NEXT (8a): turn this curve into the ONE number per hypothesis that "
      "combine actually thresholds -- the test statistic q~_mu")
