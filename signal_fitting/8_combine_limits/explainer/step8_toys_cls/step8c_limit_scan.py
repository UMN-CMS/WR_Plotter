#!/usr/bin/env python3
"""Step 8c -- the limit: scan mu until CLs crosses 0.05.

Nothing new happens in this step -- it is step 8b in a loop.  For a ladder of
hypotheses mu, compute toy-based CLs(mu); the 95% CL upper limit is the mu
where the (monotonically falling) curve crosses CLs = 0.05.

CHECK: because our "observed" dataset IS the unfluctuated MC, the toy
crossing should land on combine's OBSERVED AsymptoticLimits number for this
card (22.4 events = 0.920 fb x rate) -- at this mass (B ~ 363 events) toys
and asymptotics agree, so combine's asymptotic value doubles as the toy
reference.  (No HybridNew was run on the k5_bw50 card; the archived 10.6
k3 parity had HybridNew 37.5 +- 0.7 vs asymptotic 36.7 -- same story.)

Also drawn: the asymptotic CLs(mu) curve (step 9a's formulas) whose crossing
is combine -M AsymptoticLimits = 22.4 events.

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python step8c_limit_scan.py [ntoys]
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

ntoys = int(sys.argv[1]) if len(sys.argv) > 1 else 300
m = CardModel(mass=2000)
ref = load_reference(2000)
print(__doc__.split("\n\n")[0])

mus = np.array([12, 16, 19, 22, 25, 28, 32, 38])
print(f"\nscanning {len(mus)} hypotheses, {ntoys} toys/family each")
cls_toy, cls_err = [], []
for i, mu in enumerate(mus):
    r = m.cls_toys(float(mu), ntoys, seed=9000 + 17 * i)
    err = (r["cls"] * np.sqrt(1 / max(r["clsb"] * ntoys, 1)
                              + 1 / max(r["clb"] * ntoys, 1))
           if r["clsb"] > 0 else 0.0)
    cls_toy.append(r["cls"])
    cls_err.append(err)
    print(f"  mu = {mu:5.1f} -> CLs = {r['cls']:.3f} +- {err:.3f}")

# toy crossing by log-linear interpolation
ct = np.array(cls_toy)
pos = ct > 0
ul_toys = float(np.exp(np.interp(np.log(0.05), np.log(ct[pos])[::-1],
                                 np.log(mus[pos])[::-1])))
ul_asym = m.asymptotic_limit()
print(f"\n  toy CLs crossing:        UL = {ul_toys:.1f} events")
print(f"  asymptotic crossing:     UL = {ul_asym:.1f} events")
# our "observed" dataset is the unfluctuated MC, so both crossings are
# compared with combine's OBSERVED AsymptoticLimits number (22.40 events),
# not the median-expected (22.72) -- they differ by the small best-fit
# offset.  (toys vs asymptotic at B ~ 363: same Gaussian regime.)
obs = ref.get("fitdiag", {}).get("asymptotic_observed")
check("toy UL vs combine observed", ul_toys, obs, tol=0.08)
check("asymptotic UL vs combine observed", ul_asym, obs, tol=0.05)

fig, ax = plt.subplots(figsize=(9.5, 6))
ax.errorbar(mus, cls_toy, yerr=cls_err, fmt="o", ms=6, color="#e42536",
            label=f"toy CLs ({ntoys}/family) -- what HybridNew does")
grid = np.linspace(mus[0], mus[-1], 60)
# asymptotic curve for display (observed branch of the step-9a formulas)
from combine_from_scratch import CardModel as _CM     # noqa: E402
asym = []
for mu in grid:
    s = m.sigma_A(float(mu))
    import ROOT
    q, _ = m.qmu_tilde(float(mu))
    if q <= 0:
        asym.append(1.0)
        continue
    sq = np.sqrt(q)
    if q <= (mu / s) ** 2:
        clsb = 1 - ROOT.TMath.Freq(sq)
        clb = 1 - ROOT.TMath.Freq(sq - mu / s)
    else:
        clsb = 1 - ROOT.TMath.Freq((q + (mu / s) ** 2) / (2 * mu / s))
        clb = 1 - ROOT.TMath.Freq((q - (mu / s) ** 2) / (2 * mu / s))
    asym.append(clsb / clb if clb > 0 else 1.0)
ax.plot(grid, asym, color="#5790fc", lw=2,
        label="asymptotic CLs (step 9a) -- what AsymptoticLimits does")
ax.axhline(0.05, color="black", lw=1.2, ls="--")
ax.text(mus[0], 0.052, "CLs = 0.05", fontsize=11)
for x, col, lab in ((ul_toys, "#e42536", f"toys: {ul_toys:.1f}"),
                    (ul_asym, "#5790fc", f"asymptotic: {ul_asym:.1f}")):
    ax.axvline(x, color=col, lw=1.2, ls=":")
    ax.text(x + 0.4, 0.35, lab, color=col, fontsize=10, rotation=90)
ax.set_yscale("log")
ax.set_xlabel(r"tested signal yield $\mu$ [events]")
ax.set_ylabel(r"CLs($\mu$)")
ax.set_title("the limit = where CLs crosses 0.05  "
             "(combine observed reference: 22.4 events)")
ax.legend(fontsize=11)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"step8c_limit_scan.{ext}", dpi=150)
print(f"\nwrote {HERE}/step8c_limit_scan.png")
print("NEXT (9a): why the blue curve needs no toys at all -- the Asimov "
      "dataset and the asymptotic formulas")
