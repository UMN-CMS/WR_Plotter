#!/usr/bin/env python3
"""Step 8b -- toys and CLs: the two distributions behind every combine limit.

Fix ONE signal hypothesis mu (we use a value near the limit).  Throw two
families of pseudo-experiments and compute q~_mu for every one:

  S+B toys   Poisson draws of  nu_i(mu, theta-hat-hat(mu))   "the universe
             where the signal exists at exactly size mu"
  B-only     Poisson draws of  nu_i(0,  theta-hat-hat(0))    "the universe
             with no signal"

Then place the OBSERVED q~_mu(obs) on both histograms:

    CLs+b = P( q~ >= q_obs | s+b )     "how often would the signal universe
                                        look at least this signal-free?"
    1-CLb = P( q~ >= q_obs | b )       "...and the no-signal universe?"
    CLs   = CLs+b / (1-CLb)

Excluding at 95% CL means CLs <= 0.05.  Dividing by (1-CLb) is the CLs tweak
from the old explainer (section 3 of limit_plots_guide.md), in its general
form: when the experiment has no real sensitivity to mu, the two histograms
lie on top of each other, the ratio -> 1, and no exclusion is possible --
downward fluctuations can't exclude a signal you couldn't have seen.

This picture IS combine -M HybridNew, and the old "family of shifted
Gaussians" figure was exactly this with q~ replaced by a Gaussian yield.

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python step8b_toys_cls.py [mu] [ntoys]
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

mu = float(sys.argv[1]) if len(sys.argv) > 1 else 23.0
ntoys = int(sys.argv[2]) if len(sys.argv) > 2 else 400
m = CardModel(mass=2000)
print(__doc__.split("\n\n")[0])
print(f"\nhypothesis mu = {mu}, {ntoys} toys per family (seed 4242)")

res = m.cls_toys(mu, ntoys, seed=4242)
print(f"  q~_obs           = {res['q_obs']:.3f}")
print(f"  CLs+b            = {res['clsb']:.3f}   (S+B tail beyond q_obs)")
print(f"  1-CLb            = {res['clb']:.3f}   (B-only tail beyond q_obs)")
print(f"  CLs = ratio      = {res['cls']:.3f}   (exclude when <= 0.05)")

fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(0, max(np.percentile(res["q_sb"], 99), res["q_obs"] * 1.3,
                          8), 40)
for arr, col, lab in ((res["q_sb"], "#e42536", "S+B toys"),
                      (res["q_b"], "#5790fc", "B-only toys")):
    ax.hist(np.clip(arr, bins[0], bins[-1]), bins=bins, histtype="step",
            lw=2, color=col, label=lab)
    tail = np.mean(arr >= res["q_obs"])
    ax.hist(np.clip(arr[arr >= res["q_obs"]], bins[0], bins[-1]), bins=bins,
            color=col, alpha=0.25)
ax.axvline(res["q_obs"], color="black", lw=2, ls="--",
           label=fr"observed $\tilde q_\mu$ = {res['q_obs']:.2f}")
ax.set_yscale("log")
ax.set_xlabel(fr"$\tilde q_\mu$  at  $\mu = {mu:g}$")
ax.set_ylabel("toys")
ax.set_title("the two universes, scored by the same statistic; "
             "shaded = the tails that make CLs")
ax.text(0.60, 0.72,
        fr"CLs+b = {res['clsb']:.3f}" "\n"
        fr"1$-$CLb = {res['clb']:.3f}" "\n"
        fr"CLs = {res['cls']:.3f}",
        transform=ax.transAxes, fontsize=13,
        bbox=dict(boxstyle="round", fc="#f0f0f0", ec="grey"))
ax.legend(fontsize=11)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"step8b_toys_cls.{ext}", dpi=150)
print(f"\nwrote {HERE}/step8b_toys_cls.png")
print("NEXT (8c): repeat this at many mu, find where CLs crosses 0.05 -- "
      "that crossing IS the HybridNew limit")
