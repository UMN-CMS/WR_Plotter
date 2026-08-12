#!/usr/bin/env python3
"""Step 6b -- the same fit, read off in each of the 31 bins.

Combine's likelihood is BINNED: it compares an observed count n_i with a
predicted count nu_i, one bin at a time. In combine's notation

    nu_i = r * s_i + b_i

with r the POI (sigma x BR in fb on this card), s_i the expected signal counts
in bin i at r = 1, and b_i the expected background counts. Same card and same
post-fit values as 6a (`card_float_m2000.txt`, combine FitDiagnostics).

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _common as C
from combine_from_scratch import CardModel

hep.style.use("CMS")

m = CardModel(mass=2000)
ref = json.load(open(HERE.parent / "reference_fitdiag.json"))["prod_m2000"]
p, rate = ref["postfit"], ref["rate_per_fb"]
mu, sg = p["mu_sig"], p["sigma_sig"]
k = p["b_expo"] / 1000.0
SQ2 = math.sqrt(2.0)


def shape(x, width):
    """(s_i-per-fb, b_i) at positions x, in events per `width`."""
    bn = (math.exp(k * (m.hi - m.m_c)) - math.exp(k * (m.lo - m.m_c))) / k
    b = p["bkg_norm"] * np.exp(k * (x - m.m_c)) * width / bn
    gn = 0.5 * (math.erf((m.hi - mu) / (SQ2 * sg))
                - math.erf((m.lo - mu) / (SQ2 * sg)))
    s = rate * (np.exp(-0.5 * ((x - mu) / sg) ** 2)
                / (sg * math.sqrt(2 * math.pi))) * width / gn
    return s, b


# the 31 predicted counts, and the same model as a smooth curve
s_i, b_i = shape(m.centers, m.width)
nu = p["r"] * s_i + b_i
xs = np.linspace(m.lo, m.hi, 600)
s_x, b_x = shape(xs, m.width)
smooth = p["r"] * s_x + b_x

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.plot(xs, smooth, color=C.BLUE, lw=1.3, ls="--", alpha=0.85,
        label="S+B fit (smooth)")
edges = np.append(m.centers - m.width / 2, m.centers[-1] + m.width / 2)
ax.stairs(nu, edges, color=C.BLUE, lw=2.6,
          label=rf"$\nu_i$: the same fit in {m.nbins} bins")
ax.errorbar(m.centers, m.n, yerr=m.n ** 0.5, fmt="ko", ms=6, zorder=5,
            label=r"Data  $n_i$")

ax.set_yscale("log")
ax.set_xlim(m.lo - 40, m.hi + 40)
ax.set_ylim(0.3, 300)
C.log_yaxis_one_ten(ax)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel(f"Events / {m.width:.0f} GeV")
ax.legend(loc="upper right", fontsize=15, frameon=False)
ax.grid(alpha=0.3)

ax.text(0.035, 0.955, "ee\nResolved SR\n"
        rf"$m_{{W_R}} = {int(m.mass)}$ GeV",
        transform=ax.transAxes, fontsize=15, va="top")
ax.text(0.035, 0.06,
        r"$\nu_i = r\,s_i + b_i$"
        "\n"
        r"$r$ = POI;   $s_i$, $b_i$ = expected signal, background in bin $i$",
        transform=ax.transAxes, fontsize=14, va="bottom")

hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              lumi="59.8", com=13, fontsize=15)
C.savefig(fig, HERE / "step6b_binned")
print(f"{'bin':>4} {'n_i':>8} {'s_i':>8} {'b_i':>9} {'nu_i':>9}")
for i in (0, 15, 16, 30):
    print(f"{i+1:>4} {m.n[i]:8.1f} {s_i[i]:8.3f} {b_i[i]:9.3f} {nu[i]:9.3f}")
print(f"sum: n={m.n.sum():.1f}  s={s_i.sum():.2f} (=rate)  "
      f"b={b_i.sum():.1f}  nu={nu.sum():.1f}")
