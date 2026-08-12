#!/usr/bin/env python3
"""Step 2a -- the signal shape, drawn in the Stage-1 style.

The WR2000_N1000 MC shape (blue stairs, raw genWeight fills) with the CARD's
Gaussian on top (red curve): mu and sigma are NOT fit to this histogram --
they come from the Stage-2 linear window parameterization mu(m_WR),
sigma(m_WR) (the median Gaussian-core width over the M_N grid), evaluated at
m_WR = 2000. The light-red band is the k5 (mu +- 5 sigma) window of the optimized float
card (the running example here); the narrower k3 Stage-6 geometry is the
anchored-card / toy geometry shown for comparison in step 2b. The Gaussian is
normalized to the in-window MC count, exactly as in
1_signal_widths/gaussian/detail_gauss_fit.py.

Shape treatment (updated with the 10.9 retest + toy re-validation): on the
float cards at 1400-2600, BOTH mu and sigma are `param`-constrained at
0.3*sigma0 (they float, with a Gaussian penalty) -- nearly free (+3.5%) now
that the background slope is constrained, so the M_N width variation lives
in the model there. On the old floating-background card the same sigma prior
cost +22-35% (the signal<->background degeneracy); see step 7d. Float cards
at 2800-3200 and all anchored cards keep the shape fully fixed (the shape
nuisances collapse toy convergence in the sparser windows).
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C

HERE = Path(__file__).resolve().parent
POINT, MWR, MN = "WR2000_N1000", 2000, 1000

hep.style.use("CMS")
w = next(x for x in C.load_stage6_windows() if x["mWR"] == float(MWR))
mu, sigma = w["m_c"], w["sigma_win"]
K = 5.0                                    # optimized float-card geometry
lo, hi = mu - K * sigma, mu + K * sigma
edges, vals = C.load_signal(POINT, rebin=5)
centers = 0.5 * (edges[:-1] + edges[1:])

fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
ax.stairs(np.maximum(vals, 0.0), edges, color=C.BLUE, linewidth=1.5)

# card window: light-red band + the card Gaussian normalized to the
# in-window count (the Stage-1 convention).
ax.axvspan(lo, hi, color=C.RED, alpha=0.10, zorder=0)
sel = (centers >= lo) & (centers <= hi)
n_in = float(np.maximum(vals[sel], 0.0).sum())
bin_w = float(edges[1] - edges[0])
norm = 0.5 * (math.erf((hi - mu) / (sigma * math.sqrt(2)))
              - math.erf((lo - mu) / (sigma * math.sqrt(2))))
xs = np.linspace(lo, hi, 400)
pdf = np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
ax.plot(xs, n_in * bin_w * pdf / norm, color=C.RED, linewidth=2.4, zorder=4)
ax.axvline(mu, color="black", linestyle=":", linewidth=1.3, zorder=4)

ax.set_xlim(0, 1.4 * MWR)
ax.set_ylim(bottom=0)
ax.set_ylim(0, ax.get_ylim()[1] * 1.50)
ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
ax.set_ylabel("Events / bin")
ax.grid(alpha=0.3)
hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
              com=13, fontsize=16)
info = "\n".join([
    "ee  Resolved SR",
    C.ERA,
    rf"$M_{{W_R}}={MWR}$, $M_N={MN}$ GeV",
    "",
    rf"$\mu={mu:.0f}$ GeV,  $\sigma={sigma:.0f}$ GeV",
    rf"$k=5$ window $[{lo:.0f},{hi:.0f}]$ GeV",
    rf"$\mu,\sigma$ param-constr. $(0.3\sigma_0)$",
])
ax.text(0.04, 0.96, info, transform=ax.transAxes, fontsize=15,
        verticalalignment="top", horizontalalignment="left")
C.savefig(fig, HERE / "step2a_signal_shape")
print("next: step2b -- the same window recipe at every mass (the map)")
