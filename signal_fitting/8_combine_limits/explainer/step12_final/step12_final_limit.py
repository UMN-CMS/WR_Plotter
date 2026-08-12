#!/usr/bin/env python3
"""Step 12 -- the destination: the data-free refined run2 2018 expected limit.

Assembles the Stage-10.8 refined table (float 1400-3200 asymptotic; anchored
1000-1200 asymptotic; anchored >= 3400 HybridNew; MC-Asimov observation
everywhere -> no data touched anything) into the official-style plot: x =
800-6000 GeV, y = sigma x BR(eeqq') in fb, 1e-4..1e4, Brazil band + thin red
theory (sample sigma x 0.5 for the ee channel). No observed line -- there is
no data in this chain by construction.

Expected exclusion crossing: ~5.04 TeV (official 2018 expected: ~4.9; the
stat-only + known-background-shape setting explains being slightly stronger).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C
from xsec_limit import plot_band, BANDS  # noqa: E402  (path set in _common)

HERE = Path(__file__).resolve().parent
meta = C.load_meta()
pts = []
for r in C.load_refined():
    pts.append({
        "mWR": r["mWR"],
        "sigma": {N: float(r[f"fb_{k}"]) / 1000.0
                  for N, k in zip(BANDS, ("m2s", "m1s", "med", "p1s", "p2s"))},
        "sigma_obs": float("nan"),
        "xsec_pb": float(r["xsec_pb"]),
    })
plot_band("refined, no data", pts, HERE / "step12_final_limit",
          ykey="sigma", obskey="sigma_obs", theory=True, scale=1000.0,
          ylabel=r"$\sigma(pp \to W_R)\,\mathcal{B}(W_R \to eeq\bar{q}\,')$ (fb)",
          channel=C.CHANNEL, topology=C.TOPOLOGY, com=meta.get("com", 13),
          lumi=meta["lumi"], cl=0.95, trust_max=None,
          center="zero (MC Asimov)")
print("wrote step12_final_limit.pdf/.png -- the end of the chain")
