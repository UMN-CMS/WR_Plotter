#!/usr/bin/env python3
"""Step 1d -- the step-1a component stack with LO signal benchmarks on top.

Same stack as step1a_component_stack.py (converted run2 backgrounds, 59.8 fb^-1,
ee resolved SR, 50 GeV bins) plus the expected yield of two signal grid points,
normalized to the luminosity from the raw-genWeight signal histograms:

    N_bin = xsec[pb] * 1000 * lumi[fb^-1] * (sum w_bin / genEventSumw)

xsec (LO MadGraph, no k-factor) and genEventSumw come from
data/configs/RunII/2018/RunIISummer20UL18/RunIISummer20UL18_signal.json.
Made 2026-09-26 for the HEP-seminar slide "Simulation is weakest where the
search is most sensitive" (research-slides/talks/20261007_hep_seminar).
"""
import json
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

# (tag, colour, linestyle); add e.g. ("WR2000_N1000", "#b2182b", "-.") for the
# 2 TeV running example (2652 events; its sparse tail clutters the 4 TeV bump).
SIGNALS = C.OVERLAY_SIGNALS
lumi, com = C.lumi_com()

comps = C.load_bkg_components(rebin=5)
edges = comps["DYJets"][0]
centers = 0.5 * (edges[:-1] + edges[1:])
colors = {"Other": "#94a4a2", "Nonprompt": "#964a8b", "tt_tW": "#f89c20",
          "DYJets": "#5790fc"}
order = ["Other", "Nonprompt", "tt_tW", "DYJets"]

fig, ax = plt.subplots(figsize=(9, 7.5))
ax.stackplot(centers, *[comps[s][1] for s in order], step="mid",
             colors=[colors[s] for s in order],
             labels=[C.BKG_LABELS[s] for s in order])
bkg_handles, bkg_labels = ax.get_legend_handles_labels()

_, total, b_up, b_dn, has_syst = C.load_bkg_band(rebin=5)
lo = np.clip(total - b_dn, 1e-4, None)
ax.fill_between(centers, lo, total + b_up, step="mid", facecolor="none",
                edgecolor="#555555", hatch="/////", linewidth=0.0, zorder=5)
band_label = r"MC stat $\oplus$ syst" if has_syst else "MC stat. unc."
stat_proxy = Patch(facecolor="none", edgecolor="#555555", hatch="/////",
                   label=band_label)

sig_handles = []
for tag, color, ls in SIGNALS:
    m, n = tag[2:].split("_N")
    entry = C.signal_config_entry(tag)
    e_s, v_s = C.load_signal(tag, rebin=5)
    assert np.allclose(e_s, edges)
    if C.SEMINAR_ERA == "2024":
        # Run-3 histograms are filled with genWeight*xsec*lumi/genEventSumw per
        # flavour sample: already events. (The old renormalization by the file's
        # weighted cutflow total was wrong: that file merges the ee and mumu
        # samples, after the skim, and it scaled the overlay down by ~0.62.)
        scale = 1.0
    else:
        scale = entry["xsec"] * 1000.0 * lumi / entry["genEventSumw"]
    y = v_s * scale
    h = ax.stairs(np.clip(y, 1e-4, None), edges, color=color, linestyle=ls,
                  linewidth=2.2, zorder=6,
                  label=(fr"$(m_{{W_R}}, m_N) = ({m}, {n})$ GeV: "
                         f"{y.sum():.0f} events"))
    sig_handles.append(h)
    print(f"  {tag}: xsec {entry['xsec'] * 1000:.3g} fb, "
          f"expected ee resolved SR events {y.sum():.1f}")

ax.set_yscale("log")
ax.set_xlim(800, 6000)
ax.set_ylim(1e-3, 1e4)
C.log_yaxis_one_ten(ax)
ax.set_xlabel(r"$m_{\ell\ell jj}$ (GeV)")
ax.set_ylabel("Events / 50 GeV")
ax.legend(bkg_handles[::-1] + [stat_proxy] + sig_handles,
          bkg_labels[::-1] + [band_label] + [h.get_label() for h in sig_handles],
          loc="upper right", fontsize=14, frameon=False)
ax.text(0.03, 0.96, f"ee\nResolved SR\n{C.MC_LABEL}\nsignal: LO $\\sigma$",
        transform=ax.transAxes, fontsize=16, va="top")
C.cms_label(ax)
C.savefig(fig, HERE / "step1d_signal_overlay")
