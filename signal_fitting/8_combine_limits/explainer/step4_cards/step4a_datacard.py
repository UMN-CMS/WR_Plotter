#!/usr/bin/env python3
"""Step 4a -- the datacard itself, annotated line by line (slide-ready PDF).

Renders the real optimized float card
(production/cards/ee_resolved/card_float_m2000.txt) with a plain-language note
beside every line, colour-coded by datacard section, so each entry is
self-explanatory on a slide. The companion step4a_float_card.py draws the same
card's three SHAPES; this one is the card TEXT.

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  (plain matplotlib -- no ROOT/LCG needed, but harmless under LCG)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _common as C  # noqa: F401  (kept for savefig style parity)

HERE = Path(__file__).resolve().parent

COL = {"comment": "#8a8a8a", "count": "#1f77b4", "shape": "#2ca02c",
       "obs": "#e8820c", "proc": "#7a4fb5", "nuis": "#c8332a"}

# (kind, card_text, annotation) ; kind "div" is a section divider line
ROWS = [
    ("comment", "# ee_resolved   m_WR=2000   variant=float",
                "comment (combine ignores '#'): channel, mass, card type"),
    ("comment", "# window [1150,2700] bw=50 m_c=1927.7 sigma=157.77",
                "the k5 window + peak from step 2, at 50 GeV binning"),
    ("comment", "# rate = lumi*eff = 24.3356 ev/fb  (r in fb)",
                "the rate that turns the POI into a cross section (step 2c)"),
    ("div",),
    ("count", "imax 1",
              "number of channels = 1  (ONE mass window -- not ee vs mumu)"),
    ("count", "jmax 1",
              "number of background processes = 1  (summed MC = one smooth fn)"),
    ("count", "kmax *",
              "number of nuisance parameters = any  ('*' -> combine counts them)"),
    ("div",),
    ("shape", "shapes sig       win  ws_float_m2000.root w:sig_pdf",
              "signal shape = fixed Gaussian, object 'sig_pdf' in workspace 'w'"),
    ("shape", "shapes bkg       win  ws_float_m2000.root w:bkg_pdf",
              "background shape = the falling expo 'bkg_pdf'"),
    ("shape", "shapes data_obs  win  ws_float_m2000.root w:data_obs",
              "the observation = summed background MC histogram (no real data)"),
    ("div",),
    ("obs", "bin          win", "the single channel is named 'win'"),
    ("obs", "observation  -1", "-1 = take the observed yield from data_obs"),
    ("div",),
    ("proc", "bin      win   win", "both processes live in the channel 'win'"),
    ("proc", "process  sig   bkg", "the two processes in this channel"),
    ("proc", "process  0     1", "process id:   <= 0 = signal,   >= 1 = background"),
    ("proc", "rate     24.3356  1",
             "signal: no 'sig_pdf_norm' in w, so THIS is the yield = L*eff (x r)"),
    ("proc", "",
             "bkg: 'bkg_pdf_norm' IS in w, so that gives the yield -- 1 = multiplier"),
    ("div",),
    ("nuis", "b_expo param -2.9435 0.12435",
             "bkg SLOPE: constrained to the spectrum fit  -2.94 +/- 0.12 /TeV (step 3b)"),
    ("nuis", "bkg_pdf_norm flatParam",
             "bkg NORM: floats freely (pinned by this window's own sidebands)"),
    ("nuis", "mu_sig param 1927.7 47.331",
             "signal PEAK: constrained  1927.7 +/- 47.3   (47.3 = 0.3 x sigma_0)"),
    ("nuis", "sigma_sig param 157.77 47.331",
             "signal WIDTH: constrained  157.77 +/- 47.3   (same 0.3 x sigma_0)"),
    ("nuis", "",
             "   why: sigma_0 is the MEDIAN over M_N, but the true width varies"),
    ("nuis", "",
             "   with M_N -- the prior lets the fit absorb that spread (+3.5%)"),
]

fig, ax = plt.subplots(figsize=(15, 10.5))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.02, 0.985, "card_float_m2000.txt", family="monospace",
        fontsize=18, fontweight="bold", va="center")
ax.text(0.02, 0.945,
        "8_combine_limits/production/cards/ee_resolved/     "
        r"$-$     ee resolved, $m_{W_R}=2000$ (float card: k5 window, 50 GeV bins)",
        fontsize=12, color="#555555", va="center")
ax.axvline(0.505, ymin=0.06, ymax=0.93, color="#dddddd", lw=1)

y = 0.90
for row in ROWS:
    if row[0] == "div":
        ax.plot([0.02, 0.49], [y, y], color="#cfcfcf", lw=1.0)
        y -= 0.017
        continue
    kind, card, note = row
    ax.text(0.02, y, card, family="monospace", fontsize=12.5,
            color=COL[kind], va="center")
    ax.text(0.52, y, note, family="sans-serif", fontsize=12,
            color="#2b2b2b", va="center")
    y -= 0.0335

# key / footnotes
ax.plot([0.02, 0.98], [0.062, 0.062], color="#cfcfcf", lw=1.0)
ax.text(0.02, 0.040,
        r"$\bf{flatParam}$ = free parameter, no penalty.     "
        r"$\bf{param\ \ C\ \ E}$ = Gaussian-constrained nuisance: central value "
        r"$C$, width $E$ (a penalty for moving away from $C$).",
        fontsize=11.5, color="#2b2b2b", va="center")
ax.text(0.02, 0.014,
        r"The POI $\bf{r}$ (signal strength) is added by combine automatically "
        r"-- it multiplies the signal rate.  Combine profiles "
        r"$r$, bkg_pdf_norm, b_expo, mu_sig, sigma_sig against the three penalty terms.",
        fontsize=11.5, color="#2b2b2b", va="center")

fig.savefig(HERE / "step4a_datacard.pdf", bbox_inches="tight")
fig.savefig(HERE / "step4a_datacard.png", bbox_inches="tight", dpi=150)
print("wrote step4a_datacard.pdf/.png")
