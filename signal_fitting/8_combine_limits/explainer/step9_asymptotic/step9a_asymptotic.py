#!/usr/bin/env python3
"""Step 9a -- the asymptotic shortcut: Asimov dataset, sigma_A, and why the OLD
step2_explainer was right all along (in its regime).

Toys are expensive.  CCGV (arXiv:1007.1727) proved that in the large-sample
limit the whole step-4/5 machinery collapses to closed form, needing only ONE
number per hypothesis:

  ASIMOV DATASET   the perfectly average background-only dataset:
                   data_i = nu_i(r=0, theta-hat(0)) -- no fluctuations.
  sigma_A          evaluate q~_mu on the Asimov dataset; then
                   sigma_A = mu / sqrt(q~_mu(Asimov)) is the effective
                   Gaussian width of the mu-hat estimator.
  closed forms     with mu-hat ~ N(mu', sigma_A), CLs+b and CLb become
                   error functions, and CLs = 0.05 can be solved directly.
                   This is combine -M AsymptoticLimits.

THE BRIDGE to the old explainer (7_limit_plots/step2_explainer): its model
was "N-hat is Gaussian(mu0, sigma) with sigma = RMS(N_sp)".  That IS the
asymptotic mu-hat distribution -- so for an unbounded Gaussian estimator the
CCGV median expected limit reduces ALGEBRAICALLY to the old formula

    UL_med = sigma * ( Phi^-1(1 - alpha * Phi(0)) )  ~  1.96 sigma

This step verifies all three numerically:
  (a) sigma_A vs the step-6d curve width (they should agree ~%),
  (b) our from-scratch asymptotic UL vs combine AsymptoticLimits (22.40
      events observed on the k5_bw50 card),
  (c) the CCGV median-expected vs the old explainer's closed form with
      sigma = sigma_A.

  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python step9a_asymptotic.py
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from combine_from_scratch import CardModel, load_reference, check

import ROOT

m = CardModel(mass=2000)
ref = load_reference(2000).get("asymptotic", {})
print(__doc__.split("\n\n")[0])

# (a) sigma_A vs the likelihood-curve width
asimov = m.asimov(0.0)
mu_test = 23.0
sA = m.sigma_A(mu_test)
(_, _, _), errs, _ = m.fit_global()
print(f"\n  Asimov dataset (first 3 bins): {[round(v,1) for v in asimov[:3]]} "
      "... (no fluctuations, by construction)")
print(f"  sigma_A(mu={mu_test:.0f}) = {sA:.2f}   vs   step-6d curve width "
      f"= {errs[0]:.2f}")

# (b) the asymptotic UL from scratch (observed = the MC dataset), checked
# against combine's OBSERVED AsymptoticLimits value on the identical card.
# The small residual is the RooFit pdf-evaluation convention that also
# offsets r-hat slightly in step 6d -- it propagates into sigma_A.
ref_fd = load_reference(2000).get("fitdiag", {})
ul_obs = m.asymptotic_limit()
check("asymptotic UL(observed) vs combine", ul_obs,
      ref_fd.get("asymptotic_observed"), tol=0.05)

# (c) the bridge: CCGV median expected == the old explainer formula
alpha = 0.05
med_ccgv = m.asymptotic_limit(quantile=0.5)
med_old = sA * ROOT.TMath.NormQuantile(1 - alpha * ROOT.TMath.Freq(0.0))
print(f"\n  CCGV median expected (from scratch)     = {med_ccgv:.2f}")
print(f"  old-explainer formula with sigma_A      = {med_old:.2f}   "
      f"(= {med_old/sA:.3f} * sigma_A; the familiar ~1.96 sigma)")
check("CCGV median == old formula", med_ccgv, med_old, tol=0.02)
check("median expected vs combine 50.0%", med_ccgv, ref.get("ul_med"),
      tol=0.05)
print("  (the ~1% offset vs combine is the sigma_A evaluation-convention "
      "residual;\n   the shape of the whole band is exact -- see step 9b)")

# figure: the asymptotic mu-hat distribution with the band quantile geometry
fig, ax = plt.subplots(figsize=(10, 5.8))
x = np.linspace(-3.5 * sA, 3.5 * sA, 500)
g = np.exp(-0.5 * (x / sA) ** 2) / (sA * math.sqrt(2 * math.pi))
ax.plot(x, g, color="#5790fc", lw=2.2,
        label=fr"asymptotic $\hat\mu$ distribution: $N(0, \sigma_A={sA:.1f})$"
              "\n(background-only universe)")
for q, col in ((0.025, "#f5d800"), (0.16, "#00cc00"), (0.5, "black"),
               (0.84, "#00cc00"), (0.975, "#f5d800")):
    xq = sA * ROOT.TMath.NormQuantile(q)
    ul = m.asymptotic_limit(quantile=q)
    ax.axvline(xq, color=col, lw=1.5, ls="--")
    ax.annotate(f"q={q:g}\nUL={ul:.0f}", xy=(xq, 0.7 * max(g)),
                fontsize=9, ha="center",
                bbox=dict(boxstyle="round", fc="white", ec=col))
ax.set_xlabel(r"$\hat\mu$ [events]")
ax.set_ylabel("probability density")
ax.set_title("the old explainer's Gaussian, derived: each background-only "
             "quantile maps to one Brazil-band edge")
ax.legend(fontsize=11, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"step9a_asymptotic.{ext}", dpi=150)
print(f"\nwrote {HERE}/step9a_asymptotic.png")
print("NEXT (9b): sweep the quantile -> the full expected band, and where the "
      "asymptotic shortcut breaks (the sparse masses)")
