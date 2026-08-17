# Flavor-CR walkthrough — state and handoff

Context note for continuing the step-by-step explainer of the flavor-CR
anchored fit (started Aug 13 2026 on cmslpc). The plots here are made by
`../explain_fcr.py` (LCG_106, run from `optimization/`). Running example
everywhere: **m_WR = 2000, ee resolved, k5 window [1150, 2700], 50 GeV bins**
(the card binning — never plot these at a different bin width).

## The two-step story (the agreed framing)

1. **Flavor CR measures the tt normalization.** e-mu + jets, otherwise
   identical to the SR. tt -> e-mu twice as often as ee; DY and W_R -> eejj
   can never give e-mu, so the region is 90% tt+tW and signal-free ->
   unblinded, real data used (EGamma; user rule: **CR uses real data, the SR
   stays blind**). Count in the window, subtract non-tt, divide by the tt
   prediction:  SF_tt = (480 - 52)/453 = **0.946 ± 0.048**.
   - Naming (user asked): call it the **tt normalization scale factor
     (SF_tt)**; `rateParam` is the combine mechanism, "mu_tt" the internal
     symbol. The **transfer factor** is the separate MC ratio
     T_tt/T_cr = 190.5/452.6 ≈ 0.42 that carries the CR count into the SR.
2. **SR fit uses it.** Background split at the stack-plot seam:
   tt = MC shape (slope frozen at -3.31) x SF_tt — nothing fit to the SR;
   rest (Z+jets + other, 173 ev) = free expo fit to the SR spectrum (NOT
   summed MC). Fit = orange(frozen) + blue(free) + signal Gaussian.
   Remaining MC assumptions (discussed): tt *shape* in the window (validated
   by the flat data/MC ratio in 1b), the transfer factor (flavor counting x
   eps_e/eps_mu; lnN not yet added), the 52-ev CR contamination.
   In truth it's one simultaneous likelihood; SF_tt is profiled with the CR
   as a ±5% leash.

## Walkthrough position

- **Done and understood**: 1a/1b (regions; 1b carries the full SF_tt
  calculation on-plot: window / data 480 / non-tt 52 / tt 453 / formula),
  2a (SR split, log scale), the SF naming, the assumption audit.
- **NEXT STEP: 3_constraint.png** — the mu_tt likelihood profile ("the
  leash"): SR alone nearly flat (can't tell tt from DY), CR-alone parabola
  ±5%, combined == CR-alone (black lies on red). Then 4_effect.png
  (fcr/float expected-limit ratio, combine + homemade toys overlaid).
- 2b_cr_bin.png (the one-bin CR channel picture) was made but not yet
  walked through; it can support step 3.

Teaching style that works for this user: ONE small piece at a time, plain
words, stop and check; lead with yield x shape reading; expect pushback if
a plot is busy or an explanation leans on unexplained symbols.

## Results so far (all in the repo)

- Combine variants (results/ee_resolved): `k5_bw50_fcr` (CR=MC) and
  `k5_bw50_fcrd` (CR=EGamma data). fcr/float = 0.95-0.98 at 1400-2200,
  ~1.00 >= 2400; fcrd == fcr within 1% (free rest norm re-absorbs the tt
  shift against the blind SR — the anchor is defense, not expected
  sensitivity). Homemade twins: `6_spurious_signal_toys/card_model_toys.py`
  (+ `shared/card_sb_fit.py`), outputs run2/card/ and run2/card_fcrd/;
  homemade/combine parity 0.90-1.05 over 1400-2800 (>=3000: known
  runaway-toy RMS points). Details: optimization/README.md (flavor-CR
  sections) and 6_spurious_signal_toys/README.md.
- Agreed next analysis steps after the walkthrough: transfer-factor lnN,
  DY-CR anchor for Z+jets, stack with bconstr, mumu SR reuse of the same CR.
