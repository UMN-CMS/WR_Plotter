# run2 2018 ee — the data-free expected limit AND combine itself, from the very beginning

> **STATUS: documentation.** One merged walkthrough (Jul 2026) of the former
> `explainer/` (the analysis, step by step) and `internals_explainer/`
> (combine rebuilt from scratch, was 10.7). Read it in order: steps 1–5 build
> everything that goes *into* combine, steps 6–9 open the black box and show
> what combine *does* with a card, steps 10–12 use that understanding to
> justify the production choices and assemble the final band.

A step-by-step explainer of how the run2 2018 ee-resolved expected limit is
built **with combine only, with no data anywhere**, using the optimized
settings established in Stages 9–10.8 — interleaved with a from-scratch
rebuild of combine's statistics on the real m_WR = 2000 card, so nothing has
to be taken on faith. One directory per step, one figure per script
(signal/window plots follow the `1_signal_widths` styling: blue stairs MC,
light-red window band, red Gaussian normalized to the in-window count).
Everything is read-only against the already-produced run2 inputs and the
Stage-10.8/10.9 results.

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
for s in step*/step*.py; do python "$s"; done
```

## The chain

### Part I — what goes into combine (the analysis inputs)

| step | script | teaches |
|---|---|---|
| 1a | `step1_backgrounds/step1a_component_stack.py` | the 4 converted run2 MC samples, pre-scaled to 59.8 fb⁻¹ (50 GeV bins + MC stat band) |
| 1b | `step1_backgrounds/step1b_summed_observation.py` | the SUM is both the fit input and `data_obs` (the no-data setting) |
| 1c | `step1_backgrounds/step1c_window_on_background.py` | the m=2000 k5 window drawn on `data_obs`: 31 bins, B = 363 ev |
| 2a | `step2_signal/step2a_signal_shape.py` | the card Gaussian on the WR2000_N1000 MC, Stage-1 style; µ,σ imposed from the Stage-2 linear parameterization, `param`-constrained at this mass |
| 2b | `step2_signal/step2b_window_map.py` | µ ± kσ across the grid + the 800/6000 clamps — **regime material, see step 13** |
| 2c | `step2_signal/step2c_efficiency.py` | ε_ee = S_win/(0.5·sumw) at k5, half-N diagonal; rate = lumi×ε → r ≡ σ·B in fb |
| 3a | `step3_background_model/step3a_background_pdf.py` | the background MODEL: `bkg_pdf` = expo, norm (floats) + slope (constrained) |
| 3b | `step3_background_model/step3b_constrained_slope.py` | where the slope comes from: expo fit to the trusted spectrum [1000, 3500] → b = −2.94 ± 0.12 /TeV |
| 4a | `step4_cards/step4a_datacard.py` | **the datacard itself, annotated line by line** (slide-ready) |
| 4a′ | `step4_cards/step4a_float_card.py` | the same card as three SHAPES on the spectrum |
| 4b | `step4_cards/step4b_anchored_card.py` | the fully-fixed card: only r floats (m=4600) — regime material |
| 5a | `step5_anchor/step5a_member_fits.py` | the four trusted-spectrum family fits (expo/tail/expo2/powexp) |
| 5b | `step5_anchor/step5b_member_ratio.py` | the family spread = the model systematic |
| 5c | `step5_anchor/step5c_transport.py` | B_env per window vs the raw MC (the mcmax caveat, visibly) |

### Part II — inside combine (the statistics, rebuilt from scratch)

At this point a datacard exists. Steps 6–9 rebuild what `combine` does with
it, one piece at a time, on the **real run2 m_WR = 2000 float card**
(`k5_bw50`: 31 window bins of 50 GeV on [1150, 2700], floating expo
background, fixed Gaussian signal). Combine's POI on this card is σ·B in fb
(`rate` = 24.34 ev/fb); the steps work in **events** — r_events = r_fb ×
rate, identical in the likelihood — so every picture stays in yields. Every
step prints `CHECK` lines against genuine combine outputs on the identical
card (the answer key below), so nothing has to be taken on faith.

| step | script | teaches | key output / check |
|---|---|---|---|
| 6a | `step6_likelihood/step6a_ingredients.py` | a card = data bins + a signal shape + a background shape; the model is ν_i = r·S_i + B·f_i(b) — 3 numbers in, 31 predictions out | the ingredients, no statistics yet |
| 6b | `step6_likelihood/step6b_one_bin_poisson.py` | ONE bin's Poisson probability, read two ways: distribution (fixed ν, varying n) vs **likelihood** (fixed observed n, varying ν) | the mental flip everything rests on |
| 6c | `step6_likelihood/step6c_product_over_bins.py` | independent bins ⇒ multiply ⇒ −2lnL = 2Σ(ν_i − n_i ln ν_i); one number scores the whole spectrum | per-bin cost breakdown of three example models |
| 6d | `step6_likelihood/step6d_nll_curve.py` | scan r: the curve's minimum is the fitted yield, its Δ=1 width the error — the likelihood **is** the fit you always did | r̂ 0.02σ from FitDiagnostics; errors match to 0.3 % |
| 7 | `step7_profiling/step7_profiling.py` | re-minimizing nuisances at each r widens the curve — that width **is** the systematic | profiling widens ±1σ by ×1.44 here |
| 8a | `step8_toys_cls/step8a_test_statistic.py` | q̃_μ (CCGV eq. 16): one-sided, boundary at μ̂ < 0; large q̃ = hypothesis in trouble | q̃ curves for shifted datasets |
| 8b | `step8_toys_cls/step8b_toys_cls.py` | toys under s+b and b-only, scored by q̃; CLs = ratio of the two tails (= `HybridNew`) | at μ = 23 (combine's limit) our CLs = 0.051 |
| 8c | `step8_toys_cls/step8c_limit_scan.py` | the limit = where CLs(μ) crosses 0.05 | toy crossing 23.7 vs combine observed 22.4 (+6 %, PASS); asymptotic 22.6 (+1 %) |
| 9a | `step9_asymptotic/step9a_asymptotic.py` | Asimov dataset → σ_A → closed forms (= `AsymptoticLimits`); **the bridge**: CCGV median expected ≡ the old step2_explainer formula (0.1 %) | the old explainer was the asymptotic Gaussian corner of this machinery |
| 9b | `step9_asymptotic/step9b_expected_band.py` | sweep the b-only quantile → the Brazil band; where the shortcut breaks (sparse masses: 10.6 measured asymptotic 2.28 vs toys 3.25 at 4600) | all five band points PASS; band *shape* matches combine exactly |

### Part III — the production choices, now legible

| step | script | teaches |
|---|---|---|
| 10a | `step10_methods/step10a_undercoverage.py` | asymptotic under-covers below B≈5 (same cards, two methods) — step 9a's shortcut breaking on the real cards |
| 10b | `step10_methods/step10b_band_collapse.py` | the band collapses when b-only toys are all n=0 |
| 11a | `step11_settings/step11a_observation.py` | setting: no-data vs data-seeded (isolated, ×0.72–1.05) |
| 11b | `step11_settings/step11b_edges.py` | setting: anchored vs floating at the edges (×0.38–0.6) |
| 11c | `step11_settings/step11c_method.py` | setting: asymptotic vs HybridNew (isolated, ×0.6–1.1) |
| 11d | `step11_settings/step11d_signal_shape.py` | setting: shape priors cost +22% on a floating background but +3.5% on the slope-constrained one — the price was the degeneracy |
| 11e | `step11_settings/step11e_float_optimization.py` | setting: the 10.9 float-card scan — the two ~27% levers compound to ×0.60; binning alone does nothing |
| 12 | `step12_final/step12_final_limit.py` | the destination: the official-style data-free band, crossing ~5.04 TeV |
| 13 | `step13_regimes/step3_regime_map.py` | B_window 837→0.04: the three regimes and their failure modes (was step 3; the walkthrough now runs one card end-to-end and generalizes at the end) |

---

## The internals answer key (steps 6–9)

Every Part-II step checks itself against genuine combine outputs on the
identical m=2000 card (`card_k5_bw50_float_m2000.txt`; all numbers in
**events** = fb × rate 24.34, originals in fb kept in
`reference_fitdiag.json`):

| reference | value | from |
|---|---|---|
| best fit r̂ ± σ | −0.41 −11.0/+11.3 | `FitDiagnostics` (reference_fitdiag.json, key `k5f_m2000`) |
| observed UL (asymptotic) | 22.40 | `AsymptoticLimits` (10.9 scan) |
| expected band | 12.0 / 16.2 / 22.7 / 32.2 / 43.7 | `AsymptoticLimits` (10.9 scan) |

(The previous answer key — the k3_bw100 run3 parity card, r̂ 3.96 −18.3/+18.7,
observed 39.65, band 19.2/26.1/36.7/52.0/70.4, HybridNew 37.5 ± 0.7 — lives
on under `archived/10_limit_refinement/6_combine_parity/` and as the
`fixed_m2000` block of reference_fitdiag.json.)

**The engine:** `combine_from_scratch.py` (~300 lines, numpy + ROOT-Minuit2,
in this directory next to `_common.py`) implements the card model, the binned
Poisson −2lnL, profiling, q̃_μ, LHC-style toys (nuisances at conditional
MLEs), CLs from toys, the Asimov dataset, σ_A, and the CCGV asymptotic
limit — i.e. every ingredient of `FitDiagnostics`, `HybridNew`, and
`AsymptoticLimits` for this model class.

**Known residual:** a uniform ~+3 % offset against combine in σ_A-derived
numbers (band points, observed UL), traceable to RooFit pdf-evaluation
conventions (the same effect shifts r̂ by 0.06 σ in step 6d). The band
**shape** (each quantile / median) matches combine to ≲1 %. All checks carry
tolerances that make this explicit rather than hiding it.

**Relation to the old explainer:** `7_limit_plots/step2_explainer` remains
correct as the intuition layer for steps 8b–9a in the Gaussian regime: its
"family of shifted Gaussians" is f(μ̂ | μ) in the asymptotic limit, its CLs
tweak is the same tweak, and its UL formula is what step 9a derives from the
Asimov construction. What it could not show — because the homemade method
never needed them — are steps 6a–8a: the likelihood itself, profiling, and
the test statistic.

---

## The settings, one by one

Each setting: what it does, the alternatives, and the **measured** difference
(no hand-waving — every number below was produced in this repository).

### 1. Observation: MC Asimov (“no data”)

`data_obs` = the summed background MC in every card. With a floating
background, combine seeds its expected-band Asimov from the background-only
fit **to the observation** — so this is the only door through which data
could enter an expected limit. Closing it makes the band a statement about
the background *model*.
**Alternative:** real EGamma data (`--observed data` in Stage 9).
**Difference:** identical float cards, expected medians shift by 0.72–1.05×
(step 11a) — the 2018 data sits above the MC in the 3–5 TeV
sidebands, weakening the data-seeded expected there by up to ~40%. Anchored
cards are observation-independent (nothing floats but r), so for them the
choice does not exist.

### 2. Windows: µ±3σ for the anchored machinery, µ±5σ for the float cards

Every mass gets its fit range from the Stage-2 linear parameterization,
snapped to the bin grid, clamped to [800, 6000]. The Stage-6 geometry (and
the anchored cards) use k = 3; the **optimized float cards use k = 5 with
50 GeV bins** — the Stage-10.9 scan measured the wider sidebands to be worth
~26% by themselves (finer binning alone: nothing, 1.015–1.035). Windows
rather than the full spectrum because a 2-parameter expo only has to be
right *locally* (Stage-4 checks pass 26/26 per window).
**Caveat:** widening is safe for expo (k-sweep; and the 10.9 FitDiagnostics
toys show < 2% spurious) but not for powlaw, whose mismodeling spurious
grows with window width — the family choice and the width are a package.

### 3. POI normalization: r ≡ σ·B(eeqq̄′) in fb

`rate = lumi × eff`, with `eff = S_window / (0.5 × genEventSumw)`. The 0.5
(`--channel-bfrac`) removes the muon half of the flavor-mixed WRtoNLtoLLJJ
samples (measured 50/50 e:µ from the GenModel N-flavor genWeight shares — no
τ), making eff a genuine per-channel efficiency (0.19→0.42) and r directly
the y-axis of the official 2018 plot.
**Alternative:** the pre-July convention eff = S/sumw (a limit on the *total*
σ·B(ℓℓqq̄′)).
**Difference:** exactly ×2 on the σ axis (µ = σ/σ_theory is unchanged); the
old convention silently mislabeled the y-axis.

### 4. Background model: regime-split (the Stage-10.8 rework)

- **float (1400–3200, B = 37→1036 at k5):** expo with free norm
  (`flatParam`, measured by each window’s own sidebands) and the slope
  `param`-**constrained** to the trusted-spectrum anchor fit
  (b = −2.94 ± fit error) — the Stage-10.9 optimization, worth ~28% by
  itself and ~40% together with the k5/50 GeV window. The old fully-floating
  card was *valid* (10.6 parity 0.90–1.04) but left that sensitivity on the
  table. Toy-validated: null spurious < 2% of the medians, injection
  recovered to 1–5% (10.9 `toy_validation_table.csv`).
- **anchored (1000–1200 and ≥ 3400):** slope *and* norm fixed from a binned
  Poisson-ML fit of the summed MC over [1000, 3500] (b = −2.94/TeV,
  χ²/ndf 0.34), evaluated in the target window (B_env). Only r floats.
  Why: at 1000–1200 the window is clamped at the 800 GeV selection floor —
  no left sideband, so a floating background is collinear with the signal;
  at ≥ 3400 (B < 7) the free norm can run away on near-empty windows.
**Alternative:** floating background everywhere (the Stage-9 baseline).
**Difference:** expected median ×0.38 at 1000, ×0.68 at 1200, ×0.45–0.6 at
3400–4600 (step 11b); at 5800–6000 the refined number is honestly *higher*
(the baseline’s asymptotics under-covered — setting 6).
**Model systematic:** tail/expo2/powexp anchor members shift the median by
8–25%; quoted as grey bars ON the band (10.4 rule), not inside the CLs σ.
**Known caveat (the “mcmax” question):** in the deep tail the raw MC sits
~2–3× above every smooth member (B_env 0.37 vs windowed MC 1.05 at 4600;
step 5’s figure shows the jagged single-large-weight MC events). Stage 10.4
judged that tail to be weight noise and carried “the tail is real” as an
extra `mcmax` member; that member is not built here — if you want it, it is
one more entry in `MEMBERS` of the 10.8 workspace builder. At B ≲ 1 the
expected limit is dominated by the n = 0 discreteness, so the effect on the
band is modest, but it is the first thing to check before quoting the 5–6 TeV
points.

### 5. Signal shape: constrained on the float cards, fixed on the anchored

Gaussian(m_c, σ) from the Stage-2 linear parameterization. On the
**optimized float cards, µ and σ are `param`-constrained at 0.3σ₀**
('both030'): they float inside physical boxes (µ ± 1.5σ₀, σ ∈ [0.5, 2.5]σ₀)
with Gaussian penalties, so the M_N width variation (σ_true/σ₀ ∈
[0.67, 1.77]) and the peak-position uncertainty live inside the model.
**Why this changed (the 10.9 retest, step 11d):** the same σ-prior costs
+22–35% on a *floating*-background card (10.6, reproduced on run2) but only
+3.5% once the slope is anchor-constrained — the cost was never the prior,
it was the signal↔background degeneracy, and the slope constraint removes
it. Fully-free σ remains catastrophic.
**Where the priors apply:** float cards at **1400–2600 only**. The toy
re-validation of the final card passes there (median spurious < 3% of the
limit, core width +10%), but at 2800–3200 the two shape nuisances collapse
toy convergence (84/51/32% vs 99/88/66% fixed) and drift the survivor
median while *costing* limit — so those float cards, and all **anchored
cards**, keep the fixed shape, with the width variation as an offline bias
systematic there.

### 6. Statistical method: AsymptoticLimits, except where it breaks

AsymptoticLimits (the Asimov/Gaussian shortcut, derived from scratch in
step 9a) for B ≳ 10; **HybridNew**
(LHC-style CLs toys, `--LHCmode LHC-limits --expectedFromGrid=q`, 500
toys/point, 5 quantiles) for the anchored sparse masses.
**Difference on identical cards:** asymptotic/HybridNew medians drop to ~0.6
below B ≈ 5 (step 10) — the shortcut is anti-conservative exactly where the
limit flattens. Two artifacts to expect from the toys: quantile noise at
3400–4200 (finite toys) and genuine band collapse at ≥ 4400 (with B ≲ 0.5
almost every background-only toy observes n = 0, so all quantiles coincide).

### 7. Band convention

Combine’s expected band is centred on the background-only expectation — the
`centre = zero` convention adopted everywhere (the homemade Stage-7 default
was flipped to match; `centre = mean` would let a negative spurious bias
*tighten* the expected limit, which is anti-conservative).

### Deliberately out of scope

No systematics nuisances (stat-only), resolved ee only (no µµ, no boosted, no
channel combination), LO DY without k-factor, and the anchor parameters are
not profiled (their spread is reported next to the band; promoting them to a
`param` nuisance is the natural next hardening step).

## Bottom line

Data-free expected σ·B(eeqq̄′) with the optimized float cards: 4.96 fb @
1 TeV, 0.82 @ 2, 0.23 @ 3, ~0.12 @ 4–6; expected m_WR exclusion
**~5.04 TeV** (official 2018 expected ~4.9, with systematics; we are below
the official expected everywhere from 1.4 TeV up). The full provenance:
converted run2 MC (`20260714_run2_bkgs`) → Stage-2 windows → Stage-6 window
table → Stage-9 run2 inputs / Stage-10.9 optimized inputs (windows + rates)
→ Stage-10.8 regime-split cards → this plot.
