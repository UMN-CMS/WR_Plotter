# Stage 10 — limit refinement

> **STATUS: ARCHIVED (Jul 2026).** Study log only — nothing operational lives here. The production limit and its optimization are `../../8_combine_limits/{production,optimization}`; the 10.7 combine explainer moved to `../../8_combine_limits/internals_explainer/`. 10.1–10.6 below are the evidence base for the production design (Poisson core, prior widths, anchored estimator, asymptotic-validity boundary). See `../../LIMITS.md`.

Fixes the three problems the Stage 5–9 expected-limit chain was known to have,
**without touching any earlier stage** (all scripts here are new; they import
`bkg_fit_lib` / `sb_fit` read-only):

1. **Non-Gaussian `nsp_hist` outside a core mass window.** Only
   m_WR ≈ 1400–3200 had Gaussian toy distributions; m_WR = 1000 is broken by
   the 800 GeV selection threshold (no left sideband) and m_WR ≥ 3400 by
   window sparsity (3–11 events) — convergence collapse + survivor bias +
   quasi-discrete spike-at-zero.
2. **Fixed signal shape.** Stage 5–8 fixed the Gaussian at the Stage-2 linear
   (μ₀, σ₀); the Stage-1 U-shape (σ_true/σ₀ ∈ [0.67, 1.77] vs x = m_N/m_WR)
   means the median under-covers the x-extremes.
3. **mean+RMS + Gaussian-CLs band.** Stage 6 discarded the raw toys, forcing
   Stage 7 into a Gaussian closed form exactly where it is invalid.

A 20-agent deep review (2026-07-11) confirmed every mechanism quantitatively;
the load-bearing verified findings are summarized at the bottom.

## Setup

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```

## Shared core

### `prior_fit_lib.py` — the Stage-10 S+B fit

`fit_splusb_v2(...)` generalizes `shared/sb_fit.fit_splusb` along three axes
(hand-rolled FCN + Minuit2 via `ROOT.Math`; 1–12 ms/fit):

| axis | options | why |
|---|---|---|
| statistic | `chi2` (Stage 5–8 convention, **bit-compatible**: median \|ΔN_sig\| ≤ 0.02 on identical toys) / **`poisson`** (Baker–Cousins, **empty bins included**) | the chi² path skips `data==0` bins and needs ≥ npar+2 populated bins → occupancy-gate convergence collapse, survivor bias, and a mass-dependent zero-truncation bias (toys recover only 0.5–0.85 of injected signal). The Poisson core removes all three. |
| signal shape | (μ, σ) fixed / **Gaussian prior** `s_mu`, `s_sigma` (in GeV; `None` = free within boxes μ₀ ± 1.5σ₀, σ ∈ [0.5, 2.5]σ₀) | the U-shape width variation |
| background | free (default) / per-parameter Gaussian constraint / **fixed** (`bkg_constraints`) | the envelope's anchored estimator at untrusted masses |

Poisson mode bounds N_sig from below at −5× the observed ±2σ-core count —
without it, near-empty windows produce runaway N_sig ≈ −10³ balanced by an
inflated background (measured: RMS 6077 at m=4200 unguarded → 3.4 guarded).
Rail flags (`mu_railed`, `sigma_railed`, `nsig_railed`) are returned per fit.

### `toy_engine.py`

Input bundling (`Inputs`: background, Stage-2 windows, signal templates from
the canonical `20260624_signals`), the toy loop with **full acceptance
bookkeeping** (every toy is recorded, failures included), and **raw per-toy
CSV persistence** — the Stage-6 lesson: never keep only mean+RMS summaries.

## Steps

### 1 — `1_nsp_diagnostics/nsp_diagnostics.py`

Null toys for chi² *and* Poisson at every grid mass, raw toys saved,
Gaussianity scored (Jarque–Bera, quantile/RMS ratios r68/r95, skew) and
classified TRUSTED / MARGINAL / BROKEN with an explicit rule.

Result (ee resolved, expo): the Poisson core is strictly better everywhere —
at 2800: RMS 9.7 → 7.1, r68 0.92 → 1.00, JB p 0.0 → 0.87; at 3200: RMS
9.6 → 5.3. 3400–3600 move from BROKEN to MARGINAL (spread 3.5–3.8 vs 9–15).
m ≥ 3800 stays BROKEN on convergence (0.3–0.7) → envelope regime. m = 1000
stays BROKEN structurally (clamped window) → envelope regime.

### 2 — `2_prior_scan/prior_scan.py` + `select_prior.py`

6×6 grid of (α_μ, α_σ) prior widths (fractions of σ₀; 0 = fixed, inf = free),
**paired toys** (each Poisson draw fit by all 36 configs — config differences
carry no toy noise), null + N=9 MC-template injections at x ∈ {min, 0.2, 0.5,
0.9}, masses = trusted {1400, 2000, 2800, 3200} + edge diagnostics.

`select_prior.py` gates and ranks on the **recovery deficit**
`[mean(N=9) − mean(null)] − 9·W` where W is the template's in-window fraction:

- the additive mismodeling spurious (null mean, e.g. −19 events at 1400) sits
  identically in both cells — no signal prior can remove it, so raw bias would
  fail everything for the wrong reason;
- the out-of-window loss 9(1−W) (up to ~45 % for compressed x=min shapes) is
  bookkept downstream in the xsec efficiency, not a fit failure.

**Winner: (α_μ = 0, α_σ = 0.3)** — μ stays fixed, σ floats with a 0.3σ₀
prior. Sole config passing all gates: recovery improves (B 4.18 → 3.88) at
**zero null-spread cost** (S = 1.00). Any floating μ inflates the null spread
(S 1.28–1.58) — the Gaussian latches onto background fluctuations; free σ is
catastrophic (B ≈ 45).

### 3 — `3_injection_validation/validate_prior.py`

Fresh-seed re-test (1000 toys) of winner vs fixed vs the cheap alternatives
(static σ inflation ×1.2 / ×1.35) vs free, N ∈ {0, 9, 20}, plus the
`gauss_matched` harness closure (inject the exact fit Gaussian: any deficit
left is estimator floor, not shape mismatch). Scoreboard =
`validation_table_{tag}.csv` + deficit/spread plots.

### 4 — `4_bkg_envelope/bkg_envelope.py`

The envelope for the untrusted masses (user goal 1). Anchor fits of the summed
MC over the trusted spectrum ([1000, 3500] central expo; expo2/powexp
function variants; range variants incl. a tail-anchored [1000, 6000] fit;
an `mcmax` member = max(central, MC) over the tail covering "the tail is
real"), refit per target mass with the pivot at that window's m_c so the
parameters transport exactly. At the target mass the S+B fit runs with the
background **fixed** to the member (`bkg_constraints`) — only N_sig floats.
This one estimator fixes both ends: no left sideband needed at m=1000, no
in-window background freedom to run away at m ≥ 3400.

Per mass: `sigma_stat` (anchored-toy half68 on the central smooth
expectation), `sigma_model` (max Asimov shift across members), `sigma_theta`
(covariance-sample RMS), `mu0` (realMC Asimov = MC-shape spurious; used as
band centre only ≤ `--mu0-max` = 3300 — above that the MC tail is
single-large-weight-event noise, carried instead by the mcmax member).
**`rms_Nsp` in the prediction CSV = `sigma_stat` only** — model/theta are
forecast uncertainties and belong ON the band, not inside the CLs σ
(`sigma_total` is kept as a deliberately conservative variant column).

Closure (smooth-central toys refit with the *floating* fit vs the direct 10.1
toys): 1600–2800 pass (ratios 0.89–1.11); 3200 gives 0.66 — understood: the
direct spread there includes jagged-MC weight-noise (per-bin rel. err. ~0.3–
0.5) that the smooth generator correctly lacks.

Numbers (ee resolved): m=1000: σ_stat 35, model 59 (down-extrapolation
curvature dominates — honest), μ0 = −10 (vs +90 for the broken floating fit).
High mass: σ_stat 1.6–3.0, model 1.2–3.3, μ0(diag) +1.7–3.3.

### 5 — `5_expected_limits_v2/expected_limits_v2.py`

Regime-split band:

- **fit regime** (1400–3200): empirical 2.5/16/50/84/97.5 % quantiles of
  per-toy CLs ULs `UL_i = UL(N_i, c·σ_i)` from the raw toys (clamp-free form
  `UL = N − σ·Φ⁻¹(α·Φ(N/σ))`, positive always). No Gaussianity or
  homoskedasticity assumption; failed toys gate which quantiles are quotable.
- **counting regime** (1000–1200, ≥ 3400): exact discrete construction —
  n ~ Poisson(b_envelope), per-count Poisson-CLs UL by bisection, band =
  discrete quantiles. Deterministic; no fits, no convergence dependence.
  Envelope model/theta spread → alternative-b median columns.
- **centre = zero** by default; the spurious signal can only *widen* the band
  (`--spurious widen` takes the per-quantile max of the centred and
  bias-shifted bands) — the Stage-7 default (`--center mean`) let a negative
  fit bias *tighten* the expected limit (anti-conservative, review finding).
- overlays the Stage-7 closed form so the change is visible.

### 6 — `6_combine_parity/` (make_workspaces_parity.py, run_parity.sh, compare_parity.py)

Card-level equivalence test against CMS combine (which is a binned Poisson-ML
+ profile-likelihood-CLs machine — the convention the Stage-10 core adopted).
Cards set `rate = 1` so combine's r is the signal yield in events, directly
comparable to the 10.5 band. Three variants: `fixed` (Stage-9 model),
`prior` (`sigma_sig param σ₀ 0.3σ₀` — the 10.2 winner as a combine
nuisance), `anchored` (background slope+norm fixed to the 10.4 envelope; the
card version of the anchored estimator, for HybridNew at sparse masses).

Results (ee resolved, expo, events; `parity_table_ee_resolved.csv`):

| m_WR | combine fixed (asymp.) | v2 med | ratio | prior (asymp.) | prior (HybridNew) |
|---|---|---|---|---|---|
| 2000 | 36.7 | 40.8 | 0.90 | 49.5 | 60.8 (fixed HN: 37.5) |
| 2400 | 23.2 | 24.8 | 0.94 | 31.0 | — |
| 2800 | 16.2 | 16.4 | 0.99 | 19.9 | — |
| 3200 | 9.5 | 9.1 | 1.04 | 10.2 | — |
| 4000 | 2.9 (anchored) | 6.0 (counting) | 0.48 | HybridNew 3.25 | ratio 0.54 |
| 4600 | 2.3 (anchored) | 5.2 (counting) | 0.44 | HybridNew 3.25 | ratio 0.63 |

Conclusions:
1. **Parity achieved for the fixed model**: 0.90–1.04 (the Stage-9 era 0.78
   is gone — it was the chi²-vs-ML estimator plus the σ convention).
2. **The σ-prior as a profiled combine nuisance costs real sensitivity at low
   mass**: +35 % (asymptotic) / +62 % (HybridNew) at 2000, falling to +7 % at
   3200. The v2 per-toy band (parabolic per-toy errors, no re-profiling per
   tested r) under-prices this. ⇒ for the LIMIT model keep the shape fixed
   (combine-official) and carry the U-shape width variation as a
   bias/spurious systematic from the 10.3 tables; the floating-σ fit remains
   the right tool for the bias-methodology studies themselves.
3. **Sparse masses**: anchored-card HybridNew (3.2–3.3 events) sits ×0.5–0.6
   below the 10.5 counting band — the in-window shape carries real
   discrimination a single-bin count ignores; the counting band is the
   conservative bookend. AsymptoticLimits under-covers there (2.3 vs 3.25 at
   4600, b = 0.4) — never quote it below ~b = 5; HybridNew on these cards is
   the analysis-grade number.

### 8 — `8_run2_refined_limit/` (the refined run2 2018 ee limit, combine-only)

The Stage-10 conclusions applied to the run2 limit with nothing but combine
cards (see its own README): regime-split cards — float (1400–3200, since
Jul 2026 the **Stage-10.9 optimized card**: k5/50 GeV/slope-constrained/
signal-both030), anchored (1000–1200 left-clamped + ≥3400 sparse; background
fixed to in-script Poisson-ML anchor fits of the run2 MC, four members for
the model spread) — with HybridNew expected quantiles where AsymptoticLimits
under-covers. Result (vs the Stage-9 run2 baseline): 0.38×/0.68× at
1000/1200, 0.44–0.85× at 1400–3200, 0.45–0.6× at 3400–4600, honestly ≥1× at
5800–6000 (asymptotic under-coverage corrected); expected m_WR exclusion
~4.95 → ~5.04 TeV.

### 9 — `9_float_region_optimization/` (the 2–3.2 TeV configuration scan)

18 card configurations × 7 masses (window k3/k4/k5/a53 × binning 100/50/20 ×
background float/bconstr/bfixed/anch × signal fix/µ/σ/both-constrained), all
AsymptoticLimits on the MC Asimov. Winner **k5_bw50_bconstr(+both030)**:
geometric-mean 0.60× the k3/floating baseline; finer binning alone does
nothing; the two ~27% levers (sideband width, imported anchor slope)
compound. Signal-shape retest: with the slope constrained, profiling µ and σ
at 0.3σ₀ costs only +3.5% (vs +22% on the floating-background card — the old
10.6 cost was the signal↔background degeneracy). FitDiagnostics toy
validation (`run_sigscan_and_toys.sh`, `toy_validation_table.csv`): null
spurious < 2% of the medians, injection recovered to 1–5%, pull RMS ≈ 1
(flags: 1.17 at 2800; 66% convergence at 3200).

### 7 — `7_combine_explainer/` (combine from scratch, step by step)

Pedagogical rebuild of combine's internals on the real m=2000 parity card,
each step checked against genuine combine outputs (FitDiagnostics /
HybridNew / AsymptoticLimits). Chain: ingredients → one-bin Poisson →
product over bins → the −2lnL curve (= the familiar yield ± error) →
profiling → q̃_μ → toy CLs → the limit → the Asimov shortcut → the Brazil
band. Punchline: the old `7_limit_plots/step2_explainer` is derived as the
asymptotic Gaussian corner of this machinery. See its own README.

## Order of operations

```bash
python 1_nsp_diagnostics/nsp_diagnostics.py -v
python 2_prior_scan/prior_scan.py -v            # shard with --masses / --output-dir
python 2_prior_scan/select_prior.py -v --table 2_prior_scan
python 3_injection_validation/validate_prior.py -v
python 4_bkg_envelope/bkg_envelope.py -v
python 5_expected_limits_v2/expected_limits_v2.py -v
```

## Verified review findings this stage rests on (2026-07-11, all CONFIRMED)

| finding | number |
|---|---|
| n_ok collapse at ≥3400 is the ≥ npar+2 populated-bin gate, predicted exactly by Poisson occupancy | P(occ≤3): 0.038@3400 → 1.0@5800 matches 1−n_ok/1000 to ≤2 % |
| survivor selection biases the kept mean positive at high mass | m=5000: survivors mean +8.4 vs full-ensemble +1.3 |
| Neyman chi² zero-truncation tilts the background mass-dependently; pulls 1.17–1.57 | fixed point ×3.6 truth at μ=0.3/bin, ×0.77 at μ=3 |
| chi² toys under-recover injected signal multiplicatively (no prior can fix it) | at 3000, N_inj=10: chi² recovers 2.1, Poisson 7.9 |
| m=1000 is structural collinearity (no left sideband below the 800 GeV cut) | histogram empty < 800; window [651, 1275] clamped |
| Stage-7 σ=RMS + centre=mean conventions | combine/homemade = 0.78 only partly the pull-width factor; ~9 % is a genuine estimator (chi² vs ML) difference |
| unguarded Poisson likelihood runs away on near-empty windows | mean −234, RMS 6077 at 4200 → bounded core: RMS 3.4 |
| xsec table published into the broken regime (no --max-mass) | 3400–5000 rows from 10–45 % survivor subsets |

Caveats inherited from Stages 5–8 (unchanged): Run-2 signal shapes on a Run-3
LO-DY background (k-factor 1.0); the boosted topology remains out of scope for
in-window spurious methodology (B_window ≈ 0–4 events at every k).
