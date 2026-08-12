# FWHM parameterization for signal-width constraints in the WR S+B fit

## Motivation

- **End goal:** set limits on WR→ℓℓjj using an S+B fit on ~20 events under the peak.
- **Problem:** with ~20 events, fitting a signal PDF with free width parameters is unstable — σ runs away, eats into the background.
- **Solution:** measure the signal width from MC once, parameterize it as a function of the mass grid, and use it as a prior constraint in the fit.
- **Signal PDF choice:** bifurcated Gaussian — 3 parameters (µ, σ_L, σ_R), fewer than a double-sided Crystal Ball's ~6. Appropriate for low stats.

## Stage 0 — What we started with

- `signal_fitting/measure_fwhm.py` (moved from `fitting_v2/`) already measures on-shell FWHM, bifurcated Gaussian parameters (µ, σ_L, σ_R), and a handful of other shape descriptors per (M_WR, M_N, channel) for every signal MC point.
- Output in `signal_fitting/outputs/<era>/fwhm/` includes `fwhm_<channel>.png` trend plots and `fwhm_summary.json` with all measurements.
- Run 2 (UL18) grid: 21 WR masses (2000–6000 GeV in 200 GeV steps) × several N masses each = 390 points per channel.
- Comparison showed ee and μμ FWHM behave differently: ee grows with both x = M_N/M_WR and M_WR; μμ is broader overall and dominated by M_WR scale.

## Stage 1 — FWHM parameterization (what we built)

**Script:** `signal_fitting/fit_fwhm_parameterization.py`. Re-uses the histogram loading and FWHM computation from `measure_fwhm.py` — no duplication, same single source of truth for "how FWHM is measured."

**What it does:**

1. **Discover the signal grid** for UL18 (WR ≤ 6000).
2. **Measure FWHM per point.** For each (M_WR, M_N, channel):
   - Load the `m_lljj` histogram (resolved SR, rebin=6 → 60 GeV/bin).
   - In the on-shell window `[0.7·M_WR, 1.3·M_WR]`, find the peak bin and linearly interpolate the half-max crossings → nominal FWHM.
   - **Bootstrap** the histogram with 100 Poisson-resampled toys, rerun the FWHM routine on each, `σ_bootstrap = std(toys)`.
   - **Add the binning resolution floor in quadrature** (see next subsection): `σ²_total = σ²_bootstrap + (Δm)²/6`, with `Δm = 60 GeV`. Median total uncertainty ≈ 46 GeV (ee), 53 GeV (μμ); minimum ≈ 25 GeV (the floor itself).
3. **Restrict to `0.10 ≤ x` and `x ≤ (M_WR − 100)/M_WR`** → 369 points per channel (drops only the boosted-N regime and the near-mass-degenerate endpoint already excluded upstream).
4. **Fit 4 candidate 2-D models** to `(x, M_WR) → FWHM` with `curve_fit(sigma=σ_i, absolute_sigma=True)`, χ² = Σ[(y-ŷ)/σ]²:
   - (a) `a·x + b·M_WR`
   - (b) `a·x + b·M_WR + c`
   - (c) `M_WR·(a + b·x)`
   - (d) `a·x·√M_WR + b·√M_WR`
5. **Per-mass linear diagnostic fits** — fit `FWHM(x) = a + b·x` at each M_WR separately, plot a(M_WR) and b(M_WR) with errors.

**Results** — we adopt model (a) `a_linear` in both channels:

| Channel | Parameterization                    | χ²/ndf | best (lowest χ²/ndf) |
|---------|-------------------------------------|--------|----------------------|
| ee      | FWHM ≈ 252·x + 0.0840·M_WR  [GeV]   | 1.40   | `b_linear_int` (1.40, tied) |
| μμ      | FWHM ≈  89·x + 0.137·M_WR   [GeV]   | 1.23   | `b_linear_int` (1.13)       |

χ²/ndf is now close to 1 because the per-point uncertainties combine the Poisson bootstrap with the binning resolution floor described next; before that floor was added, χ²/ndf was ~3–4 because the high-stats points had unphysically tight bootstrap-only error bars.

**Why `a_linear` and not `b_linear_int`.** The 3-parameter form `a·x + b·M_WR + c` wins the χ²/ndf race by a small margin in μμ (1.13 vs 1.23) and is essentially tied in ee. We prefer the 2-parameter `a·x + b·M_WR` for the width prior because (i) the F-test improvement of the extra parameter is not compelling at this χ²/ndf gap, (ii) a 2-parameter form is simpler to reason about and propagate covariance for, and (iii) at the (M_WR, M_N) points relevant to the analysis the two predictions differ by less than the residual-scatter floor we add anyway. `fit_fwhm_parameterization.py` records both models in `results.json` but pins `best_model = "a_linear"` for plots and downstream consumers.

### Why bootstrap alone undercounts the FWHM uncertainty

The bootstrap models "how much would the histogram fluctuate if we reran the MC generator." Each toy Poisson-fluctuates the bin counts on the **same fixed bin grid**, so the half-max crossings move only as much as count fluctuations push them. In the limit of infinite MC stats, `σ_bootstrap → 0`.

But the half-max crossings can still only be located to **`bin_width / √12`** — the standard deviation of a uniform distribution over a single bin (the same `Δx/√12` "strip detector" resolution). This is a sub-bin position uncertainty the bootstrap can't see, because all its toys live on the same coarse bin grid as the data.

FWHM = `x_right − x_left`, and the two crossings are independent (different bins), so the variances add:

```
Var[FWHM]_binning = (Δm/√12)² + (Δm/√12)² = (Δm)²/6
σ_binning         = Δm / √6 = √2 · Δm/√12   ≈ 24.5 GeV  for Δm = 60 GeV
```

Combined with the bootstrap in quadrature (independent sources):

```
σ²_total = σ²_bootstrap + (Δm)²/6
```

This raises the floor on every point's uncertainty to ~25 GeV, which dominates at high-stats points (where `σ_bootstrap` would otherwise be ~few GeV) and is a minor correction at low-stats points (where `σ_bootstrap` is 100+ GeV). Without it, the global fit would weight the high-stats points by `1/σ²` ~30× more than they deserve, giving artificially small parameter errors and a large χ²/ndf.

### Why even the parameterization-error prediction undercounts σ_σ for downstream use

The FWHM-prediction uncertainty `√(Jᵀ cov J)` is the *parameter* error of the global 2-parameter fit. At any single (x, M_WR) it's ~1–2 GeV — the parameters `(a, b)` are pinned that tightly by 369 data points. But the **residuals to the model at individual points scatter by much more** than that, because χ²/ndf ≈ 1.3 — the linear `a·x + b·M_WR` form is mildly imperfect.

For downstream use (constraining a width prior at one specific signal point), what matters is **how far the prediction can be from truth at that point**, not how well the global parameters are determined. So we add a per-point systematic floor in quadrature on FWHM_pred, capturing the residual scatter the model can't explain.

**Derivation.** χ²/ndf = `(χ²/ndf_observed − 1) · ⟨σ²_FWHM⟩` is the average excess variance per point that the per-point error bars don't capture:

```
F² = (χ²/ndf − 1) · ⟨σ_FWHM⟩²
```

| Channel | χ²/ndf | ⟨σ_FWHM⟩ | F = √(χ²/ndf − 1) · ⟨σ_FWHM⟩ |
|---------|--------|----------|------------------------------|
| ee      | 1.40   | 46 GeV   | √0.40 · 46 ≈ **29 GeV**       |
| μμ      | 1.23   | 53 GeV   | √0.23 · 53 ≈ **25 GeV**       |

**Default: `SYSTEMATIC_FWHM_FLOOR_GEV = 25`** — channel-independent value, slightly conservative for ee and exact for μμ. Pass `--systematic-floor-gev 0` to disable (recovering the parameterization-only stat error). Channel-dependent floors are not justified at this stage of the analysis.

**Effect on the prior at WR4000_N2000 ee:**

| floor | σ_FWHM | σ_σ (gauss space) | σ_Σ (bifur space) | constrained fit behavior |
|-------|--------|-------------------|-------------------|--------------------------|
| 0 GeV (default) | 2.05 GeV | 0.87 GeV | 1.74 GeV | pinned to prior central — data has no influence |
| 25 GeV | 25.08 GeV | 10.65 GeV | 21.30 GeV | data + prior in balance; fit moves with finite resistance |

The default-zero choice means constrained fits report the parameterization-only uncertainty. Add the floor when you want the prior to act as a real Gaussian prior rather than a near-hard constraint.

**Outputs** in `signal_fitting/outputs/RunIISummer20UL18/fwhm/fits/`:

- `fit_<channel>.png` — data + best-fit overlaid per M_WR
- `per_mass_<channel>.png` — diagnostic a(M_WR), b(M_WR)
- `results.json`, `results.csv` — parameters, errors, χ²/ndf for all 4 models; per-point `(tag, x, M_WR, fwhm, fwhm_err)` table

**Backend:** ROOT Minuit2 (CMS standard), via `TGraph2DErrors + TF2 + Fit("SEX0")`. Full 2×2 parameter covariance is saved to `results.json["<channel>"]["models"]["<name>"]["covariance"]` (with `param_order`) so downstream code can compute `FWHM_err(x, M_WR) = sqrt(Jᵀ cov J)` correctly — `a` and `b` are anti-correlated.

## Stage 2 — Connecting FWHM to PDF widths (the trick)

The same FWHM measurement feeds two different PDF width parameters with **different conversion factors**:

| PDF | Width parameter | Half-max condition | FWHM relation | Divisor |
|-----|-----------------|--------------------|----------------|---------|
| Single Gaussian | σ | x − µ = ±σ·√(2 ln 2) | FWHM = 2σ·√(2 ln 2) | **σ = FWHM / 2.3548** |
| Bifurcated Gaussian | Σ ≡ σ_L + σ_R | left/right HWHM = σ_{L,R}·√(2 ln 2) | FWHM = (σ_L+σ_R)·√(2 ln 2) = Σ·√(2 ln 2) | **Σ = FWHM / 1.1774** |

The two divisors differ by a factor of 2. The bifur divisor is *half* the gauss divisor because Σ is the *sum* of two HWHM-like widths, not 2× a single Gaussian σ.

Reparameterize the bifurcated PDF for fitting:

```
Σ = σ_L + σ_R     ← total width (constrained from the FWHM fit, divisor √(2 ln 2))
Δ = σ_R − σ_L     ← asymmetry (let data decide)
```

With µ fixed to M_WR (MC says peak matches M_WR to ~0.1%), the bifurcated Gaussian has effectively **one free shape parameter Δ** in the fit — everything else is constrained or fixed. That's the structure that makes a 20-event fit stable.

### Conversion-factor history (bug found and fixed, Feb 2026)

The earlier code applied the gauss conversion (FWHM/2.3548) to **both** PDFs, so the bifur prior central value was a factor of 2 too small (Σ_pred = 196 GeV at WR4000_N2000 instead of the correct 392 GeV). Symptom in the seed scan: bifur/constrain pulls were +4–11σ across 10 seeds, indicating systematic prior–data inconsistency. After fixing the convention so the bifur uses divisor √(2 ln 2):

| | wrong: Σ_pred = 196 | correct: Σ_pred = 392 |
|---|---|---|
| bifur/constrain mean Σ (10 seeds) | 250.0 ± 15.5 | 423.4 ± 17.1 |
| bifur/constrain pulls | +4.4 to +11.1σ | +0.7 to +3.8σ (mean +1.7σ) |
| bifur ΔNLL(constrain−free) at seed 12345 | +28 | +5 |

[`fit_signal_toy.py`](fit_signal_toy.py) now stores both divisors as named constants (`FWHM_TO_GAUSS_SIGMA` and `FWHM_TO_BIFUR_SIGMA`), and `predict_fwhm` returns FWHM-space numbers; each fit function does its own per-PDF conversion. The single-Gaussian conversion was always correct.

## Stage 3 — Signal-only toy fit (~5–20 events)

Before adding background, verify the signal PDF itself is well-behaved at low stats.

**Procedure:** generate `n_events` from the MC histogram at each grid point and fit with `BifurGauss(µ, σ_L, σ_R)` under a 4-cell prior grid:

| config         | µ prior                       | width prior (Σ)                    |
|----------------|-------------------------------|-------------------------------------|
| `no_priors`    | free, init `M_WR`             | free, init `Σ_pred`                 |
| `mu_only`      | N(M_WR, σ_µ=100 GeV)          | free                                |
| `width_only`   | free                          | N(Σ_pred, σ_Σ=σ_FWHM/1.1774)        |
| `both`         | N(M_WR, σ_µ=100 GeV)          | N(Σ_pred, σ_Σ=σ_FWHM/1.1774)        |

Δ is floated everywhere with a wide N(0, 4·σ_Σ) regulator. Single-Gaussian fits use the analogous gauss-divisor; otherwise identical structure (no Δ).

### Physical motivation for the prior widths

The α coefficients used downstream aren't arbitrary numbers — they correspond to fixed information ratios between data and prior. For an unbinned MLE on a Gaussian-like PDF with N events, the asymptotic (Cramér–Rao) variances are `Var[µ] = σ²/N` and `Var[σ] = σ²/(2N)`. Converting to FWHM units (factor 2.355) and substituting N = 20 events gives the natural data-only error scales:

```
σ_µ_data    = FWHM / (2.355 · √N)         ≈ 0.095 × FWHM    (at N = 20)
σ_FWHM_data = FWHM · √(2/N) / 2.355        ≈ 0.158 × FWHM    (at N = 20)
```

These are the answers a *no-prior* Gaussian fit would give in the asymptotic regime. The Bayesian posterior precision is `1/σ²_post = 1/σ²_data + 1/σ²_prior`, so choosing a target *data:prior information ratio* fixes `σ_prior`:

```
σ_prior = σ_data / √R    where R = data-information / prior-information
```

We pick **opposite ratios for µ and width**, motivated by physics:

| parameter | data quality at N = 20 | choice          | data : prior ratio | predicted α                       | empirical α |
|-----------|------------------------|------------------|--------------------|-----------------------------------|-------------|
| **µ**     | well-localised by the events themselves | data dominates | **6 : 1**          | 0.095 × √6 ≈ **0.23**             | 0.22 (ee), 0.25 (μμ) |
| **width** | poorly determined from a few half-max crossings | prior (MC) dominates | **1 : 6**          | 0.158 / √6 ≈ **0.064**            | 0.06 (ee), 0.08 (μμ) |

Reading the table:

- **µ:** data carries 6× the information of the prior. The prior is a soft regulator — it prevents the fit from running off in pathological toys but doesn't bias the central value. Empirical α (0.22–0.25) matches the predicted 0.23.
- **width:** the prior carries 6× the information of the data. MC anchors the fit; data only adjusts. Empirical α (0.06–0.08) matches the predicted 0.064.

The asymmetry comes from the underlying physics: 20 events is enough to pin a peak's *position* (events cluster at the right place), but not its *shape* (the half-max crossings depend on the tails, which are sparsely sampled). The 6:1 ratios encode "the prior is worth roughly 1/6 or 6× of what 20 events of data would tell you."

The small ee/μμ split (α_µ = 0.22 vs 0.25; α_w = 0.06 vs 0.08) is the residual non-Gaussian correction the Cramér–Rao bound doesn't capture — the bifurcated Gaussian is asymptotic to within ~10–15%, and ee/μμ shape differences put the two channels on slightly different sides of that.

The **N-dependence is implicit**: the σ_data above is for N = 20, the headline operating point. At lower N the data scale grows (`∝ 1/√N`), so the *effective* data:prior ratio changes — the µ prior becomes information-comparable to the data at N = 5, which is one reason `no_priors` fits are harder to converge there.

### Empirical calibration

**µ-prior calibration (May 2026).** The original setup used `σ_µ_prior = sigma_peak_boot` (the bootstrap stdev of the on-shell peak position from MC, ~20–60 GeV), which produced µ-pull spreads of 0.3–0.6 across the bulk grid — the fit was *over-covering* on µ by 1.5×–3×, the prior was tighter than the 20-event data could support. Recalibrated by scanning σ_µ at 16 bulk cells:

```
sigma_mu_prior = MU_PRIOR_ALPHA[channel] * FWHM_param(M_WR, x)
   alpha_ee   = 0.22
   alpha_μμ   = 0.25
```

Drives per-cell µ-pull spread to within ±15% of 1 in 13/16 bulk cells, mean |spread−1| ≈ 0.07. The two-channel α reflects the wider μμ peaks. See `signal_fitting/scan_loose_mu_prior.py`, `signal_fitting/pull_demo.py`, `signal_fitting/pull_cell_demo.py`.

**Width-prior calibration (May 2026).** The original setup used `FWHM_param` (the parameterization a*x + b*M_WR) as the width-prior central with parametric error ⊕ 25 GeV floor. That produced Σ-pull biases of −5σ to +14σ across the grid: the parameterization can be ~100 GeV off the actual MC FWHM at individual cells, so the prior was systematically pulled away from truth. Switched the prior central to `FWHM_boot` — the per-cell bootstrap-mean MC FWHM, which equals Σ_truth × FWHM_TO_BIFUR_SIGMA up to bootstrap noise — and the σ to a proportional form referencing `FWHM_param` (matching the µ-prior σ convention):

```
FWHM_prior_central     = FWHM_boot(M_WR, x, channel)             # per-cell MC bootstrap mean
sigma_FWHM_prior       = WIDTH_PRIOR_ALPHA[channel] * FWHM_param # global parameterization
   alpha_ee   = 0.06
   alpha_μμ   = 0.08
```

Reduces Σ-pull bias range from [−5, +14]σ to [−1, +2]σ at 16 bulk cells; Σ-pull spreads land in 0.71–1.30 (most in [0.85, 1.15]). A residual +0.5–1.5σ Σ bias remains across cells — feature of the bifurcated Gaussian's parameter correlations interacting with 20 events, not removable by prior tuning. See `signal_fitting/scan_loose_width_prior.py` for the calibration, `signal_fitting/pull_demo_width.py` / `pull_cell_demo.py --param width` for pedagogy.

**Three persistent µ-bias cells** (`WR3000_N1400` ee, `WR5000_N1400` ee, `WR5000_N2400` μμ) retain ~0.5σ µ-bias under every prior — intrinsic to the bifurcated-Gaussian fit's interaction with the asymmetric MC peak shape, documented as a fit-procedure systematic for Stage 4.

**Driver:** [`pull_study.py`](pull_study.py) loops over the full 369-point grid × {ee, μμ} × `n_events ∈ {5, 10, 15, 20}` × 100 toys × 4 configs × 2 PDFs. Resume-safe (incremental CSV append + `--max-masses-per-run` chunking) — needed because of a RooFit/cppyy memory leak that crashes a single Python invocation around ~200 mass points. [`pull_study_loop.sh`](pull_study_loop.sh) wraps it: re-execs Python in chunks of 80 masses until the CSV stops growing.

**Aggregate diagnostics** (in `outputs/<era>/pull_study/`, `--plot-only --skip-per-mass` regenerates only these):

| Plot                                        | What it shows                                                                                |
|---------------------------------------------|----------------------------------------------------------------------------------------------|
| `convergence_summary_<channel>.pdf`         | Convergence rate vs `n_events`, one curve per config, per PDF. Headline.                     |
| `pull_bias_{mu,width,delta}_<channel>.pdf`  | Median pull (across toys then masses) — bias check.                                          |
| `pull_spread_{mu,width,delta}_<channel>.pdf`| 1.4826 × MAD of the pull — error-calibration check.                                          |
| `outlier_mass_scan_<channel>.pdf`           | (M_WR, M_N) scatter coloured by worst-cell convergence/bias/spread — locates problem regions.|
| `per_mass/`                                 | One small diagnostic figure per mass (slow; skip with `--skip-per-mass`).                    |

**Headline findings (Run 2 UL18, 369 mass points × 100 toys, May 2026):**

- **Bifurcated Gaussian + "Both Constrained" is the only configuration that reaches ~100 % convergence at all `n_events ≥ 5`.** Single Gaussian converges ~100 % everywhere regardless of priors; bifur with no priors fails ~50 % at `n=5` and ~13 % at `n=10`.
- **µ pull bias:** Bifur stays within ±0.3σ of zero across all 4 configs. The single Gaussian "Width Constrained" config drifts to **−3σ** at `n=20` — forcing a symmetric core onto the asymmetric MC shape pulls the fitted peak. Symptom of model misspecification, not of the pipeline.
- **Width pull bias:** all configs over-estimate the width by **+1 to +2σ** at `n=20`; the constraint suppresses this, with bifur "Both Constrained" closest to zero.
- **Pull spread:** bifur "Both Constrained" is **slightly over-constrained on µ (spread ≈ 0.3–0.5)** and **slightly under-confident on Δ (spread ≈ 1.4–1.7)**. The µ over-constraint is the more concerning: a 100 GeV µ prior is tighter than the data deserves at the masses we care about. **Open question for Stage 4: widen σ_µ to ~200 GeV or revisit µ-prior pedigree.**
- **Failure geometry:** convergence failures of the unconstrained bifur cluster along **M_N → M_WR (boosted-N / high-x)** — the on-shell window narrows there and the fit can't separate σ_L from σ_R. Worst-bias points cluster at low M_WR (~2000–3000 GeV) where the FWHM is comparable to the 60 GeV bin width.

Production setting for Stage 4: **Bifurcated Gaussian, Both Constrained, σ_µ pending revisit.**

## Stage 4 — S+B fit

Once Stage 3 passes, bolt the background on:

```
L = Poisson(N | μ·S(m) + B(m)) × priors
```

with:

- `S(m) = BifurGauss(µ=M_WR, σ_L, σ_R)`, Σ constrained from this script, Δ free.
- `B(m)` = single/double exponential, τ constrained from `fitting/fit_single_exp.py` output.
- `μ` = signal strength, the POI.

The signal-shape and background-shape studies meet **only here** — both become Gaussian priors on different parameters in the same likelihood. They never talked to each other before this.

## Where the background fit fits in

Independent track. `fitting/fit_single_exp.py` (and the newer `fit_double_exp.py`) already fits exponentials to the ℓℓ control region or sideband and reports τ ± σ_τ. That uncertainty becomes the background nuisance in Stage 4. No connection to FWHM — just a parallel pipeline.

## Error chain (detail)

```
 Poisson bin counts                       binning resolution
        │  (100 Poisson toys → FWHM)               │  (Δm/√12 per crossing,
        ▼                                          ▼   ×√2 for two crossings)
 σ_bootstrap                              σ_binning ≈ 24.5 GeV
        └───────────── added in quadrature ─────────┘
                       ▼
 σ_FWHM per signal point         ← what each data point's error bar shows
        │  (weighted least squares with ROOT, all 369 points)
        ▼
 cov(a, b) from ROOT             ← from TFitResult::CovMatrix
        │  (error propagation: Jᵀ cov J)
        ▼
 σ_FWHM,stat(x, M_WR)            ← parameter-only prediction error
        │
        │   ⊕   F = √(χ²/ndf − 1) · ⟨σ_FWHM⟩ ≈ 25 GeV
        │       (residual scatter the linear model doesn't explain)
        ▼
 σ_FWHM(x, M_WR)                 ← total per-point FWHM-prior uncertainty
        │
        ├── ÷ 2.3548  →  σ_σ for the single-Gaussian prior (σ-space)
        └── ÷ 1.1774  →  σ_Σ for the bifurcated-Gaussian prior (Σ = σ_L+σ_R space)
```

The two divisors are the FWHM-to-width relations for the respective PDFs (Stage 2 table). Combine sees `σ_Σ × 1.1774 = σ_FWHM` for the bifur, or `σ_σ × 2.3548 = σ_FWHM` for the single Gaussian.

## Current status

| Step                                       | Status                                             |
|--------------------------------------------|----------------------------------------------------|
| Signal shape measurements per grid point   | Done (`measure_fwhm.py`)                     |
| FWHM parameterization with bootstrap errors| Done (`fit_fwhm_parameterization.py`)              |
| Save full covariance in JSON               | Done (ROOT TFitResult::CovMatrix)                  |
| Background shape fits                      | Done (`fitting/fit_single_exp.py`, `fit_double_exp.py`) |
| Signal-only toy fits, single-mass diagnostics | Done (`fit_signal_toy.py`, 4-cell prior grid)   |
| Signal-only pull study, full mass grid     | Done (`pull_study.py`, 2.36 M fits — see headline findings above) |
| S+B fit + Combine datacard generator       | TODO (Stage 4)                                     |

## Files

- `signal_fitting/measure_fwhm.py` — per-point shape measurements
- `signal_fitting/fit_fwhm_parameterization.py` — FWHM(x, M_WR) fit
- `signal_fitting/fit_signal_toy.py` — single-mass single-toy 4-config comparison (gauss/bifur × prior grid)
- `signal_fitting/pull_study.py` — full-grid pull study (resume-safe, ~2.36 M fits per run)
- `signal_fitting/pull_study_loop.sh` — chunked-Python wrapper around `pull_study.py` (RooFit memory-leak workaround)
- `signal_fitting/bootstrap_demo.py` — diagnostic: original MC vs one Poisson-resampled toy
- `signal_fitting/outputs/RunIISummer20UL18/fwhm/fits/` — FWHM fit outputs
- `signal_fitting/outputs/RunIISummer20UL18/signal_toy_compare/` — single-mass 4-config plots
- `signal_fitting/outputs/RunIISummer20UL18/pull_study/` — pull-study aggregate plots + `results.csv`

### Reproducing legacy / pre-fix behavior

All on-disk outputs reflect the **current** pipeline. To regenerate older states for comparison, use these CLI flags (always write to a separate `--output-dir` so canonical outputs aren't overwritten):

| Pipeline change | Default (current) | Flag to revert |
|------------------|--------------------|----------------|
| Stage 1 binning resolution floor | enabled | `fit_fwhm_parameterization.py --no-binning-floor` |
| Toy-side residual-scatter floor (FWHM) | 25 GeV | `fit_signal_toy.py --systematic-floor-gev 0` (no floor) or `--systematic-floor-gev 50` (older 50 GeV state) |
| Bifur conversion `Σ = FWHM / 1.1774` | correct | `fit_signal_toy.py --legacy-bifur-conversion` (uses ÷2.3548, the pre-fix bug) |
| Stage 1 plotted/best model = `a_linear` | pinned | (no flag; remove the pin in `fit_fwhm_parameterization.py` to fall back to lowest-χ²/ndf selection) |

Example — reproduce the original Σ_fit = 270 single-panel bifur plot at WR4000_N2000 ee, seed 12345:
```
python signal_fitting/fit_signal_toy.py --wr 4000 --n 2000 --channel ee \
    --seed 12345 --only-config bifur_constrain \
    --legacy-bifur-conversion --systematic-floor-gev 50 \
    --output-dir signal_fitting/outputs/RunIISummer20UL18/signal_toy_compare/legacy_floor50
```

Example — disable the floor (parameterization-only stat error, corresponding to `Σ_fit ≈ 392` at WR4000_N2000 seed 12345):
```
python signal_fitting/fit_signal_toy.py --wr 4000 --n 2000 --channel ee \
    --seed 12345 --systematic-floor-gev 0 \
    --output-dir signal_fitting/outputs/RunIISummer20UL18/signal_toy_compare/no_floor
```
