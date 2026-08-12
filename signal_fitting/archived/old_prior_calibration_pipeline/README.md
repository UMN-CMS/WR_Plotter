# Single-Gaussian signal-fit calibration

Tune a single-Gaussian (no bifur) signal PDF for the W_R → ℓℓjj fit. Two
steps, nothing else.

## Plan

### Step 1 — Bootstrap centrals (existing convention)

Per (channel, mass) cell, the prior centrals are:

- `mu_central     = mu_boot`     — peak-finder bootstrap on MC histogram
- `sigma_central  = FWHM_boot / 2.3548`  — bootstrap-mean MC FWHM converted to gauss σ

Both come from Poisson resampling the MC bin counts (100 toys) and recomputing
the relevant quantity per resample. Already implemented in `fit_signal_toy.py`:
`bootstrap_peak_estimate()` and `bootstrap_fwhm_estimate()`. **No new fits on
MC. No high-stats reference fits.**

### Step 2 — Tune prior widths (the actual work)

Replace the current `α × FWHM_param` width convention with something else, then
scan to drive median pulls toward 0 and pull spreads toward 1, across all 390
masses × 2 channels.

The target is a width *function* — a single rule that gives a sensible prior
width per cell — with a small number of tunable knobs that we scan.

Candidate forms for the width function (pick one or compare a couple — TBD):

```
sigma_mu_prior   (M_WR, x, channel) = f_µ (knobs, ...)
sigma_sigma_prior(M_WR, x, channel) = f_σ (knobs, ...)
```

Some candidate `f`'s (deliberately *not* FWHM-parameterization based):

| form                                             | knobs                   | notes                                                                 |
|--------------------------------------------------|-------------------------|------------------------------------------------------------------------|
| `α × FWHM_boot`                                  | α per channel           | per-cell scale, no parameterization                                    |
| `α × mu_boot`                                    | α per channel           | mass-proportional, no width-info dependence                            |
| `α × M_WR`                                       | α per channel           | label-only — useful if FWHM_boot is noisy                              |
| `sqrt(α² × FWHM_boot² + β²)`                     | (α, β) per channel      | floor + proportional, in case very-narrow cells need a minimum width   |
| constant per channel                             | fixed value per channel | global; tests whether per-cell scaling matters at all                  |

We don't need to be exhaustive. We pick a candidate form (likely
`α × FWHM_boot`), scan α(s) on the bulk cells, check the resulting pulls
across the full grid.

## Directory layout

```
signal_fitting/
├── PIPELINE.md                (end-to-end walkthrough)
├── README.md                  (this file)
├── shared/                    (fit_signal_toy.py, measure_fwhm.py)
├── 1_signal_samples/          (signal_samples.py — Step 1)
├── 2_tune_priors/             (scan_priors.py, plot_scan_2d.py, sensitivity.py — Step 2)
├── 3_scan_full/               (scan_full.py — Step 3)
├── 4_plots/                   (plots.py, plot_alpha_demo.py, plot_fit_vs_truth.py — Step 4)
├── outputs/                   (results.csv, scan_2d_*.csv, plots/…)
└── archived/                  (retired scripts kept for history)
```

## Compute estimate

- **Prior scan**: 8 bulk cells × 2 channels × N ∈ {5, 10, 20, 50, 100} × 100 toys
  × ~5 width candidates × 1 PDF × 1 config = ~40k fits → ~30 min on LCG_106.
- **Full grid**: 390 masses × 2 channels × N ∈ {5, 10, 20, 50, 100} × 100 toys
  × 1 width choice × 1 PDF × 1 config = ~390k fits → ~4 h (chunked with the
  existing memory-leak workaround in `pull_study_loop.sh`).

## What I'm not doing (and why)

- No fit-window scan. The window stays at the current
  `[ONSHELL_WINDOW_LO_FRAC × M_WR, ONSHELL_WINDOW_HI_FRAC × M_WR]`.
- No `no_priors` / `mu_only` / `width_only` configs. Both µ and σ priors are
  always on. The only knobs are the prior widths.

## Decisions log

| date       | decision                                                | source             |
|------------|---------------------------------------------------------|--------------------|
| 2026-05-20 | Single-Gaussian PDF only; both µ + σ priors always on   | user               |
| 2026-05-20 | First pass: centrals = bootstrap peak/FWHM, prior widths = α × FWHM_boot | user |
| 2026-05-20 | First-cut α: α_µ=0.10, α_σ=0.05 (uniform), then ee=0.05/mumu=0.07 | scans |
| 2026-05-20 | First pass had structural µ-bias ~-2σ and width-bias ~+2σ at N=20 (gauss MLE landed on the windowed mean/RMS, not the peak/FWHM) | analysis |
| 2026-05-20 | Switched truth and prior centrals to windowed (mean, RMS) of MC in `[0.7,1.3]×M_WR`. Bias dropped to ~0 everywhere | user idea |
| 2026-05-20 | Switched prior-width yardstick from FWHM to RMS: σ_µ_prior = α_µ × RMS, σ_σ_prior = α_σ × RMS | user |
| 2026-05-20 | Retuned α with moment truth: α_µ saturates above ~2 (uninformative); α_σ=0.15 for both channels | 2D scan + full-grid validation |
| 2026-05-20 | Picked α_µ = 2.0, α_σ = 0.15 (both channels) with numpy PCG64 toy sampling | full-grid validation |
| 2026-05-21 | Switched moment + toy-sampling RNG from numpy to ROOT (TH1::GetMean/StdDev, TH1::GetRandom with Knuth-hashed seed); retuned α to compensate for TRandom3 vs PCG64 character | "use ROOT everywhere" goal |
| 2026-05-21 | α_µ = 1.0, α_σ = 0.20 (both channels), ROOT-RNG sampling — final under the robust half-68% pull metric | full-grid validation |
| 2026-05-28 | Switched primary calibration metric from robust half-68% to a binned Gaussian fit (`shared/pull_stats.py:gaussian_pull_fit`) across all scripts | standard HEP convention |
| 2026-05-28 | **Final: α_µ = 1.0, α_σ = 0.25 (both channels), ROOT-RNG sampling, Gaussian-fit metric** — retuned from 0.20 because the metric change moved the σ-spread at the operating point | prior scan + full-grid validation |
| 2026-05-28 | Tighter recheck scan (α_µ ∈ {0.8,…,1.2}, α_σ ∈ {0.21,…,0.29}, 25 cells × 100 toys, N=20) confirmed (1.0, 0.25): ee w-spread 1.04, µµ w-spread 0.99 — both within ±0.05 of target. Channels prefer opposite α_σ extremes (ee→0.21, µµ→0.27); 0.25 is the two-channel compromise. | `outputs/scan_2d_recheck_0p25.csv` |
| 2026-06-01 | Width-estimator cross-check (`1_signal_samples/compare_width_estimators.py`): tabulated σ_eff(float), σ_eff(sym), σ_Gauss-fit, σ_FWHM (from a smooth RooKeysPdf, no rebin), and σ_RMS-full on the native MC for all 390 masses × 2 channels × 2 topologies. **Resolved shapes are strongly non-Gaussian** — narrow core + heavy (mostly low-side) tail: median σ_RMS_full overestimates σ_eff(float) by ~39% (ee) / 32% (µµ), σ_FWHM is only ~57% (ee) / 64% (µµ) of σ_eff(float), and the full estimator span reaches ×2.5 (ee) / ×2.1 (µµ), worst at low M_N/M_WR. Boosted shapes are far more Gaussian (estimator span ≲ ×1.3). Takeaway: the *un*windowed RMS is tail-dominated and not interchangeable with σ_eff; the production truth's `[0.7,1.3]×M_WR` window is doing real work (windowed RMS tracks σ_eff(float) to a few %). | `outputs/width_estimators.csv` |
| 2026-06-01 | Per-method deep dive for σ_eff(float) (`1_signal_samples/detail_eff_gauss.py` → `outputs/width_estimators/eff_gauss/`): per-cell MC histogram with the 68.27% interval `[x_low, x_high]` drawn, plus σ_eff/M_WR, interval-edge/center, 2σ-containment, and interval-tail-fraction trends + a per-point table. Confirms a **strong low-side tail** (median fraction below x_low ≈ 0.22–0.26 vs above x_high ≈ 0.05–0.10) and **sub-Gaussian 2σ containment** (≈0.85–0.89 < 0.9545), consistent with the non-Gaussianity above. σ_eff/M_WR ≈ 0.09 (resolved) / 0.13–0.14 (boosted). The cross-method comparison plots moved to `outputs/width_estimators/comparison/`. | `outputs/width_estimators/eff_gauss/eff_gauss_table.csv` |
| 2026-06-01 | **Boosted cells restricted to the boosted regime x = M_N/M_WR ≤ 0.3** (`--boosted-max-x`, default 0.3, in both width scripts). Above it the boosted SR is unphysical and the smallest-68% window latches onto off-shell structure — at x=0.5 the interval center sat at ≈0.70 M_WR; within x ≤ 0.3 it sits at ≈0.90 M_WR, on the resonance. Drops 303/390 boosted masses per channel (keeps 87). Resolved keeps the full grid. | user |
| 2026-06-02 | **Pre-window stability is the selection criterion.** Recomputed each on-shell width for windows [0.7,1.3], [0.8,1.2], [0.85,1.15] (`width_window_stability.py`). σ_eff^on (median \|Δ\| ≈ 16%) and RMS^on (≈29%) are **window-driven** (their target integrates the windowed tails); the **iterative-core Gaussian σ (≈1.7%)** and the **RooKeysPdf FWHM (≈0.6%)** are **robust** (both key off the peak, not the tails). | `outputs/width_estimators/{gauss_fit,fwhm}/window_stability/` |
| 2026-06-02 | **Boosted redefined as x < 0.1**, discovered from disk (the production CSV stops at x≥0.1) — the genuinely highly-boosted regime, 51 masses/channel down to x=0.017. `collect_cells` now sources boosted from `discover_masses`. | user |
| 2026-06-02 | **Dropped σ_eff, σ_eff-sym, RMS, and the wide best-Gaussian fit; kept only the two robust widths** — σ_gauss^on (iterative core, `detail_gauss_fit.py`) and σ_FWHM^on (`detail_fwhm.py`), compared in `compare_gauss_fwhm.py` (σ_FWHM^on ≈ 0.75–0.85 × σ_gauss^on). Removed `compare_width_estimators.py`, `detail_eff_gauss.py`, the dead `shape_estimators` primitives, and the `eff_gauss/`, `rms/`, `comparison/` outputs. | user |

## Final calibration (2026-05-20)

For each cell, define the windowed first two moments of the MC mass histogram:

```python
fit_lo, fit_hi = ONSHELL_WINDOW_LO_FRAC * M_WR, ONSHELL_WINDOW_HI_FRAC * M_WR
centers = 0.5 * (edges[:-1] + edges[1:])
in_window = (centers >= fit_lo) & (centers <= fit_hi)
mean = sum(centers[in_window] * vals[in_window]) / sum(vals[in_window])
RMS  = sqrt(sum(vals[in_window] * (centers[in_window] - mean)**2) / sum(vals[in_window]))
```

Then for the production gauss fit:

```
mu_prior_central     = mean
sigma_prior_central  = RMS
σ_µ_prior            = 1.00 × RMS    # both channels
σ_σ_prior            = 0.25 × RMS    # both channels
```

Pulls are measured against `mu_truth = mean` and `width_truth = RMS`.

The toy event sampling uses ROOT TH1::GetRandom with a Knuth-hashed seed
(`seed × 2654435761 mod 2^32`) — see `shared/fit_signal_toy.py`
`sample_from_hist_root` for details. The Knuth hash is needed because
TRandom3 (= ROOT.gRandom) has known poor initialization for close integer
seeds. The α values above are tuned for this RNG; if you switch back to
numpy PCG64 sampling, the equivalent numpy calibration is
(α_µ = 2.0, α_σ = 0.15).

Validation at N=20 (operating point), full 390-mass grid, binned Gaussian-fit
pull metric:

| metric                    | ee      | mumu    |
|---------------------------|---------|---------|
| median µ-bias             | −0.13   | −0.16   |
| median µ-spread           | 0.94    | 0.96    |
| median width-bias         | 0.02    | 0.04    |
| median width-spread       | 1.03    | 0.98    |
| µ-spread within ±0.1 of 1 | 98%     | 100%    |
| µ-spread within ±0.2 of 1 | 100%    | 100%    |
| w-spread within ±0.1 of 1 | 88%     | 91%     |
| w-spread within ±0.2 of 1 | 99%     | 100%    |

Production CSV: [`outputs/results.csv`](outputs/results.csv) (390k rows).
Production plots: [`outputs/plots/`](outputs/plots/).

```
