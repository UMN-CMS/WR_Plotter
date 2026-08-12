# Width parameterization — interpretation

Candidates: 1a = σ_gauss^on/M_WR, 1b = σ_FWHM^on/M_WR, modeled vs x = M_N/M_WR
(and M_WR where it helps; m = (M_WR−4000)/4000).
Metric: leave-one-M_WR-out CV median |(pred−meas)/meas| (`cross_validation/cv_summary.csv`).
Ranges: resolved x ∈ [0.10, 0.98], boosted x ∈ [0.017, 0.095] — disjoint, never
interpolate/extrapolate across them.

Models compared (10): **1D in x** — pol2, pol3, pol4, physics (pole at x→1,
resolved only), spline; **2D in (x, M_WR)** — poly3+mass, fxgx, +m2, +x2m, spline2d.

## Lowest-CV model per cell
| width | ch | cat | best (lowest CV) | CV med | CV q95 |
|---|---|---|---|---|---|
| gauss | ee   | resolved | spline2d   | 2.7% |  8% |
| gauss | ee   | boosted  | fxgx       | 3.0% | 22% |
| gauss | mumu | resolved | spline2d   | 1.8% |  6% |
| gauss | mumu | boosted  | spline2d   | 2.2% |  7% |
| fwhm  | ee   | resolved | fxgx       | 4.9% | 16% |
| fwhm  | ee   | boosted  | poly3+mass | 6.7% | 20% |
| fwhm  | mumu | resolved | fxgx       | 3.7% | 12% |
| fwhm  | mumu | boosted  | +x2m       | 3.1% | 13% |

## Findings

1. **The flexible 2D models win on raw CV — but mostly by flexibility.** spline2d
   (per-M_WR interpolating spline) and fxgx (pol4 + m·pol3, 9 params) take 6 of 8
   cells, beating the compact poly3+mass by ~0.5–2 pp (e.g. gauss ee-resolved:
   spline2d 2.7% vs poly3+mass 4.9%). But spline2d is nonparametric and fxgx is
   high-order: lowest CV ≠ best *shippable* form.

2. **For a compact analytic parameterization, poly3+mass is the pragmatic pick.**
   6 params, mass-aware, within ~1–2 pp of the flexible models in every cell, and
   far more interpretable. Reach for spline2d/fxgx only if that ~1–2 pp matters and
   a non-compact form is acceptable.

3. **The M_WR dependence is real and a linear mass term removes it.** A pure-x
   cubic (pol3) leaves residual structure that tracks M_WR — resid_R2_m up to 0.39
   (gauss & fwhm ee-resolved) and 0.57 (gauss µµ-boosted). Adding b₁m + b₂xm
   (poly3+mass) drives resid_R2_m → 0.00 in *every* cell. Where pol3's R2_m is
   already ~0 (e.g. gauss µµ-resolved 0.05) the mass structure was weak to begin
   with, but the term never hurts the structure metric.

4. **Gaussian parameterizes 1.4–2.2× more smoothly than FWHM.** Best-CV gauss vs
   fwhm: ee-res 2.7 vs 4.9% (1.8×), ee-boost 3.0 vs 6.7% (2.2×), µµ-res 1.8 vs
   3.7% (2.1×), µµ-boost 2.2 vs 3.1% (1.4×). The Gaussian core width is the more
   predictable definition in every cell.

5. **Resolved vs boosted: treat separately.** Disjoint x ranges, different best
   models, and the physics pole term is resolved-only. Never share a fit across them.
   Boosted cells also have heavier tails (CV q95 up to ~20%) from their small,
   low-x samples.

6. **ee vs µµ: keep separate.** ee CV residuals run larger than µµ (e.g. gauss
   resolved 2.7 vs 1.8%); the channels aren't consistent enough to combine.

## Bottom line
The Gaussian core width is the smoother, more predictive definition (lower CV in
every cell). The raw CV minima (~1.8–3.0% gauss, ~3.1–6.7% fwhm) come from the
flexible spline2d/fxgx, but for a usable closed form **poly3+mass** is the
recommendation: it is the compact, interpretable model that kills the residual
M_WR structure and stays within ~1–2 pp of the flexible best. FWHM is usable but
~1.4–2.2× noisier and likewise wants the mass term.
