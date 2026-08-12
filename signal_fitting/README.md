# On-shell signal-width study

Measure the on-shell width of the W_R → ℓℓjj MC signal peak, pick a
**window-robust** width definition, and find a smooth low-order
parameterization of that width across the (M_WR, M_N) grid — the input a
downstream fit needs to constrain the signal shape.

Everything lives under [`1_signal_widths/`](1_signal_widths/): the scripts, the
shared helpers (in [`shared/`](shared/)), and the generated outputs are
**co-located** with the script that makes them.

> **History.** This directory previously hosted a single-Gaussian prior-calibration /
> pull study. That work is retired under
> [`archived/old_prior_calibration_pipeline/`](archived/old_prior_calibration_pipeline/)
> (its own `README.md` + `PIPELINE.md` go with it). Nothing here depends on it.

## Setup

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```

PyROOT (RooFit) + numpy + matplotlib/mplhep. Every script takes `-v`. All
input/output paths default relative to the script location, so the scripts run
from any working directory.

## Glossary

| Term | Meaning |
|---|---|
| **cell** | one `(channel, mass)` point; `channel ∈ {ee, mumu}` |
| **x** | `M_N / M_WR`, the mass-ratio abscissa for every trend plot |
| **topology** | `resolved` (full x grid) vs `boosted` (highly-boosted regime `x < 0.1`, discovered from disk). Disjoint x ranges — **never** parameterize across them |
| **on-shell window** | `[0.7, 1.3] × M_WR` baseline (`ONSHELL_WINDOW_*_FRAC` in [`shared/measure_fwhm.py`](shared/measure_fwhm.py)); stability windows are `[0.8,1.2]` and `[0.85,1.15]` |
| **σ_gauss^on** | width of the **iterative-core Gaussian** fit (fit in `[μ±2σ]`, re-seeded until μ,σ move < 2%) — keys off the peak |
| **σ_FWHM^on** | `FWHM / 2.3548` of a **RooKeysPdf** (adaptive KDE) over the window — keys off the peak |

**Why only these two widths?** A pre-window stability study (the decision below)
showed σ_eff and the unwindowed RMS shift by ~16–29 % when the window changes —
they integrate the windowed tails and are window-driven. The Gaussian-core σ
(~1.7 %) and the RooKeysPdf FWHM (~0.6 %) key off the *peak*, not the tails, and
are stable. The dropped estimators and their old outputs are gone; see the
decisions log.

## Directory layout

```
signal_fitting/
├── master_masses.csv          master signal grid (mass + topology), read by every stage
│
├── 0_signal_samples/          Stage 0 — raw MC signal-shape visuals
│   ├── signal_samples.py        per-point 1D mass histograms     -> 1d_histograms/
│   ├── signal_peak_overlay.py   per-M_WR overlay, colorbar by M_N -> peak_overlays/
│   ├── 1d_histograms/
│   └── peak_overlays/
│
├── 1_signal_widths/           Stage 1 — on-shell width study
│   ├── gaussian/                  σ_gauss^on deep dive — script + its outputs
│   │   ├── detail_gauss_fit.py
│   │   ├── gauss_fit_table.csv
│   │   ├── histograms/            per-cell MC + fitted Gaussian
│   │   ├── window_stability.csv   per-window widths + robustness verdict
│   │   ├── chi2_ndf/              ⎫ one folder per plotted quantity,
│   │   ├── sigma_over_mWR/        ⎬ files named {channel}_{topology}.{png,pdf}
│   │   ├── mu_over_mWR/           ⎪
│   │   └── robustness/            ⎭
│   ├── fwhm/                      σ_FWHM^on deep dive — script + its outputs
│   │   ├── detail_fwhm.py
│   │   ├── fwhm_table.csv
│   │   ├── histograms/            per-cell MC + KDE + half-max crossings
│   │   ├── window_stability.csv   per-window widths + robustness verdict
│   │   ├── sigma_over_mWR/
│   │   ├── peak_over_mWR/
│   │   └── robustness/
│   └── gauss_vs_fwhm/            comparison of the two widths
│       ├── compare_gauss_fwhm.py  reads the sibling gaussian/ & fwhm/ tables
│       ├── sigma_over_mWR/        σ_gauss^on & σ_FWHM^on overlaid
│       └── ratio_fwhm_over_gauss/ σ_FWHM^on / σ_gauss^on
│
├── 2_width_parameterization/  Stage 2 — model the width vs (x, M_WR)
│   │                            (reads the Stage-1 gaussian/ & fwhm/ tables)
│   ├── parameterizations/     fit all models — one folder per model
│   │   ├── parameterize_width.py   the fitter (lives with its outputs)
│   │   ├── parameterization_params.json   fitted parameters per model
│   │   ├── 1d/   pol2 pol3 pol4 physics spline   (x-only: one fit curve)
│   │   ├── 2d/   poly3+mass fxgx +m2 +x2m spline2d  (x & M_WR: curve per M_WR)
│   │   │         each model: gauss/ & fwhm/ -> {ch}_{cat}.{png,pdf}
│   │   └── all_models/        every model overlaid (2D at m=0), gauss/ & fwhm/
│   ├── cross_validation/      score all models by LOMO CV
│   │   ├── validate_models.py   the scorer (lives with its outputs)
│   │   ├── cv_summary.csv       all metrics per (width, ch, cat, model)
│   │   ├── predictions.csv      per-point measured + predicted σ (GeV)
│   │   ├── cv_comparison/       CV-median bar chart -> <width>.{png,pdf}
│   │   ├── residuals/           1d/ & 2d/ -> <model>/<width>/{ch}_{cat} (per model)
│   │   ├── pred_vs_meas/        1d/ & 2d/ -> measured vs predicted σ (y=x)
│   │   └── best_model_residual/ gauss/ & fwhm/ -> {ch}_{cat}
│   └── INTERPRETATION.md
│
└── shared/                    cross-stage modules (measure_fwhm, shape_estimators)
```

## Scripts

### Per-width deep dives (run these first — they write the tables others read)

| Script | What it does | Writes into |
|---|---|---|
| [`gaussian/detail_gauss_fit.py`](1_signal_widths/gaussian/detail_gauss_fit.py) | iterative-core Gaussian fit per cell | `gaussian/` (table + window_stability.csv + histograms + chi2_ndf/sigma_over_mWR/mu_over_mWR/robustness) |
| [`fwhm/detail_fwhm.py`](1_signal_widths/fwhm/detail_fwhm.py) | RooKeysPdf peak + FWHM per cell | `fwhm/` (table + window_stability.csv + histograms + sigma_over_mWR/peak_over_mWR/robustness) |

### Window-robustness check

Each deep dive fits its width at the seed windows `[0.70,1.30]`, `[0.80,1.20]`,
`[0.85,1.15]` and, via `shape_estimators.window_stability_report`, writes
`window_stability.csv` (per-window widths) plus a console verdict — median / max
`|estimator(window)/estimator(baseline) − 1|` per (channel, topology). The
`robustness/` plots are the visual of the same ratios. (Pre-2026-06 this lived in
a separate `width_window_stability.py`; it's now folded into the deep dives.)

### Comparison

[`compare_gauss_fwhm.py`](1_signal_widths/gauss_vs_fwhm/compare_gauss_fwhm.py) reads both tables and
writes `gauss_vs_fwhm/sigma_over_mWR/` (overlay) and
`gauss_vs_fwhm/ratio_fwhm_over_gauss/` (ratio ≈ 0.75–0.85).

### Stage 2 — width parameterization (`2_width_parameterization/`)

Models the width vs (x, M_WR). All three read the Stage-1 tables from
`1_signal_widths/{gaussian,fwhm}/` (override with `--widths-dir`) and write into
`2_width_parameterization/` (override with `--out-dir`). `parameterize_width.py`
owns the model registry (`pol2/3/4`, `physics`, `spline`, `poly3+mass`, `fxgx`,
`+m2`, `+x2m`, `spline2d`); `validate_models.py` imports it.

| Script | What it does | Writes into |
|---|---|---|
| [`parameterizations/parameterize_width.py`](2_width_parameterization/parameterizations/parameterize_width.py) | fit every candidate model per (channel, topology, width); save fitted parameters + per-model fit plots (no scoring) | `parameterizations/` (params.json + one folder per model) |
| [`cross_validation/validate_models.py`](2_width_parameterization/cross_validation/validate_models.py) | score those models by leave-one-M_WR-out CV; recommend the best per cell | `cross_validation/` (cv_summary.csv, bar charts, held-out residuals) |

Interpretation: [`INTERPRETATION.md`](2_width_parameterization/INTERPRETATION.md).

### Stage 0 — signal samples (`0_signal_samples/`)

Visual aids for the raw MC signal shapes, not part of the width chain. Both read
the master grid from [`master_masses.csv`](master_masses.csv)
and plot each point in its tagged topology (resolved → `mass_fourobject`,
`m_{ℓℓjj}`; boosted → `mass_twoobject`, `m_{ℓJ}`), at 80 GeV bins.

- [`signal_samples.py`](0_signal_samples/signal_samples.py) — one MC mass
  histogram per (channel, mass). Writes to `0_signal_samples/1d_histograms/`.
- [`signal_peak_overlay.py`](0_signal_samples/signal_peak_overlay.py) — per
  M_WR, overlays all its M_N points (unit-area normalized) on one axes,
  color-coded by M_N via a colorbar. Writes to `0_signal_samples/peak_overlays/`.

## A typical run

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
cd signal_fitting

# Stage 1 — measure the widths (writes the tables others read)
python 1_signal_widths/gaussian/detail_gauss_fit.py -v   # -> gaussian/ (table + window_stability.csv)
python 1_signal_widths/fwhm/detail_fwhm.py -v            # -> fwhm/     (table + window_stability.csv)
python 1_signal_widths/gauss_vs_fwhm/compare_gauss_fwhm.py  # -> gauss_vs_fwhm/

# Stage 2 — parameterize (reads the Stage-1 tables)
python 2_width_parameterization/parameterizations/parameterize_width.py  # fit -> parameterizations/
python 2_width_parameterization/cross_validation/validate_models.py  # score -> cross_validation/
```

`compare_gauss_fwhm.py` and all of Stage 2 read `1_signal_widths/gaussian/gauss_fit_table.csv`
and `1_signal_widths/fwhm/fwhm_table.csv`, so run the two deep dives first. To rebuild only
the summary plots from existing tables, the deep dives take `--plots-only`.

The resolved mass grid is read from `master_masses.csv` (override with
`--mass-csv`); boosted cells are discovered from the MC on disk.

## Shared infrastructure

- [`shared/shape_estimators.py`](shared/shape_estimators.py) — backbone: cell discovery /
  loading (`collect_cells`, `discover_masses`), the width primitives
  (`gaussian_core_fit`, `keys_peak_and_fwhm`, `windowed_moments`,
  `gaussian_chi2_ndf`), and the shared M_WR-colorbar plotters
  (`plot_scalar_vs_x_by_mwr`, `plot_series_vs_x`, …; all `mkdir` their output's parent).
- [`shared/measure_fwhm.py`](shared/measure_fwhm.py) — window-fraction constants, mass
  parsing (`parse_masses`), and MC histogram loading.

## Decisions log

| date | decision |
|---|---|
| 2026-06-01 | Width-estimator cross-check across σ_eff(float/sym), σ_Gauss-fit, σ_FWHM, σ_RMS-full on native MC. **Resolved shapes strongly non-Gaussian** (narrow core + heavy low-side tail): unwindowed RMS overestimates σ_eff by ~32–39 %, σ_FWHM ≈ 0.57–0.64 × σ_eff; the `[0.7,1.3]×M_WR` window does real work. Boosted far more Gaussian. |
| 2026-06-01 | **Boosted restricted to x ≤ 0.3** then, on disk, redefined as **x < 0.1** (genuinely highly-boosted; production grid stops at x ≥ 0.1). 51 masses/channel down to x ≈ 0.017. `collect_cells` sources boosted from `discover_masses`. |
| 2026-06-02 | **Pre-window stability is the selection criterion.** σ_eff^on (median \|Δ\| ≈ 16 %) and RMS^on (≈ 29 %) are window-driven; the **iterative-core Gaussian σ (≈ 1.7 %)** and **RooKeysPdf FWHM (≈ 0.6 %)** are robust. |
| 2026-06-02 | **Dropped σ_eff, σ_eff-sym, RMS, and the wide best-Gaussian fit; kept only σ_gauss^on and σ_FWHM^on.** Removed `compare_width_estimators.py`, `detail_eff_gauss.py`, dead `shape_estimators` primitives, and the `eff_gauss/`, `rms/`, `comparison/` outputs. (σ_FWHM^on ≈ 0.75–0.85 × σ_gauss^on.) |
| 2026-06-03 | Retired the old single-Gaussian prior-calibration / pull study (and its `README.md`/`PIPELINE.md`) to `archived/old_prior_calibration_pipeline/`. |
| 2026-06-03 | **Reorganized into `1_signal_widths/` with outputs co-located per method.** `gaussian/` and `fwhm/` hold their script + one folder per plotted quantity; shared scripts (comparison, parameterization, window-stability, signal-samples) sit at the root and write to `gauss_vs_fwhm/`, `parameterization/`, `<method>/window_stability/`, `signal_samples/`. The resolved mass grid moved from the archived `results.csv` to a self-contained `master_masses.csv`. |

**Bottom line (from `INTERPRETATION.md`):** the Gaussian-core width is the
smoother, more predictive definition (lower LOMO-CV residuals in every cell,
~2.4–4.9 %); FWHM is usable but ~1.2–1.9× noisier and always needs a mass term.
Treat resolved/boosted and ee/µµ separately.
