# Gauss-only signal-fit calibration — pipeline overview

End-to-end walkthrough of the scripts in this study, in the order you'd run
them. Each script is one self-contained step.

## Glossary

| Term | Meaning |
|---|---|
| **cell** | One `(channel, mass)` pair. The full grid has 390 masses × 2 channels = 780 cells. |
| **bulk cells** | 8 hand-picked cells per channel used for α tuning: M_WR ∈ {3,4,5,6} TeV × M_N/M_WR ∈ {≈0.3, ≈0.5}. "Bulk" because they're in the middle of the phase space — far from edges where kinematics get weird. |
| **curated cells** | 15 cells per channel (5 M_WR × 3 M_N) used for demo plots. Slightly bigger than the bulk set; also includes WR2000 and a high-ratio cell per M_WR. |
| **truth** | The windowed mean (µ) and RMS (σ) of the MC signal histogram in `[0.7, 1.3] × M_WR`. Used both as the prior central and as the pull denominator's reference. |

> **Note on "RMS" terminology.** Strictly, RMS = √⟨x²⟩ (about zero) while the standard deviation σ = √⟨(x − ⟨x⟩)²⟩ (about the mean). They differ by `RMS² = σ² + ⟨x⟩²`. ROOT historically named the standard-deviation getter `TH1::GetRMS()` — and so the HEP convention is to call σ "RMS". We follow that convention everywhere: every "RMS" in this study and on plot labels is the **standard deviation about the windowed mean** (i.e. ROOT's `TH1::GetStdDev`, which is what `GetRMS()` actually returns). The literal RMS-about-zero is never used.

| **α_µ, α_σ** | Prior-width knobs. The Gaussian µ-prior has σ = α_µ × RMS_truth; the σ-prior has σ = α_σ × RMS_truth. **Production: (α_µ, α_σ) = (1.0, 0.25)**. |
| **N** | Number of toy events per fit (N ∈ {5, 10, 20, 50, 100}). |

## Pipeline

```
                          ┌──────────────────────────────────────┐
                          │  signal_samples.py                   │
       Step 1   ─────────►│  Plot MC + window + mean/RMS truth   │
                          │  → plots/signal_samples/             │
                          └──────────────────────────────────────┘

                          ┌──────────────────────────────────────┐
                          │  scan_priors.py + sensitivity.py     │
       Step 2   ─────────►│  Tune (α_µ, α_σ) on 8 bulk cells     │
       (one-time)         │  → locks in (1.0, 0.25)              │
                          │  → plots/sensitivity/                │
                          └──────────────────────────────────────┘

                          ┌──────────────────────────────────────┐
                          │  scan_full.py                        │
       Step 3   ─────────►│  Sample toys + fit at chosen α       │
                          │  across all 390 masses × 5 N values  │
                          │  → outputs/results.csv               │
                          └──────────────────────────────────────┘

                          ┌──────────────────────────────────────┐
                          │  plots.py                            │
       Step 4   ─────────►│  Pull histograms + 2D scatter        │
                          │  → plots/1d_pulls/, plots/2d_pulls/  │
                          └──────────────────────────────────────┘
```

### Step 1 — Signal samples + truth values

Script: `1_signal_samples/signal_samples.py`

For each of the 15 curated cells:
1. Load the native MC mass histogram.
2. Compute the windowed mean and RMS via `ROOT.TH1::GetMean` and `GetStdDev`.
3. Plot the histogram with `[0.7, 1.3]×M_WR` shaded, mean and ±RMS lines drawn.

Output: `plots/signal_samples/{mass}/{channel}.{pdf,png}` — 60 files.

These plots show what the truth values are visually. The truth values themselves
are **not stored** anywhere — they're recomputed inline whenever Step 2 or
Step 3 needs them. (It's two lines of code, deterministic given the MC histogram.)

### Step 2 — Tune α (one-time)

Scripts: `2_tune_priors/scan_priors.py`, then `2_tune_priors/sensitivity.py`
(with optional `2_tune_priors/plot_scan_2d.py` for raw-scan heatmaps).

`scan_priors.py` runs the toy + fit loop over the 8 bulk cells × 2 channels ×
5 N values × ~50 toys, looping over a grid of (α_µ, α_σ) values. For each
(cell, N, seed) it samples the toy ONCE, then refits the same toy at every
α in the grid. This isolates the α effect from toy-to-toy fluctuations.

Output: `outputs/scan_2d_root_rng_v2.csv`
(and an extension `outputs/scan_2d_root_rng_extra.csv` filling in finer α values).

`sensitivity.py` reads these CSVs and produces three diagnostic types per N
value: 1D α-slices through the operating point, a 2D comfort-zone heatmap,
and the prior-vs-data dominance fraction.

Output: `outputs/plots/sensitivity/` — 189 files.

The result: **(α_µ, α_σ) = (1.0, 0.25)** locked in. Both channels use the
same values.

You only re-run this if you want to retune α. It's not part of a typical pass.

### Step 3 — Production toy generation

Script: `3_scan_full/scan_full.py`

For each of the 390 masses × 2 channels × 5 N values × 100 toys:
1. Load the native MC histogram (one load per cell).
2. Compute the windowed mean/RMS (same as Step 1 — two lines).
3. Sample N events from the histogram in the window using `TH1::GetRandom`
   with a Knuth-hashed seed.
4. Run the gauss fit with priors centered at mean/RMS, widths α_µ × RMS and
   α_σ × RMS.
5. Write the fit results to CSV.

Output: `outputs/results.csv` — 390,000 rows.

This is the production data file that every plot in Step 4 reads.

### Step 4 — Production plots

Scripts: `4_plots/plots.py` (single driver, multiple `--what` modes).

| `--what` mode | What it does | Files |
|---|---|---|
| `2d_pulls` | Pull-bias and pull-spread vs M_N/M_WR scatter plots, one per (channel, N, metric, param) | 80 |
| `1d_pulls` | Pull histograms + single-toy pull-demos for the 15 curated cells × 5 N × 2 channels × 2 params | 1200 |
| `all` | Both of the above | — |

Output: `outputs/plots/2d_pulls/` and `outputs/plots/1d_pulls/`.

`plots.py` reads from `outputs/results.csv` for the pull histograms; for the
single-toy demos it re-runs ONE fit using the same machinery as `scan_full.py`
(so it can draw the toy data + the fitted curve in one plot).

### Companion diagnostics (also in `4_plots/`)

| Script | Purpose | Output |
|---|---|---|
| `plot_alpha_demo.py` | Pull histogram + single-toy demo for one (channel, mass, N) at a custom α_σ. Used to compare visually how the fit reacts to different σ-prior tightness. | `outputs/plots/alpha_comparison/` |
| `plot_fit_vs_truth.py` | Per-cell median Gaussian fit (mu_fit, σ_fit) − MC truth (windowed mean, RMS) vs `x = M_N/M_WR`, in the same 2-panel (abs GeV + %) layout as `5_parameterize_priors/leave_one_out.py`. Reads `results.csv`; default N=20. | `outputs/plots/fit_vs_truth/` |

## A typical run from a fresh checkout

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

# Step 1 — signal sample plots
python signal_fitting/1_signal_samples/signal_samples.py \
    --output-dir signal_fitting/outputs/plots/signal_samples

# Step 3 — production toys
python signal_fitting/3_scan_full/scan_full.py \
    --channels ee mumu --alpha-mu 1.0 --alpha-sigma 0.25 --use-moments \
    --n-toys 100 --n-events 5,10,20,50,100 \
    --output signal_fitting/outputs/results.csv

# Step 4 — production plots
python signal_fitting/4_plots/plots.py \
    --input  signal_fitting/outputs/results.csv \
    --output-dir signal_fitting/outputs/plots \
    --what all --n-events 5 10 20 50 100 \
    --alpha-mu 1.0 --alpha-sigma 0.25 --use-moments --curated

# Step 2 — only if you want to retune α or refresh sensitivity diagnostics.
```

Total time on LCG_106 (single core): ~20 min.

## Shared infrastructure (imported by all scripts)

- `shared/measure_fwhm.py` — window-edge constants, mass parsing, MC histogram loading.
- `shared/fit_signal_toy.py` — `run_fit` (RooFit gauss), `sample_from_hist_root` (ROOT TH1::GetRandom with hashed seed), and other primitives.

## Decision points (locked in via README decisions log)

- Truth = windowed mean & RMS of MC.
- Prior centrals = truth.
- Prior widths = α_µ × RMS, α_σ × RMS.
- α = (1.0, 0.25) for both channels.
- RNG = ROOT TRandom3 with Knuth-hashed seed.
- Moments via ROOT TH1.
