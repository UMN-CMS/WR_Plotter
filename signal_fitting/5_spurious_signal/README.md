# Stage 5 — spurious-signal study

**Does the background model fake a signal when there is none?** This is the core
pre-unblinding check for a functional-form background (validation-plan check #4):
fit signal + background to a **signal-free** template and read off the fake
signal the background mismodelling produces.

For each (channel, topology), grid mass `m_WR`, and background function, fit

    background TF1 (Stage-4 recentered)  +  fixed-shape Gaussian(μ, σ)

to the **background-only MC** (DY + tt̄/tW + nonprompt + other; **no signal
injected**), inside the window `[μ − kσ, μ + kσ]` only — the same range as the
Stage-4 background fits. Only `N_sig` and the background coefficients float; the
Gaussian shape is fixed. The fitted yield is the **spurious signal** `N_sp`.

- **Window** `[μ − kσ, μ + kσ]` with μ/σ (default) from the **Stage-2 linear
  m_WR parameterization** (`2_width_parameterization/wr/window_params.json`) — one
  window per `m_WR`, as the analysis does; background recentered at that μ.
  (`--window-source measured` falls back to the per-`m_WR` median over `m_N`.)
- **Gaussian** fixed to the on-shell point's own (μ, σ) — `m_N` closest to
  `mn_frac·m_WR` (default `m_N = m_WR/2`) — from the Stage-1 fits.
- Bin errors are the expected Poisson `√(content)`, so the pull is comparable to
  the acceptance band.

**Headline metric: the pull** `= N_sp / σ(N_sig)`. A fit passes if the fake
signal is small relative to its statistical uncertainty — a common band is
`|pull| < 0.2` (tight) to `0.5` (loose).

> This is the Asimov (no-fluctuation) spurious signal. Stage 6
> (`6_spurious_signal_toys`) is the toy-based version of this same check (the
> distribution of the fake signal over Poisson toys of the background); Stage 8
> (`8_signal_injection_study`) does the complementary signal-**injection** /
> recovery study.

## Setup & run

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
python spurious_signal.py --channel ee   --topology resolved
python spurious_signal.py --channel mumu --topology resolved
```

Defaults: `--era RunIII2024Summer24 --dir 20260317_lo_dy --mn-frac 0.5 --k 3
--bin-width 100`. `--no-fit-plots` skips the per-mass diagnostics.

| Output | Contents |
|---|---|
| `pull_vs_mass/{ch}_{topo}/{fn}.*` | **headline** — spurious pull `N_sp/σ` vs `m_WR`, **one plot per function**, with ±0.2 / ±0.5 acceptance bands |
| `spurious_yield_vs_mass/{ch}_{topo}/{fn}.*` | spurious `N_sp ± σ` (events) vs `m_WR`, **one plot per function** |
| `fit_diagnostics/{ch}_{topo}/{fn}/m{mWR}.*` | the S+B fit to background-only: grey MC bkg, red B+S, blue-dashed Stage-4 background-only fit (the red−blue gap in the window is the fake signal), stat box with `N_sp`, pull, χ²/ndf |
| `spurious_table_{ch}_{topo}.csv` | window, signal tag, `N_spur(±)`, pull, `N_sp/√B`, Minuit status |

## What it shows

The 2-parameter functions (`expo`, `powlaw`) fake little (|pull| mostly < 0.5);
the flexible higher-order functions can curve up in the window to fake a small
positive signal even with a good χ²/ndf — that excess is exactly what the pull
catches. The fit diagnostic makes it visible: the red B+S curve lifts off the
blue background-only fit inside the window.

## Caveats (shared with Stage 6)

- **Method-validation only, not the final systematic:** inputs are a Run-2 signal
  width (RunIISummer20UL18) and a LO-DY background (k-factors = 1.0) that misses
  CR data/MC by ~20–30 %. The numbers are not the deliverable bias systematic
  until the DY K-factor + reshape are applied.
- The Stage-4 sideband/window method (and hence this spurious test) is only valid
  below ~3.6 TeV (resolved), where the upper sideband still exists; above that the
  peak piles into the spectrum endpoint.
- The S+B fit core lives in `signal_fitting/shared/sb_fit.py` (shared with
  Stage 6).
