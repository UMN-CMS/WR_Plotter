# Stage 6 — spurious-signal toy study

**The toy generalization of the Stage-5 Asimov spurious signal.** Stage 5 fits
the S+B model **once** to the unfluctuated background-only MC and reads off a
single fake signal `N_sp` per window. Here we draw many **Poisson toys** of the
background-only expectation and fit each, so for every window we get the
**distribution** of the fake signal — how big it typically is, how much it
scatters, and whether that scatter is consistent with the quoted `σ(N_sig)`.

For each (channel, topology), grid mass `m_WR`, and background function, the
expected pseudo-data is the **background-only MC** (no signal injected), and each
toy is a bin-wise Poisson draw of it:

    mu[bin]        = bkg_MC[bin]              # background only
    data_toy[bin]  = Poisson( mu[bin] )      # data statistics

then we fit `background TF1 (Stage-4 recentered) + fixed-shape Gaussian(μ, σ)`
**inside the window `[μ − kσ, μ + kσ]` only** — the Stage-4 range — with the
background coefficients and `N_sig` floating. The fitted yield is the toy's
spurious signal `N_sp`; its pull is `N_sp/σ_fit` (the injected signal is zero).

> Only the **data** is Poisson-fluctuated — the MC template's own statistical
> error does **not** enter the toy generation (the Poisson draw already *is* the
> data fluctuation). So the **coverage** result (pull width) is clean; the toy
> mean `⟨N_sp⟩` inherits the same jagged-template caveat as the Stage-5 Asimov
> value (and indeed `⟨N_sp⟩ ≈` the Asimov `N_sp`).

The **window** (and background recentering) and the fixed **Gaussian** are
defined exactly as in Stage 5: window from the **median** over `m_N` — the
Stage-2 linear `m_WR` parameterization (`2_width_parameterization/wr/window_params.json`)
by default — and the Gaussian uses the same linear `(μ, σ)`
(`μ_sig = m_c`, `σ_sig = σ_win`). The S+B fit core is shared with Stage 5/7 in
[`../shared/sb_fit.py`](../shared/sb_fit.py).

## Metrics (per mass, function, over the converged toys)

| metric | meaning |
|---|---|
| `⟨N_sp⟩ ± RMS` | central fake signal (`⟨N_sp⟩ ≈` Stage-5 Asimov) and its toy spread (`≈ σ(N_sig)`) |
| `pull_mean = ⟨N_sp/σ_fit⟩` | the typical fake signal **in σ** — headline; bands `±0.2` (tight) / `±0.5` (loose) |
| `pull_width = RMS(N_sp/σ)` | **coverage** — `≈ 1` if the error bars are honest |
| `frac \|pull\|>0.5` | how often a single experiment fakes a notable signal |
| `q95(\|N_sp\|)` | a conservative spurious magnitude (95th percentile) |

## Setup & run

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
python spurious_signal_toys.py --channel ee   --topology resolved
python spurious_signal_toys.py --channel mumu --topology resolved
```

Defaults: `--era RunIII2024Summer24 --dir 20260317_lo_dy --mn-frac 0.5 --k 3
--bin-width 100 --ntoys 1000 --functions expo powlaw`. Toys are ~30 ms/fit, so
1000 toys × ~30 masses × 2 functions ≈ 30 min/category — add more `--functions`
or raise `--ntoys` deliberately. `--no-toy-plots` skips the per-mass `N_sp`
histograms. Off-grid masses work via `--masses 2341 ...` (requires
`--window-source param`, the default).

| Output | Contents |
|---|---|
| `pull_mean_vs_mass/{ch}_{topo}/{fn}.*` | **headline** — toy pull mean (fake signal in σ) vs `m_WR`, ±0.2/±0.5 bands |
| `spurious_yield_vs_mass/{ch}_{topo}/{fn}.*` | `⟨N_sp⟩ ± RMS` (events) vs `m_WR`, with the Stage-5 Asimov `N_sp` overlaid |
| `pull_width_vs_mass/{ch}_{topo}/{fn}.*` | **coverage** (pull RMS) vs `m_WR`, reference line at 1.0 |
| `nsp_hist/{ch}_{topo}/{fn}/m{mWR}.*` | per-mass `N_sp` distribution over toys, with the Asimov value + toy mean |
| `spurious_toy_table_{ch}_{topo}.csv` | every `(mass, function)` summary row |

## Relation to the neighbours

- **Stage 5** (`5_spurious_signal`) is the Asimov (single-fit) version of this
  exact check — the toy `⟨N_sp⟩` should reproduce its `N_sp`.
- **Stage 8** (`8_signal_injection_study`) does the complementary **signal
  injection** (real signal MC, `N ≥ 10`). Its toy null (`N = 0`) was promoted
  here, so the Stage-8 toys default to injection only.

## Caveats (shared with Stages 5 / 8)

- **Method-validation only, not the final systematic:** Run-2 signal width on a
  Run-3 LO-DY background (k-factors = 1.0, ~20–30 % off CR data/MC). Not the
  deliverable bias systematic until the DY K-factor + reshape are applied.
- The in-window method is only valid below ~3.6 TeV (resolved); above that the
  peak piles into the spectrum endpoint and the high-mass toys lose converged
  fits (dropped by `--min-toys`).
