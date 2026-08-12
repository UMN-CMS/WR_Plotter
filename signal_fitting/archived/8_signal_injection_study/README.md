# Stage 7 — signal-injection study (real signal MC)

Does the analysis's **Gaussian signal model** recover an injected signal without
bias, and are its **error bars honest**, when the injected signal is the **true
MC shape**? This tests the signal-model bias on top of the background bias, both
**without signal-region data**. It comes in two variants:

| dir | what it does | answers |
|---|---|---|
| [`asimov/`](asimov/) | inject the **exact** expected signal, fit **once** | the central **recovery bias** |
| [`toys/`](toys/) | **Poisson-fluctuate** the pseudo-data over many toys, fit each | bias **and** the **pull width / coverage** |

Both share the S+B fit core in [`../shared/sb_fit.py`](../shared/sb_fit.py) and the
slope-constrained background functions from Stage 4, so the fit is identical; only
the pseudo-data generation differs.

## Common method (per channel, topology, grid mass `m_WR`)

1. Take the **real signal MC** for that mass — the `WR{m_WR}_N{m_N}` sample with
   `m_N` closest to `mn_frac·m_WR` (default `m_N = m_WR/2`). Same files as
   `0_signal_samples` (signal era `RunIISummer20UL18`, `master_masses.csv`).
2. Normalize to **unit area** (the MC is not xsec/lumi-scaled, but that cancels in
   the shape) → the injected signal shape.
3. Build the pseudo-data and **fit** `background TF1 (Stage-4 recentered) +
   fixed-shape Gaussian(μ, σ)` **inside the window `[μ − kσ, μ + kσ]` only** — the
   same range as the Stage-4 fits — with one floating yield `N_sig`.
   - **Window** `[μ ± kσ]` and the background recentering use the **median over
     `m_N`** (one window per `m_WR`, as the analysis does).
   - the **Gaussian** uses the **injected point's own (μ, σ)** (its Stage-1 fit),
     so `N_sig` vs the injected `N` isolates the **non-Gaussian shape** bias plus
     the in-window background/signal degeneracy. `N` is a test strength (events),
     not a predicted yield.

Injection levels: **`N = 0`** (spurious / null), **`N = 10`** (small realistic),
**`N = 1000`** (extreme — the Gaussian-vs-true-shape mismatch is unmistakable).
The **`N = 0` null/spurious toy now lives in Stage 6** (`6_spurious_signal_toys`),
so **`toys/` defaults to `N = 10` only** (pass `--inject 0` here only as a
cross-check); at `N = 1000` the in-window pull is swamped by the deterministic
shape bias (not a coverage test), so the extreme injection is left to `asimov/`.

---

## `asimov/` — deterministic injection

`data = bkg + N·shape` (no fluctuation), fit once → `N_sig`. Run:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
cd asimov
python signal_injection.py --channel ee   --topology resolved
python signal_injection.py --channel mumu --topology resolved
```

| Output (per `N{N}/`) | Contents |
|---|---|
| `N{N}/fit_diagnostics/{ch}_{topo}/{fn}/m{mWR}.*` | the S+B fit, Stage-4 style: grey MC bkg (solid) + bkg+sig (dashed), red B+S fit, blue-dashed in-window background-only fit |
| `N{N}/fitted_yield_vs_mass/{ch}_{topo}.*` | fitted `N_sig ± σ` vs `m_WR` per function, injected `N` as reference |
| `N{N}/injection_table_{ch}_{topo}.csv` | window, signal tag, `N_inj`, `N_sig(±)`, recovered `N_sig/N_inj`, pull, Minuit status |

**Lesson:** the extreme injection is clearest — at `N = 1000` the Gaussian
recovers well below 1000 because it cannot reproduce the true non-Gaussian shape
inside the window; `N = 0` is the (Asimov) spurious signal; `N = 10` checks
small-signal recovery within the large statistical error.

---

## `toys/` — Poisson-fluctuated toys (bias **and** coverage)

For each toy, the pseudo-data is a **bin-wise Poisson draw of the full
expectation**:

```
data_toy[bin] = Poisson( bkg[bin] + N·shape[bin] )       # background AND signal fluctuate
```

Because `Poisson(bkg) + Poisson(N·shape) = Poisson(bkg + N·shape)`, fluctuating
the total is identical to fluctuating background and signal independently. The MC
template's **own statistical error does not enter** the generation — the Poisson
draw already is the data's statistical fluctuation. Over the converged toys we
report, per `(m_WR, N, function)`:

- **bias** `⟨N_fit⟩ − N` (events) and **pull mean** `⟨(N_fit − N)/σ_fit⟩` (bias in σ)
- **pull width** = RMS of `(N_fit − N)/σ_fit` → **coverage** (≈ 1 if the error bars are honest)

```bash
cd toys
python signal_injection_toys.py --channel ee --topology resolved --ntoys 200
```

Defaults: `--functions expo powlaw` (the Stage-4-validated functions; toys are
~30 ms/fit, add others explicitly), `--ntoys 200`, `--min-toys 50`,
`--seed 12345` (per-point seeds derived deterministically), `--no-toy-plots`
skips the per-point pull histograms.

| Output (per category) | Contents |
|---|---|
| `pull_mean_vs_mass/{ch}_{topo}/{fn}.*` | bias-in-σ vs `m_WR`, `N` overlaid, ±0.2 (tight) / ±0.5 (loose) bands |
| `pull_width_vs_mass/{ch}_{topo}/{fn}.*` | **coverage** vs `m_WR`, `N` overlaid, reference line at 1.0 |
| `bias_vs_mass/{ch}_{topo}/{fn}.*` | `⟨N_fit⟩ − N` (events) vs `m_WR`, `N` overlaid |
| `pull_hist/{ch}_{topo}/{fn}/m{mWR}_N{N}.*` | per-point pull distribution + overlaid unit Gaussian |
| `toy_table_{ch}_{topo}.csv` | every `(m_WR, N, function)` summary row |

**What's trustworthy:** the **coverage** (pull width) result is clean — it is
driven by the data-Poisson scatter, which the MC error bars do not affect. The
**bias** (pull mean) inherits the same template-jaggedness caveat as Asimov: the
jagged MC truth is identical in every toy, so it does not average away and feeds a
coherent bias — exactly what a smooth-truth template (the deferred validation-plan
item #1) would remove. The toy-based **null/spurious** check (`N = 0`) is Stage 6
(`6_spurious_signal_toys`); inject `N ≥ 10` here.

---

## Caveats (shared)

- **Method-validation only:** Run-2 signal shape on a Run-3 LO-DY background
  (k-factors = 1.0, ~20–30 % off CR data/MC) — not the final bias systematic until
  the DY K-factor + reshape are applied.
- Valid only where the in-window method holds (resolved ≲ 3.6 TeV); above that the
  peak piles into the spectrum endpoint and the high-mass toys lose converged fits.
