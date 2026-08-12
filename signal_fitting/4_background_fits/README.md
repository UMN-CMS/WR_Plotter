# Stage 4 — background fits

Can the summed MC background (`B = DY+jets + tt/tW + nonprompt + other`) be
described by a smooth analytic function **inside** the signal window
`[μ − kσ, μ + kσ]`? This is the precondition for any window-based background
strategy — checked here per (channel, topology) and per signal-grid m_WR.

Mass observable per topology: resolved → `mass_fourobject` (m_ℓℓjj);
boosted → `mass_twoobject` (m_ℓJ). The window centre/width come **by default from
the Stage-2 linear m_WR parameterization** (`2_width_parameterization/wr/window_params.json`,
`μ(m_WR)=a+b·m_WR`, `σ_median(m_WR)=a+b·m_WR`) — the smooth window the analysis uses,
defined at off-grid masses too. Pass `--window-source measured` to instead take the
per-m_WR median over M_N of the Stage-1 fitted widths (`mu_gaus`/`sigma_gaus`).

The earlier `sideband_closure/` and `flavor_cr_fit/` sub-stages were parked in
`signal_fitting/archived/4_background_fits/` — they predate the ROOT fitter and
can be revived against `bkg_fit_lib` when needed.

## Setup

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```

## Fitting (`bkg_fit_lib.py`)

All fits are ROOT: a **TF1 chi-square fit with Minuit2 Migrad** to the histogram
bins in the window (empty bins are skipped), seeded from a weighted log-space
linear least-squares solution so Migrad starts near the minimum. Histogram I/O
stays on uproot. The model is **recentered at the window peak μ**: the slope
variable is `u = (m − μ)/1000` and power laws use `m/μ`, so `a = log f` at the
peak. This de-correlates `a` from `b` and makes Minuit's covariance accurate
instead of forced positive-definite (the fitted curve is identical to a m=0
pivot). Candidate functions:

| name | f(m) | pars |
|---|---|---|
| `expo` | `e^{a+bu}` | 2 |
| `expo2` | `e^{a+bu+cu²}` | 3 |
| `expo3` | `e^{a+bu+cu²+du³}` | 4 |
| `powlaw` | `e^a (m/μ)^b` (pure power law) | 2 |
| `powexp` | `(m/μ)^b e^{a+cu}` (power law × exp) | 3 |
| `dexp` | `e^{a₁+b₁u} + e^{a₂+b₂u}` (double exp) | 4 |

Every fit reports **three independent** quality checks — the minimal
non-redundant set; a fit **passes** only if all three hold:

1. **valid minimum** — `FitResult::IsValid()`. Because
   `IsValid() = State().IsValid() && !IsAboveMaxEdm() && !HasReachedCallLimit()`,
   this single check already requires EDM below Minuit's tolerance, the call
   limit not reached, and a Hesse-produced valid covariance.
2. **covariance accurate** — `CovMatrixStatus() == 3` (accurate, **not** merely
   forced positive-definite, which is status 2). This is the gap `IsValid()`
   leaves open and the workhorse check: every ≥3-parameter fit lands here.
3. **no parameter at limit** — no parameter within 1e-3 of a set TF1 limit.
   Only `dexp` has limits, so only `dexp` can fail this.

These three reproduce the exact pass/fail verdict of the older six-check set:
the earlier `edm_ok`, `below_call_limit`, and `hesse_ok` checks are all
subsumed by `valid_minimum` (every failure they catch already fails it). The
raw `status`/`cov_status`/`edm`/`ncalls` are still tabulated in the CSV as
diagnostics — they say *how* a fit failed, but never change the verdict.

## `in_window_fit/`

```bash
cd signal_fitting/4_background_fits/in_window_fit
python in_window_fit.py -v --diagnostics                 # ee resolved
python in_window_fit.py --channel mumu --topology boosted --diagnostics
```

Defaults: `--era RunIII2024Summer24 --dir 20260317_lo_dy --k 3 --bin-width 100`,
grid masses `[1000, 6000]`, no early stop (every mass attempted and tabulated).
All outputs are namespaced by `{channel}_{topology}`, so the four categories
coexist.

| Output | Contents |
|---|---|
| `chi2_ndf_vs_mass/{ch}_{topo}_k{k}.{png,pdf}` | in-window χ²/ndf vs m_WR per function (fixed y-range, `--chi2-ymax`, default 10) |
| `fit_uncertainty_vs_mass/{ch}_{topo}_k{k}.{png,pdf}` | relative fit uncertainty on the window yield `δB_fit/B_fit` vs m_WR |
| `params_vs_mass/{function}/{ch}_{topo}_k{k}.{png,pdf}` | each coefficient vs m_WR with its Minuit error, one panel per parameter |
| `diagnostics/{ch}_{topo}/{function}/m{mWR}.{png,pdf}` | local spectrum + fit (±1σ band); stat box with coefficients ± errors, χ²/ndf, and the **fit PASSED/FAILED(check list)** flag (`--diagnostics`) |
| `in_window_table_{ch}_{topo}.csv` | everything: window, B_MC(±), B_fit(±), χ²/ndf, coefficients ± errors, Minuit status/cov-status/EDM/NCalls, the three checks, `fit_passed` |

## What it shows (k=2, all four categories)

**With the μ-recentered parameterization, almost every fittable window passes.**
Pooled over the four categories (84 windows per function): `expo` **55/55** and
`powlaw` **55/55** (every fittable window), `expo2` **51**, `powexp` **45**,
`expo3` **44**; only `dexp` lags at **17** (its two terms are interchangeable
and a normalization can rail against a limit — a genuine degeneracy that
recentering does not remove).

This is a large change from the earlier m=0-pivot fits, where every
≥3-parameter function failed the covariance check at *every* mass. That failure
was a **parameterization artifact, not physics**: pivoting the exponent at m=0
while the window sits at t≈2–5 made `a` and `b` ~99.8% anti-correlated and the
polynomial regressors `t, t², t³` strongly collinear, so Minuit had to force the
covariance positive-definite (status 1, cov-status 2). Recentering at μ
(`u=(m−μ)/1000`, power laws use `m/μ`) de-correlates the parameters and
conditions the Hessian, so the covariance comes back accurate (cov-status 3).
The fitted curves are **identical** — same χ²/ndf, same `B_fit` — only the
parameter basis and its error matrix change. (Passing the covariance check means
the errors are *trustworthy*, not necessarily *small*; the window-yield
uncertainty `δB_fit/B_fit` is the separate quantity to weigh per function.)

The `expo`/`powlaw` fits remain good at low mass (χ²/ndf ≈ 0.7–1.7, slope `b` to
~10%). Windows still run out of background above ~4.4 TeV (resolved) /
~3.4 TeV (boosted), where the remaining masses are flagged not-fittable.
