# Stage 7 — expected-limit ("Brazil band") plots

> **STATUS: frozen cross-check (config A).** The quoted run2 limit is built by `../8_combine_limits/production/`; see `../LIMITS.md`. This homemade chain reaches only ~4 TeV (its toys need populated windows).

> New to limit plots? **`limit_plots_guide.md`** explains from scratch how the
> Stage-6 `nsp_hist` becomes these plots and how to read every feature.

Turns the **Stage-6 spurious-signal toy distributions** into an **expected 95 %
CL upper-limit-on-the-signal-yield** band vs `m_WR`. The Stage-6 toys are Poisson
draws of the background-only MC, each fit with `bkg + fixed-shape Gaussian`; the
fitted signal yield `N_sp` is the background-only distribution of the estimator —
exactly the ingredient of an expected-limit Brazil band. Here we map it into a
limit.

## Method — CL_s asymptotic, evaluated with the toy moments

For a Gaussian estimator `N_hat ~ N(mu0, sigma)` under background-only, the
one-sided **CL_s** asymptotic 95 % CL upper limit on the signal yield is

    UL(N_hat) = N_hat + sigma * Phi^{-1}( 1 - alpha * Phi(N_hat/sigma) ),  alpha = 1-CL

`UL(.)` is monotonic in `N_hat`, so the `p`-quantile of the limit distribution is
`UL` evaluated at the `p`-quantile of `N_hat`. The `±N sigma` band is therefore,
with `mu0 = <N_sp>` and `sigma = RMS(N_sp)` read straight from the Stage-6 CSV,

    UL_N = mu0 + sigma*N + sigma * Phi^{-1}( 1 - alpha * Phi(N + mu0/sigma) )
                                                    for N in {-2,-1,0,+1,+2}

i.e. exactly **"integrate the (per-toy) limit distribution from the left and read
the 2.5 / 16 / 50 / 84 / 97.5 % quantiles"**, in closed form.

- `sigma = RMS(N_sp)` — the **measured toy spread**, not the covariance-matrix fit
  error. The toys' pull width `~1.1` shows the fit errors slightly under-cover, so
  the RMS is the honest sigma.
- **CL_s** flavour keeps the `-2 sigma` edge positive (`~1.05 sigma`, vs the naive
  CL_{s+b} `median - 2 sigma` which would go negative).
- Median expected limit `= UL_0 ~ 1.96 sigma`.

The Gaussian model is exact to `<1 %` here because the Stage-6 `N_sp`
distributions are cleanly Gaussian (see `6_.../nsp_hist/`). If a raw *empirical*
CDF is ever wanted (to drop the Gaussian assumption / capture tails), Stage 6
would need to dump its per-toy `N_hat`/`sigma_hat` arrays.

### `--center mean` (default) vs `--center zero`

- **`mean`** centres the band at the toy `<N_sp>`, i.e. it integrates the *actual*
  `nsp_hist` — the fit bias is folded into the band. Faithful to "turn this
  histogram into a limit".
- **`zero`** is the textbook pure-statistical expected limit (band centred at 0);
  the spurious bias is then a *separate systematic*. Use this to compare against a
  `combine -M AsymptoticLimits` expected band.

## Run

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
python3 expected_limit.py --channel ee --topology resolved --trust-max 3400
```

Reads `../6_spurious_signal_toys/spurious_toy_table_{ch}_{topo}.csv` by default
(`--table` to point elsewhere, e.g. the `offgrid_2341/` table).

## Outputs

| path | contents |
|---|---|
| `expected_limit/{ch}_{topo}/{fn}.*` | Brazil band vs `m_WR`, one per function |
| `expected_limit_table_{ch}_{topo}.csv` | `UL_-2 … UL_+2`, `ul_med`, `ul_obs` per `(mass, fn)` |

The red points overlay the nominal-MC spurious yield `N_sp` (the Stage-5/6 Asimov
value) so the fit bias can be read against the statistical sensitivity.

## Caveats

- The `m_WR = 1000` point has a **clamped window** (hits `fit_min = 800`) → a huge,
  non-physical RMS; dropped by `--min-mass 1200` (default).
- Bands are a genuine per-mass toy measurement, so they are **jagged** — that is
  honest, not a bug.
- ee-resolved background closure is only trustworthy `≲ 3–3.4 TeV` (see Stage 4);
  `--trust-max` draws the marker. Read the band above it with care.
- The limit is in **event** units — `xsec_limit.py` (below) converts it to `σ×BR`.

# Stage 7b — cross-section limits (`xsec_limit.py`)

Converts the event-yield band into an upper limit on `σ×BR(pp → W_R → lljj)`:

    sigma_UL = N_UL / (1000 * L[fb^-1] * eff),   eff = S_fit / genEventSumw

per mass point, where `S_fit` is the raw signal-MC yield inside the Stage-6
S+B fit range `[fit_lo, fit_hi]` (read from the Stage-6 CSV, so the window is
identical by construction) and `genEventSumw` comes from the analyzer config
JSON. This pairing is exact because the signal histograms are **raw genWeight
fills** (xsec/lumi/sumw scaling was off when they were produced; verified —
per-histogram Σw² ≠ Σw matches the NanoAOD genWeight structure `±1/±0.99`,
~1 % negative weights, so `genEventSumw` ≠ event count by a real 1–3 %).
The conversion is a positive per-mass scale factor, so all five band edges
divide through and the quantiles survive.

The **RunII signals stand in for Run3**: shapes, efficiencies and `σ_theory`
come from the RunII UL18 samples/config (13 TeV, inclusive LLJJ — the
per-channel branching is absorbed by the selection), while `L` is the Run3
luminosity of the background band.

## Run

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
python3 xsec_limit.py --channel ee --topology resolved --trust-max 3400
```

Reads the **Stage-6** CSV directly (not the Stage-7 table — the band is
recomputed from the unrounded toy moments with the same `cls_band` code, and
the Stage-6 table also carries `fit_lo/hi` and `signal_tag`). Key knobs:
`--signal-era` / `--signal-dir` (default `RunIISummer20UL18` /
`20260624_signals`), `--signal-config` (default derived:
`../data/configs/RunII/2018/.../RunIISummer20UL18_signal.json`), and the same
`--center` / `--cl` / `--min-mass` as `expected_limit.py`.

## Outputs

| path | contents |
|---|---|
| `xsec_limit/{ch}_{topo}/{fn}.*` | `σ×BR` limit vs `m_WR` [TeV], official style: solid observed, dotted expected, 68/95% bands, red theory curve (fb, log-y) |
| `xsec_limit/{ch}_{topo}/{fn}_mu.*` | the same in `μ = σ/σ_theory`, line at `μ = 1` |
| `xsec_limit_table_{ch}_{topo}.csv` | eff bookkeeping + `σ` (pb) / `μ` bands per `(mass, fn)` |

The **"Observed limit"** is the CL_s limit evaluated at the unfluctuated-MC
(Asimov) `N_sp` with the **toy-RMS sigma** — the same calibrated sigma the band
uses, so observed vs expected differ only by the spurious-signal bias. (This
deviates from `expected_limit.py`'s `ul_obs`, which uses the per-fit covariance
error; that error under-covers and collapses in the sparse high-mass windows.)

The `σ_theory` crossing (equivalently `μ_UL = 1`) is the expected exclusion
reach — ee-resolved, expo: `m_WR ≈ 4.5 TeV` (well beyond the `≲3.4 TeV`
closure-trust boundary, so read it as indicative).

## Caveats

- `eff` uses the signal yield **inside the fit range** as the numerator. The
  fitted `N_sp` is the *full* Gaussian normalization, and the Gaussian template
  on a real (tailed) signal shape recovers roughly the in-range core — Stage 8
  injection with the true MC shape would measure the residual recovery factor.
- `σ_theory` and the efficiency follow the Stage-6 per-mass `signal_tag`
  (`m_N` closest to `m_WR/2`); other `m_N` change `eff` mostly via the window
  fraction (see the Stage-1 width-variation study).
