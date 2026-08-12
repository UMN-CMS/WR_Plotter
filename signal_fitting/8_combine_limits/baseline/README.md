# Stage 9 — the same limit with CMS Combine

> **STATUS: reference baseline (config B).** Deliberately un-optimized (k3 window, floating expo, fixed signal) — the comparison point for `../production/`. See `../../LIMITS.md`. (Previously `9_combine_limits/`.)

Reproduces the Stage-7/7b expected-limit machinery with
`combine -M AsymptoticLimits`, using the **same statistical model** as
`shared/sb_fit.py`: recentered falling background (expo/powlaw, slope ≤ 0,
floating norm) + fixed-shape Gaussian(m_c, sigma_win), fit to the summed
background MC in the Stage-6 window. Windows, signal tags and efficiencies are
read from the Stage-6 CSV / Stage-7b machinery — nothing re-derived.

Signal `rate` = lumi × eff = events per **1 fb** of sigma×BR, so the POI is
the cross section directly: `r_UL == sigma_UL [fb]`.

## Workflow

| step | script | env | does |
|---|---|---|---|
| 1 | `prepare_inputs.py` | LCG_106 | background TH1 + per-mass JSON (window, eff, rate, toy moments) |
| 2 | `make_workspaces.py` | container (plain PyROOT) | RooWorkspace + datacard per (mass, fn) |
| 3 | `run_limits.sh` | auto re-execs into container | steps 2 + `combine -M AsymptoticLimits` loop |
| 4 | `collect_limits.py` | LCG_106 | results → `combine_limit_table_{tag}.csv` + homemade comparison |

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
python3 prepare_inputs.py --channel ee --topology resolved
./run_limits.sh ee resolved expo,powlaw all     # or: ee resolved expo 2000
python3 collect_limits.py --channel ee --topology resolved
```

Workspaces are built **inside** the container on purpose (same ROOT version
writes and reads them); step 1 writes only plain TH1D + JSON, which is
version-safe. The container is the standard cms-analysis combine image
(CMSSW_14_1_0_pre4, combine v10) from unpacked cvmfs; `run_limits.sh` binds
`/uscms_data` so all repo paths resolve unchanged.

## Phase-1 parity check (ee resolved, expo, m_WR = 2000)

Data = background MC ("observed" = MC Asimov), stat-only. Events:

|            | −2σ | −1σ | median | +1σ | +2σ | obs |
|---|---|---|---|---|---|---|
| combine    | 19.2 | 26.1 | 36.7 | 52.0 | 70.4 | 39.7 |
| homemade (center-zero, σ=RMS) | 25.2 | 33.8 | 47.0 | 65.4 | 87.6 | 49.0 |
| ratio      | 0.76 | 0.77 | 0.78 | 0.80 | 0.80 | 0.81 |

The **uniform ~0.78 ratio** is the expected sigma-convention difference, not a
model discrepancy:

* `FitDiagnostics` on the same card: background-only norm = **219.203** =
  exactly the Stage-6 `B_window`; slope b = −2.74/TeV (sensible, cf. the
  Stage-4 component fits); best-fit signal 4.0 events vs `nsp_asimov` = 3.0.
* combine's likelihood error on the yield: **σ ≈ 20.8 events ≈ sb_fit's
  covariance error (20.2)** — the two likelihoods agree.
* the homemade band deliberately uses the **toy RMS (24.0)** instead, because
  the Stage-6 pulls showed the likelihood error under-covers (pull width 1.19
  at this mass). RMS/σ_A = 1.19 ⇒ ratio ≈ 0.84; the remaining few % is
  combine's bounded q̃_μ machinery (σ re-derived from the Asimov per tested r).

So: same model, same fit, same CLs band geometry (both floors the −2σ edge);
combine is ~20% more aggressive **because it trusts the likelihood width**
while the homemade band is toy-calibrated. Combine's expected band corresponds
to the homemade `--center zero`; the spurious bias enters only through the
MC-as-data "observed" (until Phase 3 promotes it to a nuisance).

## Phase 2 — full scan (done)

25 masses x {expo, powlaw}, collected into `combine_limit_table_{tag}.csv` and
plotted by `plot_combine_limits.py` (step 5 in the table above):

| path | contents |
|---|---|
| `plots/{tag}/{fn}.*` | official-style sigma x BR limit (same `plot_band` as Stage 7b) |
| `plots/{tag}/{fn}_mu.*` | the same in mu = sigma/sigma_theory |
| `plots/{tag}/{fn}_overlay.*` | combine band vs homemade center-zero band + median-ratio panel |

Findings (ee resolved, expo):

* **Below ~3 TeV the two methods agree to 0–30%** (homemade/combine median
  ratio 1.0–1.3), with the same band geometry — the residual is the
  toy-RMS-vs-likelihood-sigma convention established in Phase 1.
* **Above ~3.2 TeV they diverge sharply** (ratio 1.4 at 3.0 TeV, >2 at
  3.6 TeV): the windows get sparse, the asymptotic likelihood error stays
  small while the toy RMS blows up. This is the known low-count breakdown of
  the asymptotic approximation — the homemade toy band is the calibrated one
  there, and any combine number beyond ~3 TeV should eventually be validated
  with `HybridNew` toys (Phase 3). Consistent with (and adds to) the
  Stage-4 "trustworthy ≲ 3.4 TeV" closure boundary.
* combine extends to 6 TeV without dropout (no toys needed) and its band is
  smooth (per-mass Asimov, no toy jitter); expected theory crossing ~4.9 TeV
  vs the homemade ~4.5 TeV — both far beyond the trust boundary, indicative
  only.

## Roadmap
* **Phase 3** (each optional, in value order) — spurious-signal as a
  constrained additive nuisance (combine analog of `--center mean`);
  RooMultiPdf discrete profiling over expo/powlaw; true MC signal shape
  (RooHistPdf) instead of the Gaussian; lumi/eff systematics; mumu + boosted;
  `combineCards.py` channel combination; `HybridNew` toy cross-check at a few
  masses.

## Notes / gotchas

* `prepare_inputs.py` clips negative/non-finite background bins to 0 (sparse
  tail only — same treatment as the Stage-6 toy means).
* The fit range is snapped to the 100 GeV grid the way `TH1::Fit` selects bins
  (bin centers inside `[fit_lo, fit_hi]`), so combine fits exactly the bins
  `sb_fit` fit.
* combine's Gaussian yield is normalized within the fit range (≤0.3% from the
  full-Gaussian convention of `sb_fit` at k=3 — negligible).
* The container spits out a harmless numpy `_ARRAY_API` warning from
  pandas/bottleneck on startup; ignore it.

## run2 (Jul 2026)

The whole stage is run-aware: `{run2,run3}/{inputs,cards,results,plots}` +
per-run tables; `run_limits.sh CH TOPO FNS MASSES RUN [mc|data]` (the 6th arg
picks the observation; `prepare_inputs.py --with-data` dumps the real-data
histogram). The refined run2 chain (regime-split anchored cards + HybridNew)
lives in `../production/`, and
**`../explainer/`** is the step-by-step explainer of the data-free run2
expected limit with every optimized setting quantified against its
alternative, including combine's statistical internals rebuilt from scratch
as its steps 6–9 (see its README).
