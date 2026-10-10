# signal_fitting — signal shape, background model, spurious signal and limits

Everything between the WR_Plotter histograms and an expected limit for the
W_R → ℓℓjj search: the on-shell signal shape and its width (stages 0–2), the
background spectra and the analytic in-window background fits (3–4), the
spurious-signal tests of that background model (5–6), and the limits, first
with a homemade CLs machinery (7) and then with CMS Combine (8).

Every stage reads the WR_Plotter rootfiles,
`WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<set>/WRAnalyzer_*.root`. The mass
observable is the reconstructed W_R mass: resolved → `mass_fourobject`
(m_ℓℓjj), boosted → `mass_twoobject` (m_ℓJ). Regions are ee / μμ × resolved /
boosted. Most of the limit work so far is **ee resolved**.

**The quoted result** is configuration C,
[`8_combine_limits/production/refined_limit_table_ee_resolved.csv`](8_combine_limits/production/refined_limit_table_ee_resolved.csv)
(run2 2018 ee resolved, data-free expected σ·B(eeqq̄′), 1000–6000 GeV, expected
exclusion ~5.04 TeV). [`LIMITS.md`](LIMITS.md) is the one-page summary of the
three limit configurations. Read it first if you only care about limits.

## Contents

| path | what |
|---|---|
| [`0_signal_samples/`](0_signal_samples/) | raw signal MC shapes (visual aids) |
| [`1_signal_widths/`](1_signal_widths/) | on-shell width per (M_WR, M_N): Gaussian core and RooKeysPdf FWHM |
| [`2_width_parameterization/`](2_width_parameterization/) | `wr/` = the window map μ(M_WR), σ(M_WR) used downstream; `wr_mn/` = width models in (x, M_WR) |
| [`3_background_plots/`](3_background_plots/) | SR MC component shapes, DY-CR and flavor-CR data/MC |
| [`4_background_fits/`](4_background_fits/) | analytic in-window background fits (`bkg_fit_lib.py`), smoothing study |
| [`5_spurious_signal/`](5_spurious_signal/) | Asimov spurious signal: S+B fit to background-only MC |
| [`6_spurious_signal_toys/`](6_spurious_signal_toys/) | the same over Poisson toys; card-likelihood toys |
| [`7_limit_plots/`](7_limit_plots/) | config A: homemade CLs band from the Stage-6 toys (frozen cross-check) |
| [`8_combine_limits/`](8_combine_limits/) | configs B (`baseline/`) and C (`production/`), `optimization/`, `explainer/` |
| [`run2_results/`](run2_results/) | the 2018 legacy (HNWRAnalyzer) inputs, converters, legacy vs new comparison |
| [`shared/`](shared/) | fit and loader cores imported by several stages |
| [`archived/`](archived/) | retired studies (see the end) |
| [`LIMITS.md`](LIMITS.md) | the limits on one page: configs A/B/C, which table supersedes which |
| [`validation_plan.md`](validation_plan.md) | the checklist for a data-driven background estimate (Jun 2026; uses the pre-renumbering stage names) |
| [`master_masses.csv`](master_masses.csv) | the signal grid: 481 points (`mass` = `WR<m>_N<n>`, `topology` = resolved/boosted) |
| `progress_slide.tex` / `.pdf` | Jun 2026 one-slide status (old numbering: its stage 6 is the now-archived injection study) |

## Environment

PyROOT scripts run from `WR_Plotter/`. The commands below use `$PY`:

```bash
cd ~/work/WrCoffea/WR_Plotter
# Mac: Homebrew ROOT + the fitting venv
export PYTHONPATH=/opt/homebrew/lib/root
PY=.venv-fitting/bin/python          # i.e. PYTHONPATH=/opt/homebrew/lib/root .venv-fitting/bin/python <script>
# LPC: source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh; PY=python3
```

Every script resolves its default inputs and outputs relative to its own file,
so the working directory does not matter (one exception:
`run2_results/convert_run2_dycr_mll.py` must run from `WR_Plotter/`). Most
scripts take `-v`.

**Combine** runs through the `run_*.sh` wrappers. Each wrapper re-execs itself
in the apptainer combine container
(`/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest`,
binding `/uscms_data` and `/cvmfs`), sets up CMSSW in the container, builds the
workspaces in the container and runs combine. On cmslpc, run the wrapper
directly, as in [`8_combine_limits/production/README.md`](8_combine_limits/production/README.md):

```bash
cd signal_fitting/8_combine_limits/production
./run_refined.sh ee resolved 500        # container; HybridNew toys/point = 500
# then, LCG_106:
python collect_refined.py -v
```

On the Mac the container runs inside the OrbStack VM `combine`, which needs
the repo bound in. The recipe in
[`4_background_fits/smoothing/README.md`](4_background_fits/smoothing/README.md):

```bash
orb -m combine bash -c 'cd $PWD && EXTRA_BINDS="-B /Users/billy/work/WrCoffea" SKIP_HYBRID=1 ./run_smooth_limits.sh kernel'
```

The same pattern applies to production, from `production/`:
`orb -m combine bash -c 'cd $PWD && EXTRA_BINDS="-B /Users/billy/work/WrCoffea" ./run_refined.sh ee resolved 500'`.
Only `run_refined.sh` (and `run_smooth_limits.sh`, which calls it) reads
`EXTRA_BINDS`. `run_limits.sh`, `run_opt.sh`, `run_sigscan_and_toys.sh` and
`run_fcr.sh` bind only `/uscms_data` and `/cvmfs`. `run_mc.sh` does not re-exec
at all. Every wrapper skips its re-exec when it is already inside a container,
so for those, start the container yourself with the bind (here from
`optimization/`):

```bash
orb -m combine bash -c 'cd $PWD && apptainer exec -B /cvmfs -B /Users/billy/work/WrCoffea /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest bash run_fcr.sh'
```

## Conventions that cut across stages

- **Eras and output folders.** Stages 3–7 and `8_combine_limits/baseline/`
  send their outputs to a `run2/` or `run3/` sub-folder, chosen by `--era`.
  `production/` and `optimization/` have no run level: they default to run2,
  and the optimization 2024 chain uses `*_2024*` folders. The default for
  Stages 3–7 is `--era RunIII2024Summer24 --dir 20260317_lo_dy` (the LO-DY 2024 set). The
  chain that feeds configs A/B/C is run2:
  `--era RunIISummer20UL18 --dir 20260714_run2_bkgs` (the legacy 2018
  histograms converted by `run2_results/`). The `card_model_toys.py`,
  `8_combine_limits/optimization/` and `production/` defaults are already run2.
  Signal MC for stages 0–2 and 7 is `RunIISummer20UL18/20260624_signals`.
- **Window.** `[μ − kσ, μ + kσ]`, with μ and σ taken from the Stage-2 linear
  map `2_width_parameterization/wr/window_params.json` (`--window-source param`,
  the default), clamped to [800, 6000] GeV. Stages 4–6, config A and config B
  use k = 3 with 100 GeV bins. The config C float cards (1400–3200 GeV), the
  card-model toys and the explainer use k = 5 with 50 GeV bins (`k5_bw50`).
  The config C anchored cards keep the Stage-6 k = 3 windows on 100 GeV bins.
- **Binning rule.** Every plot of a spectrum that goes into a card uses the
  same bin width that combine receives for that card: 50 GeV for `k5_bw50`
  cards, 100 GeV for the k3 cards. Never use a display-only binning.
- **Limits.** No data in the expected bands (MC Asimov; the baseline also has
  a data-observed variant). Bands are centred at zero (`--center zero`, the
  default). Per-channel σ·B uses `bfrac = 0.5`, because the run2 signal samples
  are 50/50 e:μ flavor-mixed. The run2 lumi is 59.83 fb⁻¹.

## Glossary (signal width)

| term | meaning |
|---|---|
| **cell** | one `(channel, mass)` point; `channel ∈ {ee, mumu}` |
| **x** | `M_N / M_WR`, the abscissa of every width trend plot |
| **topology** | `resolved` (full x grid) vs `boosted` (x < 0.1, discovered from the MC on disk). The x ranges are disjoint, so **never** parameterize across them |
| **seed windows** | `[0.70,1.30]`, `[0.80,1.20]`, `[0.85,1.15]` × M_WR. Every Stage-1 deep dive fits all three |
| **σ_gauss^on** | width of the **iterative-core Gaussian** fit: fit in `[μ±2σ]`, re-seeded until μ and σ move < 2 %. Keys off the peak |
| **σ_FWHM^on** | `FWHM / 2.3548` of a **RooKeysPdf** (adaptive KDE) over the window. Keys off the peak |

Only these two widths survived. A window-stability study showed that σ_eff and
the RMS shift by ~16–29 % when the window changes (they integrate the tails),
while the Gaussian core (~1.7 %) and the KDE FWHM (~0.6 %) are stable.

## Stage 0 — signal samples ([`0_signal_samples/`](0_signal_samples/))

Visual aids for the raw MC signal shapes. They are not part of the width chain.
Both scripts read `master_masses.csv` and plot each point in its tagged
topology, at 80 GeV bins.

- `signal_samples.py`: one MC mass histogram per (channel, mass) →
  `1d_histograms/<mass>/<channel>.{png,pdf}`.
- `signal_peak_overlay.py`: for each M_WR, all of its M_N points overlaid at
  unit area, colored by M_N → `peak_overlays/`.

```bash
$PY signal_fitting/0_signal_samples/signal_samples.py -v
$PY signal_fitting/0_signal_samples/signal_peak_overlay.py --wr-list 2400 4000 5600
```

## Stage 1 — on-shell widths ([`1_signal_widths/`](1_signal_widths/))

Measures the width of the signal peak per cell and picks a window-robust
definition. Each script writes its outputs next to itself.

| script | does | writes |
|---|---|---|
| `gaussian/detail_gauss_fit.py` | iterative-core Gaussian per cell and seed window | `gaussian/gauss_fit_table.csv`, `window_stability.csv`, `histograms/`, `chi2_ndf/`, `sigma_over_mWR/`, `mu_over_mWR/`, `robustness/` |
| `fwhm/detail_fwhm.py` | RooKeysPdf peak and FWHM per cell and seed window | `fwhm/fwhm_table.csv`, `window_stability.csv`, `histograms/`, `sigma_over_mWR/`, `peak_over_mWR/`, `robustness/` |
| `gauss_vs_fwhm/compare_gauss_fwhm.py` | the two widths overlaid, and their ratio (≈ 0.75–0.85) | `gauss_vs_fwhm/sigma_over_mWR/`, `ratio_fwhm_over_gauss/` |
| `plot_width_variation.py` | spread of μ_gauss and σ_gauss over M_N at each M_WR, i.e. what the per-M_WR median hides (matplotlib only) | `width_variation/{sigma_abs,sigma_over_mWR,mean_abs,mean_over_mWR}/` |

Which seed-window rows are used: the per-cell plots, the robustness ratios,
`compare_gauss_fwhm.py` and Stage 2 `wr_mn/` use the `[0.8,1.2]` rows
(`BASELINE_WINDOW` / `BASE`). The window map (Stage 2 `wr/`) and Stages 4–6
use the `[0.7,1.3]` rows of `gauss_fit_table.csv` (`BASELINE_FIT_RANGE`). Each
deep dive prints the window-stability verdict and writes it to
`window_stability.csv`. `--plots-only` rebuilds the plots from the existing
tables.

```bash
$PY signal_fitting/1_signal_widths/gaussian/detail_gauss_fit.py -v   # run the two deep dives first:
$PY signal_fitting/1_signal_widths/fwhm/detail_fwhm.py -v            # everything else reads their tables
$PY signal_fitting/1_signal_widths/gauss_vs_fwhm/compare_gauss_fwhm.py
$PY signal_fitting/1_signal_widths/plot_width_variation.py -v
```

The resolved grid comes from `master_masses.csv` (`--mass-csv`). Boosted cells
(x < 0.1, `--boosted-max-x`) are discovered from the MC on disk.

## Stage 2 — width parameterization ([`2_width_parameterization/`](2_width_parameterization/))

There is no stage README. This section is the reference.

**`wr/`: the window map that everything downstream uses.** For defining a
search window only M_WR is known, so `parameterize_window.py` takes, at each
grid M_WR, the median over M_N of the Stage-1 Gaussian μ and σ (`[0.7,1.3]`
rows). It then fits straight lines μ(M_WR) and σ(M_WR) per (channel,
topology). Two σ conventions are stored: `median` (the default everywhere)
and `conservative` (the maximum over M_N after dropping x > 0.9). The script
writes `window_params.json`, plus `fit/` (data and line) and `closure/`
(predicted vs measured, residuals). `window_for_mwr.py` turns any M_WR, on or
off the grid, into μ, σ and the window edges. It can also be imported.

```bash
$PY signal_fitting/2_width_parameterization/wr/parameterize_window.py -v
$PY signal_fitting/2_width_parameterization/wr/window_for_mwr.py --mwr 2341 --channel ee --topology resolved --k 3
```

`window_params.json` is read by `4_background_fits/bkg_fit_lib.py` and
`in_window_fit.py`, by Stages 5–6, by `7_limit_plots/window_optimization/` and
by `8_combine_limits/` (`optimization/`, `explainer/`).

**`wr_mn/`: the width as a function of (x, M_WR).** Nothing downstream reads
it. It is the study behind the choice of width.

| script | does | writes (beside itself) |
|---|---|---|
| `parameterizations/parameterize_width.py` | fits the model registry to σ/M_WR for both widths: 1D in x (`pol2/3/4`, `physics`, `spline`) and 2D in (x, M_WR) (`poly3+mass`, `fxgx`, `+m2`, `+x2m`, `spline2d`) | `parameterization_params.json`, `1d/`, `2d/`, `all_models/` |
| `cross_validation/validate_models.py` | scores every model by leave-one-M_WR-out CV and recommends the best per cell (imports the registry) | `cv_summary.csv`, `predictions.csv`, `cv_comparison/`, `residuals/`, `pred_vs_meas/`, `best_model_residual/` |

```bash
$PY signal_fitting/2_width_parameterization/wr_mn/parameterizations/parameterize_width.py
$PY signal_fitting/2_width_parameterization/wr_mn/cross_validation/validate_models.py
```

Both read the Stage-1 tables (`--widths-dir`) and write beside themselves
(`--out-dir`). Bottom line, from
[`wr_mn/INTERPRETATION.md`](2_width_parameterization/wr_mn/INTERPRETATION.md):
the Gaussian core width is the smoother, more predictive definition (lower CV
in every cell). The CV minima are ~1.8–3.0 % (gauss) and ~3.1–6.7 % (fwhm),
from the flexible `spline2d`/`fxgx`. The recommended closed form is
`poly3+mass`. FWHM is ~1.4–2.2× noisier and also needs the mass term. Treat
resolved/boosted and ee/μμ separately.

## Stage 3 — background plots ([`3_background_plots/`](3_background_plots/README.md))

The summed background B = DY+jets + tt̄/tW + nonprompt + other, looked at
before choosing a fit function.

- `mc_shapes/plot_mc_backgrounds.py`: SR MC-only views (`--plots overlay stack
  individual`) plus `component_yields.csv`.
- `cr_plots/plot_control_regions.py`: DY CR and flavor (eμ) CR, stacked MC +
  data + ratio. The flavor CR is drawn once per dataset (EGamma, Muon).

Inputs are the background MC and data of `--era/--dir`. The MC is already
lumi-scaled. Outputs go to `<script dir>/<run2|run3>/`.

```bash
$PY signal_fitting/3_background_plots/mc_shapes/plot_mc_backgrounds.py -v
$PY signal_fitting/3_background_plots/cr_plots/plot_control_regions.py -v
```

## Stage 4 — background fits ([`4_background_fits/`](4_background_fits/README.md))

Can the summed MC background be described by a smooth function **inside** the
signal window? `bkg_fit_lib.py` is the fit library: TF1 χ² fits with Minuit2,
recentered at the window peak, with six functions (`expo`, `expo2`, `expo3`,
`powlaw`, `powexp`, `dexp`). A fit passes three checks: valid minimum,
accurate covariance, no parameter at a limit. Stages 5–8 import this library.

- `in_window_fit/in_window_fit.py`: per grid M_WR and function, χ²/ndf, fit
  uncertainty, parameters and diagnostics → `in_window_fit/<run2|run3>/`
  (`chi2_ndf_vs_mass/`, `fit_uncertainty_vs_mass/`, `params_vs_mass/`,
  `diagnostics/`, `in_window_table_{ch}_{topo}.csv`). The k sweeps live in
  `k2/`…`k5/`. `make_check_table.py --run run2|run3` turns the tables into
  LaTeX pass-count tables (`fit_checks/`).
- [`smoothing/`](4_background_fits/smoothing/README.md) (Sep–Oct 2026): four
  smoothers (kernel, hybrid, param, RooKeysPdf) of the jagged high-mass MC.
  They are fed opt-in into the config C cards. Effect on the limit: < 3 %. The
  folder also has the fit-statistic finding (χ²/Neyman and effective-count
  likelihoods are biased low on weighted MC, so use `TH1::Fit "L"`), the
  per-process split sets and the legacy vs new-framework comparison. Run2 ee
  resolved by default → `run2/ee_resolved*/`. The 2024 sets → `run3/`.

```bash
$PY signal_fitting/4_background_fits/in_window_fit/in_window_fit.py -v --diagnostics
$PY signal_fitting/4_background_fits/smoothing/smooth_background.py -v
$PY signal_fitting/4_background_fits/smoothing/compare_methods.py -v --toys 200
```

Window inputs: `window_params.json` (default), or the Stage-1 Gaussian table
with `--window-source measured`.

## Stage 5 — spurious signal ([`5_spurious_signal/`](5_spurious_signal/README.md))

Does the background model fake a signal? `spurious_signal.py` fits the Stage-4
function plus a fixed-shape Gaussian (μ, σ of the M_N ≈ M_WR/2 point) to the
**background-only** MC inside the window. N_sp and the pull N_sp/σ(N_sig) are
reported per mass and function. Core: [`shared/sb_fit.py`](shared/sb_fit.py).
Outputs go to `<run2|run3>/` (`pull_vs_mass/`, `spurious_yield_vs_mass/`,
`fit_diagnostics/`, `spurious_table_{ch}_{topo}.csv`). Sweeps: `compare_k.py`,
`compare_binwidth_k.py`, `rebinning_study.py` (`k2/`…`k5/`, `k_comparison/`,
`offgrid_2341/`).

```bash
$PY signal_fitting/5_spurious_signal/spurious_signal.py --channel ee --topology resolved
```

Caveat for Stages 5–6: these are method-validation numbers (Run-2 signal widths
on whichever background `--era` selects), not the final bias systematic.

## Stage 6 — spurious-signal toys ([`6_spurious_signal_toys/`](6_spurious_signal_toys/README.md))

The toy version of Stage 5. Poisson toys of the background-only MC, each fit
with the same S+B model, give the **distribution** of N_sp: mean (bias), RMS
(resolution) and pull width (coverage).

- `spurious_signal_toys.py` → `<run2|run3>/` (`pull_mean_vs_mass/`,
  `pull_width_vs_mass/`, `spurious_yield_vs_mass/`, `nsp_hist/`,
  `spurious_toy_table_{ch}_{topo}.csv`). **This table is the input of config A
  (Stage 7) and of the config B inputs.**
- `card_model_toys.py`: the same toys, fit with the combine card's likelihood
  ([`shared/card_sb_fit.py`](shared/card_sb_fit.py), 50 GeV, `k5_bw50`).
  Models: `card_float`, `card_fcr`, `card_fcrs`; `--cr-obs EGamma` labels the
  rows `card_fcrd`. Geometry comes from `8_combine_limits/optimization/inputs`.
  Outputs go to `<run>/card*/`.
- `single_toy_diagnostic.py` (one toy's S+B picture) and
  `plot_card_spurious_vs_mass.py <card_toy_table.csv>` (mean ± RMS vs mass).

```bash
$PY signal_fitting/6_spurious_signal_toys/spurious_signal_toys.py --channel ee --topology resolved
$PY signal_fitting/6_spurious_signal_toys/card_model_toys.py --ntoys 1000 --hist-adaptive -v
```

## Stage 7 — homemade limits, config A ([`7_limit_plots/`](7_limit_plots/README.md))

**Frozen cross-check.** It turns the Stage-6 toy moments (⟨N_sp⟩, RMS) into a
closed-form CLs Brazil band. Its reach is ≤ 4 TeV, because the toys die in
empty windows. New to limits? Start with
[`limit_plots_guide.md`](7_limit_plots/limit_plots_guide.md).

- `expected_limit.py`: band on the signal yield (events) →
  `<run>/expected_limit/`, `expected_limit_table_{ch}_{topo}.csv`.
- `xsec_limit.py`: events → σ×B with eff = S_window / (bfrac · genEventSumw),
  from the raw signal histograms and the signal config JSON →
  `<run>/xsec_limit/`, `xsec_limit_table_{ch}_{topo}.csv`. The config A table
  is `run2/xsec_limit_table_ee_resolved.csv`.
- Comparisons with the official 2018 result: `digitize_official2018.py`
  (→ `official2018_expected_digitized.csv`, check image `digitize_check*.png`),
  `compare_homemade_official.py`, `compare_run2_reference.py`.
- [`window_optimization/`](7_limit_plots/window_optimization/README.md): signal
  efficiency and sensitivity vs the window half-width k. Useful range k ≈ 3–5.
- `step2_explainer/` … `step5_explainer/`: the statistics explainers, one
  figure per script.

```bash
$PY signal_fitting/7_limit_plots/xsec_limit.py --era RunIISummer20UL18 --channel ee --topology resolved --trust-max 3400
```

Both scripts read `6_spurious_signal_toys/<run>/spurious_toy_table_{ch}_{topo}.csv`
(`--table` overrides it).

## Stage 8 — combine limits ([`8_combine_limits/`](8_combine_limits/))

| folder | config / role | main scripts | writes |
|---|---|---|---|
| [`baseline/`](8_combine_limits/baseline/README.md) | **B**: the config-A model as combine cards (k3, floating expo/powlaw, fixed Gaussian, 100 GeV). Reference, deliberately un-optimized | `prepare_inputs.py` → `run_limits.sh CH TOPO FNS MASSES RUN [mc\|data]` (builds workspaces with `make_workspaces.py`, then AsymptoticLimits) → `collect_limits.py`, `plot_combine_limits.py` | `<run>/{inputs,cards,results,plots}/`, `<run>/combine_limit_table_{ch}_{topo}.csv` |
| [`production/`](8_combine_limits/production/README.md) | **C**: the quoted result. Regime-split cards: anchored background at 1000–1200 and ≥ 3400 (HybridNew in the sparse tail), optimized float card at 1400–3200 (k5, 50 GeV, anchor-constrained slope, μ/σ priors ≤ 2600) | `run_refined.sh CH TOPO NTOYS` (runs `make_refined_workspaces.py`) → `collect_refined.py`; `compare_production_official.py` | `cards/`, `results/`, `refined_limit_table_ee_resolved.csv`, `plots/ee_resolved/refined.*` |
| [`optimization/`](8_combine_limits/optimization/README.md) | evidence for C's float card: the configuration scan, the FitDiagnostics toy validation, and the flavor-CR anchored variants (`k5_bw50_fcr`, `_fcrd`, `_fcrs`) | `prepare_opt.py` → `run_opt.sh` → `collect_opt.py`; `run_sigscan_and_toys.sh` → `collect_toys.py`; `prepare_fcr.py` → `run_fcr.sh` | `inputs/` (read by C, Stage 6 and the explainer), `cards/`, `results/`, `opt_table_ee_resolved.csv`, `toy_validation_table.csv` |
| [`explainer/`](8_combine_limits/explainer/README.md) | documentation: how C is built, data-free, step by step. Steps 1–5 are the inputs, 6–9 rebuild combine's statistics from scratch (`combine_from_scratch.py`), 10–12 cover the production choices, 13 the mass regimes | `step*/step*.py` | figures next to each script |

```bash
# config B, run2 (PyROOT steps with $PY, combine through the wrapper; Mac: see Environment)
$PY signal_fitting/8_combine_limits/baseline/prepare_inputs.py --era RunIISummer20UL18 --dir 20260714_run2_bkgs --with-data
(cd signal_fitting/8_combine_limits/baseline && ./run_limits.sh ee resolved expo,powlaw all run2 data)
$PY signal_fitting/8_combine_limits/baseline/collect_limits.py --run run2
# config C
(cd signal_fitting/8_combine_limits/production && ./run_refined.sh ee resolved 500)
$PY signal_fitting/8_combine_limits/production/collect_refined.py -v
# explainer (WR_SEMINAR_ERA=2024 switches to the 2024 inputs, outputs suffixed _2024)
for s in signal_fitting/8_combine_limits/explainer/step*/step*.py; do $PY "$s"; done
```

Config C depends on config B's run2 inputs: `production/` reads
`baseline/run2/inputs/` (data_obs, rates and efficiencies of the anchored
cards) and the baseline run2 table (`rMax`), plus `optimization/inputs/` for
the float cards. Rebuild B's run2 inputs before C.

`optimization/` also holds the Oct 2026 HEP-seminar 2024 chain, which its
README does not yet cover: `WR_SEMINAR_ERA=2024`, `inputs_2024*/`,
`cards_2024*/`, `results_2024*/`, `explain_fcr*.py`, `plot_fcr_limit.py`,
`make_mc_cards.py` + `run_mc.sh`, `signal_systs.py`.

### The three limit configurations

As defined in [`LIMITS.md`](LIMITS.md):

| config | what | where | table (run2 ee resolved) | status |
|---|---|---|---|---|
| **A, homemade** | Stage-6 toys → closed-form CLs band | `7_limit_plots/` | `run2/xsec_limit_table_ee_resolved.csv` | frozen cross-check, reach ≤ 4 TeV |
| **B, combine baseline** | the same model as A, as combine cards (k3 window, floating expo, fixed signal) | `8_combine_limits/baseline/` | `run2/combine_limit_table_ee_resolved.csv` | reference point, deliberately un-optimized; for method comparisons only |
| **C, combine production** | regime-split cards (anchored + optimized float, HybridNew in the sparse tail) | `8_combine_limits/production/` | `refined_limit_table_ee_resolved.csv` | **quoted** |

## run2_results — the Run 2 legacy comparison ([`run2_results/`](run2_results/))

The 2018 legacy analysis (HNWRAnalyzer) histograms
(`HNWRAnalyzer_SkimTree_LRSMHighPt_*.root`: DY LO-HT raw / reweighted /
reweighted+reshaped, TT_TW, NonPrompt, Others, SingleElectron and SingleMuon
data) and the official 2018 ee limit plot
(`1D_EE_Combined_HalfN_Limit_vs_WR.pdf`, digitized by
`7_limit_plots/digitize_official2018.py`).

| script | does | writes |
|---|---|---|
| `convert_run2_backgrounds.py` | renames the legacy WRCand-mass histograms into plotter format (DY = LO HT, reweighted and reshaped) | `rootfiles/RunII/2018/RunIISummer20UL18/20260714_run2_bkgs/`, the run2 background set of Stages 3–8 |
| `convert_run2_dycr_mll.py` | legacy ZCand mass in the resolved DY CR (run from `WR_Plotter/`) | `.../20260926_run2_bkgs_mll/` |
| `convert_run2_dycr_unshaped.py` | resolved DY CR with the raw (unshaped) DY | `.../20260929_run2_bkgs_unshaped/` |
| `compare_legacy_new.py` | legacy vs WrCoffea sets, every region: yields, data/MC closure, cutflow, new/legacy ratios. Blinding is built in | `framework_comparison/` (`yields.csv`, `closure.csv`, `cutflow.csv`, PDFs) |

```bash
$PY signal_fitting/run2_results/convert_run2_backgrounds.py -v
$PY signal_fitting/run2_results/compare_legacy_new.py
```

The sibling `4_background_fits/smoothing/compare_frameworks.py` does the ee
resolved SR only. The smoothing README's "Known issue" section records why the
legacy and the new-framework 2018 histograms cannot be mixed.

## shared — common cores ([`shared/`](shared/))

| module | role | used by |
|---|---|---|
| `shape_estimators.py` | signal-cell discovery and loading (`collect_cells`, `discover_masses`, `load_master_masses`), width primitives (`gaussian_core_fit`, `keys_fwhm_detail`, …), M_WR-colorbar plotters, `window_stability_report` | Stages 0, 1, 5, 6, `optimization/` |
| `measure_fwhm.py` | the on-shell window constants (`ONSHELL_WINDOW_*_FRAC` = 0.7/1.3), mass parsing, signal histogram loading and rebinning | Stages 0, 5, 6, 7, 8 |
| `sb_fit.py` | the TF1 S+B fit core (Stage-4 background + fixed Gaussian) and the width/signal-template loaders | Stages 5, 6, `window_optimization/` |
| `card_sb_fit.py` | `CardSB`: the combine card's likelihood rebuilt (extended binned Poisson; modes `float`, `fcr`, `fcr_shape`) | `card_model_toys.py`, `optimization/` (`explain_fcr*`, `make_mc_cards`, `plot_bkg_*`), `smoothing/compare_methods.py` |
| `syst_band.py` | total systematic envelope from the analyzer's `syst_<src><up\|down>_<region>` histograms | `optimization/` (`prepare_fcr`, `signal_systs`, `make_mc_cards`, `explain_fcr`), `explainer/_common.py` |

The background-function library itself is
[`4_background_fits/bkg_fit_lib.py`](4_background_fits/bkg_fit_lib.py).

## archived — retired

[`archived/`](archived/) holds retired work. Nothing active writes there; do
not extend it. It contains `10_limit_refinement/` (studies 10.1–10.6 that fixed
the homemade machinery; `6_combine_parity/` proved combine ≡ homemade in the
trusted regime; its README is the study log), `8_signal_injection_study/`
(superseded by the optimization FitDiagnostics toys),
`4_background_fits/{sideband_closure,flavor_cr_fit}/`, the old
single-Gaussian prior-calibration pipeline
(`old_prior_calibration_pipeline/`), `gauss_only_study/`,
`legacy_outputs_RunIISummer20UL18/` and loose demo scripts. One read-only
dependency remains: `8_combine_limits/explainer/combine_from_scratch.py` falls
back to the 10.6 parity CSV in `archived/10_limit_refinement/6_combine_parity/`.

## History

| date | change |
|---|---|
| 2026-06-01/02 | Width-estimator study: resolved shapes are strongly non-Gaussian (narrow core, heavy low tail). Window stability is the selection criterion. σ_eff, σ_eff-sym, RMS and the wide best-Gaussian fit were dropped; σ_gauss^on and σ_FWHM^on kept. Boosted redefined as x < 0.1 |
| 2026-06-03 | Width work reorganized into `1_signal_widths/`, outputs co-located per method. The resolved grid moved to `master_masses.csv`. The prior-calibration / pull study retired to `archived/old_prior_calibration_pipeline/` |
| Jun 2026 | `2_width_parameterization/` split into `wr/` (the M_WR-only window map) and `wr_mn/` (the original (x, M_WR) models). Stages 3–7 added |
| Jul 2026 | Stage outputs split into `run2/` and `run3/` by `--era`. Combine stages added; layout renumbered to a contiguous 0–8 + `archived/` (old paths: see the history note in `LIMITS.md`). `internals_explainer/` merged into `explainer/` |
| Aug–Oct 2026 | Flavor-CR anchored cards and card-model toys (Aug), background smoothing (Sep), legacy vs new-framework comparison and the 2024 seminar chain (Oct) |
