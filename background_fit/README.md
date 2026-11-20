# background_fit

The background of the signal + background fit is an analytic function fitted to the data. These scripts choose
and test that function on simulation, in the resolved SR (m_lljj) and the boosted SR (m_lJ), ee and mumu.

## Setup

1. A Python that has ROOT (PyROOT), numpy, uproot, matplotlib and mplhep. Nothing is installed from
   `WR_Plotter/`. At the LPC, in a new shell with no other environment active:

   ```bash
   source /cvmfs/sft.cern.ch/lcg/views/LCG_108/x86_64-el9-gcc13-opt/setup.sh
   ```

2. The analyzer's histograms under `rootfiles/`, one directory per era. `<dir>` is the name given to the analyzer
   with `--dir`, the same in every era.

   ```text
   rootfiles/Run3/2022/Run3Summer22/<dir>/
   rootfiles/Run3/2022/Run3Summer22EE/<dir>/
   rootfiles/Run3/2023/Run3Summer23/<dir>/
   rootfiles/Run3/2023/Run3Summer23BPix/<dir>/
   rootfiles/Run3/2024/RunIII2024Summer24/<dir>/
   ```

   Each holds `WRAnalyzer_DYJets.root`, `WRAnalyzer_tt_tW.root`, `WRAnalyzer_Nonprompt.root` and
   `WRAnalyzer_Other.root`.

3. Name the directory and the dataset. Every command below runs from `WR_Plotter/`.

   ```bash
   export FIT_DIR=<dir> FIT_DATASET=run3
   ```

## First run: the fitted background parameters of every window

Store the background the fits run on, the simulation as it is.

```bash
python background_fit/3_truths.py --truths raw --variants lo_ht
```

Fit each function in the window of every signal mass and write its parameters to a table.

```bash
python background_fit/26_window_params.py
```

The table is `background_fit/results_run3/<channel>/26_window_params/window_params_k5_lo1200.csv` (`results/` for
`FIT_DATASET=2024`), one row per background, function and signal mass. `--help` lists the options (functions,
window width, floor, masses).

| Column | Meaning |
| --- | --- |
| `channel` | `ee` or `mumu` |
| `dataset`, `lumi` | the `FIT_DATASET` and its luminosity in fb^-1 |
| `variant` | the DY sample of the background (`lo_ht`: LO, HT-binned) |
| `background` | the spectrum that is fitted: `raw`, the simulation as it is, or `kde9`, the RooKeysPdf of the nine processes |
| `function` | the function's name: `expo`, `powexp`, ... |
| `k` | the window's half-width in units of `sigma` |
| `mass` | the signal mass m_WR of this row, GeV |
| `mu`, `sigma` | mean and width of the signal peak at this mass, GeV |
| `lo`, `hi` | the fit window, GeV: `mu` -+ `k` x `sigma`, clipped at the floor and at 8000 |
| `n_window` | expected background events in the window |
| `n_core` | expected background events within +-2 `sigma` of `mu` |
| `formula` | the function in ROOT TF1 syntax: `x` in GeV, `[i]` is `p<i>`, the value is events per bin |
| `pivot` | the mass in the formula at which the normalisation is defined, GeV |
| `bin_width` | the width of a bin, GeV |
| `par_names` | the parameters' names, in the order `p0`, `p1`, ... |
| `outcome` | `pass`, or the first check the background-only fit fails: `no valid minimum`, `covariance incomplete`, `parameter at a bound`, `not falling` (`no fit`: too few populated bins) |
| `p<i>`, `p<i>_err` | parameter `i` of the background-only fit and its error, for a dataset of this luminosity |
| `cov<i><j>` | covariance of `p<i>` and `p<j>` |
| `outcome_sb` | the same verdict for the signal + background fit |
| `nsig_sb`, `nsig_sb_err` | the signal yield of the signal + background fit and its error, in events |
| `p<i>_sb` | parameter `i` of the background in the signal + background fit |

Both fits are to the expected spectrum, without statistical fluctuation. The two default functions, with
u = (x - 2000) / 1000:

| Function | Formula | `p0` | `p1` | `p2` |
| --- | --- | --- | --- | --- |
| `expo` | exp(p0 + p1 u) | log of the events per bin at 2000 GeV | slope per TeV | |
| `powexp` | exp(p0 + p2 u) (x / 2000)^p1 | log of the events per bin at 2000 GeV | power | slope per TeV |

## Settings

| Variable | Values | Default |
| --- | --- | --- |
| `FIT_DATASET` | `2024` (that era only), `run3` (2022 to 2024 summed, 2024 scaled up to cover 2025), `run3_scaled` (2024 alone at the Run 3 luminosity) | `2024` |
| `FIT_TOPOLOGY` | `resolved`, `boosted` | `resolved` |
| `FIT_DIR` | the histogram directory of step 2 | `20261015_skims_v2` |

- `config.py` holds every other setting.
- Outputs: `background_fit/results*/<channel>/<script>/`, one tree per dataset and topology.
- Every script takes `--help`.

## The chain, in order

Yields, simulation statistics and plots of the background spectrum.

```bash
python background_fit/1_spectrum.py
```

Fit every function of the menu to the simulated spectrum and compare them.

```bash
python background_fit/2_screen.py
```

Build the backgrounds the toys are thrown from (`truths` in the code): the simulation as it is, and smoothed.

```bash
python background_fit/3_truths.py
```

The smoothed backgrounds need the smoothing library (`smooth_lib`), which is not in this repository yet. Without
it, add `--truths raw` to steps 3, 4 and 6: the simulation as it is.

Throw toys from each background and fit signal + background over the full range, per function and signal mass.

```bash
python background_fit/4_toys.py --workers 8
```

Apply the pass rule to the toys of step 4 and write the verdict per function (`DECISION.md`).

```bash
python background_fit/5_decide.py
```

Fit the same toys in a window of +-k sigma around the signal mass instead of the full range.

```bash
python background_fit/6_windows.py --workers 8
```

A +-5 sigma window from 1200 GeV with 100,000 toys per cell, in a folder of its own:

```bash
python background_fit/6_windows.py --k 5 --lo 1200 --ntoys 100000 --tag _100k --workers 8
```

Compare the full-range fits of step 4 with the window fits of step 6 (`WINDOWS.md`).

```bash
python background_fit/7_compare_windows.py
```

## Figures and tables from the stored toys

These read the toys of steps 4 and 6. `--help` shows which folder each one reads.

One cell (one channel, background and signal mass): the fit to the expected spectrum, one toy with its fit, the
N_sig and pull distributions.

```bash
python background_fit/11_bias_explainer.py --channel ee --mass 2000
```

Pull mean, pull width and sigma_fit against the signal mass, per window width and function.

```bash
python background_fit/12_variants.py
```

The same quantities with one change at a time: the window width, the function, the background.

```bash
python background_fit/13_pieces.py
```

Fraction of fits passing, pull mean and pull width against the mass, at the 2024 and the Run 3 luminosity.

```bash
python background_fit/14_luminosity.py
```

The +-5 sigma window on four backgrounds (three DY samples and the RooKeysPdf).

```bash
python background_fit/15_truths_scan.py
```

One table of every window width, dataset and channel from the 100,000-toy runs (`MATRIX_100k.md`).

```bash
python background_fit/16_matrix.py
```

What starting the fit at 800, 1000 or 1200 GeV does to the fit and to the bias.

```bash
python background_fit/17_fit_floor.py
```

## The backgrounds themselves

Events above a mass and per mass slice: the simulation against the RooKeysPdf.

```bash
python background_fit/18_cumulative.py
```

The exponential of a quadratic: `fit` fits it to the summed background, `matrix` tabulates its window toys.

```bash
python background_fit/19_expo2.py fit
```

The simulation with every stored smoothed background drawn over it.

```bash
python background_fit/20_truths_overlay.py
```

The background models side by side (`models`), the bias of one function on each (`bias`), the RooKeysPdf with
one exponential tail (`sumtail`).

```bash
python background_fit/22_deck_models.py models
```

Which process causes a bump in the pull mean: one process group at a time replaced by a smooth curve.

```bash
FIT_DATASET=run3 python background_fit/23_bump.py
```

## Boosted SR

Steps 1 to 7 run on the boosted SR with `FIT_TOPOLOGY=boosted`.

Every function fitted to the boosted background from 800, 1000 and 1200 GeV (`fit`); one table of the boosted
window toys (`matrix`).

```bash
FIT_TOPOLOGY=boosted FIT_DATASET=run3 python background_fit/21_boosted.py fit
```

Figures and numbers of the boosted study: `models`, `grids`, `shapes`, and `signal` (the signal mean and width
lines, written to `signal_map_2024.json`).

```bash
FIT_TOPOLOGY=boosted FIT_DATASET=run3 python background_fit/24_boosted_deck.py models
```

Why fits fail: each toy fitted background-only, signal + background, and again from other starting points.

```bash
FIT_TOPOLOGY=boosted FIT_DATASET=run3 FIT_SIGNAL_MAP=background_fit/signal_map_2024.json python background_fit/25_fail_study.py
```

## Flavour CR fit (tried, not used)

Toys with a simultaneous fit of the SR and the e-mu flavour CR.

```bash
python background_fit/8_flavor_cr.py --workers 8
```

Compare that fit with the SR-only fit of step 4 (`FLAVOR_CR.md`).

```bash
python background_fit/9_compare_fcr.py
```
