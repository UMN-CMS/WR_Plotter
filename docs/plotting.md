# Plotting

Every command runs from `WR_Plotter/` with the analyzer's environment and takes `--era` and `--dir`,
the era and the `--dir` tag of the analyzer production under
`WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/`.

- `--era`: one era, or a combination from `data/lumi.yaml` (`2022`, `2023`, `2024`, `Run3`). A
  combination reads each era's files from `rootfiles/Run3/<year>/<era>/<dir>` and adds the
  histograms, so every era needs the same `--dir`; its luminosity is the sum. A sample or histogram
  missing from some eras is reported. Output under `plots/Run3/<year>/<year>/` or `plots/Run3/2022-2024/Run3/`.
- `--local-plots`: write under `WR_Plotter/plots/`. Without it, plots go to
  `/eos/user/<initial>/<username>/<Run>/<Year>/<Era>/<dir>/...` when `/eos` is mounted, else to
  `WR_Plotter/outputs/`. `EOSUSER` (the CERN username if not `$USER`), `EOSUSER_PATH`
  (`<initial>/<username>`), `EOS_BASE` (the whole root), `EOS_ENDPOINT` (default `eosuser.cern.ch`),
  `FORCE_EOS=1`, `FORCE_LOCAL=1` override (`wrplotter/paths.py`). Tables go to `WR_Plotter/tables/`.
- `-r`, `-v`, `-s`: regions, variables, signal samples, repeated or comma-separated.
  `--list-eras`, `--list-regions`, `--list-variables` print the choices.
- Data streams: both data files fill every region, so each region takes one: EGamma for the ee
  regions and `wr_emu_boosted_flavor_cr`, Muon for the μμ regions, `wr_resolved_flavor_cr` and
  `wr_mue_boosted_flavor_cr`.
- Blinding is the analyzer's. Signal-region data appears only from files made with `--unblind` (Run 2
  only). Files without a `blinding` record in their manifest never show signal-region data, and the
  cutflows show no data column for them.
- Systematic band: every variation the files carry (`data/systematics.yaml`); `scale` is the
  envelope of the μR, μF and μRμF variations; `pdf` and `alphas` are formed from the stored members
  at the plot binning. Variations are summed era by era; an era without a variation adds its nominal
  and is reported. `--no-syst` forces a statistical band.

```bash
cd WR_Plotter
../.venv/bin/python bin/make_stackplots.py          --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/make_cutflows.py            --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/make_signal_cutflows.py     --era RunIII2024Summer24 --dir <tag>
../.venv/bin/python bin/make_systematics_tables.py  --era RunIII2024Summer24 --dir <tag> [--per-bin]
../.venv/bin/python bin/compare_dy.py               --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/plot_event_weights.py       --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/plot_dy_zpt_reweight.py     --era RunIII2024Summer24 --dir <tag> --dy lo_ht,lo_ht_zpt --local-plots
../.venv/bin/python bin/emu_transfer_factor.py      --era Run3 --dir <tag> --local-plots
../.venv/bin/python bin/make_signal_plots.py        --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/compare_signal_eras.py      --era RunIII2024Summer24 --era 2022 --era 2023 --dir <tag> --local-plots
../.venv/bin/python bin/signal_event_plots.py       fill --era RunIII2024Summer24 --point WR4000_N2100 --flavour El --out <hists>
../.venv/bin/python bin/signal_event_plots.py       plot --out <hists> --era RunIII2024Summer24 --dir <tag> --local-plots
```

| Command | Makes | Output |
| --- | --- | --- |
| `make_stackplots.py` | data over the stacked backgrounds for every region and variable, signal overlays in the signal regions, the systematic band; uniform bins (`--run2-binning`: the Run 2 m(W_R) fit bins) | `plots/<...>/<dir>/stack_dy_<tag>/<region>/<variable>_<region>.pdf` (`stack_dy_<tag>_run2_binning/`) |
| `make_cutflows.py` | cutflow tables (LaTeX, text) and plots per topology and channel, weighted and raw, with efficiencies | `tables/<...>/<dir>/cutflows_dy_<tag>/`, `plots/.../cutflows_dy_<tag>/` |
| `make_signal_cutflows.py` | yields and efficiencies per (m_WR, m_N) point | `tables/<...>/<dir>/signal_outputs/signal_cutflows/` |
| `make_signal_plots.py` | one era's signal: A × ε vs m_N and over the grid, signal in the CRs, σ and k, mass shapes and resolution, kinematics, cutflow plots | `plots/<...>/<dir>/signal_outputs/signal_plots/`, `tables/.../signal_plots/` |
| `plot_signal_theory_uncertainties.py` | scale, PDF, α_S and PS changes of the signal-region yields vs m_WR, from a `--systs scale pdf ps` signal set (`--theory-dir`) | `plots/<...>/<dir>/signal_outputs/signal_theory/` |
| `make_systematics_tables.py` | per region, at the fit binning: the up/down change of each process's yield under every source, the largest per-bin change, a dash where a source does not move a process, the absent sources named; `--per-bin` adds bin tables | `tables/<...>/<dir>/systematics_dy_<tag>/systematics_<variable>_<region>.{txt,md,tex}`, `systematics.{txt,md,csv}` |
| `compare_dy.py` | DY samples alone and inside the total background: the named comparisons of `data/dy_comparisons.yaml` or a `--dy` list | `plots/.../dy_compare/[<comparison>/]` (`dy_compare_run2_binning/`) |
| `plot_event_weights.py` | the MC event-weight distributions per region, total and each factor | `plots/.../event_weights/` |
| `plot_dy_zpt_reweight.py` | the DY Z-pT payload and, with `--dir`, the reweighted over the unreweighted DY in the DY CRs' pT(ll) | `plots/.../dy_zpt_reweight/` |
| `emu_transfer_factor.py` | the eμ transfer factor SR / flavour CR on tt̄ and its closure below 800 GeV | `plots/<...>/<dir>/emu_transfer_factor_*.pdf`, `tables/<...>/<dir>/emu_transfer_factor_*.tex` |
| `compare_signal_eras.py` | signal efficiencies and shapes across eras | `plots/<...>/<ref dir>/signal_outputs/signal_compare/` |
| `signal_event_plots.py` | event-level signal plots that need generator particles or objects before a cut (the AN-19-083 Figs. 3-5, 10-12, 83-87 analogues): `fill` runs one mass point and flavour through the analyzer's corrections and object selection, `plot` draws every filled point; needs the analyzer checkout (`wrcoffea`, its configs and filesets) | `plots/<...>/<dir>/signal_outputs/event_level/` |

## Stack plots

- Signal regions overlay the era's three reference points (m(W_R) = 2, 4 and 6 TeV,
  `cutflow_signals_<topology>` in `data/lumi.yaml`); `-s signal_WR4000_N2100,...` overrides.
- `--run2-binning`: the m(W_R) observables in the Run 2 fit bins (AN-19-083), resolved m(ℓℓjj) 800,
  1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000 GeV, boosted m(ℓJ) 800, 1000, 1200, 1500,
  1800, 8000 GeV, drawn as events per bin; `--per-gev` divides by the bin width.
- `--plot-config <yaml>`: another plot-settings file. `--extra-label`: a line under the region text.
  `--dry-run` lists the plots.

## Drell-Yan samples

Every DY sample is a file in `--dir`: the default (`lo_ht`, LO HT-binned, not reweighted) is
`WRAnalyzer_DYJets.root`; any other `WRAnalyzer_DYJets_<TAG>.root`, from `run_analysis.py <era>
DYJets --dy <TAG>` (`lo_inc`, `nlo_inc`, `nlo_mll`, `nlo_njet`, `nlo_ptll`) or `--dy-reweight zpt` /
`zpt+reshape` (`lo_ht_zpt`, `lo_ht_zpt_reshape`). `--dy TAG` picks one (repeat or comma-separate for
one plot set each); `--dy TAG=DIR` reads it from another directory. Sets made before 2026-10-12
kept each variant as `WRAnalyzer_DYJets.root` in `<dir>_<TAG>_dy`, read when the file is not in
`--dir` (the old tags `lo_ht_zptonly`, `lo_ht_reweighted` still work), and hold the reweighted DY in
`<dir>` itself: `--dy lo_ht_zpt_reshape=<dir>`. A DY file whose reweighting (from its event-weight
histograms) does not match its tag is refused. `nlo_ptll` is not a DY prediction (unstitched slices).

`compare_dy.py` comparisons (`--comparison`, default every one; the first sample is the reference):

| Comparison | Samples | Question |
|---|---|---|
| `all` | `lo_ht`, `lo_ht_zpt_reshape`, `lo_inc`, `nlo_inc`, `nlo_mll`, `nlo_njet` | every DY prediction together |
| `lo` | `lo_ht`, `lo_inc`, `lo_ht_zpt_reshape` | HT slicing vs inclusive LO; the Z-pT reweight and reshape |
| `nlo` | `nlo_inc`, `nlo_mll`, `nlo_njet` | the NLO samples against the inclusive one |
| `lo_vs_nlo_inclusive` | `lo_inc`, `nlo_inc` | the generator difference alone |
| `lo_ht_vs_nlo` | `lo_ht`, `nlo_inc`, `nlo_mll`, `nlo_njet` | the default DY against the NLO samples |

Outputs under `dy_compare/[<comparison>/]`: `abs/<region>/` (each sample with the ratio to the
reference), `shape/` (unit yield), `stat/` (relative statistical uncertainty and N_eff per bin),
`total/` (the total background with each sample; data and Data/Bkg. in control regions),
`total_scaled/` (control regions, each prediction scaled to the data yield). `--plots core` (default)
draws the fit observable, m_ll, pT(ll), the reshape jet pT, the subleading jet, m_jj and the recoil
pT in the DY CRs and the fit observable and m_ll in the SRs; `--plots all` every configured pair.

`plot_dy_zpt_reweight.py` outputs: `qcd_ratio.pdf` (NLO/LO vs generator Z pT with the scale, PDF and
α_S variations and the Run 2 ratio), `qcd_ratio_full_range.pdf`, `ew_correction.pdf`, `weight.pdf`
(QCD × EW), `spectra.pdf` (the LO and NLO spectra), `applied/<region>.pdf` (with `--dir`).

## Cutflows

`make_cutflows.py`: yields {cumulative, one cut alone} × {weighted, unweighted} and efficiencies
(running and relative) per topology and channel; `--topology`, `--channel`, `--signal WR4000_N2100`,
`--no-plots`. Outputs `cutflow_<variant>_<topology>_<channel>.tex`, `cutflows.txt`, `preview.pdf`
(with pdflatex) and `plots/.../cutflows_dy_<tag>/<topology>_<channel>/<variant>.pdf`. The cut sequence
and labels come from the files and `data/cutflows.yaml`; blinded rows are marked.

`make_signal_cutflows.py`: every `WRAnalyzer_signal_WR<m>_N<m>.root` in `--dir`, one topology per
run (`--boosted` for boosted). `--mode fixed-wr` (one table per WR mass, a column per N), `fixed-n`,
`wr-compare` (one column per WR mass at its highest N), `channel-compare` (ee and μμ side by side);
`--format terminal|latex`; `--efficiency running|relative`; `--onecut`. Every mass point must have the
same cut sequence.

## Signal outputs

Every signal output of a production sits in `<dir>/signal_outputs/<part>/`:

| Part | Made by |
|---|---|
| `signal_plots/` | `make_signal_plots.py`: `efficiency/<channel>_WR<m>.pdf` (A × ε vs m_N: resolved, boosted, sum), `efficiency_map/`, `cr_fraction/` (signal in the DY and flavour CRs over σ × L), `xsec/`, `kfactor/`, `mass/`, `resolution/` (median / m_WR and σ_eff / median), `kinematics/<region>/`, `cutflow/`; tables `efficiency.csv`, `latex/signal_eff_<channel>_WR<m>.tex` |
| `signal_theory/` | `plot_signal_theory_uncertainties.py`: `<topology>_<channel>_mN100.pdf` and `_mNhalf.pdf` (the lightest m_N at each m_WR, and m_N ≈ m_WR / 2), `theory_uncertainties.csv` |
| `event_level/` | `signal_event_plots.py plot --era <era> --dir <dir>` |
| `signal_cutflows/` | `make_signal_cutflows.py` |
| `signal_compare/<era>_vs_<era>.../` | `compare_signal_eras.py`: `efficiency/`, `resolution/`, `mass/<region>/`, `cutflow/`, `kinematics/`; tables `efficiency.csv`, `scale_factors.csv`, `cutflow.csv`, `summary.txt`, `efficiency_tables.txt`, `latex/` |

`compare_signal_eras.py`: each `--era ERA[=DIR]` is an era or combination read from `DIR`, else from
`--dir`; the first is the reference of every ratio. Efficiency = signal-region yield over σ × L of the
channel's flavour, with σ from the analyzer's `data/configs/<...>/<Era>_signal.json`
(`--analyzer-root`, or `--fileset ERA=PATH`); a flavour-mixed `WRtoNLtoLLJJ` sample counts half per
channel; the summary warns (`norm_check`) when `no_cuts` / σL differs from skim × filters by more than
10 %. Each efficiency is split into `no_cuts` × selection × scale factors. `--points` (default the
`cutflow_signals` of `data/lumi.yaml`), `--label`, `--name`, `--no-kinematics`, `--no-pdf`.

## Event weights and transfer factor

`plot_event_weights.py`: per region `<region>.pdf` (log₁₀|w| per MC group, w > 0 solid, w < 0 dashed,
N, N_eff and the negative fraction in the legend), `components/<region>/<component>.pdf` (each
factor, unit-normalized) and `tables/.../event_weights/<region>.txt`. With several `--dy` the DY
samples are also overlaid under `event_weights/dy_samples/`.

`emu_transfer_factor.py`: tt+tW only (`WRAnalyzer_tt_tW.root`), resolved and boosted (`--resolved`,
`--boosted`); plots `emu_transfer_factor_{resolved,boosted}[_tf]_<Era>.pdf`,
`emu_transfer_factor_boosted_{ejet,mujet}[_tf]_<Era>.pdf`, tables `.tex` and `preview.pdf`. The
study mode `--tf-study --input WRAnalyzer_TTbar.root` fits below and above 800 GeV on a production
made with `--study transfer-factor --split-groups tt_tW`; 800 GeV must stay a bin edge.

## Plot settings and style

| File | Used by |
|---|---|
| `default.yaml` | every era; 2022 and 2023 use it as is |
| `run3_subera.yaml` | Run3Summer22, Run3Summer22EE, Run3Summer23, Run3Summer23BPix |
| `RunIII2024Summer24.yaml` | RunIII2024Summer24, the 2024 and Run3 combinations |
| `RunIISummer20UL18.yaml` | RunIISummer20UL18 |

Every file except `default.yaml` starts with `inherit: default` and lists only the values that
differ. An era uses its own file if one exists, else the one named by `plot_settings` in
`data/lumi.yaml`, else `default.yaml`.

- Uniform bins: each axis starts at its selection cut on a bin edge and ends about where the
  simulation does; what lies beyond is folded into the last bin. η follows the acceptance: 0.3 bins
  over |η| < 2.4 for muons and jets, 0.5 over |η| < 2.5 for electrons. m(ℓℓjj) runs 800 to 8000 GeV in
  100 GeV bins with the y axis down to 1e-4.
- Stack axes: the top is 10× the highest bin, raised only as far as the CMS label, region text and
  legend need; the bottom is a decade below the smallest bin of the stack or the data and below the
  smallest signal peak, never below 0.1 with data unless the configured bottom is lower.
- Fonts: figures are 10 inches wide with no text under 20 pt (axis titles 26 pt, tick labels 22 pt;
  `wrplotter/plotting_helpers.py` FONT_SIZE_*), 5.7 pt at 0.45\textwidth in the AN.
- CMS style: the CMS label inside the frame, top left, "Work in progress" and "Simulation" where no
  data is drawn; luminosity and energy above the frame, right. Data first in the legend with Garwood
  errors and horizontal bars on variable-width bins; the band is "Sim. stat. (+ syst.) unc.".
  Backgrounds and signal overlays use the CMS (Petroff) palette; limit plots use the CAT band colours,
  #228b22 (68 %) and #ffcc00 (95 %). The event-weight (13 pt) and cutflow (14 pt) legends need
  enlarging before a plot goes into a note at half a text width.

`scripts/plot_jet_veto_maps.py` draws the analyzer's jet veto maps into `plots/jet_veto_maps/<era>.pdf`.

Tests: `cd WR_Plotter && ../.venv/bin/python -m pytest tests -q` (offline, under a minute).
