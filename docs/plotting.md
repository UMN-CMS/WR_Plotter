# Plotting

Every command runs from `WR_Plotter/` with the analyzer's environment and reads one analyzer
production, `rootfiles/<Run>/<Year>/<Era>/<dir>/`. `--help` lists every option of a command; this
page lists the ones that change what is produced.

| Option | Meaning |
| --- | --- |
| `--era` | one era, or a combination from `data/lumi.yaml` (`2022`, `2023`, `2024`, `Run3`): the eras' histograms are added and the luminosities summed; every era is read from its own `rootfiles/.../<dir>`, so all need the same `--dir`. A sample or histogram missing from some eras is reported. |
| `--dir` | the analyzer's `--dir` tag |
| `--local-plots` | plots under `plots/`; without it they go to EOS when `/eos` is mounted, else to `outputs/` (`wrplotter/paths.py` reads `EOSUSER`, `EOS_BASE`, `FORCE_LOCAL`). Tables always go to `tables/`. |
| `-r`, `-v`, `-s` | regions, variables, signal samples; repeat or comma-separate. `--list-regions`, `--list-variables`, `--list-eras` print the choices. |
| `--dy TAG[=DIR]` | the Drell-Yan sample, next section |

Outputs mirror the input path: `plots/<Run>/<Year>/<Era>/<dir>/...` and `tables/<Run>/<Year>/<Era>/<dir>/...`;
a combination writes to `plots/Run3/<year>/<year>/` or `plots/Run3/2022-2024/Run3/`.

- Each region takes one data stream: EGamma for the ee regions and `wr_emu_boosted_flavor_cr`, Muon
  for the μμ regions, `wr_resolved_flavor_cr` and `wr_mue_boosted_flavor_cr`.
- Blinding is the analyzer's. Signal-region data appears only from files made with its `--unblind`
  (Run 2 only); files without a `blinding` record in their manifest show no signal-region data.
- The systematic band holds every variation the files carry (`data/systematics.yaml`): `scale` is the
  envelope of the μR, μF and μRμF variations, `pdf` and `alphas` are formed from the stored members at
  the plot binning. A combination adds variations era by era; an era without one adds its nominal
  and is reported.

## Drell-Yan samples

| Tag | File in `--dir` | Made by |
| --- | --- | --- |
| `lo_ht` (default: LO HT-binned, not reweighted) | `WRAnalyzer_DYJets.root` | `run_analysis.py <era> DYJets` |
| `lo_ht_zpt`, `lo_ht_zpt_reshape` | `WRAnalyzer_DYJets_<tag>.root` | `--dy-reweight zpt`, `zpt+reshape` |
| `lo_inc`, `nlo_inc`, `nlo_mll`, `nlo_njet`, `nlo_ptll` | `WRAnalyzer_DYJets_<tag>.root` | `--dy <tag>` |

`--dy TAG` picks one; several tags give one plot set each; `TAG=DIR` reads the file from another
directory. Sets made before 2026-10-12 kept each variant as `WRAnalyzer_DYJets.root` in
`<dir>_<TAG>_dy/`, which is still read when the file is not in `--dir`. A file whose reweighting does
not match its tag is refused. `nlo_ptll` is unstitched slices, not a prediction.

## Commands

### Stack plots

```bash
../.venv/bin/python bin/make_stackplots.py --era RunIII2024Summer24 --dir <tag> --local-plots
```

Data over the stacked backgrounds for every region and variable, signal overlays in the signal
regions, the systematic band. Output: `plots/<...>/<dir>/stack_dy_<tag>/<region>/<variable>_<region>.pdf`.

- `--run2-binning`: the m(W_R) observables in the Run 2 fit bins, into `stack_dy_<tag>_run2_binning/`;
  `--per-gev` divides by the bin width.
- `-s`: the overlaid points; default the era's 2, 4 and 6 TeV reference points
  (`cutflow_signals_<topology>` in `data/lumi.yaml`).
- `--no-syst`: statistical band only. `--plot-config`: another plot-settings file. `--extra-label`: a
  line under the region text. `--dry-run` lists the plots.

### Cutflows

```bash
../.venv/bin/python bin/make_cutflows.py --era RunIII2024Summer24 --dir <tag> --local-plots
```

Per topology and channel: cumulative and one-cut yields, weighted and raw, with running and relative
efficiencies. Output: `tables/<...>/<dir>/cutflows_dy_<tag>/cutflow_<variant>_<topology>_<channel>.tex`,
`cutflows.txt` and `preview.pdf`; plots `plots/<...>/<dir>/cutflows_dy_<tag>/<topology>_<channel>/<variant>.pdf`.

- `--topology`, `--channel`, `-s` (the signal columns). `--no-plots`, `--no-tables`, `--no-pdf`.
- Row labels and the data stream per channel: `data/cutflows.yaml`. Blinded rows are marked.

### Signal cutflows

```bash
../.venv/bin/python bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir <tag>
```

Yields and efficiencies per (m_WR, m_N) point, from every `WRAnalyzer_signal_WR<m>_N<m>.root` in
`--dir`; resolved, or boosted with `--boosted`. Output:
`tables/<...>/<dir>/signal_outputs/signal_cutflows/signal_<table>_<topology>_<channel>_<variant>.tex`,
`signal_cutflows.txt`, `preview.pdf`.

- `--mode fixed-wr | fixed-n | wr-compare | channel-compare`, `--wr-mass`, `--efficiency running|relative`,
  `--onecut`. `--channel` prints one channel to the terminal instead (`--format latex` to write it).

### Signal plots

```bash
../.venv/bin/python bin/make_signal_plots.py --era RunIII2024Summer24 --dir <tag> --local-plots
```

One era's signal: A × ε against m_N and over the grid, the signal in the control regions, σ and k,
mass shapes and resolution, kinematics, cutflow plots. Output:
`plots/<...>/<dir>/signal_outputs/signal_plots/{efficiency,efficiency_map,cr_fraction,xsec,kfactor,mass,resolution,kinematics,cutflow}/`,
tables `tables/<...>/<dir>/signal_outputs/signal_plots/efficiency.csv` and `latex/`.

- `--points` (default the era's reference points), `--max-shapes`, `--no-kinematics`, `--no-pdf`.
- Cross sections come from the analyzer's `data/configs/<...>/<Era>_signal.json` (`--analyzer-root`,
  or `--fileset` for a run made with another fileset).

### Signal across eras

```bash
../.venv/bin/python bin/compare_signal_eras.py --era RunIII2024Summer24 --era 2022 --era 2023 --dir <tag> --local-plots
```

Signal efficiencies and shapes across eras; each `--era ERA[=DIR]` is one set, the first the
reference of every ratio. Output:
`plots/<...>/<ref dir>/signal_outputs/signal_compare/<era>_vs_<era>.../{efficiency,resolution,mass,cutflow,kinematics}/`,
tables `efficiency.csv`, `scale_factors.csv`, `cutflow.csv`, `summary.txt`, `efficiency_tables.txt`, `latex/`.

- `--points`, `--label` (one per set), `--name`, `-o`, `--no-kinematics`, `--no-pdf`.
- Cross sections and skim counts come from the analyzer checkout (`--analyzer-root`, `--fileset ERA=PATH`).

### Signal theory uncertainties

```bash
../.venv/bin/python bin/plot_signal_theory_uncertainties.py --era RunIII2024Summer24 --dir <tag> --theory-dir <systs tag> --local-plots
```

Scale, PDF, α_S and parton-shower changes of the signal-region yields against m_WR, from a signal set
made with `--systs scale pdf ps` (`--theory-dir`, default `--dir`). Output:
`plots/<...>/<dir>/signal_outputs/signal_theory/<topology>_<channel>_mN100.pdf`, `_mNhalf.pdf`, `theory_uncertainties.csv`.

### Signal event-level plots

```bash
../.venv/bin/python bin/signal_event_plots.py fill --era RunIII2024Summer24 --point WR4000_N2100 --flavour El --out <hists>
../.venv/bin/python bin/signal_event_plots.py plot --out <hists> --era RunIII2024Summer24 --dir <tag> --local-plots
```

The signal plots that need generator particles or objects before a cut (the AN-19-083 Figs. 3-5,
10-12 and 83-87 analogues). `fill` runs one mass point and flavour through the analyzer's corrections
and object selection into `<hists>`; `plot` draws every filled point. Needs the analyzer checkout.
Output: `plots/<...>/<dir>/signal_outputs/event_level/`.

### Systematics tables

```bash
../.venv/bin/python bin/make_systematics_tables.py --era RunIII2024Summer24 --dir <tag>
```

Per region, at the plot binning (the fit binning for the mass observables): the change of each
process's yield under every variation the files carry, the largest per-bin change, the absent
sources named. Output: `tables/<...>/<dir>/systematics_dy_<tag>/systematics_<variable>_<region>.{txt,md,tex}`
and `systematics.{txt,md,csv}` for all regions.

- `--per-bin` adds the per-bin tables. `-v` adds variables beyond the mass observable; `-s` the
  tabulated signals. `--dry-run` lists the tables.

### Drell-Yan comparisons

```bash
../.venv/bin/python bin/compare_dy.py --era RunIII2024Summer24 --dir <tag> --local-plots
```

The DY samples alone and inside the total background, the first sample of each comparison the
reference. Output: `plots/<...>/<dir>/dy_compare/[<comparison>/]{abs,shape,stat,total,total_scaled}/<region>/<variable>_<region>.pdf`
(`dy_compare_run2_binning/` with `--run2-binning`).

| Comparison | Samples | Question |
| --- | --- | --- |
| `all` | `lo_ht`, `lo_ht_zpt_reshape`, `lo_inc`, `nlo_inc`, `nlo_mll`, `nlo_njet` | every DY prediction together |
| `lo` | `lo_ht`, `lo_inc`, `lo_ht_zpt_reshape` | HT slicing against inclusive LO; the Z-pT reweight and reshape |
| `nlo` | `nlo_inc`, `nlo_mll`, `nlo_njet` | the NLO samples against the inclusive one |
| `lo_vs_nlo_inclusive` | `lo_inc`, `nlo_inc` | the generator difference alone |
| `lo_ht_vs_nlo` | `lo_ht`, `nlo_inc`, `nlo_mll`, `nlo_njet` | the default DY against the NLO samples |

- `--comparison` picks named comparisons (default every one, `data/dy_comparisons.yaml`); a `--dy` list
  compares those samples instead.
- `abs`: each sample with the ratio to the reference; `shape`: unit yield; `stat`: relative statistical
  uncertainty and N_eff per bin; `total`: the total background with each sample, with data in the
  control regions; `total_scaled`: control regions, each prediction scaled to the data yield.
- `--plots core` (default: the fit observables and the variables the DY corrections act on, in the DY
  CRs and SRs) or `all`.

### Drell-Yan Z-pT reweight

```bash
../.venv/bin/python bin/plot_dy_zpt_reweight.py --era RunIII2024Summer24 --dir <tag> --dy lo_ht,lo_ht_zpt --local-plots
```

The era's Z-pT payload (`config.yaml` `dy_zpt_reweight`; `--payload` another, `--reference` the
Run 2 overlay) and, with `--dir`, the reweighted over the unreweighted DY in the DY control
regions' pT(ll) (`--dy before,after`). Output: `plots/<...>/[<dir>/]dy_zpt_reweight/{qcd_ratio,qcd_ratio_full_range,ew_correction,weight,spectra}.pdf`
and `applied/<region>.pdf`.

### Drell-Yan jet-pT reshape

```bash
../.venv/bin/python bin/plot_dy_reshape.py --era RunIII2024Summer24 --dir <tag> --local-plots
```

The era's reshape payload (`config.yaml` `dy_reshape`; `--payload` another, `--tag` names its
outputs): (data − non-DY MC) / DY MC against the leading AK4 jet pT in the resolved DY control
regions and the AK8 jet pT in the boosted ones, ee and μμ combined, with the channel difference as a
band. `--after DIR` adds the DY CR2 closure stacks of a `--study dy-cr-split` production before and
after the reshape derived in CR1. Output: `plots/<...>/[<dir>/]dy_reshape/ratio_{resolved,boosted}[_<tag>].pdf`
and `cr2_closure/<region>/<variable>_<region>_{before,after}.pdf`.

### Event weights

```bash
../.venv/bin/python bin/plot_event_weights.py --era RunIII2024Summer24 --dir <tag> --local-plots
```

The MC event-weight distributions per region: log₁₀|w| per group (w > 0 solid, w < 0 dashed; N,
N_eff and the negative fraction in the legend) and each factor of the weight. Output:
`plots/<...>/<dir>/event_weights/<region>.pdf`, `components/<region>/<component>.pdf`, `dy_samples/`
with several `--dy`; `tables/<...>/<dir>/event_weights/<region>.txt`.

- `--no-components`: the total weight only.

### eμ transfer factor

```bash
../.venv/bin/python bin/emu_transfer_factor.py --era Run3 --dir <tag> --local-plots
```

The eμ transfer factor, signal region over flavour control region on tt̄+tW
(`WRAnalyzer_tt_tW.root`), and its closure; resolved and boosted (`--resolved`, `--boosted`).
Output: `plots/<...>/<dir>/emu_transfer_factor_{resolved,boosted}[_tf]_<Era>.pdf`,
`emu_transfer_factor_boosted_{ejet,mujet}[_tf]_<Era>.pdf`; tables `.tex` and `preview.pdf`.

- `--tf-study --input WRAnalyzer_TTbar.root`: the `_nomass` regions of a production made with
  `--study transfer-factor --split-groups tt_tW`, fitted below and above 800 GeV, which must stay a
  bin edge. `--rebin`, `--variable-bins`.

### Jet veto maps

```bash
../.venv/bin/python scripts/plot_jet_veto_maps.py
```

The analyzer's jet veto maps, `plots/jet_veto_maps/<era>.pdf`.

## Plot settings and style

| File | Used by |
| --- | --- |
| `default.yaml` | every era; 2022 and 2023 use it as is |
| `run3_subera.yaml` | Run3Summer22, Run3Summer22EE, Run3Summer23, Run3Summer23BPix |
| `RunIII2024Summer24.yaml` | RunIII2024Summer24, the 2024 and Run3 combinations |
| `RunIISummer20UL18.yaml` | RunIISummer20UL18 |

- Every file except `default.yaml` starts with `inherit: default` and lists only what differs. An era
  uses its own file, else the one `plot_settings` names in `data/lumi.yaml`, else `default.yaml`.
- Uniform bins start at the selection cut and end about where the simulation does; what lies beyond
  is folded into the last bin. m(ℓℓjj) runs 800 to 8000 GeV in 100 GeV bins. The Run 2 fit bins of
  `--run2-binning`: resolved m(ℓℓjj) 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000 GeV;
  boosted m(ℓJ) 800, 1000, 1200, 1500, 1800, 8000 GeV, drawn as events per bin.
- Stack y axis: the top is 10× the highest bin, raised only as far as the labels need; the bottom is a
  decade below the smallest stacked or data bin and below the smallest signal peak.
- Figures are 10 inches wide with no text under 20 pt (`wrplotter/plotting_helpers.py`).
- CMS style: the label inside the frame, top left ("Work in progress"; "Simulation" where no data is
  drawn), luminosity and energy above the frame on the right; data first in the legend with Garwood
  errors; the band "Sim. stat. (+ syst.) unc."; the CMS (Petroff) palette, the CAT band colours for
  limits.

Tests: `cd WR_Plotter && ../.venv/bin/python -m pytest tests -q` (offline, under a minute).
