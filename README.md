# WR Plotter

Plotting tools for the WR analysis. Takes ROOT histogram files produced by the WrCoffea analyzer and generates stacked MC + data plots, DY comparison overlays, signal closure studies, and transfer factor plots.

[docs/plotting.md](docs/plotting.md) is the one-page overview: which command makes which plot or table, where the outputs go, and the systematics tables (`make_systematics_tables.py`), which this README does not cover.

## Table of Contents
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Making Stackplots](#making-stackplots)
  - [Output Locations](#output-locations)
  - [Filtering by Region](#filtering-by-region)
  - [Filtering by Variable](#filtering-by-variable)
  - [Signal Overlays](#signal-overlays)
  - [Eras, Years and Run 3](#eras-years-and-run-3)
  - [Drell-Yan Samples](#drell-yan-samples)
  - [Blinding](#blinding)
- [Command Reference](#command-reference)
- [Other Scripts](#other-scripts)
  - [make_cutflows.py](#make_cutflowspy)
  - [Signal outputs](#signal-outputs)
  - [make_signal_cutflows.py](#make_signal_cutflowspy)
  - [compare_signal_eras.py](#compare_signal_eraspy)
  - [make_signal_plots.py](#make_signal_plotspy)
  - [plot_signal_theory_uncertainties.py](#plot_signal_theory_uncertaintiespy)
  - [compare_dy.py](#compare_dypy)
  - [plot_dy_zpt_reweight.py](#plot_dy_zpt_reweightpy)
  - [plot_event_weights.py](#plot_event_weightspy)
  - [emu_transfer_factor.py](#emu_transfer_factorpy)
- [Removed Options](#removed-options)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
  - [EOS / CERNBox Setup](#eos--cernbox-setup)
  - [Plot Settings YAML](#plot-settings-yaml)
- [Getting Started](#getting-started)

---

## Quick Start

### Prerequisites
Run at least `RunIII2024Summer24` in the WrCoffea analyzer. An analysis set is one `--dir` of the
analyzer; the following files should exist in it:
```
rootfiles/Run3/2024/RunIII2024Summer24/<dir>/WRAnalyzer_DYJets.root
rootfiles/Run3/2024/RunIII2024Summer24/<dir>/WRAnalyzer_tt_tW.root
rootfiles/Run3/2024/RunIII2024Summer24/<dir>/WRAnalyzer_Nonprompt.root
rootfiles/Run3/2024/RunIII2024Summer24/<dir>/WRAnalyzer_Other.root
rootfiles/Run3/2024/RunIII2024Summer24/<dir>/WRAnalyzer_EGamma.root
rootfiles/Run3/2024/RunIII2024Summer24/<dir>/WRAnalyzer_Muon.root
rootfiles/Run3/2024/RunIII2024Summer24/<dir>/WRAnalyzer_signal_WR4000_N2100.root
rootfiles/Run3/2024/RunIII2024Summer24/<dir>/WRAnalyzer_signal_WR4000_N100.root
```
Each file has a manifest next to it, `WRAnalyzer_<sample>_systematics.json`. The plotter reads
the dataset names (to check the Drell-Yan sample) and the blinding record from it.

---

### Making Stackplots

Plot all variables in all analysis regions:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --local-plots
```
This produces resolved and boosted plots for all control and signal regions. Signal regions
show the background stack with a signal overlay; their data is blinded by the analyzer (see
[Blinding](#blinding)).

List available eras:
```bash
python3 bin/make_stackplots.py --list-eras
```

Pass the same `--dir` that was given to the analyzer. Without `--dir` the files are read from the
era directory itself.

The uncertainty band includes every systematic variation the files carry (`data/systematics.yaml`
lists the sources; `scale` is the envelope of the three QCD-scale variations). A nominal-only
production gives a statistical band only, without warnings. When only some samples carry variations,
the others contribute their nominal and are reported with a warning, since the band then leaves
their systematic uncertainties out. `--no-syst` forces a statistical band.

---

### Output Locations

With `--local-plots`, plots are saved locally, one set per Drell-Yan sample:
```
plots/<Run>/<Year>/<Era>/<dir>/stack_dy_<tag>/<region>/<variable>_<region>.pdf
```
Example:
```
plots/Run3/2024/RunIII2024Summer24/20261008_skims_v2/stack_dy_lo_ht/wr_mumu_resolved_dy_cr/mass_fourobject_wr_mumu_resolved_dy_cr.pdf
```

Without `--local-plots`, plots are uploaded to EOS/CERNBox. See [EOS / CERNBox Setup](#eos--cernbox-setup) for configuration.

Tables (cutflows, event-weight summaries) are always written locally, under
`tables/<Run>/<Year>/<Era>/<dir>/`.

---

### Filtering by Region

Plot a specific region with `-r`:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 -r resolved_dy_cr --local-plots
```
List available regions and the data stream each one uses:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 --list-regions
```
Both data files fill every region (the analyzer does no overlap removal), so each region takes its
data from the stream of its trigger only: EGamma for the ee regions and `wr_emu_boosted_flavor_cr`;
Muon for the μμ regions, `wr_resolved_flavor_cr` (the eμ trigger is the muon trigger) and
`wr_mue_boosted_flavor_cr`.

---

### Filtering by Variable

Plot a specific variable with `-v`:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 -r resolved_dy_cr -v pt_leading_jet --local-plots
```
Comma-separated lists work too:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 -v pt_leading_jet,pt_leading_lepton --local-plots
```
List available variables:
```bash
python3 bin/make_stackplots.py --list-variables
```

---

### Signal Overlays

In signal regions, a default signal sample is overlaid automatically (depends on era and region topology). Override with `-s`:
```bash
# Single signal sample
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 -s signal_WR6000_N3100 --local-plots

# Multiple signal samples (all overlaid on every SR)
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 -s signal_WR4000_N2100,signal_WR4000_N100 --local-plots
```
Samples that don't have histograms for a given region are silently skipped.

---

### Eras, Years and Run 3

`--era` (every script) takes a single era, a year, or all of Run 3:

| `--era` | Combines | Output under |
|---|---|---|
| `Run3Summer22`, `Run3Summer22EE`, `Run3Summer23`, `Run3Summer23BPix`, `RunIII2024Summer24` | one era | `plots/Run3/<year>/<era>/` |
| `2022`, `2023`, `2024` | the eras of that year | `plots/Run3/<year>/<year>/` |
| `Run3` | all five eras | `plots/Run3/2022-2024/Run3/` |

A combination reads each era's files from its own directory, `rootfiles/Run3/<year>/<era>/<dir>`, and adds the
histograms; MC is already normalized to each era's luminosity, and the combined luminosity is the sum of the eras'
(`data/lumi.yaml`, where a combination lists only its `sub_eras`). So every era needs the same `--dir`. A sample
missing in some of the eras is reported with a warning, because the combination would undercount it; so is a
histogram missing from some eras' files. Systematic variations are summed era by era: an era without a variation
adds its own nominal (the source does not affect it, or that era was produced without it, which is reported). A
combination without its own `data/plot_settings` file uses the one named by `plot_settings` in `data/lumi.yaml`.

All of them in one go:
```bash
for era in Run3Summer22 Run3Summer22EE Run3Summer23 Run3Summer23BPix RunIII2024Summer24 2022 2023 2024 Run3; do
    python3 bin/make_stackplots.py --era "$era" --dir 20261008_skims_v2 --local-plots
done
```

---

### Drell-Yan Samples

Every Drell-Yan sample has its own file in `--dir`. `--dy TAG` (`make_stackplots.py`, `compare_dy.py`,
`make_cutflows.py`, `plot_event_weights.py`) picks one by its tag (`data/dy_samples.yaml`):

- the default sample (`lo_ht`: LO HT-binned, not reweighted) is `WRAnalyzer_DYJets.root`;
- any other is `WRAnalyzer_DYJets_<tag>.root`: `lo_ht_zpt` and `lo_ht_zpt_reshape` (the LO HT-binned
  DY with the Z-pT reweight, and with the jet-pT reshape too: analyzer `--dy-reweight zpt` or
  `zpt+reshape`), `lo_inc`, `nlo_inc`, `nlo_mll`, `nlo_njet`, `nlo_ptll` (analyzer `--dy <tag>`);
- sets made before 2026-10-12 kept each sample as `WRAnalyzer_DYJets.root` in `<dir>_<tag>_dy`
  (`lo_ht_zpt` also as `lo_ht_zptonly` or `<dir>_noreshape`, `lo_ht_zpt_reshape` as
  `lo_ht_reweighted`; these old names still work as tags); a tag whose file is not in `--dir` is read
  from there;
- `--dy TAG=DIR` reads it from another directory.

Every other sample always comes from `--dir`. A DY file whose manifest lists the datasets of a
different DY sample is refused. Repeat `--dy` (or comma-separate) to make one set per sample:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --dy lo_ht,lo_inc --local-plots
```

---

### Blinding

Blinding is done by the analyzer, not the plotter. By default the analyzer never writes
signal-region histograms for data, and stops the data ee/μμ cutflows before the m(ll) > 200 GeV cut;
its manifest records this under `blinding`. Run 3 cannot be unblinded; Run 2 data made with the
analyzer's `--unblind` (manifest `blinding.applied: false`) is shown in the signal regions.

Data files made before analyzer-side blinding have no `blinding` record. Their signal-region data is
never read, and the cutflows show no data column for them (with a note) until the data is rerun.

---

## Command Reference

### make_stackplots.py

| Flag | Short | Arguments | Description |
|------|-------|-----------|-------------|
| `--era` | | `<era_name>` | **Required.** Era to process (e.g., `RunIII2024Summer24`) |
| `--dir` | | `<subdir>` | Analysis set: subdirectory under the input/output paths |
| `--dy` | | `<tag>[=<dir>]` | Drell-Yan sample(s); one plot set each. Repeat or comma-separate |
| `--region` | `-r` | `<name>` | Region(s) to plot. Repeat or comma-separate |
| `--variable` | `-v` | `<name>` | Variable(s) to plot. Repeat or comma-separate |
| `--signal` | `-s` | `<sample>` | Signal sample(s) to overlay on SR plots. Repeat or comma-separate |
| `--local-plots` | | | Save to `plots/` instead of EOS |
| `--plot-config` | `-c` | `<yaml>` | Custom plot settings YAML |
| `--run2-binning` | | | m(W_R) observables in the Run 2 fit bins instead of uniform bins; plots go to `stack_dy_<tag>_run2_binning/` |
| `--per-gev` | | | With `--run2-binning`, events per GeV instead of per bin (Run 2 drew events per bin) |
| `--no-syst` | | | Statistical band only, even if the files carry variations |
| `--extra-label` | | `<text>` | Extra line under the region label |
| `--dry-run` | | | List the plots that would be made and exit |
| `--list-eras` | | | List eras and exit |
| `--list-regions` | | | List regions for era and exit |
| `--list-variables` | | | List variables and exit |

### Examples
```bash
# All regions and variables
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --local-plots

# Single region, single variable
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 -r resolved_dy_cr -v pt_leading_jet --local-plots

# Multiple regions and variables
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 -r resolved_dy_cr,boosted_sr -v pt_leading_jet,mass_dilepton --local-plots

# One set per DY sample
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --dy lo_ht,lo_inc --local-plots

# Custom signal overlay
python3 bin/make_stackplots.py --era RunIII2024Summer24 --dir 20261008_skims_v2 -s signal_WR4000_N2100,signal_WR4000_N100 --local-plots
```

---

## Other Scripts

### make_cutflows.py

Cutflow tables and plots from the analyzer's cutflow histograms, for every topology (resolved,
boosted) and channel (ee, μμ, eμ):

- yields: {cumulative, one cut alone} × {weighted, unweighted};
- efficiencies (weighted only): running and relative-to-previous for cumulative, single-cut for
  one-cut.

```bash
python3 bin/make_cutflows.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --local-plots
python3 bin/make_cutflows.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --topology resolved --channel mumu --signal WR4000_N2100 --no-plots
```
Outputs:
```
tables/<Run>/<Year>/<Era>/<dir>/cutflows_dy_<tag>/cutflow_<variant>_<topology>_<channel>.tex
tables/<Run>/<Year>/<Era>/<dir>/cutflows_dy_<tag>/cutflows.txt     # every table as text
tables/<Run>/<Year>/<Era>/<dir>/cutflows_dy_<tag>/preview.pdf      # when pdflatex is available
plots/<Run>/<Year>/<Era>/<dir>/cutflows_dy_<tag>/<topology>_<channel>/<variant>.pdf
```
The cut sequence is read from the files. The data column comes from the channel's trigger stream
(`data/cutflows.yaml`; cut labels live there too); rows the analyzer blinded are marked as such.
Weighted MC rows use the base event weight without lepton scale factors, so the last row sits a few
percent above the signal-region histograms.

### Signal outputs

Every signal output of a production lives in one folder, `<plots>/<Run>/<Year>/<Era>/<dir>/signal_outputs/`, with
the tables under `tables/<Run>/<Year>/<Era>/<dir>/signal_outputs/` (tables never go to EOS):

| Subfolder | Made by |
|---|---|
| `signal_plots/` | `make_signal_plots.py` (efficiencies, cross sections, mass shapes, kinematics, cutflow plots) |
| `signal_theory/` | `plot_signal_theory_uncertainties.py` (scale, PDF, α_S, PS vs m_WR) |
| `event_level/` | WrCoffea `scripts/signal_event_plots.py plot --era <era> --dir <dir>` (generator-level and object-level plots) |
| `signal_cutflows/` | `make_signal_cutflows.py` (tables only) |
| `signal_compare/<era>_vs_<era>.../` | `compare_signal_eras.py` (under the reference era's production) |

### make_signal_cutflows.py

Signal-only cutflow tables: yields plus running and relative efficiencies (single-cut with `--onecut`)
for every `WRAnalyzer_signal_WR<m>_N<m>.root` in `--dir`. A signal-only directory works too, and a year
or `--era Run3` sums the eras. One topology per run (resolved, or `--boosted`):

| `--mode` | Tables |
|---|---|
| `fixed-wr` | one per WR mass, one column per N mass |
| `fixed-n` | one per N mass, one column per WR mass |
| `wr-compare` | one, one column per WR mass at its highest N |
| `channel-compare` | one per WR mass, ee and μμ side by side |

```bash
# LaTeX for ee and μμ (fixed-wr + wr-compare), with preview.pdf
python3 bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir 20261008_skims_v2
# One channel, printed
python3 bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --channel mumu --wr-mass 4000
python3 bin/make_signal_cutflows.py --era Run3 --dir 20261008_skims_v2 --channel ee --boosted --mode fixed-n --efficiency running
```
Outputs (LaTeX; `-o` overrides the directory):
```
tables/<Run>/<Year>/<Era>/<dir>/signal_outputs/signal_cutflows/signal_<table>_<topology>_<channel>_<variant>.tex
tables/<Run>/<Year>/<Era>/<dir>/signal_outputs/signal_cutflows/signal_cutflows.txt     # every table as text
tables/<Run>/<Year>/<Era>/<dir>/signal_outputs/signal_cutflows/preview.pdf             # when pdflatex is available
```
`--mode` and `--format` (`terminal`, `latex`) override the defaults above; `--efficiency running|relative`
keeps only that efficiency. As in `make_cutflows.py`, the cut sequence and its labels come from the files
and `data/cutflows.yaml`; every mass point must have the same sequence, or the script stops with an error.

### compare_signal_eras.py

Compare the signal samples of several eras (Run 2 vs Run 3, or the Run 3 years): acceptance × efficiency
and the shape of the signal-region mass. Each `--era ERA[=DIR]` is an era or a combination (`2022`,
`Run3`, ...) read from `DIR`, else from `--dir`; the first is the reference of every ratio. Only the signal
files and their manifests are read, so signal-only directories work:
```bash
python3 bin/compare_signal_eras.py --era RunIII2024Summer24 --era 2022 --era 2023 --dir 20261008_signals --local-plots
python3 bin/compare_signal_eras.py --era RunIISummer20UL18=<dir> --era 2022 --era 2023 --era 2024 --dir 20261008_signals --local-plots
```
- Efficiency = signal-region yield (every analyzer weight) / (σ × L of the channel's lepton flavour). σ is
  the cross section of the datasets in the file's manifest, read from the analyzer's
  `data/configs/<Run>/<Year>/<Era>/<Era>_signal.json` (`--analyzer-root` for another checkout, or
  `--fileset ERA=PATH` for the fileset a run used); a flavour-mixed `WRtoNLtoLLJJ` sample counts half for
  ee and half for μμ. σ must be the one the analyzer normalized with: when the weighted `no_cuts` / σL
  differs from skim × filters (unweighted counts) by more than 10 %, the summary warns (`norm_check`
  column), e.g. after the configs' cross sections change.
- Cutflow efficiencies use the base weight. The rows ee and μμ share (`no_cuts`, the jet cuts) are divided
  by σ × L of all flavours, the rest by that of the channel's flavour, so `no_cuts` shows what the skim and
  the event filters (jet veto map included) keep and the last row is the efficiency before lepton SFs.
- Each efficiency is split into `no_cuts` × selection × scale factors; `no_cuts` into skim × filters (with
  the analyzer's skimmed signal filesets), and the scale factors into reco/ID/isolation/trigger/pileup
  (the analyzer's `event_weights/<region>/components`).
- Mass peak: median, 16/84 % quantiles and σ_eff (half the narrowest interval holding 68.3 %).

Outputs under `<plots>/<Run>/<Year>/<Era>/<dir>/signal_outputs/signal_compare/<era>_vs_<era>.../` of the reference
(`-o DIR` writes plots and tables there instead):

| Folder / file | Content |
|---|---|
| `efficiency/<topology>_<channel>_WR<m>.pdf` | A × ε vs m_N per m_WR, resolved, boosted and `combined` (their sum), ratio to the reference |
| `resolution/<topology>_<channel>_WR<m>.pdf` | median / m_WR and σ_eff / median of the SR mass vs m_N |
| `mass/<region>/<point>.pdf` | SR mass of every point common to all sets, unit area |
| `cutflow/<topology>_<channel>_<point>.pdf` | Efficiency after each cut (`--points`) |
| `kinematics/<region>/<variable>_<point>.pdf` | The other SR variables, unit area (`--points`; `--no-kinematics` skips) |
| `tables/.../signal_outputs/signal_compare/<...>/efficiency.csv`, `scale_factors.csv`, `cutflow.csv`, `summary.txt` | The numbers; the summary averages each set's ratios (and those of the efficiency's factors) to the reference over the shared points, and lists the mean scale factors of every set |
| `tables/.../efficiency_tables.txt`, `latex/` | A × ε of every set and the ratios to the reference at the shared points: text with stat. errors, and CMS-style LaTeX tables per topology, channel and m_WR with `preview.pdf` (`--no-pdf` skips the compilation) |

`--label` (once per `--era`) sets the legend labels and `--name` the output folder; two sets of one era (e.g.
the UL18 official and pre-legacy samples, run into different directories) are given as `--era ERA=DIR` twice.

`--points` defaults to the `cutflow_signals` of `data/lumi.yaml` that every set has (`all` = every common point).

### make_signal_plots.py

Signal plots of one era from its signal files alone (the single-era counterpart of
`compare_signal_eras.py`; definitions in `wrplotter/signal_compare.py` and `wrplotter/signal_plots.py`):
```bash
python3 bin/make_signal_plots.py --era RunIII2024Summer24 --dir 20261012_skims_v2 --local-plots
python3 bin/make_signal_plots.py --era RunIII2024Summer24 --dir 20261012_skims_v2 --points WR4000_N2100 WR6000_N100 --local-plots
```
Under `<plots>/<Run>/<Year>/<Era>/<dir>/signal_outputs/signal_plots/`:

| Folder | Content |
|---|---|
| `efficiency/<channel>_WR<m>.pdf` | A × ε (SR yield over σ × L of the flavour) vs m_N at each m_WR: resolved SR, boosted SR, their sum (AN-19-083 App. B) |
| `efficiency_map/<topology>_<channel>.pdf` | A × ε over the (m_WR, m_N) grid |
| `cr_fraction/<topology>_<channel>_WR<m>.pdf` | signal in the DY and flavour CRs over σ × L (the flavour CRs over both flavours, as a point's file merges them), vs m_N |
| `xsec/`, `kfactor/` | σ_LO and σ_LO × k (fb) and k vs m_N, from the analyzer's signal config (AN Fig. 6) |
| `mass/` | SR mass of several m_N at one m_WR, and of every m_WR at m_N ≈ 0.05 / 0.5 m_WR, unit area |
| `resolution/` | median / m_WR and σ_eff / median of the SR mass peak vs m_N |
| `kinematics/<region>/`, `cutflow/` | the other SR variables (unit area) and the cumulative cutflow efficiency of `--points` (default: the `cutflow_signals` of `data/lumi.yaml`) |

Tables under `tables/<...>/<dir>/signal_outputs/signal_plots/`: `efficiency.csv` (per point and channel: A × ε, the CR fractions,
the peak median and σ_eff) and `latex/signal_eff_<channel>_WR<m>.tex` with `preview.tex`. `-o DIR` writes everything
under `DIR`; `--max-shapes N` limits the curves per mass plot; `--no-kinematics` skips the per-point plots.

### plot_signal_theory_uncertainties.py

The change of each signal region's yield under the QCD scale (envelope), PDF, α_S and parton-shower ISR / FSR
variations against m_WR (AN-19-083 Fig. 129), from a signal set made with `run_analysis.py ... Signal --systs scale
pdf ps`. The weights are normalized per sample by the analyzer, so these are acceptance effects only:
```bash
python3 bin/plot_signal_theory_uncertainties.py --era RunIII2024Summer24 --dir 20261013_skims_v2 \
    --theory-dir 20261013_signal_theory --local-plots
```
`--theory-dir` is the set with the variations and `--dir` the production whose signal outputs get the plots
(default: the `--theory-dir`). Plots `<plots>/<...>/<dir>/signal_outputs/signal_theory/<topology>_<channel>_<mN100|mNhalf>.pdf` (the lightest m_N at each m_WR,
and m_N ≈ m_WR / 2); the table `tables/<...>/<dir>/signal_outputs/signal_theory/theory_uncertainties.csv` holds every point, region
and source.

### compare_dy.py

Compare Drell-Yan samples, on their own and inside the total background: the named comparisons of
`data/dy_comparisons.yaml`, each into its own folder (every one by default, or the ones `--comparison`
names; the first sample is the reference), or two or more `--dy` samples, the first the reference:
```bash
python3 bin/compare_dy.py --era RunIII2024Summer24 --dir 20261013_skims_v2 --local-plots
python3 bin/compare_dy.py --era RunIII2024Summer24 --dir 20261013_skims_v2 --comparison all,lo_ht_vs_nlo --plots all --local-plots
python3 bin/compare_dy.py --era RunIII2024Summer24 --dir 20261013_skims_v2 --dy lo_ht,lo_inc -r resolved_dy_cr -v mass_fourobject --local-plots
```

| Comparison | Samples (reference first) | Question |
|---|---|---|
| `all` | `lo_ht`, `lo_ht_zpt_reshape`, `lo_inc`, `nlo_inc`, `nlo_mll`, `nlo_njet` | Every DY prediction together (`lo_ht_zpt` and `nlo_ptll` left out) |
| `lo` | `lo_ht`, `lo_inc`, `lo_ht_zpt_reshape` | HT slicing vs inclusive LO; what the Z-pT reweight + jet-pT reshape do |
| `nlo` | `nlo_inc`, `nlo_mll`, `nlo_njet` | The NLO samples against the inclusive one (`nlo_ptll` left out: unstitched) |
| `lo_vs_nlo_inclusive` | `lo_inc`, `nlo_inc` | The generator difference alone, both inclusive |
| `lo_ht_vs_nlo` | `lo_ht`, `nlo_inc`, `nlo_mll`, `nlo_njet` | The default DY against the NLO samples |

Every DY sample is a file in `--dir`: `WRAnalyzer_DYJets.root` for the default `lo_ht`, `WRAnalyzer_DYJets_<TAG>.root` for any other (sets made before 2026-10-12: `<dir>_<TAG>_dy/`, still read). With `--comparison`, `--dy TAG=DIR` only says where a sample is;
a comparison whose sample cannot be found is skipped and reported, the others are drawn. `--plots core`
(the default) draws the `core` set of `data/dy_comparisons.yaml`: in the DY CRs the fit observable, m_ll,
pT(ll), the jet pT the reshape uses, the subleading jet, m_jj and the recoil pT; in the SRs the fit observable
and m_ll. `--plots all` draws every configured pair; `-r`/`-v` restrict either.
Every variable is drawn in uniform bins (the plot settings' `rebin`); `--run2-binning` draws the m(W_R)
observables in the Run 2 fit bins (`rebin_variable`) and writes to `dy_compare_run2_binning/` instead.
`lo_ht` is the LO HT-binned DY without the Z-pT reweight and the jet-pT reshape (the analyzer default
since 2026-10-12; `lo_ht_zpt` and `lo_ht_zpt_reshape` add them); sets made before then hold the reweighted DY
in `<dir>`, read with `--dy lo_ht_zpt_reshape=<dir>`. A DY file whose reweighting (read from its event-weight histograms) is not
the one its tag stands for is refused.

Outputs under `<plots>/<Run>/<Year>/<Era>/<dir>/dy_compare/[<comparison>/]`:

| Folder | Content |
|---|---|
| `abs/<region>/` | DYJets of each sample, with the ratio to the reference |
| `shape/<region>/` | The same, each sample scaled to unit yield |
| `stat/<region>/` | The statistical power of each sample per bin: its relative statistical uncertainty (left axis) and effective number of simulated events N_eff = (Σw)²/Σw² (right axis); N_eff over the plotted range in the legend |
| `total/<region>/` | Total background built with each DY sample; data and Data/Bkg. per sample in control regions, totals compared to the reference in signal regions |
| `total_scaled/<region>/` | Control regions only: as `total`, each prediction scaled to the data yield (factor in the legend) |

### plot_dy_zpt_reweight.py

The DY Z-pT reweighting of an era, drawn from the payload the analyzer applies (`config.yaml`
`dy_zpt_reweight`, or `--payload`), with the converted Run 2 ratio overlaid (`--reference`, default:
the payload the EW correction was copied from; `none` to skip):
```bash
python3 bin/plot_dy_zpt_reweight.py --era RunIII2024Summer24 --local-plots
python3 bin/plot_dy_zpt_reweight.py --era RunIII2024Summer24 --dir 20261012_skims_v2 --dy lo_ht,lo_ht_zpt --local-plots
```
Outputs under `<plots>/<Run>/<Year>/<Era>/[<dir>/]dy_zpt_reweight/`:

| File | Content |
|---|---|
| `qcd_ratio.pdf` | QCD NLO / LO vs generator Z pT up to 1 TeV with the scale, PDF and α_S variations and the Run 2 ratio (AN-19-083 Fig. 14) |
| `qcd_ratio_full_range.pdf` | The same over the payload's whole range, log x |
| `ew_correction.pdf` | The EW factor with its three uncertainties |
| `weight.pdf` | QCD × EW, the weight the analyzer gives a DY event, with the total uncertainty and the two factors |
| `spectra.pdf` | The inclusive LO and NLO generator spectra the ratio came from, with NLO / LO |
| `applied/<region>.pdf` | With `--dir`: the reweighted over the unreweighted DY (`--dy before,after`) in the reco pT(ll) of each DY CR, against the payload |

The before/after pT(ll) stack plots of AN-19-083 Figs. 15-18 are `make_stackplots.py`'s with
`--dy lo_ht,lo_ht_zpt -r resolved_dy_cr,boosted_dy_cr -v pt_dilepton`.

### plot_event_weights.py

MC event-weight distributions per region, from the analyzer's `event_weights/` histograms (MC only):
```bash
python3 bin/plot_event_weights.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --local-plots
python3 bin/plot_event_weights.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --dy lo_ht,lo_inc -r resolved_sr --local-plots
```
Outputs:
```
<plots>/<Run>/<Year>/<Era>/<dir>/event_weights/<region>.pdf
<plots>/.../event_weights/components/<region>/<component>.pdf
tables/<Run>/<Year>/<Era>/<dir>/event_weights/<region>.txt
```
- `<region>.pdf`: log₁₀|w| of the region's nominal weight per MC group (signals too in signal
  regions); w > 0 solid, w < 0 dashed; the legend gives N, N_eff and the fraction of negative weights.
- `components/`: each factor of that weight, unit-normalized per group: the generator weight and the
  σ·L/Σw normalization (log₁₀), then pileup, the lepton scale factors, DY reshaping etc., and
  `corrections`, their product (weight = generator weight × normalization × corrections).
- `<region>.txt`: the same numbers as text (N, N_eff, Σw, negative fraction; mean, RMS and
  fraction outside the axis of every component).

With two or more `--dy` samples the DY samples are also overlaid on their own under
`event_weights/dy_samples/`. Files made before the analyzer wrote these histograms need an analyzer
rerun; the script says so and exits.

### emu_transfer_factor.py

Compute SR / flavor-CR transfer factors for tt+tW (resolved and boosted; `--resolved` or
`--boosted` for one of them):
```bash
python3 bin/emu_transfer_factor.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --local-plots
python3 bin/emu_transfer_factor.py --era Run3 --dir 20261008_skims_v2 --local-plots
```
`--era` is required and takes a single era, a year (`2022`, `2023`, `2024`) or `Run3` (see
[Eras, Years and Run 3](#eras-years-and-run-3)); a combination adds the histograms of its eras, each
read from `rootfiles/<Run>/<Year>/<era>/<dir>`, before the ratio is formed and fitted. Only tt+tW MC
is read (`WRAnalyzer_tt_tW.root`), so there is no `--dy` option and no DYJets file is needed.

Outputs (`[_tf]` in study mode, which also adds `_lowzoom` views of each plot):
```
<plots>/<Run>/<Year>/<Era>/<dir>/emu_transfer_factor_{resolved,boosted}[_tf]_<Era>.pdf
<plots>/<Run>/<Year>/<Era>/<dir>/emu_transfer_factor_boosted_{ejet,mujet}[_tf]_<Era>.pdf
tables/<Run>/<Year>/<Era>/<dir>/emu_transfer_factor_{resolved,boosted}[_tf].tex
tables/<Run>/<Year>/<Era>/<dir>/emu_transfer_factor_boosted_perchannel[_tf].tex
tables/<Run>/<Year>/<Era>/<dir>/preview.pdf
```
`<plots>` is `plots/` with `--local-plots`, otherwise EOS (`outputs/` when EOS is not mounted).

Transfer-factor study (fit below and above 800 GeV separately). Requires
analyzer output produced with `--study transfer-factor --split-groups tt_tW`,
which writes `<region>_nomass` histograms and a separate `WRAnalyzer_TTbar.root`:
```bash
python3 bin/emu_transfer_factor.py --era RunIII2024Summer24 --dir tf_study --tf-study --input WRAnalyzer_TTbar.root --local-plots
```
`--input` names a `WRAnalyzer_<sample>.root` file in each era's `--dir` directory; without it, study
mode reads `WRAnalyzer_TTbar.root` when every era has it, else `WRAnalyzer_tt_tW.root`.
800 GeV must stay a bin edge after rebinning (the default `--variable-bins`
edges do); the script refuses otherwise. Older files that only contain
`_tf_low` regions are still read by summing them with the nominal region
(decided per era file in a combination).

---

## Removed Options

- `make_stackplots.py --unblind`: the analyzer decides (see [Blinding](#blinding)).
- `make_stackplots.py --name`: it was unused.
- `compare_dy.py --mode lo-nlo|mll-vs-ht|cross-era|three-way`: these read files that no longer exist;
  use `--dy` with the samples to compare.
- `make_cutflow_tables.py`: replaced by `make_cutflows.py`.
- `compare_dy_cr_data.py`: replaced by `compare_dy.py` (`total/` and `total_scaled/`).

---

## Repository Structure

```
WR_Plotter/
├── bin/                              # CLI scripts
│   ├── make_stackplots.py            #   Stacked MC + data plots, one set per DY sample
│   ├── make_cutflows.py              #   Cutflow tables and plots (yields, efficiencies)
│   ├── compare_dy.py                 #   Drell-Yan sample comparisons
│   ├── plot_dy_zpt_reweight.py       #   The DY Z-pT reweighting payload and its effect
│   ├── plot_event_weights.py         #   MC event-weight distributions and components
│   ├── make_signal_cutflows.py       #   Signal cutflow tables per (WR, N) mass point
│   └── emu_transfer_factor.py        #   tt+tW SR/CR transfer factors (resolved + boosted)
├── scripts/                          # Utility scripts
│   └── plot_jet_veto_maps.py         #   Jet veto map figure
├── wrplotter/                        # Core library
│   ├── config.py                     #   Load lumi, plot settings, DY samples, cutflow config
│   ├── paths.py                      #   File I/O, EOS upload, repo_root(), plots_base()
│   ├── inputs.py                     #   Analysis-set file locations, DY samples, manifests (blinding)
│   ├── regions.py                    #   Analysis region definitions and data streams
│   ├── variables.py                  #   Physics variable definitions
│   ├── sample_groups.py              #   Sample grouping and styling
│   ├── stack.py                      #   Histograms of one stack plot (MC, data, signals, systematics)
│   ├── systematics.py                #   Systematic variations and the uncertainty band
│   ├── cutflow.py                    #   Cutflow tables (reading, data alignment, text/LaTeX)
│   ├── cutflow_plots.py              #   Cutflow yield and efficiency plots
│   ├── dy_compare.py                 #   DY overlay and total-background comparison plots
│   ├── event_weights.py              #   Event-weight distributions and components
│   ├── histogram_cache.py            #   Cached ROOT files and key lists, read_hist, load-and-rebin
│   ├── histogram_utils.py            #   Histogram rebinning and manipulation
│   ├── plotting_helpers.py           #   Matplotlib/mplhep CMS plot formatting
│   └── cli_utils.py                  #   CLI helpers (parse_multi, --dy, region selection)
├── data/                             # Configuration files
│   ├── lumi.yaml                     #   Luminosity, run, year, CoM per era
│   ├── sample_groups.yaml            #   Sample grouping and colors
│   ├── dy_samples.yaml               #   Drell-Yan samples: label, colour, dataset pattern
│   ├── cutflows.yaml                 #   Cutflow data stream per channel and cut labels
│   ├── variables.yaml                #   Histogram axis definitions
│   ├── region_shorthands.yaml        #   Shorthand aliases for regions
│   ├── systematics.yaml              #   Systematic uncertainty names
│   └── plot_settings/                #   rebin/xlim/ylim: default.yaml + per-era overrides
├── tests/                            # Unit tests (pytest)
├── rootfiles/                        # Input ROOT histograms (from analyzer)
│   └── <Run>/<Year>/<Era>/<dir>/     #   e.g., Run3/2024/RunIII2024Summer24/20261008_skims_v2/
├── plots/                            # Output plots (created by --local-plots)
├── tables/                           # Output tables
├── pyproject.toml
└── README.md
```

---

## Configuration

### EOS / CERNBox Setup

Without `--local-plots`, plots upload to `/eos/user/<first-char>/<username>/...`. The EOS username defaults to `$USER`. If your CERN username differs from your local login (e.g., LPC username `bjackson` but CERN username `wijackso`), set one of these in `~/.bashrc`:

```bash
# Option 1: CERN username (builds path as first-char/username)
export EOSUSER=wijackso        # -> /eos/user/w/wijackso/...

# Option 2: Full path segment
export EOSUSER_PATH=w/wijackso # -> /eos/user/w/wijackso/...

# Option 3: Override the entire EOS root
export EOS_BASE=/eos/user/w/wijackso
```

Additional environment variables:

| Variable | Description |
|----------|-------------|
| `EOS_ENDPOINT` | xrdfs/xrdcp hostname (default: `eosuser.cern.ch`) |
| `FORCE_EOS` | Set to `1` to use EOS even if `/eos` is not mounted |
| `FORCE_LOCAL` | Set to `1` to always write locally instead of EOS |

### Plot Settings YAML

`data/plot_settings/` sets the rebinning and axis ranges per region and variable:

| File | Used by |
|---|---|
| `default.yaml` | every era; 2022 and 2023 use it as is |
| `run3_subera.yaml` | Run3Summer22, Run3Summer22EE, Run3Summer23, Run3Summer23BPix |
| `RunIII2024Summer24.yaml` | RunIII2024Summer24, and the 2024 and Run3 combinations |
| `RunIISummer20UL18.yaml` | RunIISummer20UL18 |

Every file except `default.yaml` starts with `inherit: default` and lists only the values that differ, so a
change made in `default.yaml` reaches every era that does not override that value. An era uses its own file
if one exists, else the one named by `plot_settings` in `data/lumi.yaml`, else `default.yaml`. Example override:

```yaml
inherit: default

wr_resolved_flavor_cr:
  pt_leading_jet: { rebin: 4, xlim: [0, 600], ylim: [1, 1e6] }
```

Override the whole file with `--plot-config <path>` (it may use `inherit:` too).

Every variable is drawn in uniform bins (reviewed 2026-10-13 on the 2024 production):

- Each axis starts at its selection cut on a bin edge (lepton pT at 60 or 50 GeV, jet pT at 40, AK8 pT at 200,
  m(ℓℓ) at 400 or 200, LSF3 just under 0.75, Δφ at 2) and ends about where the simulation does; what lies
  beyond is folded into the last bin.  η follows the object's acceptance: 0.3 bins over |η| < 2.4 for muons and
  jets, 0.5 over |η| < 2.5 for electrons and the mixed flavour CR.
- Widths are round and set per region class by the statistics: 10-25 bins across the bulk of each distribution,
  coarser in the boosted flavour CRs (a few hundred events).
- m(ℓℓjj) runs over the whole fit range, 800-8000 GeV in 100 GeV bins, with the y axis down to 1e-4, so the
  simulated and data tails are in view in every region.
- `--run2-binning` draws the m(W_R) observables in the Run 2 fit bins instead (AN-19-083; HNWR_Plotter
  `python/mylib.py: RebinWRMass`): resolved m(ℓℓjj) edges 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200,
  8000 GeV; boosted m(ℓJ) edges 800, 1000, 1200, 1500, 1800, 8000 GeV (`rebin_variable`).  Drawn as events per
  bin, as in Run 2; `--per-gev` divides by the bin width.

Stack y axes follow the content: the top is 10x the highest bin, raised only as far as the CMS label, region
text and legend need; the bottom is a decade below the smallest bin of the stack or the data, never below 0.1
with data (unless the configured bottom is lower, as for m(ℓℓjj)).

Fonts: every figure is 10 inches wide with no text under 20 pt (legends, region text, the luminosity; axis
titles 26 pt, tick labels 22 pt; `wrplotter/plotting_helpers.py` FONT_SIZE_*).  That prints at 5.7 pt when a
figure is placed at 0.45\textwidth in the AN, above the FigGuidelines minimum of half the 11 pt body text.

---

## Getting Started

If you cloned WrCoffea and the WR_Plotter submodule is empty:
```bash
git submodule update --init --recursive
```
Or clone with `--recursive`:
```bash
git clone --recursive https://github.com/UMN-CMS/WrCoffea.git
```

Create a branch in the submodule:
```bash
cd WR_Plotter
git checkout -b branch_name
git push -u origin branch_name
```

Install dependencies:
```bash
python3 -m pip install .
```

### Grid Proxy
```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```

### ROOT (LCG)
At LPC:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```
At UMN:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos8-gcc11-opt/setup.sh
```

### Running Tests
```bash
cd WR_Plotter
python -m pytest -v
```
