# WR Plotter

Plotting tools for the WR analysis. Takes ROOT histogram files produced by the WrCoffea analyzer and generates stacked MC + data plots, DY comparison overlays, signal closure studies, and transfer factor plots.

## Table of Contents
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Making Stackplots](#making-stackplots)
  - [Output Locations](#output-locations)
  - [Filtering by Region](#filtering-by-region)
  - [Filtering by Variable](#filtering-by-variable)
  - [Signal Overlays](#signal-overlays)
  - [Unblinding](#unblinding)
- [Command Reference](#command-reference)
- [Other Scripts](#other-scripts)
  - [compare_dy.py](#compare_dypy)
  - [make_cutflow_table.py](#make_cutflow_tablepy)
  - [signal_closure.py](#signal_closurepy)
  - [transfer_factor_tt_tW.py](#transfer_factor_tt_twpy)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
  - [EOS / CERNBox Setup](#eos--cernbox-setup)
  - [Plot Settings YAML](#plot-settings-yaml)
- [Getting Started](#getting-started)

---

## Quick Start

### Prerequisites
Run at least `RunIII2024Summer24` in the WrCoffea analyzer. The following files should exist:
```
rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_DYJets.root
rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_tt_tW.root
rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_Nonprompt.root
rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_Other.root
rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_EGamma.root
rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_Muon.root
rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_signal_WR4000_N2100.root
rootfiles/Run3/2024/RunIII2024Summer24/WRAnalyzer_signal_WR4000_N100.root
```

---

### Making Stackplots

Plot all variables in all analysis regions:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 --local-plots
```
This produces resolved and boosted plots for all control and signal regions. Signal regions are blinded by default with a signal overlay shown instead of data.

List available eras:
```bash
python3 bin/make_stackplots.py --list-eras
```
Currently `RunIISummer20UL18` and `RunIII2024Summer24` are confirmed to work.

If you used `--dir` in the analyzer, pass the same subdirectory here:
```bash
python3 bin/make_stackplots.py --era RunIISummer20UL18 --dir my_directory --local-plots
```

---

### Output Locations

With `--local-plots`, plots are saved locally:
```
plots/<Run>/<Year>/<Era>/<Region>_<Dataset>/<Variable>_<Region>.pdf
```
Example:
```
plots/Run3/2024/RunIII2024Summer24/resolved_dy_cr_EGamma/pt_leading_jet_resolved_dy_cr.pdf
```

Without `--local-plots`, plots are uploaded to EOS/CERNBox. See [EOS / CERNBox Setup](#eos--cernbox-setup) for configuration.

---

### Filtering by Region

Plot a specific region with `-r`:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr --local-plots
```
List available regions:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 --list-regions
```

---

### Filtering by Variable

Plot a specific variable with `-v`:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr -v pt_leading_jet --local-plots
```
Comma-separated lists work too:
```bash
python3 bin/make_stackplots.py --era RunIII2024Summer24 -v pt_leading_jet,pt_leading_lepton --local-plots
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
python3 bin/make_stackplots.py --era RunIII2024Summer24 -s signal_WR6000_N3100 --local-plots

# Multiple signal samples (all overlaid on every SR)
python3 bin/make_stackplots.py --era RunIII2024Summer24 -s signal_WR4000_N2100,signal_WR4000_N100 --local-plots
```
Samples that don't have histograms for a given region are silently skipped.

---

### Unblinding

Signal regions are blinded by default. Unblind with `--unblind` (safe for RunII):
```bash
python3 bin/make_stackplots.py --era RunIISummer20UL18 --unblind --local-plots
```
Run3 unblinding is blocked.

---

## Command Reference

### make_stackplots.py

| Flag | Short | Arguments | Description |
|------|-------|-----------|-------------|
| `--era` | | `<era_name>` | **Required.** Era to process (e.g., `RunIII2024Summer24`) |
| `--region` | `-r` | `<name>` | Region(s) to plot. Repeat or comma-separate |
| `--variable` | `-v` | `<name>` | Variable(s) to plot. Repeat or comma-separate |
| `--signal` | `-s` | `<sample>` | Signal sample(s) to overlay on SR plots. Repeat or comma-separate |
| `--local-plots` | | | Save to `plots/` instead of EOS |
| `--unblind` | | | Show data in signal regions |
| `--dir` | | `<subdir>` | Subdirectory under input/output paths |
| `--name` | | `<suffix>` | Append suffix to filenames |
| `--plot-config` | `-c` | `<yaml>` | Custom plot settings YAML |
| `--variable-rebin` | | | Use variable-width bins from YAML |
| `--list-eras` | | | List eras and exit |
| `--list-regions` | | | List regions for era and exit |
| `--list-variables` | | | List variables and exit |

### Examples
```bash
# All regions and variables
python3 bin/make_stackplots.py --era RunIII2024Summer24 --local-plots

# Single region, single variable
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr -v pt_leading_jet --local-plots

# Multiple regions and variables
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr,boosted_sr -v pt_leading_jet,mass_dilepton --local-plots

# Custom signal overlay
python3 bin/make_stackplots.py --era RunIII2024Summer24 -s signal_WR4000_N2100,signal_WR4000_N100 --local-plots

# Unblind RunII with custom directory
python3 bin/make_stackplots.py --era RunIISummer20UL18 --unblind --dir my_analysis --local-plots
```

---

## Other Scripts

### compare_dy.py

Compare DYJets histograms in three modes:

```bash
# LO vs NLO within one era
python3 bin/compare_dy.py --mode lo-nlo --era RunIII2024Summer24

# 2024 NLO mll-binned vs 2022 LO HT-binned
python3 bin/compare_dy.py --mode mll-vs-ht --era 2022

# Compare DYJets between two eras
python3 bin/compare_dy.py --mode cross-era --era RunIII2024Summer24 --ref-era RunIISummer20UL18
```

### make_cutflow_table.py

Generate a LaTeX cutflow table from analyzer output:
```bash
python3 make_cutflow_table.py --era RunIII2024Summer24
```

### signal_closure.py

Run2 vs Run3 signal closure study (in `scripts/`):
```bash
python3 scripts/signal_closure.py
```

### transfer_factor_tt_tW.py

Compute SR / flavor-CR transfer factors for tt+tW (in `scripts/`):
```bash
python3 scripts/transfer_factor_tt_tW.py
```

---

## Repository Structure

```
WR_Plotter/
├── bin/                              # Production CLI scripts
│   ├── make_stackplots.py            #   Stacked MC + data plots
│   └── compare_dy.py                 #   DY comparison overlays (LO/NLO, cross-era)
├── scripts/                          # One-off analysis scripts
│   ├── signal_closure.py             #   Run2 vs Run3 signal closure
│   └── transfer_factor_tt_tW.py      #   SR/CR transfer factors
├── wrplotter/                        # Core library
│   ├── config.py                     #   Load lumi, kfactors, plot settings
│   ├── io.py                         #   File I/O, EOS upload, repo_root()
│   ├── regions.py                    #   Analysis region definitions
│   ├── variables.py                  #   Physics variable definitions
│   ├── sample_groups.py              #   Sample grouping and styling
│   ├── histo.py                      #   Histogram loading and rebinning (high-level)
│   ├── histogram_utils.py            #   Histogram rebinning and manipulation (low-level)
│   ├── plotting_helpers.py           #   Matplotlib/mplhep CMS plot formatting
│   └── cli_utils.py                  #   CLI helpers (parse_multi, setup_logging)
├── data/                             # Configuration files
│   ├── lumi.json                     #   Luminosity, run, year, CoM per era
│   ├── kfactors.yaml                 #   MC scale factors
│   ├── plot_settings/                #   Per-era rebin/xlim/ylim YAML configs
│   │   ├── RunIII2024Summer24.yaml
│   │   ├── RunIISummer20UL18.yaml
│   │   └── ...
│   └── sample_groups/                #   Per-era sample grouping and colors
│       ├── base.yaml
│       ├── RunIII2024Summer24.yaml
│       └── ...
├── tests/                            # Unit tests (pytest)
│   ├── test_config.py
│   ├── test_regions.py
│   ├── test_histogram_utils.py
│   ├── test_plotting_helpers.py
│   └── test_cli_utils.py
├── test/                             # Development/validation studies
│   ├── mll_study/                    #   Dilepton mass optimization
│   └── ...                           #   Cross-era comparisons, SF validation
├── rootfiles/                        # Input ROOT histograms (from analyzer)
│   └── <Run>/<Year>/<Era>/           #   e.g., Run3/2024/RunIII2024Summer24/
├── plots/                            # Output plots (created by --local-plots)
├── make_cutflow_table.py             # Cutflow LaTeX table generator
├── pytest.ini
├── requirements.txt
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

Each era has a YAML file in `data/plot_settings/` controlling rebinning and axis ranges per region and variable. Example:

```yaml
wr_resolved_flavor_cr:
  pt_leading_jet:
    rebin: 4
    xlim: [0, 600]
    ylim: [1, 1e6]
```

Override with `--plot-config <path>`.

---

## Getting Started

If you cloned WrCoffea and the WR_Plotter submodule is empty:
```bash
git submodule update --init --recursive
```
Or clone with `--recursive`:
```bash
git clone --recursive git@github.com:UMN-CMS/WrCoffea.git
```

Create a branch in the submodule:
```bash
cd WR_Plotter
git checkout -b branch_name
git push -u origin branch_name
```

Install dependencies:
```bash
python3 -m pip install -r requirements.txt
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
