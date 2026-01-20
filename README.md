# WR Plotter Documentation

This repository provides tools for processing and plotting WR background, data, and signal events. 

## Table of Contents
- [Quick Start](#quick-start) – Get started making stack plots
  - [Prerequisites](#prerequisites) – Required ROOT files
  - [Stackplots](#stackplots) – Basic plotting commands
  - [Output Locations](#output-locations) – Where plots are saved
  - [Plotting Specific Regions](#plotting-specific-regions) – Filter by region
  - [Plotting Specific Variables](#plotting-specific-variables) – Filter by variable
  - [Unblinding](#unblinding) – Show data in signal regions
- [Command Reference](#command-reference) – Complete flag reference and examples
- [Repository Structure](#-repository-structure) – Overview of how the codebase is organized
- [Getting Started](#getting-started) – Installation and environment setup
---

## Quick Start

### Prerequisites
Check that you have ran over at least RunIII2024Summer24 in the main analyzer. The following files should exist:
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

### Stackplots

There is one command that will make stackplots of all variables in all analysis regions,
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 --local-plots
```
This will make plots of the resolved and boosted variables in the control and signal regions. 

The signal region is blinded by default, and a signal sample is shown instead of data.

To see avaliable eras, run
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 --list-eras
```
Right now `RunII2Summer20UL18` and `RunIII2024Summer24` are confirmed to work.

If you used the `--dir` argument in `bin/run_analysis.py` (so that the files are saved under `dir/`), you can use the same argument here to point to those ROOT files
```
python3 bin/make_stackplots.py --era RunIISummer20UL18 --dir my_directory  --local-plots
```

---

### Output Locations

The `--local-plots` flag saves plots locally in the `plots/` directory,
```
plots/<Run>/<Year>/<Era>/<Region>_<Dataset>/<Variable>_<Region>.pdf
```
Example:
```
plots/Run3/2024/RunIII2024Summer24/resolved_dy_cr_EGamma/pt_leading_jet_resolved_dy_cr.pdf
```

CERNBox plots (without `--local-plots`) are uploaded to your EOS area. The exact path depends on your CERNBox configuration.

All plots are saved in **PDF format** and organized by:
- Region name and primary dataset (e.g., `resolved_dy_cr_EGamma`)
- Individual plot files named by variable and region (e.g., `pt_leading_jet_resolved_dy_cr.pdf`)

---

### Plotting specific regions
To run over a particular region, include the `-r` flag,
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr  --local-plots
```
To see avaliable regions,
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 --list-regions
```
---

### Plotting specific variables
To plot a particular variable, include the `-v` flag,
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr -v pt_leading_jet  --local-plots
```
it also accepts list arguments, i.e.
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr -v pt_leading_jet,pt_leading_lepton  --local-plots
```
To see all avaliable variables,
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 --list-variables
```

---
### Unblinding
To default, the signal region is blinded. To unblind, include the `--unblind` flag (safe for `RunII`),
```
python3 bin/make_stackplots.py --era RunIISummer20UL18 --unblind  --local-plots
```

---

## Command Reference

### make_stackplots.py Flags

| Flag | Short | Arguments | Description |
|------|-------|-----------|-------------|
| `--era` | | `<era_name>` | **Required.** Specify the era (e.g., RunIII2024Summer24, RunIISummer20UL18) |
| `--region` | `-r` | `<region_name>` | Filter to specific region(s). Can repeat or use comma-separated list |
| `--variable` | `-v` | `<var_name>` | Filter to specific variable(s). Can repeat or use comma-separated list |
| `--local-plots` | | | Save plots locally to `plots/` instead of uploading to CERNBox |
| `--unblind` | | | Show data in signal regions (default: blinded) |
| `--dir` | | `<directory>` | Use subdirectory under input/output paths |
| `--name` | | `<suffix>` | Append suffix to output filenames |
| `--plot-config` | | `<yaml_file>` | Custom YAML file with rebin/xlim/ylim settings |
| `--list-eras` | | | List all available eras and exit |
| `--list-regions` | | | List all available regions for specified era and exit |
| `--list-variables` | | | List all available variables and exit |

### Examples

```bash
# Basic usage - all regions and variables for an era
python3 bin/make_stackplots.py --era RunIII2024Summer24 --local-plots

# Single region, single variable
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr -v pt_leading_jet --local-plots

# Multiple regions and variables (comma-separated)
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr,boosted_sr -v pt_leading_jet,mass_ll --local-plots

# Multiple regions and variables (repeated flags)
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr -r boosted_sr -v pt_leading_jet -v mass_ll --local-plots

# Unblind signal regions with custom directory
python3 bin/make_stackplots.py --era RunIISummer20UL18 --unblind --dir my_analysis --local-plots

# List available options
python3 bin/make_stackplots.py --list-eras
python3 bin/make_stackplots.py --era RunIII2024Summer24 --list-regions
python3 bin/make_stackplots.py --era RunIII2024Summer24 --list-variables
```

---

## 📂 Repository Structure

The repository follows a clean architecture separating user-facing scripts, core library code, configuration, and data storage.

### Directory Overview

```
bin/         # User-facing CLI scripts (production workflows)
python/      # Core analysis library (public API)
src/         # Low-level plotting and histogram utilities
data/        # Configuration files (YAML/JSON)
rootfiles/   # Input ROOT histograms from analyzer
plots/       # Generated output plots (PDF format)
test/        # Development and validation scripts
```

### Key Directories

**`bin/`** - Production Scripts
- [`make_stackplots.py`](bin/make_stackplots.py) - Main script for generating stacked histogram plots
- [`compare_dy.py`](bin/compare_dy.py) - Compare LO vs NLO DYJets samples
- [`compare_22_24_dy.py`](bin/compare_22_24_dy.py) - Compare 2022 vs 2024 data
- Thin wrappers around the core library functionality

**`python/`** - Core Library
- [`config.py`](python/config.py) - Load and manage YAML/JSON configurations (lumi, kfactors, plot settings)
- [`plotter.py`](python/plotter.py) - Main `Plotter` class orchestrating the plotting workflow
- [`regions.py`](python/regions.py) - Define analysis regions (resolved/boosted, control/signal regions)
- [`variables.py`](python/variables.py) - Define physics variables to plot (mass, pT, eta, phi, etc.)
- [`io.py`](python/io.py) - File I/O utilities and EOS/CERNBox integration
- [`histo.py`](python/histo.py) - Histogram loading and rebinning functions
- [`sample_groups.py`](python/sample_groups.py) - Sample organization and styling

**`src/`** - Utilities
- [`histogram_utils.py`](src/histogram_utils.py) - Histogram rebinning and manipulation
- [`plotting_helpers.py`](src/plotting_helpers.py) - Matplotlib styling and CMS plot formatting

**`data/`** - Configuration Files
- `plot_settings/` - Per-era plot configurations (rebin, x/y limits)
  - [`RunIII2024Summer24.yaml`](data/plot_settings/RunIII2024Summer24.yaml)
  - [`RunIISummer20UL18.yaml`](data/plot_settings/RunIISummer20UL18.yaml)
- `sample_groups/` - Sample grouping and colors
- [`lumi.json`](data/lumi.json) - Luminosity values per era
- [`kfactors.yaml`](data/kfactors.yaml) - MC scale factors

**`rootfiles/`** - Input Data
- Organized by `Run/Year/Era/` (e.g., `Run3/2024/RunIII2024Summer24/`)
- Contains ROOT histograms from the upstream WrCoffea analyzer
- Background samples: DYJets, tt_tW, Nonprompt, Other
- Data samples: EGamma, Muon
- Signal samples: WR4000_N2100, WR4000_N100

**`plots/`** - Output Directory
- Generated PDF plots organized by era and region
- Created when using `--local-plots` flag
- Structure: `plots/Run/Year/Era/Region_Dataset/Variable_Region.pdf`

**`test/`** - Development Scripts
- Analysis optimization studies (e.g., `mll_study/`)
- Cross-era comparison scripts
- Background fraction studies
- Scale factor validation

---

## Getting Started
If you have cloned WrCoffea and the WR_Plotter submodule is empty, run this from the WrCoffea repo
```bash
git submodule update --init --recursive
```
Alternatively, next time clone the repo with the `--recursive` flag
```bash
git clone --recursive git@github.com:UMN-CMS/WrCoffea.git
```

Then go to the submodule and make a new branch
```
cd WR_Plotter
git checkout -b branch_name
git push -u origin branch_name
```

Install the required packages:
```bash
python3 -m pip install -r requirements.txt
```

### Grid UI
To authenticate for accessing grid resources, use:
```bash
voms-proxy-init --rfc --voms cms -valid 192:00
```

### ROOT
To enable ROOT functionality, source the appropriate LCG release:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```
If using UMN’s setup, use:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_104/x86_64-centos8-gcc11-opt/setup.sh
```
