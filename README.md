# WR Plotter Documentation

Welcome to the WR plotter submodule! This repository provides tools for processing and plotting WR background, data, and signal events. Below, you’ll find instructions on setting up the environment, and how to make some stack plots.

## Table of Contents
- [Quick Start](README.md#quick-start) – How to make stack-plots.
- [Repository Structure](README.md#repository-structure) – Overview of how the repository is organized.
- [Getting Started](README.md#getting-started) – Instructions for installing and setting up the plotter.
---

## Quick Start

### Prerequisites
First, check that you have ran over at least RunIII2024Summer24 in the main analyzer. The following files should exist:
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

To make stack plots, run
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 --local-plots
```
This will make stackplots of all major variables in all analysis regions (resolved/boosted control/signal regions). The signal region is blinded.

The `--local-plots` flag saves plots locally in the repository. Removing it will upload them to CERNBox.

### Output Locations

**Local plots** (with `--local-plots`):
```
plots/<Run>/<Year>/<Era>/<Region>_<Dataset>/<Variable>_<Region>.pdf
```
Example:
```
plots/Run3/2024/RunIII2024Summer24/resolved_dy_cr_EGamma/pt_leading_jet_resolved_dy_cr.pdf
```

**CERNBox plots** (without `--local-plots`):
Plots are uploaded to your EOS area. The exact path depends on your CERNBox configuration.

All plots are saved in **PDF format** and organized by:
- Region name and primary dataset (e.g., `resolved_dy_cr_EGamma`)
- Individual plot files named by variable and region (e.g., `pt_leading_jet_resolved_dy_cr.pdf`)

To see avaliable eras, run
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 --list-eras
```


### Plotting specific regions
To run over a particular region, include the `-r` flag,
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr  --local-plots
```
To see avaliable regions,
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 --list-regions
```

### Plotting specific variables
To plot a particular variable, include the `-v` flag,
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr -v pt_leading_jet  --local-plots
```
which also accepts list arguments, i.e.
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 -r resolved_dy_cr -v pt_leading_jet,pt_leading_lepton  --local-plots
```
To see avaliable variables,
```
python3 bin/make_stackplots.py --era RunIII2024Summer24 --list-variables
```

### Unblinding
To default, the signal region is blinded. To unblind, include the `--unblind` flag,
```
python3 bin/make_stackplots.py --era RunIISummer20UL18 --unblind  --local-plots
```

If you used the `--dir` argument in `bin/run_analysis.py` (so that the files are saved under `dir/`), you can use the same argument here
```
python3 bin/make_stackplots.py --era RunIISummer20UL18 --dir my_directory  --local-plots
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
This repository is structured to separate executable scripts, core analysis logic, and documentation.

```
bin/        # user-facing CLI entrypoints (tiny wrappers).
data/       # configs & static metadata (tracked, human-editable)
python/     # the importable library (public API)
scripts/    # developer & maintenance utilities
rootfiles/  # input ROOT/hist files 
test/       # Holds test and development scripts.
```

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
