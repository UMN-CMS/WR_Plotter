# WR Plotter Documentation

Welcome to the WR plotter submodule! This repository provides tools for processing and plotting WR background, data, and signal events. Below, you’ll find instructions on setting up the environment, and how to make some stack plots.

## Table of Contents
- [Repository Structure](README.md#repository-structure) – Overview of how the repository is organized.
- [Getting Started](README.md#getting-started) – Instructions for installing and setting up the plotter.
- [Examples](README.md#examples) – How to make control region plots.
---

## 📂 Repository Structure
This repository is structured to separate executable scripts, core analysis logic, and documentation.

```
bin/        # user-facing CLI entrypoints (tiny wrappers).
data/       # configs & static metadata (tracked, human-editable)
docs/       # Contains documentation markdown.
python/     # the importable library (public API)
scripts/    # developer & maintenance utilities
rootfiles/  # input ROOT/hist files 
test/       # Holds test and development scripts.
```

---

## Getting Started
Begin by cloning the repository:
```bash
git clone git@github.com:UMN-CMS/WrCoffea.git
cd WrCoffea
```
Create and source a virtual Python environment:
```bash
python3 -m venv wr-env
source wr-env/bin/activate
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

## Examples

### Prerequisites
First, ensure you have ran over at least Run3Summer22 in the main analyzer. The following files should exist:
```  
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_Diboson.root
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_DYJets.root
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_TW.root 
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_EGamma.root
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_Muon.root
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_SingleTop.root
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_Triboson.root
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_TTbar.root
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_TTbarSemileptonic.root
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_TTV.root
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_TW.root
rootfiles/Run3/2022/Run3Summer22/WRAnalyzer_WJets.root
```
If you also have these files for `Run3Summer22EE`, then you can plot all of 2022.

### Making a simple control region plot
If these files exist, then try making a single plot with 
```
python3 bin/plot_control_regions.py --era Run3Summer22 -r wr_mumu_resolved_dy_cr -v mass_dilepton
```
This will make a stack plot of the dimuon mass in the Drell-Yan control region.

### Other examples
To plot multiple variables,
```
python3 bin/plot_control_regions.py --era Run3Summer22 -r wr_mumu_resolved_dy_cr -v mass_dilepton,mass_fourobject
```

To make plots for both the electron and muon channels, remove the `-r` argument
```
python3 bin/plot_control_regions.py --era Run3Summer22 -v mass_dilepton
```

To make plots for all variables, remove the `-v` argument
```
python3 bin/plot_control_regions.py --era Run3Summer22
```

If you used the `--dir` argument in `bin/run_analysis.py` (so that the files are saved under `dir/`), you can use the same argument here
```
python3 bin/plot_control_regions.py --era Run3Summer22 --dir dy_nlo
```

To run over all of 2022 (Run3Summer22 and Run3Summer22EE combined), use
```
python3 bin/plot_control_regions.py --era 2022
```

To make plots of the flavor sideband, use
```
python3 bin/plot_control_regions.py --era 2022 --dir dy_ht --cat flavor_cr
```
