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

### Making control region plots
To make control region plots 
```
python3 bin/plot_control_regions.py --era 2022 --dir dy_ht
```
One can also specify `--era Run3Summer22` or `--era Run3Summer22EE`.

The `--dir` argument will look for ROOT files saved in that directory, so `--dir` must match whatever was used with `bin/run_analysis.py`.

Note that in order for this to run, ROOT files must exist for the following samples:
```
Run3Summer22/WR_Analyzer/DYJets.root
Run3Summer22/WR_Analyzer/TTbar.root
```
TBC
