# WR Plotter

Plots and tables from the WrCoffea analyzer's ROOT files: stacked data and simulation, cutflows,
Drell-Yan comparisons, signal efficiencies and shapes, transfer factors, systematics tables.
Every command, its options and its outputs: [docs/plotting.md](docs/plotting.md).

## Setup

The plotter is the `WR_Plotter` submodule of WrCoffea and runs in the analyzer's environment:

```bash
git clone --recursive -b billy_dev https://github.com/UMN-CMS/WrCoffea.git && cd WrCoffea
git submodule update --init --recursive        # if the submodule is empty
python3.12 -m venv .venv && .venv/bin/pip install -e ".[xrootd,test]"
cd WR_Plotter && ../.venv/bin/python -m pytest tests -q
```

Inputs are the analyzer's outputs, `rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_<sample>.root`,
each with a manifest `WRAnalyzer_<sample>_systematics.json` next to it. A stack needs the four
background files, the two data files and, for the signal regions, the signal files.

## Commands

```bash
cd WR_Plotter
../.venv/bin/python bin/make_stackplots.py          --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/make_cutflows.py            --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/make_signal_cutflows.py     --era RunIII2024Summer24 --dir <tag>
../.venv/bin/python bin/make_systematics_tables.py  --era RunIII2024Summer24 --dir <tag>
../.venv/bin/python bin/compare_dy.py               --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/plot_event_weights.py       --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/emu_transfer_factor.py      --era Run3 --dir <tag> --local-plots
```

- `--dir` is the analyzer's `--dir`. `--era` takes one era or a combination (`2022`, `2023`, `2024`,
  `Run3`), which sums the eras' histograms; every era then needs the same `--dir`.
- `--local-plots` writes under `plots/`; without it plots go to EOS (`/eos/user/<i>/<user>/...`) when
  `/eos` is mounted, else to `outputs/`. Tables always go to `tables/`.
- `-r` and `-v` pick regions and variables; `--list-eras`, `--list-regions`, `--list-variables` print
  the choices.

## Data

| File | Holds |
| --- | --- |
| `data/lumi.yaml` | eras and combinations: run, year, luminosity, plot-settings file, signal points |
| `data/plot_settings/` | rebin, axis ranges per region and variable: `default.yaml` plus override files with `inherit: default` |
| `data/sample_groups.yaml`, `data/dy_samples.yaml` | sample groups, colours, the DY variants and their file tags |
| `data/cutflows.yaml` | the data stream per channel and the row labels |
| `data/systematics.yaml`, `data/variables.yaml`, `data/region_shorthands.yaml` | the systematic sources of the band, the variables, region aliases |
