# Plotting

What this page answers: which WR_Plotter command makes which plot or table from the analyzer's
output, and where the results go. The plotter is the `WR_Plotter/` submodule (branch `billyDev`);
its own [README](../README.md) has the options of every script except
`make_systematics_tables.py`, which only this page documents.
Verified on 2026-10-12.

Every command runs from `WR_Plotter/` with the analyzer's virtual environment and takes `--era`
and `--dir`, the era and the `--dir` tag of the analyzer production under
`WR_Plotter/rootfiles/<Run>/<Year>/<Era>/<dir>/`. Add `--local-plots` to write under
`WR_Plotter/plots/` instead of EOS. Without it, plots go to
`/eos/user/<initial>/<username>/<Run>/<Year>/<Era>/<dir>/...` when `/eos` is mounted, else to
`WR_Plotter/outputs/`. `EOSUSER` (the CERN username, if not `$USER`), `EOSUSER_PATH` (the
`<initial>/<username>` part), `EOS_BASE` (the whole `/eos/user/...` root) and `FORCE_LOCAL=1` (never
EOS) override this (`wrplotter/paths.py`). Tables go to `WR_Plotter/tables/`, never to EOS.

```bash
cd WR_Plotter
../.venv/bin/python bin/make_stackplots.py          --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/make_cutflows.py            --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/make_signal_cutflows.py     --era RunIII2024Summer24 --dir <tag>
../.venv/bin/python bin/make_systematics_tables.py  --era RunIII2024Summer24 --dir <tag> [--per-bin]
../.venv/bin/python bin/compare_dy.py               --era RunIII2024Summer24 --dir <tag> --dy lo_ht,nlo_inc --local-plots
../.venv/bin/python bin/plot_event_weights.py       --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/emu_transfer_factor.py      --era Run3 --dir <tag> --local-plots
../.venv/bin/python bin/compare_signal_eras.py      --help
```

| Command | Makes | Output |
| --- | --- | --- |
| `make_stackplots.py` | Data over the stacked backgrounds for every region and variable, signal overlays in the signal regions, the systematic band from every variation the files carry (`data/systematics.yaml`) | `plots/<...>/<dir>/stack_dy_<tag>/<region>/<variable>_<region>.pdf` |
| `make_cutflows.py` | Cutflow tables (LaTeX, text) and plots per topology and channel, weighted and raw, with efficiencies | `tables/<...>/<dir>/cutflows_dy_<tag>/`, `plots/.../cutflows_dy_<tag>/` |
| `make_signal_cutflows.py` | Yields and efficiencies per (m_WR, m_N) point | `tables/<...>/<dir>/signal_cutflows/` |
| `make_systematics_tables.py` | Per region, at the fit binning of the mass observable: the up/down change of each process's yield under every systematic source (backgrounds, their total, signal points), the largest per-bin change, a dash where a source does not move a process (a produced variation equal to the nominal, such as a placeholder, is a row of dashes), the sources absent from the files named under the legend; `--per-bin` adds bin-by-bin tables | `tables/<...>/<dir>/systematics_dy_<tag>/systematics_<variable>_<region>.{txt,md,tex}` and the combined `systematics.{txt,md,csv}` |
| `compare_dy.py` | Two or more DY samples (`--dy lo_ht,nlo_inc,...`) alone and inside the total background | `plots/.../dy_compare/` |
| `plot_event_weights.py` | The MC event-weight distributions per region, total and each factor | `plots/.../event_weights/` |
| `emu_transfer_factor.py` | The eμ transfer factor SR / flavour CR on tt̄ and its closure below 800 GeV (`--tf-study` with a `--study transfer-factor` production) | `plots/<...>/<dir>/emu_transfer_factor_*.pdf` (no subfolder), `tables/<...>/<dir>/emu_transfer_factor_*.tex` |
| `compare_signal_eras.py` | Signal efficiencies and shapes across eras | see `--help` |

- **Eras.** `--era` takes one analysis era or a combination from `data/lumi.yaml`: `2022`,
  `2023`, `2024`, `Run3`. A combination sums the sub-eras' histograms, so every sub-era needs the
  same `--dir`. `RunIII2025Summer25` is not in `data/lumi.yaml` yet, so the plotter does not take it.
- **DY sample.** The default is the analyzer's default DY sample (LO HT-binned). `--dy TAG` reads
  another variant from `<dir>_<TAG>_dy`, or `--dy TAG=DIR` from an explicit directory.
  `lo_ht_noreweight` is the default sample without the DY Z-pT reweight and jet-pT reshape; make it
  with `bin/run_analysis.py <era> DYJets --no-dy-zpt-reweight --dir <dir>_lo_ht_noreweight_dy`
  (same analyzer version as `<dir>`), then `--dy lo_ht,lo_ht_noreweight`. Nothing checks that the
  flag was set: the directory name is the only record.
- **Binning.** The plot settings (`data/plot_settings/`) follow the Run 2 analysis: the m(W_R)
  observables use the variable fit-bin edges and are drawn as events per bin. The systematics
  tables use the same binning, so their numbers are the ones the datacards see.
- **Blinding.** Signal-region data appears only from data files the analyzer made with
  `--unblind` (Run 2 only). Files without a `blinding` record in their manifest predate the
  policy and never show signal-region data.
- **Systematic band.** Variations are summed era by era; an era whose file lacks a variation adds
  its nominal (never zero) and is reported. `scale` is the envelope of the μR, μF and μRμF
  variations; `pdf` and `alphas` are formed from the stored PDF members at the plot binning.

`scripts/plot_jet_veto_maps.py` draws the jet veto maps of the analyzer's `data/jsonpog/` payloads
into `WR_Plotter/plots/jet_veto_maps.png`.

Tests: `cd WR_Plotter && ../.venv/bin/python -m pytest tests -q` (350 tests, offline, about 20 seconds).
