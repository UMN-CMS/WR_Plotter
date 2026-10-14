# Plotting

What this page answers: which WR_Plotter command makes which plot or table from the analyzer's
output, and where the results go. The plotter is the `WR_Plotter/` submodule (branch `billyDev`);
its own [README](../README.md) has the options of every script except
`make_systematics_tables.py`, which only this page documents.
Verified on 2026-10-13.

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
../.venv/bin/python bin/compare_dy.py               --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/plot_event_weights.py       --era RunIII2024Summer24 --dir <tag> --local-plots
../.venv/bin/python bin/plot_dy_zpt_reweight.py     --era RunIII2024Summer24 --dir <tag> --dy lo_ht,lo_ht_zpt --local-plots
../.venv/bin/python bin/emu_transfer_factor.py      --era Run3 --dir <tag> --local-plots
../.venv/bin/python bin/compare_signal_eras.py      --help
```

| Command | Makes | Output |
| --- | --- | --- |
| `make_stackplots.py` | Data over the stacked backgrounds for every region and variable, signal overlays in the signal regions, the systematic band from every variation the files carry (`data/systematics.yaml`); uniform bins (`--run2-binning`: the Run 2 m(W_R) fit bins) | `plots/<...>/<dir>/stack_dy_<tag>/<region>/<variable>_<region>.pdf` (`stack_dy_<tag>_run2_binning/`) |
| `make_cutflows.py` | Cutflow tables (LaTeX, text) and plots per topology and channel, weighted and raw, with efficiencies | `tables/<...>/<dir>/cutflows_dy_<tag>/`, `plots/.../cutflows_dy_<tag>/` |
| `make_signal_cutflows.py` | Yields and efficiencies per (m_WR, m_N) point | `tables/<...>/<dir>/signal_outputs/signal_cutflows/` |
| `make_signal_plots.py` | Signal plots of one era: A × ε vs m_N and over the grid, signal in the CRs, σ and k, mass shapes and resolution, kinematics, cutflow plots per m_WR | `plots/<...>/<dir>/signal_outputs/signal_plots/`, `tables/<...>/<dir>/signal_outputs/signal_plots/` |
| `plot_signal_theory_uncertainties.py` | Scale, PDF, α_S and PS changes of the signal-region yields vs m_WR, from a `--systs scale pdf ps` signal set (`--theory-dir`) | `plots/<...>/<dir>/signal_outputs/signal_theory/` |
| `make_systematics_tables.py` | Per region, at the fit binning of the mass observable: the up/down change of each process's yield under every systematic source (backgrounds, their total, signal points), the largest per-bin change, a dash where a source does not move a process (a produced variation equal to the nominal, such as a placeholder, is a row of dashes), the sources absent from the files named under the legend; `--per-bin` adds bin-by-bin tables | `tables/<...>/<dir>/systematics_dy_<tag>/systematics_<variable>_<region>.{txt,md,tex}` and the combined `systematics.{txt,md,csv}` |
| `compare_dy.py` | DY samples alone and inside the total background: the named comparisons of `data/dy_comparisons.yaml` (every one by default; `--comparison all,lo,nlo,lo_vs_nlo_inclusive,lo_ht_vs_nlo` picks) or a `--dy` list; the `core` plot set by default (`--plots all`: every pair); uniform bins (`--run2-binning`: the Run 2 m(W_R) fit bins) | `plots/.../dy_compare/[<comparison>/]` (`dy_compare_run2_binning/`) |
| `plot_event_weights.py` | The MC event-weight distributions per region, total and each factor | `plots/.../event_weights/` |
| `plot_dy_zpt_reweight.py` | The DY Z-pT reweighting payload (QCD NLO/LO with its variations and the Run 2 ratio, the EW factor, their product, the generator spectra) and, with `--dir`, the reweighted over the unreweighted DY in the DY CRs' pT(ll) | `plots/.../dy_zpt_reweight/` |
| `emu_transfer_factor.py` | The eμ transfer factor SR / flavour CR on tt̄ and its closure below 800 GeV (`--tf-study` with a `--study transfer-factor` production) | `plots/<...>/<dir>/emu_transfer_factor_*.pdf` (no subfolder), `tables/<...>/<dir>/emu_transfer_factor_*.tex` |
| `compare_signal_eras.py` | Signal efficiencies and shapes across eras | `plots/<...>/<ref dir>/signal_outputs/signal_compare/` (see `--help`) |

- **Signal outputs.** Every signal output of a production sits in `<dir>/signal_outputs/<part>/` (plots and
  tables alike), including the event-level plots of WrCoffea `scripts/signal_event_plots.py plot --era <era>
  --dir <dir>`; the README's "Signal outputs" section lists the parts.
- **Eras.** `--era` takes one analysis era or a combination from `data/lumi.yaml`: `2022`,
  `2023`, `2024`, `Run3`. A combination sums the sub-eras' histograms, so every sub-era needs the
  same `--dir`. `RunIII2025Summer25` is not in `data/lumi.yaml` yet, so the plotter does not take it.
- **DY sample.** Every DY sample is a file in `--dir`: the default (LO HT-binned; since 2026-10-12
  not reweighted, at its NNLO-normalized cross section) is `WRAnalyzer_DYJets.root`, any other
  `WRAnalyzer_DYJets_<TAG>.root`, written there by `bin/run_analysis.py <era> DYJets --dy <TAG>
  --dir <dir>` (`lo_inc`, `nlo_inc`, `nlo_mll`, `nlo_njet`, `nlo_ptll`) or `--dy-reweight zpt` /
  `zpt+reshape` (`lo_ht_zpt`, `lo_ht_zpt_reshape`: the Z-pT reweight, and the jet-pT reshape too).
  `--dy TAG=DIR` reads it from another directory. Sets made before 2026-10-12 kept each variant as
  `WRAnalyzer_DYJets.root` in `<dir>_<TAG>_dy`, which is read when the file is not in `--dir`
  (the old tags `lo_ht_zptonly` and `lo_ht_reweighted` still work), and hold the reweighted DY in
  `<dir>` itself: read it with `--dy lo_ht_zpt_reshape=<dir>`. The plotter reads which
  reweighting a DY file carries from its event-weight histograms and refuses a requested tag
  that does not match (for the default sample, when no `--dy` names it, it only warns).
  `nlo_ptll` is not a DY prediction (its slices are added unstitched).
- **Binning.** The plot settings (`data/plot_settings/`) follow the Run 2 analysis: the m(W_R)
  observables use the variable fit-bin edges and are drawn as events per bin (`compare_dy.py`
  uses the uniform bins unless `--run2-binning`). The systematics
  tables use the same binning, so their numbers are the ones the datacards see.
- **Blinding.** Signal-region data appears only from data files the analyzer made with
  `--unblind` (Run 2 only). Files without a `blinding` record in their manifest predate the
  policy and never show signal-region data.
- **Systematic band.** Variations are summed era by era; an era whose file lacks a variation adds
  its nominal (never zero) and is reported. `scale` is the envelope of the μR, μF and μRμF
  variations; `pdf` and `alphas` are formed from the stored PDF members at the plot binning.
- **CMS figure style** (FigGuidelines twiki, PubGuidelines, the CAT plotting pages; 2026-10-12).
  Every plot carries the CMS label inside the frame, top left, with "Work in progress" (the
  thesis-endorsement label) and "Simulation" where no data is drawn; luminosity and energy above the
  frame, right (`plotting_helpers.cms_label`, then `fit_header` last). Data come first in the legend,
  with Garwood (Poisson) errors and horizontal bars only on variable-width bins; the uncertainty band is
  "Sim. stat. (+ syst.) unc." ("MC" is not used as a noun). Backgrounds and signal overlays use the
  CMS (Petroff) palette; the limit plots in `signal_fitting/` use the CAT band colours, #228b22 (68 %)
  and #ffcc00 (95 %). Two things to fix before a plot goes into the AN or the thesis:
  the event-weight (13 pt) and cutflow (14 pt) legends would fall below half the body-text size at
  half a text width, so enlarge them; and the caption of a variable-bin plot (m_lljj, m_lJ in the Run 2
  fit bins) must say that the horizontal bars show the bin widths.

`scripts/plot_jet_veto_maps.py` draws the jet veto maps of the analyzer's `data/jsonpog/` payloads
into `WR_Plotter/plots/jet_veto_maps.png`.

Tests: `cd WR_Plotter && ../.venv/bin/python -m pytest tests -q` (350 tests, offline, about 20 seconds).
