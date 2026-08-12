# Stage 3 — background plots

The plots used to choose and validate a smooth analytic background model before
applying it to the data sidebands. The model targets the **summed** background

    B = DY+jets + tt/tW + nonprompt + other.

Worked per region: **ee / μμ × resolved / boosted**. The mass observable is the
reconstructed W_R mass in each topology (same as the signal study):
resolved → `mass_fourobject` (m_ℓℓjj); boosted → `mass_twoobject` (m_ℓJ) — the
boosted regions have no four-object mass.

Inputs: the LO-DY MC + data under
`rootfiles/Run3/2024/RunIII2024Summer24/20260317_lo_dy/`
(`WRAnalyzer_{DYJets,tt_tW,Nonprompt,Other,EGamma,Muon}.root`). The MC histograms
are already scaled to the era luminosity at production (the stack pipeline applies
no lumi factor; all k-factors are 1.0 for this era), so they are summed directly.

## Setup

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```

## Sub-stages

### `mc_shapes/` — inspect the MC component shapes in the SR (run first)

The first thing to look at before picking a fit function. One script
([`mc_shapes/plot_mc_backgrounds.py`](mc_shapes/plot_mc_backgrounds.py)) makes
three MC-only views per signal region (pick with `--plots`, default all three):

| View (`--plots`) | What it shows | Writes |
|---|---|---|
| `overlay` | total / DY+jets / tt+tW / nonprompt+other as step curves on log y | `mc_shapes/component_overlay/{channel}_{topology}.{png,pdf}` + `all_regions` |
| `stack` | classic CMS stacked, shaded MC (DY/tt+tW/nonprompt/other) + stat band, no data (SR is blinded) | `mc_shapes/stack/{channel}_{topology}.{png,pdf}` + `all_regions` |
| `individual` | one filled plot per single process (just DYJets, just tt+tW, ...) + stat band | `mc_shapes/individual/{process}/{channel}_{topology}.{png,pdf}` |

Also writes `mc_shapes/component_yields.csv` (integrals per component).

```bash
cd signal_fitting/3_background_plots/mc_shapes
python plot_mc_backgrounds.py -v                 # all three views
python plot_mc_backgrounds.py --plots stack -v   # just the stacked plots
```

Defaults: `--era RunIII2024Summer24 --dir 20260317_lo_dy --bin-width 100`,
x-range `[800, 5000]` resolved / `[800, 3500]` boosted (the SR mass threshold up;
override with `--xmin/--xmax`). CMS-styled (`hep.style.use("CMS")`,
`hep.cms.label`); process colors, labels, and stack order come from
`data/sample_groups.yaml`, and the stacking/shading recipe mirrors
`wrplotter.plotting_helpers.plot_stack`.

**What it shows.** tt+tW dominates in every SR; DY+jets is the next-largest and
the steepest; nonprompt+other is small (largest in the boosted ee channel). The
spectra fall smoothly out to ~3 TeV (resolved) / ~2 TeV (boosted), beyond which
the MC runs out of statistics — that smooth region is the fit range to target.

### `cr_plots/` — data/MC validation in the control regions

Where the background model is checked against data: the DY CR (Z-peak enriched)
and the flavor e-μ CR (tt/tW enriched). One script
([`cr_plots/plot_control_regions.py`](cr_plots/plot_control_regions.py)) makes the
standard CMS plot per CR — stacked shaded MC + data points + a Data/Sim. ratio
panel — for the mass observable. CRs are unblinded, so data is shown.

| Output | Regions |
|---|---|
| `cr_plots/dy_cr/{region}_{dataset}.{png,pdf}` | `wr_{ee,mumu}_{resolved,boosted}_dy_cr` |
| `cr_plots/flavor_cr/{region}_{dataset}.{png,pdf}` | `wr_resolved_flavor_cr`, `wr_{emu,mue}_boosted_flavor_cr` |

The flavor CR is an e-μ region recorded in both the EGamma and Muon datasets, so
it is plotted once per dataset (same MC, different overlaid data) — hence the
`_{egamma,muon}` suffix.

```bash
cd signal_fitting/3_background_plots/cr_plots
python plot_control_regions.py -v
```

This reuses the analysis stack machinery (`load_and_rebin`, the region / variable
/ plot-config / sample-group loaders) so binning (`data/plot_settings`), colors,
ordering, and dataset handling all match `bin/make_stackplots.py`. The MC band is
statistical only (no systematics collected here). The data/MC/ratio drawing is a
small local function (`plot_data_mc`) rather than `plotting_helpers.plot_stack`,
because that function's `hep.histplot(histtype="band")` band is unavailable in
LCG_106's mplhep; the band is drawn with `ax.stairs` instead.

**What it shows.** The flavor CR (tt/tW) ratio sits near 1. The DY CR ratio runs
~1.2–1.3 high at low mass — expected for this **LO** DY sample (it needs the
NLO/HT k-factor), which is exactly the kind of data/MC behavior these plots are
here to expose.
