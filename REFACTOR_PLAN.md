# Plotter refactor + analyzer blinding — handoff (2026-10-08)

**Status (2026-10-08, end of day): code, tests, docs and smoke validation are done in both repos; nothing committed.** Remaining items are in section 8.

Work across two repos, **nothing committed** in either:
- plotter: `~/work/WrCoffea/WR_Plotter` (branch `billyDev`)
- analyzer: `~/work/WrCoffea` (branch `billy_dev`)

The analyzer tree also holds ~138 files of the user's own unrelated uncommitted work
(skim code, data configs, `bin/run_analysis.py`, `tests/test_save_hists.py`, ...).
Never revert, stash or reset it. Nothing committed may mention the AI assistant (messages, content, file names); local hooks refuse it.

Python: run both test suites with the analyzer venv:
```
cd ~/work/WrCoffea            && .venv/bin/python -m pytest tests -q
cd ~/work/WrCoffea/WR_Plotter && ../.venv/bin/python -m pytest tests -q
```
`WR_Plotter/.venv` also has pytest; either venv works.

---

## 1. What the user asked for

Work on the 2024 reskim (`rootfiles/Run3/2024/RunIII2024Summer24/20261008_skims_v2`):

1. Cutflow tables, weighted and unweighted, with signal-region (SR) data blinded.
2. Stack plots with SR data blinded.
3. DY sample comparisons, both as overlays of the DY samples and inside the stack plots.
4. Every 2024 systematic. **Deferred**: nominal only for now; systematics come after a rerun with `--systs`.
5. (added later) Event-weight distribution plots.
6. (added later) Cutflow plots and efficiency plots, as well as tables.
7. (added later) **Do the blinding in the analyzer**, not the plotter.

### Decisions already made (don't re-ask)
- **Cutflow blinding:** data in ee and μμ is blinded from the m(ll) > 200 cut onward.
  - Resolved: rows `mll_gt200` and `mll_gt400`.
  - Boosted: rows `mll_gt200_boosted` and `mlj_gt800_boosted`.
  - em is a flavor CR and is never blinded.
  - The rule applies to cumulative and onecut, weighted and unweighted alike.
- **DY in stack plots: "Both".**
  - A full stack-plot set per DY sample.
  - A data vs total-background plot with one line per DY sample (CRs only; SRs compare the totals to each other).
  - DY-only overlays as well.
- **The analyzer owns blinding.**
  - Data signal regions are never written.
  - Data ee/μμ cutflows are truncated.
  - Run 3 cannot be unblinded; `--unblind` is allowed for Run 2 only.
- **Event weights: "All"** (answered 2026-10-08). Both readings:
  - the total: per region and per MC group, log₁₀|w| of the per-region nominal weight, w>0 and w<0 kept separately; legend shows N, N_eff and the negative fraction;
  - the components: per region and per MC group, one distribution per factor of that weight
    (genWeight, σ·L/Σw, pileup, DY Z-pT, signal k-factor, each lepton SF, LSF SF, DY reshape, and the product of the correction factors).
- **`bin/compare_dy_cr_data.py`: delete it** (answered 2026-10-08); `compare_dy.py` replaces it.
- **2018 `--unblind` reruns: yes, but not now** (answered 2026-10-08). The user is reskimming 2018; rerun the Run 2 workflows that need observed SR data with `--unblind` after that reskim.

---

## 2. Facts established (don't re-derive)

**Output files.** The analyzer writes `rootfiles/<Run>/<Year>/<Era>/<dir>/WRAnalyzer_<group>.root`.
- Groups: DYJets, tt_tW, Nonprompt, Other, EGamma, Muon, `signal_WR<m>_N<m>`.
- MC is already scaled to lumi × xsec.
- A sidecar `WRAnalyzer_<group>_systematics.json` sits next to each file (the "manifest").

**Region → data stream.** Both data files fill every region, with no overlap removal. Use exactly one stream per region:
- EGamma: ee regions, `wr_emu_boosted_flavor_cr`.
- Muon: μμ regions, `wr_resolved_flavor_cr` (its eμ trigger is the muon trigger), `wr_mue_boosted_flavor_cr`.

The old plotter drew three wrong-stream flavor-CR plots; that is fixed now.

**Regions.**

| Region | Selection |
|---|---|
| resolved DY CR | 60 < mll < 150, mlljj > 800 |
| resolved SR / flavor CR | mll > 400, mlljj > 800 |
| boosted SR / flavor CRs | mll > 200, mlJ > 800 |

No 200–400 sideband region exists; it is visible only in the cutflow.

**Cutflows.** Stored at `cutflow_{resolved,boosted}/{ee,mumu,em}/{cumulative,onecut}[_unweighted]`, with bin labels = cut names.
- `onecut` means that cut applied alone (not N−1).
- The last cumulative ee/μμ row is the SR. The last em row is the flavor CR (for boosted, only `wr_emu_boosted_flavor_cr`).
- Weighted MC bins exclude lepton SFs, so they sit a few % above the region histograms.
- The data column comes from the channel's trigger stream:

  | | ee | μμ | em |
  |---|---|---|---|
  | resolved | EGamma | Muon | Muon |
  | boosted | EGamma | Muon | EGamma |

**DY samples.**
- Analyzer `--dy` tags: lo_ht (the default for every era), lo_inc, nlo_inc, nlo_mll, nlo_njet, nlo_ptll.
- The DY file is always `WRAnalyzer_DYJets.root`, so a variant is identified only by its directory.
- Current 2024 directories:

  | Directory | DY sample |
  |---|---|
  | `20261008_skims_v2` | lo_ht (`DYto2L-4Jets_Bin-HT-*`) |
  | `20261008_skims_v2_lo_inc_dy` | lo_inc |

  NLO has not been rerun on skims_v2.

**State of `20261008_skims_v2`.**
- Nominal only: no `syst_*` folders. `20261004_full_systs` is the newest set with systematics.
- Its data predates analyzer blinding: the SRs are filled and the cutflows are full.

**Systematics in 2024.**
- Produced: lumi (normalization only), pileup, the lepton SF family, jer, jes (Total), elescale, elesmear, scale (MuR/MuF/MuRF), ps (ISR/FSR).
- Placeholders: muscale; musmear is identical to nominal.
- Not available: PDF.
- ROOT naming: `syst_<label lower-case>_<region>/<var>_syst_<...>_<region>`. A (region, label) pair is written only where it was produced; regions it can't affect are listed as "unaffected" in the manifest.

---

## 3. Done — analyzer (WrCoffea, uncommitted)

Written by a subagent that was stopped before it got to tests and docs. The code is complete.

- **`wrcoffea/regions.py`**
  - Blinding policy:
    - `SIGNAL_REGIONS`.
    - `is_signal_region()`, which covers variants such as `_sr_nomass`.
    - `BLINDED_CUTFLOW_STEPS`.
    - `blind_cutflow_steps()`, which raises if the stop step is missing.
    - `UNBLINDABLE_RUNS = {"RunII"}` and `check_unblind_allowed(era)`.
    - `blinded_region_names()`.
  - `region_names(..., blind=)` and `build_regions(..., blind=)` drop the SRs.
- **`wrcoffea/systematics.py`:** `production_config(..., blind=)` records `"blind"`. `parse_production_config` defaults it to False for legacy payloads.
- **`wrcoffea/analyzer.py`**
  - `WrAnalysis(..., unblind=False)`. For data, `blind = self._blind and not is_mc`, so no SR regions are built (and so no event lists) and cutflows are truncated.
  - Calls `check_unblind_allowed` for unblinded data.
  - `event_weights` is filled only for MC, only in the nominal pass, and not in `compute_sumw` (`--unskimmed`) mode.
- **`wrcoffea/histograms.py`**
  - `fill_event_weights()`: `Regular(200, -8, 2)` axis `log10_abs_weight`, unit-weight fills, split by sign.
  - `fill_cutflows` / `fill_boosted_cutflows` take `blind=`.
- **`wrcoffea/save_hists.py`**
  - `DatasetCoverage.blinded` and `expected_regions()` exclude SRs for blinded data. Blinded data that filled an SR raises.
  - `blinding_record()` feeds the manifest's `"blinding"`.
  - `event_weights` is written like the cutflows.
- **`bin/run_analysis.py`**
  - `--unblind` flag, checked via `check_unblind_allowed`, with a warning when used.
  - `normalize_by_sumw` skips `event_weights`.

**Contract the plotter relies on:**
- ROOT: `event_weights/<region>/positive` and `event_weights/<region>/negative`, TH1D over log₁₀|w|, MC only.
- ROOT: `event_weights/<region>/components/<name>` (MC only, unit-weight counts):
  `genweight_positive` / `genweight_negative` (log₁₀|genWeight|, `Regular(400, -12, 8)`),
  `normalization` (log₁₀ of xsec·lumi·1000/sumw, same axis), then one linear-axis
  (`Regular(600, 0, 3)`, axis `factor`) histogram per correction factor: `pileup`, `dy_zpt`,
  `signal_kfactor`, `dy_reshape`, the lepton-SF labels (`MuonRecoSf`, ..., `LsfSf`), and
  `corrections` = their product. genweight × normalization × corrections = the total.
- Blinded data files have no `wr_*_sr*` folders. Their ee/μμ cutflows stop before `mll_gt200` / `mll_gt200_boosted`; the axis is shorter, not zero-filled.
- Manifest top-level key:
  ```
  "blinding": {"applied": bool,
               "signal_regions": [...],
               "cutflow_truncated_before": {"cutflow_resolved": {"ee": "mll_gt200", "mumu": "mll_gt200"},
                                            "cutflow_boosted": {"ee": "mll_gt200_boosted", "mumu": "mll_gt200_boosted"}}}
  ```
  `applied` is true only for data of a blinded run.

## 4. Done — analyzer follow-up (2026-10-08)

- **Weight components** (user answer "All"):
  - `build_event_weights` keeps the individual factors (`Weights(n, storeIndividual=True)`); its arithmetic is unchanged.
  - `_event_weight_factors()` gives genweight, normalization and every other `Weights` entry.
  - `_lepton_sfs_for_region(..., factors=)` and `_dy_reshape_for_region(..., factors=)` record the per-region factors.
  - `fill_event_weights(..., factors)` fills `components/` (`histograms.py`: `WEIGHT_COMPONENT_*`, `CORRECTIONS_COMPONENT`).
- **Tests:** 675 pass.
  - New `tests/test_event_weight_hists.py` covers the components and checks per event and per region that their product equals the filled weight.
  - The 19 blinding failures are fixed, and the blinding tests are added in `test_regions.py`, `test_systematics.py::TestBlinding` and `test_run_analysis.py`.
  - The cutflow-sequence test no longer imports plotter table scripts.
- **Docs:** `docs/run_analysis.md` (new Blinding section, output contents) and the README output/flags sections.
- **Smoke run** (`20261008_smoke_blinding`, 1 file × 1 chunk per dataset: EGamma, Muon, DYJets, tt_tW, WR4000_N2100, WR4000_N100; plus `20261008_smoke_blinding_lo_inc_dy`). It confirmed:
  - no SR folders in data;
  - data ee/μμ cutflows truncated, em full;
  - the manifest `blinding` record;
  - `event_weights/` with components in MC, each component's entry count equal to the total's.

  MC normalization in these sets is meaningless (one file each, full Σw), so ignore data/MC there.

## 5. Done — plotter (WR_Plotter, uncommitted)

**New config**
- `data/dy_samples.yaml`
  - Lists each DY tag with its label, colour and a `datasets` regex; the default is lo_ht.
  - A non-default sample is read from `<dir>_<tag>_dy`, or from DIR with `--dy TAG=DIR`.
  - A directory whose manifest datasets match a different tag is refused.
- `data/cutflows.yaml`: the data stream per (topology, channel), and per-cut `table` (LaTeX) / `plot` (mathtext) labels.

**New modules**
- `wrplotter/inputs.py`
  - `Inputs` and `DYSample`; `make_inputs(era, dir, dy_specs)` returns one `Inputs` per DY sample.
  - `read_manifest`, `blinding_recorded`, `sr_data_unblinded`, `detect_dy_tag`.
- `wrplotter/systematics.py`
  - Moved out of `make_stackplots`: `SYST_ENVELOPES`, `sample_variations`, `total_variations`, `band_error2`.
  - Detects production from the `syst_*` folders. A nominal-only file gives no band and no warnings. A source missing in one region means "unaffected" there, and the nominal is used.
- `wrplotter/stack.py`
  - `Binning`, `AxisConfig` / `axis_config(vcfg)`, `HistLoader`, `StackHists`, `build_stack()`, `order_for_stack`, `hist_yield`.
  - SR data is read only if the manifest says `blinding.applied == false`; otherwise the plot is marked blinded.
- `wrplotter/cutflow.py`
  - `Cutflow`, `read_cutflow` (labels taken from the file), `CutflowTable`, `build_table()`, `efficiency()`, `format_text()`, `format_latex()`.
  - The data column is aligned by cut name; NaN means blinded. A legacy data file (no blinding record) gives no data column, with a note.
  - Already sanity-run on the real skims_v2 files, and it works.
- `wrplotter/cutflow_plots.py`
  - `plot_yields`: weighted cutflows as a stack + data + ratio; unweighted as per-group lines.
  - `plot_efficiency`: running, relative or single-cut efficiency.
  - Blinded rows are shaded.
- `wrplotter/dy_compare.py`: `plot_dy_overlay` (absolute or normalized, with ratio to the reference) and `plot_total_comparison` (data vs the total with each DY sample).
- `wrplotter/event_weights.py`: `read_distribution` (N_eff from `mass_dilepton` sums) and `plot_weight_distributions`.

**Changed**
- `wrplotter/regions.py`
  - `Region(name, primary_dataset, tlatex_alias)`: the `unblind_data` field is removed.
  - One stream per region (11 regions).
  - `is_signal_region` uses the regex `_sr(_|$)`.
  - New `topology` property; requests are de-duplicated.
- `wrplotter/plotting_helpers.py`
  - `plot_stack(..., syst_variations={syst: {"up": Hist, "down": Hist}})` replaces the nested `syst_hists`.
  - The CMS label always uses `data=True`.
  - New `cms_label`, `region_text`, `signal_label`, `log_ylim`.
- `wrplotter/config.py`: `load_dy_samples()`, `load_cutflow_config()`. Their caches are cleared in `tests/conftest.py`.
- `wrplotter/histogram_cache.py`
  - `sublumis` is optional; it was unused.
  - The k-factor era is now found under `--dir` too (it checks the parent directory).
- `wrplotter/cli_utils.py`: `add_dy_arg`, `add_selection_args`, `add_output_args`, `select_regions`, `select_variables`, `dy_specs`.
- `wrplotter/paths.py`: `plots_base(run, year, era, subdir, local=)`.
- `bin/make_stackplots.py`, rewritten:
  - `--dy` (repeatable) makes one full set per DY sample, written to `<base>/stack_dy_<tag>/<region>/<var>_<region>.pdf`.
  - Removed: `--unblind` (the analyzer decides) and `--name` (it was unused).
  - Added: `--no-syst`.
  - **Written but never run yet.**

## 6. Done — plotter follow-up (2026-10-08)

- `bin/make_cutflows.py` (renamed from `make_cutflow_tables.py` with `git mv`, so the rename is staged), `bin/compare_dy.py` (rewritten, adds `total_scaled/`), `bin/plot_event_weights.py` (total, components, text summary under `tables/.../event_weights/`, DY-only overlays with ≥2 `--dy`).
- `bin/compare_dy_cr_data.py` is deleted, as the user approved.
- `wrplotter/event_weights.py`: `read_components`, `component_stats`, `plot_component`, `format_summary`, `component_label`.
- Small library fixes:
  - `cutflow_plots`: the efficiency floor is now 1e-7, and the ratio panel says "No data shown".
  - `dy_compare`: Data/Bkg. errors come from the data variances; SRs get the "Data blinded" corner label.
  - `cutflow.table_variant`.
- **Tests:** 246 pass. The 8 old-API failures are fixed. New: `test_inputs`, `test_stack`, `test_systematics`, `test_cutflow`, `test_event_weights`.
- **README:** new scripts, the `--dy` convention, the output layout, blinding in the analyzer, and the removed options.
- **Validated:**
  - All four scripts ran on `20261008_skims_v2` (legacy data: no cutflow data column, with a note; SR plots blinded).
  - They also ran on the smoke set with lo_ht and lo_inc: a cutflow data column with blinded rows, event-weight and component plots, and DY overlays.
  - PDFs were inspected.

## 7. Open questions for the user
None open; the three earlier questions are answered in section 1.

## 8. Remaining / for the user
- **Rerun the 2024 data with the new analyzer**, so cutflows get a data column, and the MC as well, so the event-weight plots exist for `20261008_skims_v2`.
- **After the 2018 reskim**, rerun Run 2 data with `--unblind` where observed SR data is needed. Analyzer-output readers of 2018 data SRs (all existing 2018 sets are unblinded today, so nothing breaks yet):
  - `signal_fitting/run2_results/compare_legacy_new.py`: on blinded data its SR yields go silently NaN, and `cutflow_table` raises at `labels.index('mll_gt200')`.
  - `signal_fitting/8_combine_limits/baseline/prepare_inputs.py --with-data --era RunIISummer20UL18`: raises on blinded data. It opens the output with RECREATE first, so the crash wipes the committed `run2/inputs/<tag>.root`.
- **Physics notes from the validation:**
  - **Muon reco SF on the WR4000_N2100 signal:** in the μμ resolved SR it averages 0.918 per event (DY 0.957, tt 0.974; about 40% of signal events in the 0.895–0.90 bin, i.e. about 0.947 per TeV muon). This explains why the μμ signal cutflows run about 12% above the SR yields.
  - **Resolved DY CR:** data/bkg is about 1.25–1.3 with LO DY, for both lo_ht and lo_inc (boosted DY CR about 1.0).
- **Done 2026-10-08 (second pass, user request):**
  - Plot ranges, all eras: DY-CR `mass_dilepton` is now 60–150 GeV (was 60–120); boosted-SR `mass_twoobject` is now 800–8000 GeV (was 800–4000).
  - `data/kfactors.yaml` is removed, with all its plumbing:
    - `load_kfactors` and `get_kfactor`;
    - `load_and_rebin`'s `era_for_scale` / `get_kfactor_fn` / `scales`;
    - `HistLoader` / `build_stack` `scales`;
    - the three `signal_fitting` callers (`plot_control_regions.py`, `explain_fcr.py`, `plot_window_stack.py`);
    - the tests and the README.
    Every factor in it was 1.0, so no plot changes. LO DY carries its k-factor in the analyzer configs (`xsec_source.kfactor` = 1.22289).
  - Smoke outputs deleted.
- **Bug in the user's uncommitted code (not fixed):** `signal_fitting/8_combine_limits/optimization/explain_fcr.py:74` sets `INPUTS = INPUTS` on the default 2018 branch, so the script (and `plot_window_stack.py`, which imports it) raises `NameError` unless `WR_SEMINAR_ERA=2024`.
- **Done 2026-10-08 (third pass, user request):**
  - Removed `bin/signal_closure.py` and `scripts/validate_skims.py` (`git rm`). Untracked `tables/` (`git rm -r --cached`; files kept, `.gitignore` already covers it).
  - Era combinations:
    - `data/lumi.yaml` gains `2024` (= RunIII2024Summer24) and `Run3` (all five eras, label "Run 3", year "2022-2024").
    - A combination lists only `sub_eras`; `load_lumi` derives `sub_lumis` and `lumi` (Run3 = 172.685 fb⁻¹), checks any written values, and refuses nested combinations and mixed √s.
    - `plot_settings:` borrows another era's file.
    - `config.era_label` sets the plot label.
    - `Inputs.files_for` warns once when a sample exists in only some eras.
    - `HistLoader` goes through `files_for`.
  - Verified: Run3 = sum of the five eras exactly (in-range bins; the loader's overflow marker is 1.0 per load).
  - 249 plotter tests pass.
- **Done 2026-10-08 (fourth pass, approved refactors; script rewrites by helpers in progress):**
  - Overflow:
    - `rebin_histogram` folds overflow into the last bin as before, but flags it with the `overflow_hint` attribute instead of writing a fake 1.0 into the overflow bin.
    - `histogram_utils.add_hists` / `sum_hists` keep the flag.
    - `for_drawing(h)` puts the arrow marker only on the copy being drawn (plot_stack and signal_fitting's plot_control_regions).
    - Sums with `flow=True` are exact now.
  - Cleanups:
    - removed `extract_hist_data`, `load_sample_groups_raw`, `load_and_rebin(sublumis=)`, `paths.output_dir` (folded into `plots_base`) and `pytest.ini` (pyproject has the same settings);
    - `SIGNAL_COLORS` / `SIGNAL_STYLES` are now public;
    - the package `__init__` is a docstring only.
  - Plot settings:
    - `data/plot_settings/default.yaml` plus override-only files `run3_subera.yaml`, `RunIII2024Summer24.yaml`, `RunIISummer20UL18.yaml` (`inherit: default`);
    - resolution: own file → `plot_settings` in lumi.yaml → default;
    - 2066 → 812 lines; every era's effective settings verified identical.
  - Rule:
    - assistant-file ignore entries moved out of both `.gitignore` files into `.git/info/exclude`, and `wrplotter.egg-info/` and `outputs/` added to the ignores;
    - local `pre-commit` and `commit-msg` hooks in all five workspace repos refuse any mention;
    - the rule is in each workspace project's memory;
    - pushed history still has Co-Authored-By trailers (needs the user's decision).
- **Done 2026-10-08 (script rewrites):**
  - `bin/make_signal_cutflows.py`:
    - 803 → 282 lines on `wrplotter/cutflow.py`; cut names come from the files;
    - modes fixed-wr / fixed-n (newly implemented) / wr-compare / channel-compare;
    - yields bit-identical to the old script;
    - the analyzer test now checks `wrplotter.cutflow.read_cutflow` reads back the analyzer chains.
  - Library additions:
    - `inputs.make_inputs(..., with_dy=False)` for signal-only or tt-only sets, plus `Inputs.signal_points()`;
    - `cutflow.signal_cutflows()` and `write_preview` / `compile_preview` (now also used by `make_cutflows.py`, output byte-identical);
    - `latex_count` uses scientific notation below 0.01.
  - `bin/emu_transfer_factor.py`:
    - loads through `make_inputs(..., with_dy=False)` + `load_and_rebin` per era file; `--era` is required and takes years and Run3;
    - plots go to `plots_base` (EOS unless `--local-plots`);
    - 2024 fits and tables bit-identical to the old script.
    - Note: its constant chi2 fit is biased low at low statistics (per-bin errors scale with the ratio).
  - Tests: analyzer 681, plotter 285 pass.
- **Resolved DY CR study:** done (NLO inclusive and jet-binned DY on skims_v2 in `20261008_skims_v2_nlo_{inc,njet}_dy`).
  - LO needs ×1.23 (ee) / ×1.27 (μμ) on top of the 1.223 k-factor, falling with Z pT and leading-jet pT.
  - The same holds in every Run 3 era (data/bkg 1.22–1.27).
  - NLO matches the data in the resolved DY CR (inclusive 0.96 / 1.00, jet-binned 0.99 / 1.03), but its effective SR event counts are 13–70× below LO HT, so NLO stays a cross-check.
  - AN-19-083 handled this with a Z-pT correction plus a data-driven jet-pT reshape and per-year DY rate parameters.
- **Done 2026-10-08 (fifth pass, stale docs):**
  - WrCoffea `README.md`: all 14 `--systs` families; layout without the missing `test/`, with `combine_framework/` and `studies/`; complete `bin/`, `wrcoffea/` and docs lists; every `skim.py` subcommand.
  - `docs/run_analysis.md` (era list, `--systs`, `--study dy-cr-split`, where `skipbadfiles` is set), `docs/adding_a_new_era.md` (both `ERA_SUBDIRS` maps, the lumi.yaml and plot-settings step), `docs/skimming.md` (dropped the contradictory "LPC Condor not wired yet" section), `docs/filesets.md` and the `scripts/skimmed_fileset.py` header (output paths).
  - `docs/repo_review.md`: status table as of 2026-10-08 (R5 closed; R2, R4, R7, R8 partly closed; R1, R3, R6 open).
  - WR_Plotter: README `pytest.ini` line; removed the unused `data/sample_groups/2022.yaml`; `signal_fitting/README.md` rewritten to map every stage.
  - The Mac combine pattern for wrappers without `EXTRA_BINDS` (`orb -m combine bash -c 'cd $PWD && apptainer exec -B /cvmfs -B /Users/billy/work/WrCoffea <image> bash run_fcr.sh'`) was checked up to the wrapper's container guard (`APPTAINER_CONTAINER` set, repo and cvmfs visible); combine itself was not run.
- **signal_fitting stage docs, stale and not fixed** (deferred while signal_fitting is in active use; found by a 2026-10-08 read-through, re-check each when fixing):
  - `7_limit_plots/README.md`: says `--center mean` is the default (the code default is `zero`); its paths lack the `run2/`/`run3/` level; it refers to the archived Stage 8 injection.
  - `8_combine_limits/explainer/README.md`: its step-6 script names do not exist (on disk: `step6a_data_and_model` … `step6h_the_question`); `step7_limit/` and the new step1d, 1e, 2d and 3d–3g are undocumented. `LIMITS.md` calls it a 29-script walkthrough; there are 48.
  - `8_combine_limits` wrappers: no README has a Mac recipe; only `run_refined.sh` reads `EXTRA_BINDS`; `run_mc.sh` never re-execs into the container.
  - `8_combine_limits/optimization/README.md`: misses the 2024 seminar chain (`WR_SEMINAR_ERA`, `inputs_2024*` / `cards_2024*` / `results_2024*`, `explain_fcr*.py`, `plot_fcr_limit.py`, `make_mc_cards.py` with `run_mc.sh`, `signal_systs.py`).
  - `6_spurious_signal_toys/README.md`: lists only `card_float` and `card_fcr` (the code also has `card_fcrs`, `card_fcrd` via `--cr-obs`, and `--cr-shape`); refers to the archived `8_signal_injection_study`; `single_toy_diagnostic.py` and `plot_card_spurious_vs_mass.py` are undocumented.
  - `5_spurious_signal/README.md`: the same archived reference; paths lack `run2/`/`run3/`; `compare_k.py`, `compare_binwidth_k.py` and `rebinning_study.py` are not described.
  - `3_background_plots` and `4_background_fits` READMEs: paths lack `run2/`/`run3/`; Stage 4 misses `make_check_table.py`, `fit_checks/` and the k sub-folders; the smoothing README's file list omits `dy_statistics.py`, `explain_kde_setting.py` and `plot_diagnostics.py`.
  - `8_combine_limits/baseline/README.md`: old numbering ("Stage 9", production "Stage 10.8", also in the script headers); "step 5 in the table above", but the table stops at step 4; its examples run the run3 default while config B's quoted table is run2.
  - Docstrings: `detail_gauss_fit.py` and `detail_fwhm.py` cite `outputs/width_estimators/...` and `signal_fitting/1_signal_samples/...`; `shared/measure_fwhm.py` cites removed scripts; `shared/sb_fit.py` calls Stage 6 "signal-injection".
  - `validation_plan.md` and `progress_slide.tex` use the old stage names (marked as dated in the top-level README).
- **Later:** systematics band check after a `--systs` rerun (old section 6 item 7).

## 9. Review 2026-10-09
Two read-only reviews plus profiling of `make_stackplots.py` on the 2024 sets. Line numbers as of 2026-10-09 (before the first batch below). [SF] = touches `signal_fitting/` behaviour.

**Done 2026-10-09 (first batch, uncommitted):**
- **Era combinations (item 1):** `systematics.sample_variations` takes per-file nominals and sums each variation file by file, a file without it adding its own nominal. Files lacking sources that the sample's other eras have are reported once. On the review's mixed pair (2024 systs + 2022EE nominal) the μμ resolved DY CR band is 20–31 % (was 83–107 %), the ee resolved SR 6–30 % (was 74–104 %).
- **Silent drops (items 2 and 3, without the UL18 list):**
  - `histogram_cache.read_hist` / `has_key` / `keys_in_file`: unreadable files raise, and a key missing from some eras' files is reported (`report_partial`).
  - This is used by `load_and_rebin`, the stack, the cutflow and event-weight readers, and `_sources_in_file`.
  - The stack reports an MC sample with no file, and one without variations while others have them.
- **`--per-gev` ratio errors (item 4):** `plotting_helpers.data_ratio` uses the data variances. Default (events per bin) plots are unchanged.
- **Speed:** every read checks the cached key list first. The full 2024 `20261004_full_systs` set takes 36 s (about 11 min before).
- **Boosted plot list:** `config.get_var_cfg` falls back to `common_variables` for resolved regions only, so 200 plots are planned and 200 saved (was 284 planned, 84 skipped).
- **Tests:** `tests/test_regions.py` second class renamed (6 tests run again); new tests for each fix; 335 pass.
- **Checked on real files:**
  - every plot of the 2024 systematics set equals the old single-era logic (200 plots, 7600 variation arrays, identical);
  - stack plots for 2024 and Run3, cutflows (Run3), event weights, `compare_dy` and `emu_transfer_factor` all run cleanly.
- **Still open from this review:** the UL18 sources in `data/systematics.yaml` [SF], items 5–8, the rest of the performance list (parallel rendering, multi-`--dy` reloads, rebin loop, EOS mkdir), the cleanup list and the display choices.

**Wrong or misleading numbers**
1. **Era combinations with systematics:** a variation missing in one era counts as zero, not as that era's nominal. `systematics.py:93,104-108` substitutes the nominal only when every era lacks the key; `stack.HistLoader` sums over the eras that have it.
   - Measured on a 2024-systs + 2022EE-nominal pair: band 83–107 % per bin instead of 20–31 % (μμ resolved DY CR, m_lljj).
   - Triggers whenever eras differ in `--systs`/`--omit-systs`; today only 2024 has a systs set.
   - Fix: per-era variations with that era's nominal substituted; flag eras whose manifest lacks the source.
2. **Band completeness is never checked against the manifests.** A stack sample without variations contributes its nominal (`total_variations`).
   - DY from a nominal-only directory plus the rest from a systs set: the DY CR band falls from 25–43 % to about 1 %, and the legend still says "stat. + syst.".
   - `data/systematics.yaml` lacks the UL18-only sources (`lsfsf`, `dyzptqcdscale/pdf/alphas`, `dyzptew1-3`, `dyreshapestat/channel`), which `systematics.py:97-100` drops silently.
   - The folder regex (`systematics.py:38`) cannot match labels with `_` (split JES; latent).
   - [SF] `shared/syst_band.py`, `make_mc_cards.py`, `signal_systs.py` and `prepare_fcr.py` read the same list.
3. **Silent drops.** A missing key or an unreadable file in one era is skipped at DEBUG level:
   - `histogram_cache.py:44-49`, `cutflow.py:75-78,91-95`, `event_weights.py:75-79,213-216`;
   - an unreadable file makes the band stat-only (`systematics.py:43-46`).
   - Fix: one `read_hist` helper (raise on unreadable, warn on partial presence) used by the four readers.
4. **`--per-gev` ratio-panel data errors** use `sqrt(values)` of the per-GeV contents (`plotting_helpers.py:204`): too large by √(bin width), ×14–77.
   - Fix: use `sqrt(variances)`.
   - The plot-setting ylims are not adapted for `--per-gev` either.
5. **`emu_transfer_factor.py --input WRAnalyzer_EGamma.root` reads SR data with no blinding check** (`:130-144`, `:184-212`).
   - Live today: no data file has a blinding record, and all have filled SRs.
6. **Transfer-factor fit** (`chi2_fit_constant`; errors are proportional to the observed ratio, `:242-250`):
   - 2024 boosted ee: −3.6 %, with one N_eff = 1 bin carrying χ² 266 of 307.
   - Toys: −1.7 % resolved-like, −4.0 % boosted-like. The ratio of integrals is unbiased.
   - The quoted error ignores χ²/ndf (16.7/8 and 27.7/8 with the Run 2 edges).
   - The default binning is the uniform rebin, not the Run 2 edges (`rebin_spec`, `:623-639`).
   - Fix: ΣSR/ΣCR as the central value, Run 2 edges by default, error scaled by √(χ²/ndf).
7. **Unweighted cutflow tables print Data/Bkg. against raw MC counts** (`cutflow.py:354,153,381`; `cutflow_plots.py:102-106`). Hidden until the data is rerun with the blinding analyzer.
8. **The DY-sample guard** (`inputs.py:118-127,154-160`) matches only 2024 dataset names; every other era passes unchecked.

**Performance** (measured 2026-10-09, laptop under load)
- **Systematics sets:**
  - Looking up a missing folder costs ~46 ms (uproot lists the whole 371-folder file for its error message); a hit costs 0.06 ms.
  - Full 2024 `20261004_full_systs` set: ~3 s per plot, about 11 min.
  - With a cached per-file key set checked before `f[key]`: 47 s. All arrays were identical on the 6 plots checked.
- **Rendering** is the rest: the nominal 2024 set takes 31 s serial and 6 s with 11 processes; the Run3 combination takes 36 s serial. Parallel workers in `make_stackplots.py` would give about 5×.
- **84 of 284 planned plots cannot exist:** the resolved-only `common_variables` (jet, dijet, 3- and 4-object) apply to boosted regions (`config.py:180-186`). Fix: key the block by topology.
- **Smaller:**
  - multi-`--dy` runs re-read every non-DY sample per DY sample;
  - the `rebin_histogram` per-bin loop takes 1.2 ms on 5000 bins, against 0.03 ms with bincount;
  - `save_figure` runs `xrdfs mkdir -p` for every figure.

**Cleanup**
- `tests/test_regions.py:107,136`: two `TestRegionProperties` classes, so 6 tests never run.
- `--verbose` sets the root logger to DEBUG: 16k matplotlib findfont lines per plot.
- **Script scaffolding helpers:** `inputs_or_exit`, `era_info`, `save_and_close`, `tables_base`, `make_parser`, `plot_plan` and shared LaTeX tables.
  - About +70 library and −170 script lines.
  - Also fixes the "Run3" vs "Run 3" labels and the unguarded EOS saves in `emu_transfer_factor.py`.
- **`emu_transfer_factor.py`:** about 300 removable lines:
  - its own PDF preview, without a pdflatex check;
  - four near-duplicate plotting functions;
  - two LaTeX writers;
  - scipy `curve_fit` for a weighted mean.
- **Dead code:**
  - `Region/SampleGroup/Variable.print`;
  - `Variable.nbins/xmin/xmax` plus ~96 lines of `variables.yaml` (wrong anyway);
  - the unused `*_leading_loose_lepton` variables;
  - `Inputs.with_dy`, `CutflowTable.name`;
  - the `default_signals` fallback (Run 3 masses);
  - single-stream leftovers in `regions.py`.
- **CLI:**
  - `--signal WR4000_N2100` silently shows no overlay in `make_stackplots`, which wants `signal_…`;
  - topology flags and `-o` differ between scripts;
  - scripts exit 0 when every plot was skipped.
- **Packaging and docs:**
  - `pyproject.toml` declares coffea (unused) and omits scipy (used).
  - The README still advertises signal closure studies (`:3`), and its test command differs from the one used here (`../.venv/bin/python -m pytest tests -q`).

**Display choices** (not bugs; decide later)
- **Era correlations:** every source, lumi included, is treated as 100 % correlated across eras. Run 2 treated JER, trigger SFs and the LSF SF as uncorrelated. The effect on the band is small: those shifts are ≤ 1–2 %, and lumi is 1.4 % vs ~0.93 %.
- **Symmetric band:** each source uses max(|up|, |down|). For example, the 2024 DY CR scale shift of +24…40 % / −17…25 % is drawn as ±24…40 %.
- **Ratio-panel errors:** the ratio panel uses symmetric √N, so empty data bins show as 0 ± 0 (479 such bins in the 2024 CR plots). The main panel uses Garwood intervals.
- **`compare_dy`:** its totals have no MC stat error (nlo_inc: 1–18 % per bin in the ee DY CR).
- **Cutflows:**
  - Weighted rows exclude the lepton SFs: the last row is 5–10 % above the region yields (2–13 % for signal).
  - Checked 2026-10-09 on all 137 2024 signal points: event counts in the last row equal the SR exactly; with the lepton SFs switched off (one point rerun), the weighted SR yield equals the last row to 1e-14. SR/last-row medians: resolved ee 1.006, μμ 0.894; boosted ee 0.991, μμ 0.875.
  - Signal efficiencies are relative to the post-skim `no_cuts` row (68–81 % of σ×L). Read as A×ε they are 1.23–1.47× too high, and the row label omits the skim.
