# Stage 4 / smoothing — a smooth MC background for the limit inputs

> **Question (Sep 2026).** The high-mass tail of the background MC is jagged:
> single bins jump by factors of several because one MC event with a large
> weight landed there. That event is real, but it is not a *typical* event, and
> the bin content it makes is a poor estimate of the expected yield. Can we
> replace the raw histogram by a smooth expectation, and what does that do to
> the limit?

Everything here is the run2 2018 **ee resolved** SR
(`RunIISummer20UL18/20260714_run2_bkgs`, `wr_ee_resolved_sr/mass_fourobject`,
native 10 GeV bins, MC already lumi-scaled, sumw2 stored). Outputs live in
`run2/ee_resolved/`.

## What the raw MC looks like

| component | Σ events | Σ sumw2 | N_eff = Σ²/Σw² | last populated bin | negative bins |
|---|---|---|---|---|---|
| tt+tW | 650.9 | 65.2 | 6500 | 3985 GeV | 1 |
| DY+jets (LO HT, reshaped) | 361.7 | 197.4 | 663 | 7865 GeV | 8 |
| Nonprompt | 19.0 | 5.7 | 63 | 6315 GeV | 1 |
| Other | 41.4 | 50.4 | 34 | 4225 GeV | 23 |

Relative MC-stat error of the summed 50 GeV bins: ~0.4 at 3–3.5 TeV, ~0.55 at
3.5–4 TeV, ~0.8 above 4 TeV. tt+tW has uniform weights and simply runs out at
4 TeV; DY, Nonprompt and Other are **mixed-weight** samples — a common
population of low-weight events plus rare high-weight ones (DY: ~0.008
events per MC event in the high-HT slices vs ~0.2 for a low-HT-slice event
that leaks to high mass). The rare heavy events *are* the jaggedness.

Two facts that shape everything below:

- **The k5 window sums are already well constrained** (N_eff 48–250; at 3200
  B = 39.4 ± 2.9, at 4600 B = 3.5 ± 0.5). Smoothing mainly changes the *shape
  inside the window* — spikes near the signal peak feed the spurious signal —
  not the window yield.
- **The raw MC enters the limit in three places**: it is `data_obs` of every
  combine card (the float cards' fitted B follows the raw window integral),
  it is the Poisson mean of every toy generator (so ⟨N_sp⟩ inherits the
  template noise), and it is what the production anchor is fit to. Anchored
  cards (1000–1200, ≥ 3400) ignore `data_obs` — only the anchor fit sees it.

## The four smoothers (`smooth_lib.py`)

All work per component on the native 10 GeV grid, force zero below the
800 GeV threshold, and record their diagnostics in `smoothed_<method>.json`.

| method | what it does | free parameters (defaults) | normalization |
|---|---|---|---|
| **kernel** | local-quadratic Gaussian-kernel regression in u = ln m; per target bin the bandwidth is the smallest h with kernel-weighted N_eff ≥ n_target. Kernel weights only (no 1/σ² — Neyman bias). Bins beyond the last MC event stay as zeros → decays with the MC, no extrapolation. | degree 2, h ∈ [0.05, 0.4], n_target 20 | rescaled to the raw total (≤ 1.5 %) |
| **hybrid** | raw MC below a trust mass m_T (first 50 GeV point where the running N_eff over the next 500 GeV drops below 20), an expo tail above, linear blend over the 200 GeV before m_T. | trust N_eff 20 / 500 GeV, fit [m_T−1000, last populated], expo, Poisson-ML | raw below m_T, the fit above (tail not rescaled) |
| **param** | one full-range fit per component, family = best Pearson χ²/ndf among expo / powexp / expo2 that pass `bkg_fit_lib`'s checks. The per-component twin of the production anchor. | [800, last populated], pivot 2000, Poisson-ML | the fit's own (≤ 0.3 % off the raw total) |
| **kde** | `RooKeysPdf` (adaptive KDE) on a weighted `RooDataSet` with one entry per populated bin (weight = content), in m (ln m until Oct 5 2026; the Sep study results below used ln m), mirrored at 800 GeV. | ρ = 1, MirrorLeft, no resampling | shape only: rescaled to the raw total |

Defaults m_T (hybrid): tt+tW 2550, DY 2900, Nonprompt 1400, Other 1200 GeV —
for the two small components "hybrid" is essentially the parametric fit.
Family picked by `param`: expo2 for tt+tW and DY, expo for Nonprompt and Other.

### Which fit statistic — a finding in itself

The tail fits can be done three ways, and on this MC they are *not*
equivalent (`smooth_lib.fit_component`, `stat=`):

| stat | what it is | DY fit/raw | Nonprompt | Other | verdict |
|---|---|---|---|---|---|
| `chi2` | `bkg_fit_lib.fit_model`: χ² with the observed (Neyman) bin errors, populated bins only | 1.00 (expo) / 0.90 (expo2) | 0.51 | 0.43 | **biased low** wherever N_eff per bin is small: bins that fluctuated down get tiny errors and pull the fit down (DY 3200 window: 2.2 vs 5.9 raw) |
| `poisson` | `TH1::Fit "L"`: Poisson likelihood on the (fractional) contents, all bins — the production anchor's convention | 1.00 | 1.00 | 1.00 | unbiased normalization; treats every bin as one count regardless of its N_eff (a single heavy event outvotes twenty light ones). **Default.** |
| `neff` | scaled-Poisson "effective count" likelihood: n_i = v_i/w_i events with expectation f_i/w_i, w_i = var_i/v_i per bin | 0.92 | 0.69 | 0.45 | statistically right for a *single-weight* sample (identical to `poisson` for tt+tW), **biased low on mixed-weight samples**: the rare heavy population is absent from most bins, and those bins carry the power. The deficit measures how much yield the "real but not typical" events carry: DY 8 %, Nonprompt 31 %, Other 55 %. |

ROOT's `"WL"` (weighted likelihood) only corrects the covariance — the central
values are those of `"L"`. `chi2` and `neff` stay available as *diagnostic*
scan variants (`diag_*`) and are excluded from the smoothing systematic.

The same mixed-weight structure shows up in the hybrid DY seam: in
[2700, 2900] the four 50 GeV bins are each 2–5σ *below* the expo tail
(they are low-weight, N_eff ≈ 18 per bin) while the spike at 3150 (one event,
weight 0.2) is above it. The smooth expectation must sit above the "typical"
events because the rare heavy ones carry real expected yield.

## Scripts and outputs

```bash
export PYTHONPATH=/opt/homebrew/lib/root; PY=../../../.venv-fitting/bin/python   # LPC: source LCG_106
$PY smooth_background.py -v            # smoothed_<method>.{root,json} + plots/spectrum_*, neff, seam_*
$PY smooth_background.py -v --scan     # scan/<method>__<variant>.{root,json}  (58 variants)
$PY compare_methods.py -v --toys 200   # pull_stats.csv, window_yields.csv, smoothing_systematic.csv, toy_closure_{kernel,param}.csv
# combine side (OrbStack VM; on cmslpc drop the orb/EXTRA_BINDS wrapper):
orb -m combine bash -c 'cd $PWD && EXTRA_BINDS="-B /Users/billy/work/WrCoffea" SKIP_HYBRID=1 ./run_smooth_limits.sh kernel'
$PY ../../8_combine_limits/production/collect_refined.py -v --cards-dir run2/ee_resolved/cards_kernel/ee_resolved \
    --results-dir run2/ee_resolved/results_kernel/ee_resolved --output-dir run2/ee_resolved --table-suffix _kernel
$PY ../../6_spurious_signal_toys/card_model_toys.py --bkg-hist-file run2/ee_resolved/smoothed_kernel.root \
    --masses 2000 2600 3200 --models card_float --output-dir run2/ee_resolved/toys_kernel
$PY compare_limits.py -v               # limit_ratio.csv + plots/limit_ratio.*
```

`smoothed_<method>.root` holds one TH1D per component (content = smoothed,
error = √raw sumw2), `<comp>_raw`, `bkg_smooth`, `bkg_raw`, and **`bkg_native`**
(= `bkg_smooth` clipped ≥ 0, no errors) — the drop-in twin of
`8_combine_limits/optimization/inputs/ee_resolved.root:bkg_native`.

Plumbing added for the study, all opt-in, default paths byte-identical
(checked: `--bkg-source raw` reproduces the 74 production cards exactly):

- `8_combine_limits/production/make_refined_workspaces.py`: `--bkg-source
  raw|smooth:<method>`, `--smooth-dir`, `--anchor-source same|raw`,
  `--anchor-norm env|window` (anchored norm = transported fit or the window
  yield of the histogram), `--masses`;
  the smoothed `bkg_native` rebinned ×10 / ×5 replaces `data_obs_full` /
  `bkg_native`.
- `run_refined.sh`: `MRW_ARGS`, `CARDS_DIR`, `RES_DIR`, `SKIP_HYBRID`,
  `EXTRA_BINDS` environment overrides. `collect_refined.py`: `--cards-dir`,
  `--table-suffix`.
- `6_spurious_signal_toys/card_model_toys.py`: `--bkg-hist-file`.
- `run_smooth_limits.sh` (here): one method → cards + combine, masses
  1000 / 2000 / 2600 / 3200 / 3400 / 4000 / 4600 by default.

## How the methods are judged without a truth

1. **Pulls** (raw − smooth)/σ per 50 GeV bin, Pearson σ² = w_i·smooth with
   the bin's MC weight (Neyman σ = √sumw2 is also tabulated but is dominated
   by the same low-fluctuation bias as the χ² fit). Mean, RMS, linear trend,
   |pull| > 3 list → `pull_stats.csv`, `plots/pulls_<method>.*`.
2. **Window yields** in every production window (k5 50 GeV float windows
   1400–3200 from `optimization/inputs`, k3 100 GeV windows elsewhere from
   `baseline/run2/inputs`), raw ± √sumw2 vs each method vs the production
   anchor's B_env → `window_yields.csv`, `plots/window_yields.*`.
3. **Hyperparameter systematic**: max |ΔB|/B over each method's scan
   variants (kernel: n_target × h_min × degree; hybrid: trust N_eff × family
   × trust width; param: family × range; kde: ρ × mirror × resampling) and the
   between-method spread of the defaults → `smoothing_systematic.csv`.
4. **Self-consistency toys**: a smooth default plays the truth, every bin is
   fluctuated as w_i·Poisson(s_i/w_i) with the raw per-bin weight
   (`TRandom3`), all smoothers are re-run, bias/RMS of the window yields →
   `toy_closure_<truth>.csv`. (With the kernel as truth the expectation dies
   with the MC, so beyond ~3400 the tail methods look "biased high" — use the
   param truth there.)
5. Split-sample / jackknife checks are **not possible**: only lumi-scaled
   histograms with sumw2 exist, no per-event weights or ids.

## Findings (Sep 23 2026)

**1. All four smoothers agree with the raw MC where it matters, and the
expected limit does not move.** Window yields (production windows) are
within ±3 % of raw for every method up to 3200 GeV, with the raw MC-stat
error 3–8 %. The combine expected medians (asymptotic on both sides, masses
1000 / 2000 / 2600 / 3200 / 3400 / 4000 / 4600, `plots/limit_ratio.*`,
`limit_ratio.csv`):

| method | median ratio smooth / raw | note |
|---|---|---|
| kernel | 0.99–1.02 | |
| hybrid | 0.99–1.00 | |
| param | 0.98–1.04 | expo2 tails for tt+tW / DY |
| kde | 0.99–1.00 | despite a 2–3× too-high tail (see 4) |
| raw through the same path | 1.000 everywhere | regression: the 74 production cards are byte-identical, the limits reproduce to all digits |
| hybrid, anchors fit on raw MC | anchored masses 1.000 exactly | anchored cards never see `data_obs`; only the anchor fit does |

The card-model spurious signal on the smoothed templates (float cards, 1000
toys, `toys_<method>/`) is consistent with zero everywhere: Asimov N_sp at
2000 / 2600 / 3200 is −0.2 / +0.4 / +1.2 on the raw template (σ ≈ 11 / 5 /
4) and −1.3…+2.2 / −1.6…+0.4 / 0.0…+0.9 on the smooth ones — the
jagged-template bias in the float regime was already negligible. The 3200
runaway (RMS 4.4 raw → 6.7–10.1 smooth, ~2 % non-converging toys) is a
low-B fit instability, not template noise: smoothing does not cure it.

**2. Where the smoothers differ from each other: only beyond the MC.**
Hyperparameter systematic (max |ΔB|/B over each method's scan variants,
`smoothing_systematic.csv`): hybrid ≤ 1 % to 2600, 5 % at 3200, 17–60 % at
3400–4000; param 2–7 % to 3200, 16–23 % at 3400–4000; kernel 5–25 % at
2000–3200 (the bandwidth floor is the knob) and > 60 % beyond; kde 5–13 %
then > 35 %. Between-method spread of the defaults: 1–3 % to 2600, 7 % at
3200, 17–60 % at 3400–4000. Self-consistency toys with the param default as
truth (200 toys, `toy_closure_param.csv`): every method (and the raw
histogram) is unbiased to ≤ 3 % up to 4000 with RMS 2–13 %; kde is +14…+38 %
above 3400 (its tail leak). With the kernel as truth the tail methods look
"+10 %" above 3400 only because that truth dies with the MC.

**3. The pulls say what a smooth expectation can and cannot be.**
Pearson pulls of the summed background, 50 GeV bins, [800, 6000]
(`pull_stats.csv`): kernel mean +0.12 RMS 1.50, hybrid −0.19 / 1.52, param
−0.10 / 1.00, kde −1.13 / 1.56 with a −10σ trend (the tail leak). Per
component, tt+tW is textbook (RMS 0.9), DY RMS 1.4 with the 3150 and 3700
GeV spikes as the only |pull| > 3, Nonprompt and Other have *negative* means
(−0.6, −1.8 for hybrid): the smooth expectation sits above their typical
low-weight events because the rare heavy events carry 31 % / 55 % of their
yield (the `neff` deficit above). That is the correct behaviour — the
expectation includes the heavy events — and it is why no fit statistic that
weights bins by their N_eff can be used for these samples.

**4. RooKeysPdf is the wrong tool for a steeply falling spectrum.** Its
adaptive bandwidth widens where the density is low, so it leaks yield into
the tail: window yields 1.2× raw at 3400, 1.4× at 4000, 2× at 4600, 10× at
6000 (`plots/spectrum_sum.*`). ρ = 0.5 halves the leak but does not remove
it; MirrorLeft vs NoMirror is a ≤ 5 % effect below 3200. It leaves the limit
unchanged only because the anchored cards ignore it. Not recommended.

**5. The "mcmax" caveat, resolved.** In the sparse tail the per-component
smoothed yields (kernel / hybrid / param) are 0.7–1.1× the raw window MC,
i.e. 1.4–2.4× the production anchor's transported B_env (3400: 6.6–7.5 vs
5.2; 4000: 2.0–2.4 vs 1.2; 4600: 0.73–0.90 vs 0.37). The anchor is a single
expo on the *sum* over [1000, 3500], dominated by tt+tW (slope −3.4/TeV),
while the tail is DY (−2.5/TeV): the composition change is real, not weight
noise, and re-fitting the anchor on the smoothed spectrum does not change it
(B_env 0.36–0.44 at 4600). What it costs: cards whose anchored normalization
is the smoothed window yield instead of B_env (`--anchor-norm window`,
labels `*_normwin`) give expected medians ×1.07 / 1.10 / 1.08 (hybrid) and
×1.12 / 1.15 / 1.12 (param) at 3400 / 4000 / 4600 — and ×1.09 / 1.18 / 1.14
for the jagged raw window. So the tail *is* real at the factor-2 level in
the background, but at B ≲ 1 the limit is n = 0 dominated and the effect on
the quoted points is 7–15 %, inside the 10–23 % model-spread bars the
production already draws there. A per-component anchor (tt expo + DY expo,
each on its own range) would remove the discrepancy at the source.

**Recommendation.** If a smoothed background is wanted as the limit input,
use **hybrid** (raw below the per-component trust mass, Poisson-ML expo tail
above): it is the least model-dependent where the MC is good, the most
stable under its own hyperparameters (≤ 5 % to 3200), unbiased on the
self-consistency toys, and changes the production limit by < 1.5 %. The
study's real yield is negative in the float regime — the raw jagged MC was
not hurting the expected limit — and positive in the anchored tail, where it
quantifies a background under-prediction the anchor construction hides.

## Known issue: legacy vs new-framework 2018 histograms (Oct 1 2026)

The production run2 limit (and everything above this section) uses
`20260714_run2_bkgs`, a key-rename of the legacy HNWRAnalyzer 2018 histograms.
Per-process curves need a WrCoffea rerun, and the two frameworks do **not**
give the same SR histograms (`compare_frameworks.py`, ee resolved SR,
`framework_comparison.csv`):

| component | legacy | new, Oct 1 rerun (`20261001_split_bkgs`) | new, Sep 6 files (`dyreshape_20260907`) |
|---|---|---|---|
| tt+tW SR total | 651 | 539 (×0.83) | 539 |
| DY+jets SR total | 362 | **264 (×0.73)** | 162 (×0.45) — stale, see below |
| Nonprompt | 19.0 | 23.1 | 23.1 |
| Other | 41.4 | 37.7 | 37.6 |
| expo slope [1000, 3500] | | agrees within errors for every component | |
| N_eff | | 2–4× more effective MC entries than legacy (tt, Other) | |

- **The Sep 6 DY files are stale.** They predate the Sep 7 commits that wired
  the UL18 DY corrections into the analyzer (Z-pT reweight, reshape, LHE-sum
  fix). With the Oct 1 rerun the DY control regions close: data/MC = 0.998
  (ee) and 1.001 (μμ), against 1.48 with the Sep 6 files; Oct 1 / Sep 6 DY =
  1.52 in both DY CRs and 1.63 in the SRs. tt+tW is bit-identical between the
  two runs. The earlier "new-framework DY is ×0.45 of legacy" was this.
- What remains is a common ×0.83 selection difference (seen in data and in
  tt, see `signal_fitting/run2_results/compare_legacy_new.py`) and a
  DY-specific ×0.88 in the SR on top of it.
- The production cards use the legacy histogram as `data_obs`, so new-framework
  curves cannot be mixed with them: a per-component anchor would be 17–27 %
  below the observation. Legacy per-process histograms cannot be made (the
  converter only has the four merged groups).

**Decision (Oct 1 2026): everything from here on — the per-process split,
its smoothing, any anchor work — runs on the new-framework histograms.**

## Per-process split (Oct 1 2026)

Nonprompt and Other are each a mix of processes with different shapes and
weights. The analyzer now writes one file per process
(`wrcoffea/cli_utils.py:_PHYSICS_SUBGROUPS`, both RunII-UL and Run3 dataset
names): Nonprompt → `WJets`, `TTSemiLep`, `SingleTop`; Other → `Diboson`,
`Triboson`, `ttV`; tt_tW → `TTbar`, `tW` (kept merged here).

```bash
cd ~/work/WrCoffea     # local run on the Mac; needs a valid proxy (~/bin/renew-proxy.sh)
for s in Diboson Triboson ttV SingleTop WJets TTSemiLep tt_tW; do
  caffeinate -i .venv/bin/python bin/run_analysis.py RunIISummer20UL18 $s \
      --dir 20261001_split_bkgs --max-workers 4 --xrootd-timeout 120
done
caffeinate -i .venv/bin/python bin/run_analysis.py RunIISummer20UL18 DYJets \
    --dir 20261001_split_bkgs --max-workers 4 --xrootd-timeout 600 --chunksize 50000
# (composite alternative: ... RunIISummer20UL18 bkg --dy lo_ht --split-groups Nonprompt Other)
```

Three things that cost time and are worth knowing: the laptop sleeps after one
idle minute on battery and every in-flight XRootD read then "expires"
(`caffeinate -i` holds it awake; a closed lid on battery still sleeps);
single-sample runs reject `--dy` for non-DY samples (the 2018 default MC tag
is already `dy_lo_ht`); and the high-HT DY chunks need smaller chunks and a
longer read timeout (250k-event chunks time out at 120 s with no sleep
involved). Run times with 4 workers: 1–3 min for the small ones, 8 min each
for tt semileptonic and tt+tW, 19 min for DY.

```bash
$PY smooth_background.py --bkg-dir 20261001_split_bkgs --components split8 --methods kde --no-plots
$PY plot_step1_kde.py --input run2/ee_resolved_split8/smoothed_kde.root \
    --output-dir run2/ee_resolved_split8/plots/step1
```

Component sets (`smooth_lib.COMPONENT_SETS`): `legacy4` (default), `split8`,
`np_other6`, `other3`, `nonprompt3`; outputs go to `run2/ee_resolved_<set>/`.

| process | SR yield | N_eff | last MC event | note |
|---|---|---|---|---|
| tt+tW | 538.7 ± 4.8 | 12500 | | uniform weights |
| DY+jets | 264.2 ± 8.9 | 890 | | HT-binned, mixed weights |
| diboson | 28.9 ± 3.3 | 77 | 2.2 TeV | ~0.4 events per MC event: the source of the old "Other" spikes |
| W+jets | 14.2 ± 2.1 | 48 | 5.4 TeV | HT-binned: two bands of bin contents (10⁻³ and 10⁻¹–1) |
| tt semileptonic | 8.4 ± 0.6 | 189 | 3.9 TeV | one weight (0.045 per event) |
| ttV | 6.04 ± 0.08 | 6400 | 3.8 TeV | the only well-populated small process |
| triboson | 2.67 ± 0.36 | 55 | 3.0 TeV | |
| single top | 0.52 ± 0.15 | 12 | 1.7 TeV | 12 MC events: no density estimate is meaningful |

**The combined curve** (`plots/step1/combined.*`, and as a histogram
`run2/ee_resolved_split8/smoothed_kde.root:bkg_smooth`; `bkg_native` is the same
clipped ≥ 0) is the sum of the eight per-process RooKeysPdf curves, each
normalized to its own MC yield, so the total (863.7) is conserved by
construction. Against the binned total MC:

| m_lljj [GeV] | MC | summed curve | curve / MC |
|---|---|---|---|
| 800–1200 | 615.8 ± 9.3 | 615.5 | 1.00 |
| 1200–2000 | 223.7 ± 5.4 | 223.3 | 1.00 |
| 2000–3000 | 22.0 ± 1.1 | 21.4 | 0.97 |
| 3000–3500 | 1.46 ± 0.29 | 1.93 | 1.33 |
| 3500–4500 | 0.64 ± 0.13 | 1.13 | 1.76 |
| 4500–6000 | 0.15 ± 0.07 | 0.35 | 2.33 |

It describes the MC to 3 TeV and overshoots above, the RooKeysPdf tail leak
again (numbers for the m axis, Oct 5 2026). The overshoot comes from the two
large processes: above 4.5 TeV the tt+tW curve holds 0.12 events against 0.04
in its MC, the DY curve 0.18 against 0.11; the six small processes add 0.05
against 0.002.

What the split shows (RooKeysPdf, step-1 plots): ttV and tt+tW are followed
well until the usual tail overshoot; for diboson, W+jets and tt semileptonic
the curve runs below the isolated heavy-event bins and spreads their yield
over the tail, so in the stack of the six small processes the summed curve
sits 3–10× above most bins between 2 and 3.5 TeV (which contain only ttV).
The merged Nonprompt / Other histograms hid that structure.

## 2024 per-process set and the sum-vs-single comparison (Oct 4 2026)

Same split on `RunIII2024Summer24` (skims on Wisconsin, LO-inclusive DY, the
era's default MC tag): `20261004_split_bkgs`, run as
`chain24.sh`-style single-sample runs with `--max-workers 4 --xrootd-timeout
600 --chunksize 50000` under `caffeinate -i` (tt+tW 66 files = 66 min, DY 22,
tt semileptonic 17, W+jets 16, the rest 1–5 min; no read failures).

```bash
$PY smooth_background.py --era RunIII2024Summer24 --bkg-dir 20261004_split_bkgs --components split8 --methods kde --no-plots
$PY plot_step1_kde.py --input run3/ee_resolved_split8/smoothed_kde.root --output-dir run3/ee_resolved_split8/plots/step1
$PY compare_sum_vs_single.py --input run3/ee_resolved_split8/smoothed_kde.root
```

Every per-process plot now lists the RooKeysPdf settings it was made with
(`plot_step1_kde.py` reads them from `smoothed_kde.json`): x = m on
[800, 8000] GeV (ln m until Oct 5 2026 — the user judged ln m an unnecessary
complication; `explain_kde_variable.py` draws both for one process), MirrorLeft,
ρ = 1, one entry per populated 10 GeV bin with
weight = bin content, the number of entries and of negative bins dropped, and
the MC yield the curve is normalized to. The settings are identical for every
process and for the single curve.

| process (2024, ee resolved SR) | SR yield | N_eff | KDE entries |
|---|---|---|---|
| tt+tW | 1329 ± 6 | 42 000 | 279 |
| DY+jets (LO inclusive) | 451 ± 17 | 740 | 149 |
| diboson | 36.8 ± 2.8 | 178 | 104 |
| W+jets | 23.4 ± 4.4 | 29 | 34 |
| tt semileptonic | 21.2 ± 1.5 | 188 | 104 |
| triboson | 5.74 ± 0.18 | 1000 | 227 |
| ttV | 4.88 ± 0.21 | 560 | 187 |
| single top | 0.88 ± 0.44 | 4 | 9 |

**Sum of the 8 curves vs one RooKeysPdf of the summed MC**
(`compare_sum_vs_single.py` → `plots/step1/sum_vs_single.*`, `sum_vs_single.csv`,
`single_kde.root`):

| m_lljj [GeV] | MC | sum of 8 | single | sum / MC | single / MC |
|---|---|---|---|---|---|
| 800–1200 | 1320 ± 15 | 1322 | 1321 | 1.00 | 1.00 |
| 1200–2000 | 489 ± 11 | 486 | 487 | 0.99 | 1.00 |
| 2000–3000 | 56.8 ± 4.0 | 55.7 | 55.9 | 0.98 | 0.99 |
| 3000–3500 | 2.45 ± 0.61 | 3.45 | 3.19 | 1.41 | 1.30 |
| 3500–4500 | 2.17 ± 0.84 | 2.70 | 2.53 | 1.25 | 1.17 |
| 4500–6000 | 1.67 ± 1.55 | 1.95 | 1.95 | 1.17 | 1.16 |

The two curves agree with each other to 1 % below 3 TeV and to 8 % above;
both overshoot the MC by 1.2–1.4× at 3–4.5 TeV, where the MC itself is 25–90 %
uncertain (single LO-inclusive DY events weigh 0.3–1.5 events there). For
RooKeysPdf the per-process split does not change the total curve: the
estimate is linear in the entries, and only the per-process adaptive
bandwidths differ. On the 2018 set (`run2/ee_resolved_split8/`) the same
comparison gives 1.76 vs 1.56 at 3.5–4.5 TeV and 2.33 vs 2.08 at 4.5–6 TeV —
the single curve leaks slightly less into the tail because its pilot density
is set by the full spectrum. (With the original ln m axis the 2024 numbers
were 1.64/1.49 and 1.36/1.26, the 2018 ones 2.25/2.02 and 3.76/3.06: ln m
widened the tail bumps in GeV and leaked more.)

## One entry per MC event (`--save-events`, Oct 5 2026)

The RooKeysPdf entries above are populated 10 GeV bins (position = bin centre,
weight = content) because the analyzer only wrote histograms. It now also
writes the per-event SR lists: `run_analysis.py ... --save-events` adds TTrees
`events_<region>` (branches `mass`, `weight`; resolved `mass_fourobject`,
boosted `mass_twoobject`) to each output file (`wrcoffea/histograms.py:
record_events`, `save_hists.py:_save_event_lists`; 541 tests pass; checked on
2024 diboson: tree sums equal the histogram integrals and the histograms are
unchanged). Reruns: `20261005_split_events` under both eras (same per-sample
commands as before plus `--save-events`; two chains at a time — four at once
were ~6× slower per sample; the laptop must not sleep: `caffeinate -i`, lid
open or AC).

```bash
$PY smooth_background.py --era <era> --bkg-dir 20261005_split_events --components split8 --entries events
$PY plot_step1_kde.py --input run3/ee_resolved_split8_events/smoothed_kde.root --output-dir run3/ee_resolved_split8_events/plots/step1
$PY compare_sum_vs_single.py --input run3/ee_resolved_split8_events/smoothed_kde.root     # single curve = one KDE of ALL events
$PY compare_entries.py --bins run3/ee_resolved_split8/smoothed_kde.root --events run3/ee_resolved_split8_events/smoothed_kde.root --process total
```

`smooth_lib.smooth_kde_events` builds the dataset with one entry per event
(negative-weight events dropped from the dataset, kept in the normalization,
as for the bins). Outputs `run{2,3}/ee_resolved_split8_events/`; overlays
`plots/step1/compare_entries_<process>.*`.

**Result: the per-event and per-bin curves are indistinguishable**, for every
process and for the sum, in both years:

| sum of 8 / MC | 3–3.5 TeV | 3.5–4.5 TeV | 4.5–6 TeV | single curve |
|---|---|---|---|---|
| 2024, entries = bins | 1.41 | 1.25 | 1.17 | 1.30 / 1.17 / 1.16 |
| 2024, entries = events | 1.38 | 1.21 | 1.17 | 1.19 / 1.13 / 1.13 |
| 2018, entries = bins | 1.33 | 1.76 | 2.33 | 1.26 / 1.56 / 2.08 |
| 2018, entries = events | 1.34 | 1.67 | 2.16 | 1.23 / 1.32 / 1.84 |

The bin-centre shortcut was accurate: the bumps are ≥ 30 GeV wide so a ≤ 5 GeV
displacement is invisible, and the entry count enters the bump width only as
N^(−1/5), which the adaptive step then largely undoes. What the event lists do
provide is the per-event information itself (ee resolved SR):

| process | 2018 MC events (negative-weight) | 2024 MC events (negative-weight) |
|---|---|---|
| tt+tW | 15 407 (428) | 65 004 (1807) |
| DY+jets | 6 594 (0) | 1 015 (0) |
| ttV | 6 469 (0) | 3 010 (424) |
| triboson | 77 (1) | 3 672 (148) |
| diboson | 80 (0) | 276 (0) |
| tt semileptonic | 198 (2) | 262 (1) |
| W+jets | 151 (0) | 42 (0) |
| single top | 14 (1) | 11 (2) |

The 2024 LO-inclusive DY is 1015 MC events for 451 weighted (0.44 per event):
its tail is a handful of heavy events by construction. Negative weights are
the NLO samples (ttV, triboson, single top, and 2.8 % of tt+tW); their net
weight is small (tt+tW 2018: −0.06 of 539) but they are now identifiable
event by event.

## 2024 on skims_v2 with LO HT-binned DY (Oct 8 2026)

The 2024 sets above were made on the old skims with LO-inclusive DY, whose
ee SR is 1015 MC events for 451 weighted: its tail was a handful of events
weighing 0.3–1.5 each. All eight processes were rerun on skims_v2 (JEC V4,
lumi 109.70 fb⁻¹, the rebuilt configs) with `dy_lo_ht`, the new 2024
default, into `20261008_split_events` (same per-sample command as before,
`--max-workers 4 --xrootd-timeout 600 --chunksize 50000 --save-events`;
DY 54 min, tt+tW 43 min, W+jets 45 min, the rest 3–18 min, two runs at a
time).

### Which DY sample has the most events in the SR?

`dy_statistics.py` → `run3/dy_options/dy_statistics_wr_ee_resolved_sr.csv`,
`plots/dy_options_wr_ee_resolved_sr.*` (yield and N_eff per 50 GeV). The NLO
inclusive and jet-binned files are `20261008_skims_v2_{nlo_inc,nlo_njet}_dy`
(another session's runs), the m(ll)-binned one `20261008_split_events_nlo_mll_dy`.
N_eff = (Σw)²/Σw², the number of equal-weight events with the same MC error.

| DY option (ee resolved SR) | generated | MC events in SR | SR yield | N_eff | N_eff 3–4 TeV | N_eff > 4 TeV |
|---|---|---|---|---|---|---|
| LO inclusive | 1026 M | 1 015 | 475 | 740 | 2.6 | 2.7 |
| **LO HT-binned** | 1170 M | 57 525 | 496 | **17 100** | **644** | **203** |
| NLO inclusive (FxFx) | 977 M | 2 543 | 655 | 247 | 0 | 0.6 |
| NLO jet-binned (FxFx 0/1/2J) | 2452 M | 12 396 | 642 | 1 290 | 7.6 | 7.1 |
| NLO m(ll)-binned (POWHEG) | 1092 M | 397 508 | 597 | 15 900 | 150 | 49 |
| NLO pT(ll)-binned | 766 M | — | — | — | — | — |

- LO HT-binned has the most *effective* events, overall and 4–13× more than
  any other option above 3 TeV. NLO pT(ll) is not usable as is: its four
  samples are nested (pT(ll) > 100/200/400/600) and the analyzer does not
  stitch them.
- The m(ll)-binned set has 7× more MC events in the SR but most of them are
  worthless there: above 3 TeV in m_lljj, 43 % of its DY yield is 55 events
  from the 400–800 GeV bin (0.022 each) and 29 % is 591 events from
  800–1500, while 175 000 events with m(ll) > 4 TeV carry 0.4 %. The SR
  tail is moderate-m(ll) DY with hard jets, which HT binning targets.
- POWHEG makes the second jet in the shower: the m(ll)-binned tail is 2.3×
  below LO HT above 4 TeV (0.38 vs 0.86 events).
- All three NLO options give 20–30 % more SR DY than LO, although LO is
  already scaled to the NLO inclusive cross section (k = 1.223) — a DY CR
  normalization question, not addressed here.
- Provenance: another session mapped an untested 2024 DY Z-pT payload in
  `config.yaml` between ~21:03 and 21:31. The LO-HT DY run that started at
  21:22 was killed and rerun at 21:53; the m(ll) (workers started 20:32) and
  jet-binned (20:50) runs had no worker restarts and are clean. The
  rerun DY is identical to the morning's `20261008_skims_v2` DY (57 525
  events, 495.88).

### The new per-process set

```bash
$PY smooth_background.py --era RunIII2024Summer24 --bkg-dir 20261008_split_events --components split8 \
    --entries events --output-dir run3/ee_resolved_split8_events_v2
$PY plot_step1_kde.py --input run3/ee_resolved_split8_events_v2/smoothed_kde.root --output-dir run3/ee_resolved_split8_events_v2/plots/step1
$PY compare_sum_vs_single.py --input run3/ee_resolved_split8_events_v2/smoothed_kde.root
```

| process (ee resolved SR) | MC events (neg.) | yield | N_eff | old skims: events / yield | why it changed |
|---|---|---|---|---|---|
| tt+tW | 65 004 (1807) | 1342 | 42 400 | 65 004 / 1329 | |
| DY+jets | 57 525 (0) | 496 | 17 100 | 1 015 / 451 | LO HT instead of LO inclusive |
| diboson | 276 (0) | 58.3 | 197 | 276 / 36.8 | reference xsecs (WZ 54.7 vs 29.1 pb, WW 125.8 vs 80.2, ZZ 16.7 vs 12.8) |
| W+jets | 1 733 (0) | 28.9 | 98 | 42 / 23.4 | HT × M(lν)-binned W+jets (old N_eff 29) |
| tt semileptonic | 387 (1) | 21.7 | 279 | 262 / 21.2 | |
| ttV | 13 288 (3367) | 14.8 | 1 870 | 3 010 / 4.88 | ttW (`TTLNu-1Jets`, 0.24 pb) added |
| triboson | 3 672 (148) | 5.74 | 1 010 | 3 672 / 5.74 | |
| single top | 11 (2) | 1.08 | 4 | 11 / 0.88 | |

Tree sums equal the histogram integrals for every process.

**Sum of the 8 curves vs the MC** (`sum_vs_single.csv`):

| m_lljj [GeV] | MC | sum of 8 | single | sum / MC | single / MC | old skims: sum / MC (MC ± stat) |
|---|---|---|---|---|---|---|
| 800–1200 | 1371 ± 8 | 1369 | 1370 | 1.00 | 1.00 | 1.00 |
| 1200–2000 | 525 ± 5 | 526 | 525 | 1.00 | 1.00 | 1.00 |
| 2000–3000 | 65.7 ± 1.4 | 64.7 | 65.2 | 0.98 | 0.99 | 0.98 |
| 3000–3500 | 4.40 ± 0.25 | 5.11 | 4.84 | 1.16 | 1.10 | 1.38 (2.45 ± 0.61) |
| 3500–4500 | 2.50 ± 0.20 | 2.95 | 2.68 | 1.18 | 1.07 | 1.21 (2.17 ± 0.84) |
| 4500–6000 | 0.48 ± 0.08 | 0.91 | 0.67 | 1.90 | 1.40 | 1.17 (1.67 ± 1.55) |

The MC tail is now measured to 6–17 % instead of 25–90 %, and against it the
RooKeysPdf overshoot is significant: 1.9× above 4.5 TeV. The old "1.17" there
compared to an MC of 1.67 ± 1.55 made of a few heavy LO-inclusive DY events.
Per process (curve / MC):

| process | 3–3.5 TeV | 3.5–4.5 TeV | 4.5–6 TeV | last MC event |
|---|---|---|---|---|
| tt+tW | 1.70 / 1.37 ± 0.19 | 0.93 / 0.66 ± 0.15 | 0.33 / 0.13 ± 0.07 | 5.2 TeV |
| DY+jets | 2.89 / 2.73 ± 0.13 | 1.67 / 1.60 ± 0.08 | 0.47 / 0.32 ± 0.04 | 8 TeV |
| diboson | 0.22 / 0 | 0.12 / 0 | 0.01 / 0 | 3.0 TeV |
| the other five | 0.30 / 0.29 | 0.24 / 0.25 | 0.09 / 0.03 | |

DY now follows its MC to 4–6 % up to 4.5 TeV and leaks above; tt+tW
overshoots from 3 TeV (uniform weights, it simply runs out); diboson's
curve spreads its last events (3 TeV) over 3–4.5 TeV. With this MC the DY
tail may not need smoothing below ~4.5 TeV at all; tt+tW and the sparse small
processes do — the tail-treatment decision should be made on this set.

## Files

```
smooth_lib.py  smooth_background.py  compare_methods.py  compare_limits.py  run_smooth_limits.sh
run2/ee_resolved/
  smoothed_{kernel,hybrid,param,kde}.{root,json}      scan/  (58 variants)
  pull_stats.csv  window_yields.csv  smoothing_systematic.csv  toy_closure_{kernel,param}.csv
  limit_ratio.csv  refined_limit_table_ee_resolved_<label>.csv
  cards_<label>/  results_<label>/  toys_{raw,kernel,hybrid,param,kde}/
  plots/  spectrum_<comp|sum>  neff  seam_<comp>  pulls_<method>  window_yields
          smoothing_systematic  limit_ratio  ee_resolved_<label>/ (collect_refined bands)
plot_step1_kde.py  compare_sum_vs_single.py  compare_entries.py  dy_statistics.py
run{2,3}/ee_resolved_split8{,_events}/        per-process RooKeysPdf (old 2024 skims for run3)
run3/ee_resolved_split8_events_v2/            2024 skims_v2, LO HT DY (20261008_split_events)
run3/dy_options/                              DY option statistics table + plot
```
