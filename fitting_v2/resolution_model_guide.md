# `m_lljj` Resolution Model Guide

A simple, physically-motivated parametric model of the four-object invariant mass
resolution `σ(m_lljj)` for `W_R → ℓ N → ℓ ℓ jj` signal MC, implemented in
[`fitting_v2/resolution_model.py`](resolution_model.py).

## Purpose

`signal_shape_study.py` produces empirical FWHM/RMS curves vs `M_N/M_WR` showing
two features that beg a physical explanation:

1. A **U-shape** in width — narrowest near `M_N/M_WR ≈ 0.6`, broader at both extremes.
2. A **systematic ee vs μμ FWHM gap** that grows with `M_WR` and is largest in the
   boosted-N regime.

This study answers: "given the kinematics of the `W_R → ℓ N → ℓ ℓ jj` cascade and
basic CMS-style per-component resolution functions, can we *reproduce* the observed
width vs `M_N/M_WR` trend with a handful of free parameters?"

The goal is **understanding**, not a precision fit — a back-of-envelope model where
the fitted parameters are interpretable as "the muon constant term is ~X%, the jet
constant term is ~Y%", and the model curve sits sensibly on top of the data.

## Kinematic decomposition

In the `W_R` rest frame (≈ lab frame for our purposes), the available energy
`M_WR` is partitioned among three "buckets" of decay products as a function of
`(M_WR, M_N)`.

Note: `M_WR` and `M_N` are not measured from data — they are the
generated masses from the signal MC sample labels (e.g.
`WRAnalyzer_signal_WR4000_N1900.root` → `M_WR = 4000`, `M_N = 1900`).
The energy decomposition is a purely analytical calculation from these
input parameters; no histograms are opened. The `energy_decomposition.pdf`
plot evaluates these formulas on a dense grid of `M_N/M_WR` from 0.05
to 0.99 for each WR mass.

### Step 1: $W_R \to \ell_1 N$ (two-body decay)

In the $W_R$ rest frame, the primary lepton and $N$ are back-to-back.
Conservation of energy and momentum gives:

$$E_{\ell_1} + E_N = M_{W_R} \qquad \text{(energy conservation)}$$

$$|\vec{p}_{\ell_1}| = |\vec{p}_N| \qquad \text{(momentum conservation)}$$

For the lepton (massless, $m_\ell \approx 0$): $E_{\ell_1} = |\vec{p}_{\ell_1}|$.
For the neutrino: $E_N^2 = |\vec{p}_N|^2 + M_N^2$.

Since $|\vec{p}_{\ell_1}| = |\vec{p}_N|$, substitute $E_{\ell_1}$ for $|\vec{p}_N|$:

$$E_N^2 = E_{\ell_1}^2 + M_N^2$$

From energy conservation: $E_N = M_{W_R} - E_{\ell_1}$. Square both sides:

$$E_N^2 = M_{W_R}^2 - 2 M_{W_R} E_{\ell_1} + E_{\ell_1}^2$$

Set the two expressions for $E_N^2$ equal:

$$E_{\ell_1}^2 + M_N^2 = M_{W_R}^2 - 2 M_{W_R} E_{\ell_1} + E_{\ell_1}^2$$

The $E_{\ell_1}^2$ terms cancel:

$$M_N^2 = M_{W_R}^2 - 2 M_{W_R} E_{\ell_1}$$

Solve for $E_{\ell_1}$:

$$\boxed{E_{\ell_1} = \frac{M_{W_R}^2 - M_N^2}{2 M_{W_R}}}$$

Monotonically *decreases* with $M_N$ — softens as $N$ gets heavier.

And from energy conservation:

$$E_N = M_{W_R} - E_{\ell_1} = \frac{M_{W_R}^2 + M_N^2}{2 M_{W_R}}$$

### Step 2: Boost factor of $N$

The $N$ has energy $E_N$ and mass $M_N$, so its Lorentz boost factor is:

$$\gamma_N = \frac{E_N}{M_N} = \frac{M_{W_R}^2 + M_N^2}{2 M_{W_R} M_N}$$

### Step 3: $N \to \ell_2 q q'$ (three-body decay)

This is a three-body decay mediated by an off-shell $W_R^*$. In the $N$
rest frame, the total energy available is $M_N$, shared among the
secondary lepton and two quarks.

For a three-body decay of massless final-state particles, the
phase-space-averaged energy of each particle is $M_N / 3$
(equipartition). This is exact for pure phase space and approximate
to ~10% for the V−A matrix element. So in the $N$ rest frame:

$$\langle E_{\ell_2}^* \rangle \approx \frac{M_N}{3}, \qquad \langle E_{jj}^* \rangle \approx \frac{2 M_N}{3}$$

Boosting to the $W_R$ rest frame (≈ lab frame), the average energies
scale by $\gamma_N$:

$$\boxed{E_{\ell_2} \approx \gamma_N \cdot \frac{M_N}{3} = \frac{M_{W_R}^2 + M_N^2}{6 M_{W_R}}}$$

$$\boxed{E_{jj} \approx \gamma_N \cdot \frac{2 M_N}{3} = \frac{M_{W_R}^2 + M_N^2}{3 M_{W_R}}}$$

$E_{jj}$ monotonically *increases* with $M_N$ — hardens as $N$ gets heavier.

### Cross-check: energy conservation

$$E_{\ell_1} + E_{\ell_2} + E_{jj} = \frac{M_{W_R}^2 - M_N^2}{2 M_{W_R}} + \frac{M_{W_R}^2 + M_N^2}{6 M_{W_R}} + \frac{M_{W_R}^2 + M_N^2}{3 M_{W_R}}$$

Common denominator $6 M_{W_R}$:

$$= \frac{3(M_{W_R}^2 - M_N^2) + (M_{W_R}^2 + M_N^2) + 2(M_{W_R}^2 + M_N^2)}{6 M_{W_R}} = \frac{6 M_{W_R}^2}{6 M_{W_R}} = M_{W_R} \;\checkmark$$

### Summary of results

| Component | Formula | Behavior vs $M_N/M_{W_R}$ |
|---|---|---|
| Primary lepton $E_{\ell_1}$ | $\frac{M_{W_R}^2 - M_N^2}{2 M_{W_R}}$ | Decreases (softens as $N$ gets heavier) |
| Secondary lepton $E_{\ell_2}$ | $\frac{M_{W_R}^2 + M_N^2}{6 M_{W_R}}$ | Gently increases |
| Jet system $E_{jj}$ | $\frac{M_{W_R}^2 + M_N^2}{3 M_{W_R}}$ | Increases (hardens as $N$ gets heavier) |

### Why this matters for the peak width

The connection between the energy decomposition and the `m_lljj` peak
width is that **detector resolution is a function of energy**. If you
know the typical energy of each component, you know how well the
detector measures it — and therefore how much it contributes to the
width of the reconstructed `m_lljj` peak.

Looking at `energy_decomposition.pdf`, as `M_N/M_WR` increases (left
to right):

- The primary lepton (solid lines) gets **softer** — carries less energy.
- The jet system (dash-dot lines) gets **harder** — carries more energy.

Now apply what we know about CMS detector resolution:

- **Electrons**: resolution is roughly constant fractionally at high
  energy (ECAL constant term ~few%). Whether the electron carries 500
  GeV or 2000 GeV, the fractional smearing is about the same. The
  absolute smearing (`σ = E × σ/E`) scales linearly with energy — hard
  electrons contribute more to the `m_lljj` width than soft ones, but
  just proportionally.
- **Muons**: resolution gets **worse** at high momentum (sagitta term,
  `σ/p ∝ p`). The fractional smearing itself grows with energy. The
  absolute smearing grows even faster — like `p²`. A 2 TeV muon is
  smeared much more than twice as badly as a 1 TeV muon.
- **Jets**: resolution is roughly constant fractionally at high energy
  (~5–20% constant term). Same scaling as electrons — absolute smearing
  proportional to energy.

This immediately explains the two observed FWHM trends:

**Why ee FWHM increases monotonically:** At low `M_N/M_WR`, the primary
electron is hard but well-measured (small fractional resolution). The
jets are soft, so they contribute little absolute smearing. Total peak
width is small. As `M_N` grows, jets harden and their absolute smearing
increases steadily — FWHM rises monotonically.

**Why mumu FWHM is flat:** At low `M_N/M_WR`, the primary muon is hard
**and** poorly measured (large fractional resolution that grows with
momentum). This already makes the peak wide. As `M_N` grows, the muon
softens (getting *better* measured) while jets harden (getting worse).
One contribution goes down, the other goes up — they approximately
cancel, and the FWHM stays flat.

**Why the channels converge at high `M_N/M_WR`:** The primary lepton
energy drops toward zero regardless of flavor. The jet system dominates
— and jets are identical between ee and mumu. So both channels approach
the same width.

The rest of this study makes this argument quantitative: we assign
parametric resolution functions to each component (next section), combine
them in quadrature, and fit the free parameters to the measured FWHM to
check whether this picture actually reproduces the data.

## Per-component resolution functions

Each component has a CMS-style fractional resolution with one or two free parameters.
The functional forms below are standard CMS parameterizations; the free parameters
are floated in our fit rather than fixed to published values, since they act as
effective parameters absorbing geometric and pairing effects beyond pure detector
resolution.

### Electron (ECAL-dominated)

The CMS ECAL energy resolution is parameterized as
([CMS, JINST 8 (2013) P09009](https://arxiv.org/abs/1306.2016), Eq. 1;
[CMS ECAL TDR, CERN/LHCC 97-33](https://cds.cern.ch/record/349375/files/ECAL_TDR.pdf);
[CMS, arXiv:2403.15518](https://arxiv.org/abs/2403.15518) for Run 2 update):

$$\frac{\sigma(E)}{E} = \frac{S}{\sqrt{E}} \oplus \frac{N}{E} \oplus C$$

where $S$ is the stochastic term (test beam: $S = 2.8\%$), $N$ is the noise term,
and $C$ is the constant term (test beam: $C = 0.3\%$). At TeV energies the noise
term is negligible and the constant term dominates. We simplify to:

$$\frac{\sigma(E)}{E} = \sqrt{\left(\frac{a_e}{\sqrt{E}}\right)^2 + b_e^2}$$

- $a_e$ [$\sqrt{\text{GeV}}$]: stochastic term
- $b_e$: constant term, expected ~1% at TeV scales

Electrons get *better* with energy until the constant term dominates.

### Muon (sagitta-dominated at high $p_T$)

The CMS muon $p_T$ resolution at high momentum is parameterized as
([CMS, JINST 15 (2020) P02027](https://arxiv.org/abs/1912.03516), Figs. 7–9 —
the dedicated high-$p_T$ muon paper with measurements up to 1.8 TeV;
[CMS, JINST 7 (2012) P10002](https://arxiv.org/abs/1206.4071) for Run 1;
[CMS, JINST 13 (2018) P06015](https://arxiv.org/abs/1804.04528) for Run 2):

$$\frac{\sigma(p_T)}{p_T} = a \cdot p_T \oplus b$$

The linear term $a \cdot p_T$ reflects the sagitta measurement: at high $p_T$ the
track curvature is small, so the fractional uncertainty on the curvature (and
therefore $p_T$) grows linearly with $p_T$. CMS measures resolution better than
10% up to $p_T = 1$ TeV in the barrel. We use:

$$\frac{\sigma(p)}{p} = \sqrt{(a_\mu \cdot p)^2 + b_\mu^2}$$

- $a_\mu$ [1/GeV]: linear term — expected $\sim 1$–$2 \times 10^{-4}$ /GeV (so $\sigma/p \sim 10$–20% at 1 TeV)
- $b_\mu$: constant term, expected ~1%

Muons get *worse* with momentum because of the linear sagitta term — this is the
key physics that drives the ee vs $\mu\mu$ difference at low $M_N/M_{W_R}$ (where the
primary lepton is hardest).

### Jet / hadronic system

The jet energy resolution follows the same functional form as the ECAL, with a
stochastic and constant term:

$$\frac{\sigma(E)}{E} = \sqrt{\left(\frac{a_j}{\sqrt{E}}\right)^2 + b_j^2}$$

- $a_j$ [$\sqrt{\text{GeV}}$]: stochastic term
- $b_j$: constant term, expected ~5% (dominant at TeV scales)

This is the resolution for the **summed** jet system, so it absorbs JES, JER, and
pairing effects together.

## Combination into $\sigma(m_{\ell\ell jj})$

Treat the three component-energy smearings as independent and add in quadrature:

$$\sigma(m_{\ell\ell jj})^2 \approx \sigma(E_{\ell_1})^2 + \sigma(E_{\ell_2})^2 + \sigma(E_{jj})^2$$

where each $\sigma(E_i) = E_i \times (\sigma/E)_i$ evaluated at the typical energy from the
kinematic decomposition. This is the simplest non-trivial model: it ignores
correlations and the geometric `(∂m/∂E_i)` factors, but it captures the right
scaling and is the "envelope of expectation" for what determines the `m_lljj` width.

The channel selection enters only through which lepton resolution function is used
for `E_ℓ1` and `E_ℓ2`:

- **ee channel**: both leptons use the electron model.
- **μμ channel**: both leptons use the muon model.
- The jet term is **identical** in both channels.

## Fit strategy

### What the fit does

The model has no polynomials or exponentials — the "fit function" is the
physics model itself: kinematics → resolution → quadrature sum. The 6
parameters ($a_e, b_e, a_\mu, b_\mu, a_j, b_j$) control the shape of
the resolution functions, and the fitter finds the values that make the
model predictions match the measured FWHM across the full signal grid.

### Worked example: one data point

To make this concrete, here is exactly what happens when the fitter
evaluates one data point (WR4000, N1900, ee channel) with trial
parameter values $a_e = 0.03$, $b_e = 0.05$, $a_\mu = 5 \times 10^{-5}$,
$b_\mu = 0.20$, $a_j = 6.0$, $b_j = 0.19$:

**Step 1** — Compute component energies from kinematics (no free
parameters here):

$$E_{\ell_1} = \frac{4000^2 - 1900^2}{2 \times 4000} = 1549 \text{ GeV}$$

$$E_{\ell_2} = \frac{4000^2 + 1900^2}{6 \times 4000} = 817 \text{ GeV}$$

$$E_{jj} = \frac{4000^2 + 1900^2}{3 \times 4000} = 1634 \text{ GeV}$$

**Step 2** — Apply the resolution functions. It's ee, so use the
electron model for both leptons and the jet model for the jets:

$$\sigma(E_{\ell_1}) = 1549 \times \sqrt{\left(\frac{0.03}{\sqrt{1549}}\right)^2 + 0.05^2} = 1549 \times 0.0501 = 77.6 \text{ GeV}$$

$$\sigma(E_{\ell_2}) = 817 \times \sqrt{\left(\frac{0.03}{\sqrt{817}}\right)^2 + 0.05^2} = 817 \times 0.0501 = 40.9 \text{ GeV}$$

$$\sigma(E_{jj}) = 1634 \times \sqrt{\left(\frac{6.0}{\sqrt{1634}}\right)^2 + 0.19^2} = 1634 \times 0.236 = 386 \text{ GeV}$$

**Step 3** — Quadrature sum gives the predicted peak width:

$$\sigma(m_{\ell\ell jj}) = \sqrt{77.6^2 + 40.9^2 + 386^2} = \sqrt{6022 + 1673 + 148996} = 396 \text{ GeV}$$

**Step 4** — Compare to the measured FWHM at this grid point (454 GeV).
The residual $(454 - 396) / \delta$ contributes to $\chi^2$.

The fitter repeats this for all 108 points (54 grid points × 2
channels), sums the squared residuals, and adjusts the 6 parameters
to minimize the total:

$$\chi^2 = \sum_{i=1}^{108} \left(\frac{\text{FWHM}_{\text{measured},i} - \sigma_{\text{model},i}}{\delta_i}\right)^2$$

### Why shared jet parameters matter

The jet parameters $(a_j, b_j)$ are **shared** between ee and $\mu\mu$.
The jets in a $W_R \to \ell\ell jj$ event are the same regardless of
lepton flavor — same quarks, same jet energies, same detector response.
So if the fit sees that $\mu\mu$ is wider than ee at low $M_N/M_{W_R}$,
it **cannot** explain that difference through the jet parameters. It is
forced to attribute it to the lepton parameters — specifically to
$a_\mu$ being nonzero.

Without this constraint, there would be a degeneracy: the fit could
make $\mu\mu$ wider by either increasing the muon resolution terms or
increasing a separate $\mu\mu$-jet resolution. Sharing $(a_j, b_j)$
breaks that degeneracy and gives the fit its discriminating power.

### Fit configuration

The script runs **two parallel fits per era**: one against the RMS
column and one against the bin-scan FWHM column from
`shape_summary.json`. This is deliberate — we already know the FWHM
bin-scan estimator is noisy in the boosted-N regime, so running both
fits side-by-side quantifies how much the noise distorts the fitted
parameters.

Each fit:

- **Joint $\chi^2$ across both channels and all WR/N grid points** for the era.
- **Shared jet parameters** $(a_j, b_j)$ between ee and $\mu\mu$ (since the jet system
  is physically identical), with **separate** lepton terms $(a_e, b_e)$ and
  $(a_\mu, b_\mu)$.
- **6 free parameters per fit**: $a_e, b_e, a_\mu, b_\mu, a_j, b_j$.
- **108 data points** per fit (54 grid points × 2 channels for Run 3;
  188 for Run 2).
- **Backend**: `scipy.optimize.curve_fit` ($\chi^2$ least-squares with errors),
  following the pattern in [`fit_signal.py:577`](fit_signal.py#L577) (`run_scipy_fit`).
  Pure numpy/scipy — no RooFit / PyROOT required.
- **Errors**: per-point uncertainty is the statistical error
  ($\text{width} / \sqrt{2N}$) added in quadrature with a 5% modeling-uncertainty
  floor ($0.05 \times \text{width}$), so that $\chi^2$ reflects modeling discrepancy
  rather than being dominated by the tiny statistical errors on ~10k-event samples.

## Inputs

[`fitting_v2/outputs/<era>/signal_shape_study/shape_summary.json`](outputs/RunIII2024Summer24/signal_shape_study/shape_summary.json)
must exist. Run [`signal_shape_study.py`](signal_shape_study.py) for the era first
if it doesn't.

The script filters to a per-era WR mass list:

| Era | WR masses included |
|---|---|
| `RunIII2024Summer24` | `[2000, 4000, 6000]` |
| `RunIISummer20UL18`  | `[1000, 2000, 3000, 4000, 5000, 6000]` |

These match the `COMPARISON_WR_MASSES_BY_ERA` dict in
[`signal_shape_study.py`](signal_shape_study.py#L49-L60). The list lives in a
top-of-file dict; adding new eras is a one-line change.

## Outputs

### `resolution_model.py` outputs

Written to `fitting_v2/outputs/<era>/resolution_model/`:

1. **`energy_decomposition.pdf`** — channel-independent. $E_{\ell_1}$,
   $E_{\ell_2}$, $E_{jj}$ vs $M_N/M_{W_R}$ for each WR mass overlaid.
   Pure kinematics, no fit needed.

2. **`resolution_model_rms_peak_<channel>.pdf`** — predicted
   $\sigma(m_{\ell\ell jj})$ curves on top of measured RMS$_{\text{peak}}$
   points, one curve per WR mass. The primary result plot.

3. **`resolution_model_fwhm_<channel>.pdf`** — same as (2) but on the
   full-histogram FWHM axis, plus a dashed curve from the RMS peak fit
   scaled by 2.355 for comparison.

4. **`component_contribution_<channel>.pdf`** — overlay of
   $\sigma(E_{\ell_1})$, $\sigma(E_{\ell_2})$, $\sigma(E_{jj})$, and
   their quadrature sum vs $M_N/M_{W_R}$ for one representative WR mass
   (default WR4000). Shows which component drives the width in each
   kinematic regime. Fitted parameters are annotated on the plot.

5. **`resolution_model_fit.json`** — two parameter blocks (`rms_fit`
   and `fwhm_fit`) with fitted values, uncertainties, and $\chi^2$/ndf.

### `signal_shape_study.py` outputs

Written to `fitting_v2/outputs/<era>/signal_shape_study_core/`:

1. **`overlay_WR<mass>_<channel>.pdf`** — normalized shape overlays per
   WR mass group, with bifurcated Gaussian ±3σ window shaded under each
   histogram.

2. **`overlay_WR<mass>_lowN_<channel>.pdf`** — zoomed overlays for
   WR4000 and WR6000 with $M_N \leq 1900$ GeV, showing the bimodal
   structure and window isolation clearly.

3. **`fit_plots/fit_<signal>_<channel>.pdf`** — per-mass-point
   diagnostic plots showing the histogram, bifurcated Gaussian fit
   curve, ±3σ window (shaded), $M_{W_R}$ vertical line, and annotated
   fit parameters ($\mu$, $\sigma_L$, $\sigma_R$, $\chi^2$/ndf,
   RMS$_{\text{peak}}$, FWHM$_{\text{peak}}$).

4. **Shape parameter trend plots** vs $M_N/M_{W_R}$ (WR8000 excluded):
   - `shape_mean_peak_<channel>.pdf` — on-shell peak mean
   - `shape_rms_peak_<channel>.pdf` — on-shell peak RMS
   - `shape_fwhm_peak_<channel>.pdf` — on-shell peak FWHM
   - `shape_rms_<channel>.pdf` — full-histogram RMS (for comparison)
   - `shape_fwhm_<channel>.pdf` — full-histogram FWHM
   - `shape_mean_<channel>.pdf`, `shape_rms_core_<channel>.pdf`,
     `shape_skewness_<channel>.pdf`

5. **`shape_summary.json`** — all shape parameters per signal point per
   channel, including `mean_peak`, `rms_peak`, `fwhm_peak`,
   `bifur_mu`, `bifur_sigma_l`, `bifur_sigma_r`, `bifur_chi2_ndf`,
   plus the full-histogram `mean`, `rms`, `fwhm`, and the iterative
   `rms_core` with its window bounds.

## Usage

```bash
# Run 3 (sparse WR grid, 3 anchors)
python fitting_v2/resolution_model.py --era RunIII2024Summer24 -v

# Run 2 (dense WR grid, 6 anchors — much stronger M_WR scaling constraint)
python fitting_v2/resolution_model.py --era RunIISummer20UL18 -v
```

Common flags:
- `--era`: required; one of the keys in `WR_MASSES_BY_ERA`
- `--dir`: optional subdirectory under input/output paths (passed through to
  `signal_shape_study.py` JSON lookup)
- `--rep-wr-mass`: WR mass for the component-contribution plot (default 4000)
- `-v` / `--verbose`: enable DEBUG logging

## Decisions baked into the model

| Choice | Default | Why |
|---|---|---|
| Fit target | Both RMS and FWHM, in parallel | RMS is the physics answer; FWHM fit is the diagnostic. Both parameter blocks in the output JSON; both curves on the FWHM overlay so distortion is visible. |
| Eras | Run 3 + Run 2 | Run 3 sparse grid is the active dataset; Run 2 dense grid provides much stronger `M_WR` scaling constraint. |
| ee/μμ coupling | Joint fit, shared `(a_j, b_j)`, separate lepton terms | Jets are physically identical between channels — sharing them disentangles lepton from jet resolution cleanly. |
| Free parameters | 6: `a_e, b_e, a_μ, b_μ, a_j, b_j` | Minimum to test the three-component picture. |
| Secondary-lepton avg energy | `⟨E_ℓ2*⟩ = M_N/3` (equipartition) | Simplest sensible choice for a 3-body decay. |
| Boost effects | Ignored (treat W_R as at rest) | LHC boost is a small per-event smearing on top, not a systematic shift in scale. |

## Sanity checks

The script logs the following at INFO level so the model can be eyeballed
before trusting any fit:

1. **Energy conservation**: `E_ℓ1 + E_ℓ2 + E_jj = M_WR` to numerical precision
   at sample grid points.
2. **Boundary behavior**: `E_ℓ1(M_N/M_WR=0) = M_WR/2`, `E_ℓ1(M_N/M_WR=1) = 0`.
3. **Unit resolution check** with CMS-like reference values
   (`b_e=0.01, b_μ=0.01, a_μ=1.5e-4, b_j=0.05`, all stochastic terms = 0):
   evaluate `σ(m_lljj)` at WR4000 N1900 and confirm it lands in the right
   ballpark (a few hundred GeV) before fitting.
4. **Fit convergence**: χ²/ndf < ~5 (we want "captures the trend", not a
   precision fit). All fitted parameters should be positive; `b_e, b_j` should
   land within a factor of ~2 of CMS-published constant terms.
5. **Overlay plot**: predicted curves should track the U-shape, with the ee
   curve sitting slightly below the μμ curve at low `M_N/M_WR`. If the model
   can't reproduce the U-shape qualitatively, the quadrature-sum assumption is
   too crude and we'd need to revisit (e.g. weight by `E_i/M_WR`, or compute
   real partial derivatives).
6. **RMS-vs-FWHM parameter comparison**: a side-by-side table of the six
   fitted parameters from the two fits per era is printed at the end. Lepton
   constant terms `b_e, b_μ` should be roughly stable; jet `b_j` may shift more
   if the FWHM fit is dominated by the noisy peak shape.

## Results (RunIII2024Summer24)

### Peak-window measurement strategy

The `m_lljj` distribution at low `M_N/M_{W_R}` is bimodal: an on-shell
peak near $M_{W_R}$ plus a low-mass bump from boosted-N reconstruction
failures (see `overlay_WR6000_lowN_ee.png` and
`overlay_WR6000_lowN_mumu.png` in the `signal_shape_study_core/`
output). Computing the RMS or FWHM over the full histogram gives a
width dominated by the separation between the two modes, not the
detector resolution of the on-shell peak.

To isolate the on-shell peak, we fit a **bifurcated Gaussian** (a
Gaussian with separate left and right widths $\sigma_L$ and $\sigma_R$)
within a fixed window $[0.7 \times M_{W_R},\; 1.3 \times M_{W_R}]$,
seeded at $M_{W_R}$. The fit finds $\mu$, $\sigma_L$, $\sigma_R$,
and $\chi^2$/ndf. We then compute the histogram-based mean, RMS, and
FWHM **within the bifurcated Gaussian window**
$[\mu - 3\sigma_L,\; \mu + 3\sigma_R]$. These are called `mean_peak`,
`rms_peak`, and `fwhm_peak`.

Per-mass-point fit diagnostic plots showing the histogram, bifurcated
Gaussian curve, ±3σ window, and fitted parameters are in
`signal_shape_study_core/fit_plots/`.

### Fit quality

Two parallel fits are run: one against `rms_peak` (the on-shell peak
RMS), one against the original full-histogram FWHM.

| Fit target | χ²/ndf | Points | Interpretation |
|---|---|---|---|
| **RMS peak** | **1.26** | 108 | Excellent fit — model reproduces the on-shell peak width across the full grid. |
| FWHM (full histogram) | 4.17 | 108 | Reasonable — noisier because the bin-scan FWHM estimator has bin-to-bin fluctuations. |

The RMS peak fit is the primary result. By measuring the peak width
within the bifurcated Gaussian window, the reconstruction tails are
excluded and the model can focus on pure detector resolution. All 108
points (54 grid points × 2 channels) are included — no ratio cuts or
excluded mass points needed.

### Fitted parameters (RMS peak fit)

| Parameter | Value | Physical meaning |
|---|---|---|
| $a_e$ | ~0 | ECAL stochastic term — has no leverage at TeV energies. |
| $b_e$ | 7.7% | ECAL constant term (effective — absorbs geometric factors). |
| $a_\mu$ | $4.66 \times 10^{-5}$ /GeV | Muon sagitta term. Gives $\sigma/p \sim 5\%$ at 1 TeV. |
| $b_\mu$ | 11.2% | Muon constant term. |
| $a_j$ | 4.1 $\sqrt{\text{GeV}}$ | Jet stochastic term. |
| $b_j$ | 14.2% | Jet constant term (effective — absorbs JES, JER, pairing). |

The muon sagitta term $a_\mu = 4.66 \times 10^{-5}$ /GeV agrees to
within 1% between the RMS peak fit and the FWHM fit ($4.62 \times
10^{-5}$), confirming it is the most robustly determined parameter.
The jet parameters $(a_j, b_j)$ are shared between ee and $\mu\mu$
and are identical by construction.

### On-shell peak width: ee vs $\mu\mu$

The `rms_peak` trend plots (`shape_rms_peak_ee.pdf` and
`shape_rms_peak_mumu.pdf`) and the resolution model overlays
(`resolution_model_rms_peak_ee.pdf` and
`resolution_model_rms_peak_mumu.pdf`) show the two key observations
cleanly:

**ee: monotonically increasing.** The on-shell peak gets wider as
$M_N/M_{W_R}$ increases. This is driven by the jet system hardening —
as $M_N$ grows, more energy goes into the jets (see
`energy_decomposition.pdf`, dash-dot $E_{jj}$ curves rising), and the
jet constant term ($b_j = 14.2\%$) turns that into a larger absolute
width. The primary electron is well-measured at all energies ($b_e
= 7.7\%$, roughly constant fractionally), so it contributes a small,
falling component that doesn't compensate. In
`component_contribution_ee.pdf`: the blue curve (primary lepton, ~150
GeV at low ratio) falls while the purple curve (jets, ~240 → 400 GeV)
rises. The total (black) follows the jets upward.

**$\mu\mu$: flat.** The on-shell peak width barely changes across the
full $M_N/M_{W_R}$ range. At low ratio, the primary muon is hard and
poorly measured — the sagitta term $a_\mu \cdot p$ makes the fractional
resolution degrade linearly with momentum. In
`component_contribution_mumu.pdf`: the blue curve (primary muon) starts
at ~290 GeV — **twice** the ee electron contribution at the same
energy. As $M_N$ grows, the muon softens (blue falls) while jets harden
(purple rises). The two effects approximately cancel, and the total
(black) stays flat through the crossover at ratio ~0.55, then gently
rises.

**Convergence at high ratio.** At high $M_N/M_{W_R}$, the primary
lepton energy drops toward zero in both channels (see
`energy_decomposition.pdf`, right edge). The jet system dominates — and
jets are identical between channels. The ee and $\mu\mu$ curves
converge.

### The single parameter responsible

The muon sagitta term $a_\mu$ is the one parameter that creates the
entire ee vs $\mu\mu$ difference. Electrons don't have an analogous term
because ECAL energy resolution is calorimetric (improves or stays flat
with energy), not tracking-based. The fitted $a_\mu \approx 4.7 \times
10^{-5}$ /GeV is consistent between the RMS peak fit and the FWHM fit,
confirming it is a stable detector property.

### Full-histogram RMS: the reconstruction-tail U-shape

The full-histogram RMS (without the peak windowing) shows a U-shape
with a steep left-hand rise at low $M_N/M_{W_R}$ — visible in
`shape_rms_ee.pdf` and `shape_rms_mumu.pdf`. This is **not** detector
resolution; it is driven by reconstruction failures in the boosted-N
topology:

- When $N$ is light and highly boosted, its decay products are
  collimated. The resolved-SR jet pairing can fail (merge jets, pick
  wrong pair), producing events with reconstructed $m_{\ell\ell jj}$
  far below $M_{W_R}$.
- These events form a long low-side tail. At WR6000 N700 ee: the
  on-shell peak has RMS$_{\text{peak}}$ = 465 GeV, but the
  full-histogram RMS is 1997 GeV — the tail inflates it by 4×.
- The `m_lljj` distribution is visibly bimodal at the most extreme
  points (see `fit_plots/fit_WR6000_N700_ee.png` and
  `overlay_WR6000_lowN_ee.png`).

The peak-windowed RMS eliminates this contamination, which is why the
`rms_peak` trend plots show clean monotonic (ee) or flat ($\mu\mu$)
behavior instead of the U-shape.

## Limitations

- **Quadrature sum is too simple if it fails**: ignoring `(∂m/∂E_i)` partial
  derivatives means each component is assumed to contribute its own absolute
  resolution to `m_lljj`. This is right for a back-to-back system but wrong
  for a fully-connected 4-body invariant mass.
- **Lab-frame ≠ rest-frame**: the script ignores the boost from the
  parton-level frame to the lab. For `M_WR ≳ 4 TeV` at 13.6 TeV `pp`, the
  W_R is *not* produced at rest, but the per-event boost is a smearing on
  top of the kinematic scaling, not a shift.
- **Equipartition for 3-body N decay** is an O(10%) approximation; a real
  V−A matrix-element-weighted average would shift `⟨E_ℓ2*⟩` slightly.
- **Primary/secondary lepton labeling is not observable.** The model
  assigns "primary" to the lepton from `W_R → ℓ_1 N` and "secondary"
  to the lepton from `N → ℓ_2 q q'`, but reconstructed events only have
  leading and subleading leptons ordered by `p_T`. At low `M_N/M_WR` the
  primary is almost always the leading lepton; at high `M_N/M_WR` the
  secondary can be harder and they swap. This doesn't affect the model
  because both leptons are the same flavor (same resolution function)
  and the quadrature sum `σ(E_ℓ1)² + σ(E_ℓ2)²` is symmetric in the
  labeling. The model only needs the *typical energy scale* of each
  bucket, which the two-body/three-body kinematics provides correctly
  regardless of which lepton ends up leading in `p_T`.
- **No correlations**: real `m_lljj` resolution has correlated components
  (e.g. JES affects both jets coherently), which the quadrature sum drops.
- **Jet term is "summed jet system"**, not per-jet, so it doesn't break out
  effects from pairing failures in boosted topologies — those instead show
  up as a tail in the data that the RMS still sees but a Gaussian model can't.
