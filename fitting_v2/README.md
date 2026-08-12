# fitting_v2 — Background & Signal+Background Fitting

Binned extended maximum-likelihood fits to MC m(lljj) distributions using
RooFit (PyROOT). Supports background-only and signal+background (S+B) closure
tests for the W_R &rarr; lljj search.

## Environment

Requires PyROOT. On lxplus or LPC, source the LCG view:

```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
```

All scripts should be run from the repository root.

## Scripts

| Script | Purpose |
|--------|---------|
| `signal_window.py` | Compute signal mean & RMS from MC histogram; define fitting window |
| `fit_background.py` | Fit background (or S+B) model in the signal window |
| `fit_signal.py` | Parametric signal shape fits (Gaussian, Crystal Ball, etc.) |
| `fit_utils.py` | Shared library: histogram loading, model builders, fitting, plotting |

## Workflow

### Step 1: Define the signal window

`signal_window.py` loads the signal MC m(lljj) histogram and computes `TH1::GetMean()`
and `TH1::GetRMS()` directly (no parametric fit). The fitting window is:

```
[mean - n_sigma * RMS,  mean + n_sigma * RMS]
```

```bash
python fitting_v2/signal_window.py \
    --era RunIII2024Summer24 \
    --dir 20260223_nlo_dy \
    --signal WR2000_N1100 \
    --verbose
```

Outputs per channel:
- `fitting_v2/outputs/<era>/signal_window/<signal_tag>/window_<channel>.json` — mean, RMS, window edges
- `fitting_v2/outputs/<era>/signal_window/<signal_tag>/signal_<channel>.pdf` — signal histogram with mean (dashed vertical line), RMS (horizontal bar), and shaded fitting window

Key options:
- `--n-sigma`: window half-width in RMS units (default: 3.0)
- `--rebin`: merge bins before computing statistics (default: 2)
- `--channel`: run only `ee` or `mumu` (default: both)

### Step 2: Fit the background

`fit_background.py` loads the window JSON from Step 1, sums all MC background
histograms, and fits an analytic model in the full window (no sidebands, no
blinding).

#### Background-only fit

```bash
python fitting_v2/fit_background.py \
    --era RunIII2024Summer24 \
    --dir 20260223_nlo_dy \
    --signal WR2000_N1100 \
    --verbose
```

#### Signal+background (S+B) closure test

Add `--inject-signal` to inject signal MC into the pseudodata and fit a
combined S+B model. The background shape is the same analytic model; the signal
template is a `RooHistPdf` built from the signal MC histogram (no parametric
assumption).

```bash
# Full signal injection (mu = 1):
python fitting_v2/fit_background.py \
    --era RunIII2024Summer24 \
    --dir 20260223_nlo_dy \
    --signal WR2000_N1100 \
    --inject-signal --verbose

# Partial injection (mu = 0.1):
python fitting_v2/fit_background.py \
    --era RunIII2024Summer24 \
    --dir 20260223_nlo_dy \
    --signal WR2000_N1100 \
    --inject-signal --inject-mu 0.1 --verbose
```

The fitted signal strength is reported as:

```
mu_fitted = n_sig_fitted / n_sig_expected
```

where `n_sig_expected` is the signal MC yield in the window at mu=1. A closure
test passes when `mu_fitted` recovers the injected `mu` within uncertainties.

#### Background model options

```bash
# Double exponential:
python fitting_v2/fit_background.py \
    --era RunIII2024Summer24 \
    --dir 20260223_nlo_dy \
    --signal WR2000_N1100 \
    --model double-exp --verbose
```

| Model | Formula | Parameters |
|-------|---------|------------|
| `single-exp` (default) | N * exp(c * m) | c, N_bkg |
| `double-exp` | N1 * exp(c1 * m) + N2 * exp(c2 * m) | c1, c2, N1, N2 |

Note: the double-exp is degenerate in single-region fits (corr(N1, N2) ~ -1)
because the two slopes are only ~20% apart. A simultaneous fit with a flavor
control region (Stage 2) would be needed to resolve the components.

### Step 3 (optional): Parametric signal shape studies

`fit_signal.py` fits parametric models to the signal MC to study the signal
shape. This is independent of the window-based workflow above.

```bash
python fitting_v2/fit_signal.py \
    --era RunIII2024Summer24 \
    --dir 20260223_nlo_dy \
    --signal WR2000_N1100 \
    --model crystal-ball --verbose
```

Available models: `gaussian`, `crystal-ball`, `voigtian`, `breit-wigner`, `bifur-gauss`.

Each fit runs both RooFit (binned MLE) and scipy (chi2 least-squares) for
cross-validation. Results and overlay plots are saved per model.

## Output structure

```
fitting_v2/outputs/<era>/
    signal_window/<signal_tag>/
        window_ee.json      window_mumu.json
        signal_ee.pdf/png   signal_mumu.pdf/png
    background/<signal_tag>/
        fit_ee.json     fit_ee.pdf     fit_ee.png
        fit_mumu.json   fit_mumu.pdf   fit_mumu.png
    sb_injection/<signal_tag>/
        fit_ee.json     fit_ee.pdf     fit_ee.png
        fit_mumu.json   fit_mumu.pdf   fit_mumu.png
    signal_shape/<signal_tag>/<model>/
        fit_ee.json     fit_ee.pdf     fit_ee.png
        fit_mumu.json   fit_mumu.pdf   fit_mumu.png
```

### JSON output format

**signal_window**: `window_<channel>.json`
```json
{
  "metadata": { "era", "channel", "signal", ... },
  "signal_stats": { "mean", "rms", "n_events", "skewness", "kurtosis" },
  "fit_window": { "lo", "hi", "n_sigma" }
}
```

**fit_background (bkg-only)**: `fit_<channel>.json`
```json
{
  "metadata": { "era", "channel", "model", "mode": "bkg-only", "fit_window", ... },
  "roofit": {
    "fit_status": { "status", "edm", "cov_quality", "min_nll" },
    "parameters": { "<name>": { "value", "error", "error_lo", "error_hi", "range" } },
    "correlation_matrix": [[...]],
    "goodness_of_fit": { "chi2_per_ndf", "ndf", "n_bins", "chi2" }
  }
}
```

**fit_background (S+B)**: adds a `signal_injection` block:
```json
{
  "metadata": { "mode": "S+B", ... },
  "roofit": { ... },
  "signal_injection": {
    "inject_mu": 1.0,
    "n_sig_expected": 9677.8,
    "n_sig_fitted": 9660.3,
    "n_sig_error": 127.4,
    "mu_fitted": 0.998,
    "mu_error": 0.013
  }
}
```

## Common CLI options

These options are shared across all scripts (defined in `fit_utils.py`):

| Option | Default | Description |
|--------|---------|-------------|
| `--era` | (required) | Era tag, e.g. `RunIII2024Summer24` |
| `--dir` | auto | Input directory override |
| `--signal` | (required) | Signal tag, e.g. `WR2000_N1100` |
| `--channel` | both | `ee` or `mumu`; omit for both |
| `--topology` | `resolved` | `resolved` or `boosted` |
| `--rebin` | 2 | Rebin factor (original bins are 10 GeV) |
| `--n-sigma` | 3.0 | Window half-width in sigma |
| `--mass-range` | `800 6000` | Observable range in GeV |
| `--output-dir` | auto | Override output directory |
| `--verbose` | off | Enable debug logging |

## Technical notes

- All models use **extended PDFs** (`RooExtendPdf` or `RooAddPdf` of extended
  components) so the likelihood constrains both shape and normalization.
- PyROOT garbage collection can cause segfaults when Python deletes C++ objects
  still referenced by RooFit. All intermediate RooFit objects use
  `ROOT.SetOwnership(obj, False)` to prevent this.
- The S+B signal template is a **`RooHistPdf`** (non-parametric), built directly
  from the signal MC histogram. No parametric assumption is needed for the
  signal shape in the S+B fit.
- Fit plots use `mplhep` with CMS style. Upper panel shows data + fit curve
  (+ background component for S+B). Lower panel shows pulls per bin.
