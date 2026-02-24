# Background Fitting

Analytic background estimation for the four-object invariant mass (m_lljj) distribution in the W_R -> lljj signal region, upstream of the CMS Combine framework.

## Overview

The fitting procedure uses RooFit (via PyROOT) to perform binned maximum-likelihood fits of analytic functions to the total MC background m_lljj spectrum. This is exploratory work to determine the optimal functional form and stable fit parameters before formal limit-setting with Combine.

## Models

| Model | Function | Parameters | Script |
|-------|----------|------------|--------|
| Single exponential | f(m) = exp(c * m) | c | `fit_single_exp.py` |
| Double exponential | f(m) = N1 * exp(c1 * m) + N2 * exp(c2 * m) | c1, c2, N1, N2 | `fit_background.py` |
| Power-law exponential | f(m) = m^a * exp(c * m) | a, c | `fit_background.py` |

The single exponential serves as a baseline; the double exponential is the target model for the final background estimate. The power-law exponential offers similar flexibility to the double-exp without the degeneracy issues. All models use extended PDFs (RooExtendPdf) so the likelihood constrains both shape and normalization.

## Background composition

The summed MC background is dominated by two processes whose m_lljj spectra have different slopes:

- **tt+tW** dominates at low mass (~75% at 1000 GeV), but its fraction decreases with mass
- **DYJets** grows from ~25% at 1000 GeV to ~50% around 2500-3000 GeV, consistent with its shallower exponential slope
- Above ~3000 GeV, statistics become sparse (< 1 event/bin) and the fractional composition fluctuates
- **Other** and **Nonprompt** are small (~5% combined) throughout the fit range

This composition motivates the double exponential: the total spectrum is a mixture of a steep component (tt+tW-dominated) transitioning to a shallower one (DYJets-dominated) with increasing mass. Use `--component-fits` to fit each component individually and visualize the composition.

## Stages

1. **Single-region fit** (current): Fit the m_lljj distribution in a single signal region (e.g., ee resolved SR) with each analytic model. Iterate on parameter initialization and ranges until the fit converges with reasonable uncertainties.

2. **Simultaneous SR + flavor CR fit**: Fit the signal region and flavor control region simultaneously using `RooSimultaneous`. The flavor CR constrains the softer exponential component, exploiting the e-mu symmetry of flavor-symmetric backgrounds (tt, tW).

3. **Signal injection / closure test**: Inject a small fraction of MC signal onto the background, mask the signal window, and refit. Validates that the background model is unbiased in the presence of a signal.

4. **Signal strength fit**: Extend the model to background + mu * signal, where mu is the signal strength and the signal shape comes from MC templates. Scan over W_R mass points.

5. **Sensitivity estimate**: Compute the expected uncertainty on the signal strength (sigma_mu / mu) as a function of W_R mass to identify the most sensitive mass range.

## Scripts

| Script | Purpose |
|--------|---------|
| `fit_single_exp.py` | Single exponential fits (total + per-component) |
| `fit_background.py` | Double exponential and power-law exponential fits |
| `fit_utils.py` | Shared infrastructure (histogram loading, RooFit helpers, plotting) |

## Usage

### Single exponential

```bash
# Both channels, default settings:
python fitting/fit_single_exp.py --era RunIII2024Summer24 --dir 20260223_lo_dy

# Custom mass range and binning:
python fitting/fit_single_exp.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
    --mass-range 1000 4000 --rebin 10

# With per-component fits and composition plot:
python fitting/fit_single_exp.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
    --component-fits
```

### Double / power-law exponential

```bash
# Double exponential:
python fitting/fit_background.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
    --model double-exp

# Power-law exponential:
python fitting/fit_background.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
    --model pow-exp

# Scan double-exp initializations:
python fitting/fit_background.py --era RunIII2024Summer24 --dir 20260223_lo_dy \
    --model double-exp --scan
```

### Shared arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--era` | *(required)* | MC era (must match a key in `data/lumi.yaml`) |
| `--dir` | *(required)* | Input subdirectory name |
| `--channel` | *(both)* | Lepton channel: `ee` or `mumu`. If omitted, runs both. |
| `--topology` | resolved | Event topology: `resolved` or `boosted` |
| `--mass-range` | 800 6000 | Observable range [GeV] for the fit |
| `--rebin` | 20 | Rebin factor applied to the histogram |
| `--verbose` | off | Enable debug logging |

### Script-specific arguments

| Script | Argument | Description |
|--------|----------|-------------|
| `fit_single_exp.py` | `--component-fits` | Fit DYJets and tt+tW individually and plot composition vs mass |
| `fit_background.py` | `--model` | Fit model: `double-exp` (default) or `pow-exp` |
| `fit_background.py` | `--scan` | (double-exp only) Scan initialization grid and report all minima |

## Output

Each fit produces files in `fitting/outputs/<era>/<model>/<mass-range>_<binning>/`:

- `fit_<channel>.json` — fit results: parameter values, uncertainties, correlation matrix, chi2/ndf, and metadata
- `fit_<channel>.pdf` — publication-quality plot (CMS style via mplhep)
- `fit_<channel>.png` — rasterized version for quick inspection

With `--component-fits` (single-exp only), additional files are produced:
- `fit_DYJets_<channel>.json/.pdf/.png` — per-component fit
- `fit_tt_tW_<channel>.json/.pdf/.png` — per-component fit
- `composition_<channel>.json/.pdf/.png` — background composition vs mass

### Plot contents

- **Upper panel**: Binned MC background (points with error bars) overlaid with the fitted analytic curve. Includes CMS preliminary label, luminosity, region identification, and a fit information box showing the model equation, fitted parameter values, and chi2/ndf.
- **Lower panel**: Pull distribution (data - fit) / error per bin, with shaded +/-1 sigma and +/-2 sigma reference bands.

## Notes

- Fits treat MC as pseudodata with Poisson bin errors, matching the statistical model that will be used on real data.
- RooFit exponential models are sensitive to parameter initialization and allowed ranges. The double exponential in particular may require staged fitting (fix one component, fit the other, then release all parameters).
- The background histograms are read from ROOT files in `rootfiles/` and summed across all MC sample groups for the requested era/channel/topology.
- **The signal region is blinded.** All fits are performed on summed MC background histograms, not observed data. Real data in the SR must not be used.
