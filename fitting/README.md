# Background Fitting

Analytic background estimation for the four-object invariant mass (m_lljj) distribution in the W_R -> lljj signal region, upstream of the CMS Combine framework.

## Overview

The fitting procedure uses RooFit (via PyROOT) to perform binned maximum-likelihood fits of analytic functions to the total MC background m_lljj spectrum. This is exploratory work to determine the optimal functional form and stable fit parameters before formal limit-setting with Combine.

## Models

| Model | Function | Parameters |
|-------|----------|------------|
| Single exponential | f(m) = exp(c * m) | c |
| Double exponential | f(m) = frac * exp(c1 * m) + (1-frac) * exp(c2 * m) | c1, c2, frac |

The single exponential serves as a baseline; the double exponential is the target model for the final background estimate.

## Stages

1. **Single-region fit** (current): Fit the m_lljj distribution in a single signal region (e.g., ee resolved SR) with each analytic model. Iterate on parameter initialization and ranges until the fit converges with reasonable uncertainties.

2. **Simultaneous SR + flavor CR fit**: Fit the signal region and flavor control region simultaneously using `RooSimultaneous`. The flavor CR constrains the softer exponential component, exploiting the e-mu symmetry of flavor-symmetric backgrounds (tt, tW).

3. **Signal injection / closure test**: Inject a small fraction of MC signal onto the background, mask the signal window, and refit. Validates that the background model is unbiased in the presence of a signal.

4. **Signal strength fit**: Extend the model to background + mu * signal, where mu is the signal strength and the signal shape comes from MC templates. Scan over W_R mass points.

5. **Sensitivity estimate**: Compute the expected uncertainty on the signal strength (sigma_mu / mu) as a function of W_R mass to identify the most sensitive mass range.

## Usage

```bash
python fitting/fit_background.py \
    --era RunIII2024Summer24 \
    --channel ee \
    --topology resolved \
    --model single-exp \
    --mass-range 800 6000 \
    --rebin 20
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--era` | RunIII2024Summer24 | MC era (must match a key in `data/lumi.yaml`) |
| `--channel` | ee | Lepton channel: `ee` or `mumu` |
| `--topology` | resolved | Event topology: `resolved` or `boosted` |
| `--model` | single-exp | Fit model: `single-exp` or `double-exp` |
| `--mass-range` | 800 6000 | Observable range [GeV] for the fit |
| `--rebin` | 20 | Rebin factor applied to the histogram |
| `--verbose` | off | Enable debug logging |

## Output

Each fit produces three files in `fitting/outputs/<era>/`:

- `fit_<model>.json` — fit results: parameter values, uncertainties, correlation matrix, chi2/ndf, and metadata
- `fit_<model>.pdf` — publication-quality plot (CMS style via mplhep)
- `fit_<model>.png` — rasterized version for quick inspection

### Plot contents

- **Upper panel**: Binned MC background (points with error bars) overlaid with the fitted analytic curve. Includes CMS preliminary label, luminosity, region identification, and a fit information box showing the model equation, fitted parameter values, and chi2/ndf.
- **Lower panel**: Pull distribution (data - fit) / error per bin, with shaded +/-1 sigma and +/-2 sigma reference bands.

## Notes

- Fits use `SumW2Error(True)` to correctly handle weighted MC events (applies Hessian correction for the effective number of entries).
- RooFit exponential models are sensitive to parameter initialization and allowed ranges. The double exponential in particular may require staged fitting (fix one component, fit the other, then release all parameters).
- The background histograms are read from ROOT files in `rootfiles/` and summed across all MC sample groups for the requested era/channel/topology.
