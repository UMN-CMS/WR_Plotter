# Signal Fit Studies: RooFit vs Scipy Comparison

This document records findings from comparing RooFit and scipy-based signal
shape fits on WR -> lljj signal MC, and the changes made to align them.

## Overview

`fit_signal.py` fits a parametric model to the signal MC m_lljj distribution
to characterize the peak shape and define a blinding window. It runs two
independent fits:

- **RooFit**: using PyROOT's RooFit framework
- **Scipy cross-check**: using scipy PDFs (norm, crystalball, voigt_profile, etc.)

The two fitters originally disagreed significantly on fitted parameters,
chi2/ndf, and blinding windows. This document explains why and what was done
about it.

## The two fit methods

### Chi-squared (least-squares)

For each bin, compute how far off the model is from the data, in units of
the uncertainty:

    residual = (observed - expected)^2 / error^2

Sum this over all bins — that's chi-squared. The best-fit parameters minimize
this sum. Bins with large errors (few events) contribute less; bins with small
errors (many events) contribute more. Empty bins are skipped entirely.

### Maximum Likelihood Estimation (MLE)

Instead of asking "how far off is each bin?", ask "given these parameter
values, how probable is it that I'd observe exactly the data I see?" The
probability for each bin follows Poisson statistics:

    P(n | mu) = mu^n * exp(-mu) / n!

where n is the observed count and mu is the model prediction. The likelihood
is the product of these probabilities across all bins, and MLE finds the
parameters that maximize it.

"Binned" means we work with histogram bins rather than individual events.
"Extended" means the total event count is predicted by the model (via n_sig),
not fixed.

## Sources of disagreement (default MLE mode)

### 1. Fit objective

RooFit maximizes Poisson likelihood (MLE). Scipy minimizes chi-squared.

The key practical difference is how they treat bins with few or zero events:

- **Chi-squared** skips empty bins entirely, and heavily downweights
  low-count bins (large error -> small contribution). It essentially fits
  only the peak region where most events are.

- **MLE** never skips bins. An empty bin where the model predicts mu events
  contributes -mu to the log-likelihood. MLE uses the *absence* of events
  in the tails as information, which pulls the model wider to cover the
  tails, resulting in larger fitted widths.

This is the dominant source of disagreement — RooFit consistently finds
wider signals than scipy in the default MLE mode.

### 2. Minimizer algorithm

RooFit uses Minuit2 migrad (a quasi-Newton method, the standard minimizer
in particle physics). Scipy's `curve_fit` uses Trust Region Reflective.

Both are iterative algorithms that search for the minimum of their respective
objective functions. They usually find the same answer, but can diverge when
multiple parameter combinations give similar fit quality (e.g., the Crystal
Ball sigma-alpha degeneracy: a wide Gaussian core with a gentle tail gives a
similar shape to a narrow core with an aggressive tail).

### 3. MC weights

Each MC event carries a weight (xsec * lumi / sum_w_gen). When ROOT fills a
weighted histogram, it tracks two quantities per bin:

- **sumw** (sum of weights): the bin content. E.g., 50 events with weight 10
  gives sumw = 500.
- **sumw2** (sum of weights squared): 50 events with weight 10 gives
  sumw2 = 50 * 10^2 = 5000. The bin error is sqrt(sumw2) = ~71.

Why sumw2 tracks uncertainty: heavier weights mean fewer raw events were used
to reach the bin content, so the measurement is less precise. If all weights
were 1 (real data), sumw2 = N and the error is sqrt(N) — ordinary Poisson
statistics.

The two fitters use these differently:

- **RooFit's MLE** only uses sumw (the bin content). The Poisson formula
  P(n | mu) has no input for "how uncertain is this bin" — it just takes the
  count and assumes it's exact. A bin showing 500 weighted events is treated
  as if you literally counted 500 events (precision ~sqrt(500) = 22), even
  though the true precision is sqrt(sumw2) = 71.

- **Scipy's chi-squared** uses both sumw (as the observed value in the
  numerator) and sqrt(sumw2) (as the error in the denominator). This
  correctly reflects the true statistical precision of the weighted MC.

### 4. ndf in chi2/ndf

The signal only occupies a portion of the [800, 6000] GeV fit range — roughly
155 bins have content, the other ~105 are empty. The ndf differs because:

- **Scipy's chi2/ndf** is evaluated at its fit minimum, counting only
  non-empty bins (ndf = 150-152 depending on model).
- **RooFit's chi2/ndf** is a separate diagnostic computed *after* the MLE
  fit (post-hoc), counting all bins in range (ndf = 255-257).

The ndf variation within each fitter (e.g., 255 vs 257) comes from different
models having different numbers of free parameters:

- 3 params (Gaussian, Breit-Wigner): ndf = 260 - 3 = 257
- 4 params (Bifurcated Gaussian, Voigtian): ndf = 260 - 4 = 256
- 5 params (Crystal Ball): ndf = 260 - 5 = 255

## Chi-squared fit mode (--fit-method chi2)

To make the two fitters directly comparable, `--fit-method chi2` applies
three changes:

### 1. RooFit objective: MLE -> chi2

RooFit switches from `fitTo` (MLE) to `chi2FitTo` with
`DataError(RooAbsData::SumW2)`, so both fitters minimize chi-squared
using the same sqrt(sumw2) bin errors.

### 2. Scipy minimizer: Trust Region Reflective -> Minuit2

The scipy cross-check switches from `curve_fit` (Trust Region Reflective)
to iminuit, which wraps the same Minuit2 migrad used by RooFit. The model
PDFs still come from scipy (norm, crystalball, voigt_profile, etc.).

### 3. Empty bin handling

Empty bins have zero error, which causes chi2FitTo to fail with division
by zero. These bins are assigned a very large error (1e10) so chi2FitTo
effectively skips them, matching iminuit's behavior. The post-hoc chi2
diagnostic and ndf count also exclude empty bins.

### Result

After these changes, both fitters agree on:
- Fitted parameters (mean, sigma, etc.)
- Blinding windows
- ndf

The residual chi2/ndf difference (~8%) comes from RooCBShape vs
scipy.stats.crystalball evaluating to slightly different values at the
same parameters — they are independent implementations of the same formula.

## Available models

| Model | Parameters | Description |
|-------|-----------|-------------|
| gaussian | mean, sigma, n_sig | Simple symmetric peak |
| crystal-ball | mean, sigma, alpha, cb_n, n_sig | Gaussian core + power-law low-mass tail (default) |
| voigtian | mean, width, sigma, n_sig | Breit-Wigner convolved with Gaussian |
| breit-wigner | mean, width, n_sig | Pure relativistic resonance |
| bifur-gauss | mean, sigma_lo, sigma_hi, n_sig | Asymmetric Gaussian with independent left/right widths |

## Usage

```bash
# Default MLE mode (RooFit MLE + scipy chi2 cross-check)
python fitting_v2/fit_signal.py \
    --era RunIII2024Summer24 \
    --dir 20260319_nlo_dy_systs \
    --signal WR2000_N1100

# Chi2 mode (both fitters use chi2 + Minuit2, MC normalized to N_eff)
python fitting_v2/fit_signal.py \
    --era RunIII2024Summer24 \
    --dir 20260319_nlo_dy_systs \
    --signal WR2000_N1100 \
    --fit-method chi2

# Try a different model
python fitting_v2/fit_signal.py \
    --era RunIII2024Summer24 \
    --dir 20260319_nlo_dy_systs \
    --signal WR2000_N1100 \
    --model voigtian
```
