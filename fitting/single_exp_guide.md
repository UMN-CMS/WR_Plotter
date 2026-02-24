# Single Exponential Fit: How It Works

Step-by-step explanation of the single exponential background fitting pipeline.

## 1. Input: MC background histogram

The script loads ROOT files for every background process (DYJets, tt+tW, Other, Nonprompt), extracts the m_lljj histogram from each, and sums them into one total background histogram. This summed histogram is what we fit — it represents what we'd expect to see in data (before unblinding).

## 2. Rebinning

The original histogram has 10 GeV bins. `--rebin N` merges every N bins together. So `--rebin 10` gives 100 GeV bins. This controls the tradeoff:

- Too fine (20 GeV) → many bins with <1 event → terrible chi2
- Too coarse (500 GeV) → lose shape information → slope drifts
- Sweet spot: 100 GeV bins

## 3. The model: what is f(m)?

The single exponential model is:

```
f(m) = exp(c * m)
```

where `c` is a negative number (the slope). More negative = steeper falloff. This is a **shape** — it describes *how* events are distributed across mass, not *how many* there are.

To also fit the total number of events, we use an **extended PDF** (RooExtendPdf). This wraps the shape with a second parameter `n_bkg`:

```
L = Poisson(N_observed | n_bkg) * Product_over_bins[ f(m_i) ]
```

So the fit determines two things simultaneously:

- **c**: the exponential slope (shape)
- **n_bkg**: the total expected number of background events (normalization)

## 4. The likelihood

RooFit performs a **binned extended maximum-likelihood fit**. For each bin `i` with observed count `n_i`, the likelihood contribution is:

```
L_i = Poisson(n_i | mu_i)
```

where `mu_i` is the expected count in that bin from the model (integral of `n_bkg * f(m)` over the bin width). The total negative log-likelihood is:

```
-ln(L) = sum_i [ mu_i - n_i * ln(mu_i) + ln(n_i!) ]
```

Minuit2 minimizes this NLL by varying `c` and `n_bkg`.

## 5. What "fit converges" means

- **Status = 0**: Minuit found a minimum successfully
- **EDM** (estimated distance to minimum): should be tiny (~1e-8)
- **Covariance quality = 3**: the full Hessian matrix is positive-definite, so the parameter uncertainties are reliable

## 6. Goodness of fit: chi2/ndf

After fitting, we check how well the model describes the data:

```
chi2 = sum_i [ (n_i - mu_i)^2 / sigma_i^2 ]
```

Divided by ndf = (number of non-empty bins) - (number of parameters). For single-exp, that's `n_bins - 2`.

- chi2/ndf ~ 1 means the model describes the data well
- chi2/ndf >> 1 means systematic deviations (the model doesn't capture the true shape)

## 7. The pull plot

The lower panel of each fit plot shows per-bin pulls:

```
pull_i = (n_i - mu_i) / sigma_i
```

where `sigma_i` uses asymmetric Garwood (Poisson) errors — the upper error if the fit undershoots, the lower error if it overshoots. If the model is good, pulls should scatter randomly within +/-2 with no systematic trend.

## 8. Why chi2/ndf depends on binning

The single exponential is one smooth curve, but the true background is a **mixture** of two exponentials (tt+tW steep + DYJets shallow). At fine binning, the per-bin deviations from the single-exp are resolved and chi2 blows up. At coarser binning, those deviations get averaged out within each bin, and chi2 improves — but at the cost of losing sensitivity to the underlying shape.

The 100 GeV bin width happens to be coarse enough that bin-by-bin fluctuations are manageable, but fine enough that the slope `c` still converges to its true value (~-0.0029).

## 9. What the background composition tells us

Running with `--component-fits` fits DYJets and tt+tW individually with single exponentials, and produces a composition plot showing each process's fraction of the total vs mass. This is critical for interpreting the total single-exp fit.

### The two true slopes are different

The component fits for [800, 6000] GeV at 100 GeV bins (ee channel) give:

| Component | c (slope) | n_bkg (yield) | Fraction | chi2/ndf |
|-----------|-----------|---------------|----------|----------|
| tt+tW | -0.00323 | 1180 | 71% | 3.59 |
| DYJets | -0.00235 | 399 | 24% | 1.19 |
| Other + Nonprompt | — | ~80 | ~5% | — |
| **Total** | **-0.00291** | **1659** | 100% | **1.98** |

tt+tW falls steeply (more negative `c`) and dominates at low mass. DYJets falls more slowly and its relative fraction grows with mass — from ~24% at 800 GeV toward ~50% above 2500 GeV.

### The total slope is a compromise

The fitted total slope of -0.00291 sits right between the two component slopes (-0.00323 and -0.00235). It's essentially a yield-weighted average, pulled closer to tt+tW because that process contributes 71% of the events.

### Why chi2/ndf = 1.98 and not 1.0

A single exponential cannot perfectly describe a mixture of two exponentials with different slopes. This produces a characteristic systematic pattern in the pulls:

- **Low mass (~800-1200 GeV)**: tt+tW dominates, so the true distribution falls steeply. The compromise slope (-0.00291) is too shallow → the fit **undershoots** the data → **positive pulls**.
- **Mid mass (~1200-2500 GeV)**: the compromise slope happens to match the effective mixture → pulls are small.
- **High mass (~2500+ GeV)**: DYJets becomes a larger fraction and the true distribution flattens. The compromise slope is too steep → the fit **overshoots** → **negative pulls**.

This S-shaped pull pattern is the signature of fitting a single exponential to a two-component mixture. The chi2/ndf of ~2 is not a statistical fluke or a binning artifact — it reflects a real physics effect.

### Why DYJets fits better than tt+tW individually

DYJets is well-described by a single exponential on its own (chi2/ndf = 1.19). tt+tW is less so (chi2/ndf = 3.59) — it likely has internal structure from combining top pair production and single top (tW), which have slightly different kinematics but are merged into one sample group.

### Bottom line

The composition information confirms that the single-exp chi2/ndf of ~2 is a fundamental limitation of the model, not something that can be fixed by adjusting the binning or mass range. The true background is a mixture of two exponentials, and a single exponential can only approximate it. This is what motivates the double exponential model.
