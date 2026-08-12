# Toy Pseudodata Study Guide

## Purpose

Generate many toy pseudodata distributions from the MC background to
understand the expected range of fitted signal strength under the
**background-only hypothesis**. This replaces the single MC-based fit
with a distribution of outcomes, accounting for both Poisson counting
statistics and MC statistical uncertainties.

## Why Toys?

The MC background histogram uses **weighted events** — bins can have
fractional content (e.g., 0.3 events). Real data has **integer counts**
(0 or 1 events per bin). A single fit to the MC gives one estimate of
n_sig, but we need the *distribution* of possible outcomes to understand
our sensitivity.

## Statistical Method: Poisson-Gamma Compound

For each toy, for each bin of the MC histogram:

1. **Extract MC prediction**: `mu_i` (bin content) and `sigma_i` (bin error = sqrt(sumw2))
2. **Compute effective parameters**:
   - `k_eff = mu_i^2 / sigma_i^2` — effective unweighted event count
   - `theta = sigma_i^2 / mu_i` — effective weight per event
3. **Fluctuate the true rate**: `lambda_i ~ Gamma(shape=k_eff, scale=theta)`
   — this accounts for MC statistical uncertainty
4. **Sample observed count**: `n_i ~ Poisson(lambda_i)`
   — this gives integer counts like real data
5. **Empty bins**: if `mu_i <= 0`, set `n_i = 0`

This is equivalent to a **Negative Binomial** distribution and is the standard
Barlow-Beeston approach for propagating MC stat uncertainty into toys.

### Why two steps?

- The **Gamma step** fluctuates the "true" background rate within the MC
  uncertainty. If the MC has large weights (few events with large weight),
  `k_eff` is small and the fluctuation is large.
- The **Poisson step** produces integer counts from the fluctuated rate,
  mimicking real data collection.

## Usage

### Prerequisites

Run `signal_window.py` first to produce the signal window JSON:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

python fitting_v2/signal_window.py \
    --era RunIII2024Summer24 \
    --dir 20260223_nlo_dy \
    --signal WR2000_N1100
```

### Quick Test (10 toys, one channel)

```bash
python fitting_v2/toy_study.py \
    --era RunIII2024Summer24 \
    --dir 20260223_nlo_dy \
    --signal WR2000_N1100 \
    --channel ee --n-toys 10 --seed 42 --verbose
```

### Full Run (100 toys, both channels)

```bash
python fitting_v2/toy_study.py \
    --era RunIII2024Summer24 \
    --dir 20260223_nlo_dy \
    --signal WR2000_N1100 \
    --n-toys 100
```

### CLI Arguments

| Argument     | Default      | Description |
|--------------|--------------|-------------|
| `--era`      | (required)   | Era tag |
| `--dir`      | (auto)       | Input directory |
| `--signal`   | (required)   | Signal tag, e.g. `WR2000_N1100` |
| `--channel`  | both         | `ee` or `mumu`; both if omitted |
| `--n-toys`   | 100          | Number of toy datasets |
| `--model`    | `single-exp` | Background model |
| `--seed`     | 42           | Random seed for reproducibility |
| `--rebin`    | 2            | Rebin factor |
| `--n-sigma`  | 3.0          | Window half-width in sigma units |
| `--verbose`  | off          | Enable debug logging |

## Output

```
fitting_v2/outputs/<era>/toy_study/<signal_tag>/
  toys_<channel>.json               # Full results (all toys + summary)
  dist_n_sig_<channel>.pdf/png      # Distribution of fitted n_sig
  dist_mu_<channel>.pdf/png         # Distribution of fitted mu
  dist_n_sig_err_<channel>.pdf/png  # Distribution of n_sig uncertainty
  pulls_<channel>.pdf/png           # Pull distribution with Gaussian overlay
```

### JSON Structure

```json
{
  "metadata": {
    "era": "RunIII2024Summer24",
    "channel": "ee",
    "signal_tag": "WR2000_N1100",
    "model": "single-exp",
    "n_toys": 100,
    "seed": 42,
    "n_converged": 98,
    "n_failed": 2,
    "fit_window": [500.0, 3000.0],
    "n_sig_expected": 123.4
  },
  "summary_stats": {
    "n_sig":       { "mean", "median", "std", "q16", "q84", "q2_5", "q97_5", "min", "max" },
    "n_sig_error": { ... },
    "mu":          { ... },
    "pulls":       { ... }
  },
  "toys": [
    { "toy_index": 0, "fit_status": 0, "cov_quality": 3, "n_sig": ..., "n_sig_error": ..., ... },
    ...
  ]
}
```

## Interpreting Results

### Pull Distribution

The pull is defined as `n_sig_fitted / sigma(n_sig)` under the
background-only hypothesis (truth = 0). A well-calibrated fit produces:

- **Mean ~ 0**: no systematic bias in the signal extraction
- **Sigma ~ 1**: the error estimate is correctly calibrated

If sigma > 1, the errors are underestimated. If sigma < 1, they are
overestimated (conservative).

### Signal Strength Distribution

The `mu` distribution shows how much signal the fit would "find" in each
toy. Under background-only, this should scatter around 0. The width
tells you the expected statistical sensitivity:

- **68% interval** (q16 to q84): the typical range of mu values
- If |mu| < 2*sigma consistently, we cannot distinguish signal from background

### Convergence Failures

Some toys may fail to converge (status != 0 or covQual != 3),
especially for low-statistics channels or extreme Poisson fluctuations.
These are tracked separately and excluded from summary statistics. A
failure rate above ~5% may indicate the model is not robust.
