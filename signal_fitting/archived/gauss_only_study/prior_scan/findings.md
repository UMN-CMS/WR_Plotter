# Prior-scan findings (2026-05-20)

Setup: gauss PDF, both µ and σ priors always on. Centrals = bootstrap
(`mu_boot`, `FWHM_boot/2.3548`). Prior widths are the new function:

```
sigma_mu_prior    = alpha_mu    × FWHM_boot
sigma_sigma_prior = alpha_sigma × FWHM_boot / 2.3548   (gauss σ units)
```

Scan grid: 7 α_µ × 8 α_σ × 5 N × 50 toys × 8 bulk cells × 2 channels = 224,000
fits, completed in 4.5 min.

## Headline recommendation

**(α_µ, α_σ) = (0.10, 0.05) for both channels at N=20.** Single-pair, no
channel split.

Bulk-aggregated spread fitness `√((µ_spread-1)² + (σ_spread-1)²)`:

| channel | N=5 | N=10 | N=20 | N=50 | N=100 |
|---------|-----|------|------|------|-------|
| ee      | 0.44| 0.41 | 0.27 | 0.61 | 1.43  |
| mumu    | 0.40| 0.32 | 0.18 | 0.27 | 1.13  |

(Lower is better. Production target is N=20, so 0.27 / 0.18 there are the
relevant numbers.)

## Per-cell at the candidate (N=20, α_µ=0.10, α_σ=0.05)

```
channel  mass            µ_bias  µ_spr   σ_bias  σ_spr
ee       WR3000_N1000    -1.75   1.04    +1.48   1.20
ee       WR3000_N1400    -2.08   0.92    +1.03   0.87
ee       WR4000_N1200    -2.51   1.08    +2.33   1.47
ee       WR4000_N2000    -2.16   0.81    +0.98   0.86
ee       WR5000_N1400    -2.77   1.18    +2.58   1.51
ee       WR5000_N2400    -2.56   0.91    +1.70   1.09
ee       WR6000_N1800    -1.94   1.09    +2.46   1.46
ee       WR6000_N3000    -2.54   0.92    +2.04   1.25
mumu     WR3000_N1000    -1.22   0.95    +1.02   0.93
mumu     WR3000_N1400    -1.32   0.86    +0.72   0.73
mumu     WR4000_N1200    -1.43   0.92    +1.00   0.85
mumu     WR4000_N2000    -1.76   0.92    +0.96   0.82
mumu     WR5000_N1400    -0.76   1.08    +1.55   1.08
mumu     WR5000_N2400    -2.37   0.95    +1.25   0.88
mumu     WR6000_N1800    -0.99   1.01    +1.26   0.96
mumu     WR6000_N3000    -1.81   0.97    +1.36   0.97
```

* **µ-pull spread is well-calibrated**: ee 0.81–1.18, mumu 0.86–1.08 across
  the 8 bulk cells. Spread fitness ~1 on µ.
* **σ-pull spread is well-calibrated in mumu** (0.73–1.08) but **under-covers
  in 3 ee cells**: WR4000_N1200 (1.47), WR5000_N1400 (1.51), WR6000_N1800
  (1.46). These are the high-rise-asymmetry cells; σ_fit scatter exceeds
  σ_err by ~50% in those cells.
* Bias remains structural (µ: −1 to −3σ depending on cell; σ: +1 to +3σ).

## Sensitivity to (α_µ, α_σ) choice at N=20

| pair             | ee bulk-median spread fitness | mumu bulk-median |
|------------------|-------------------------------|------------------|
| (0.10, 0.05)     | 0.26                          | 0.13             |
| (0.10, 0.07)     | 0.45                          | 0.10             |
| (0.15, 0.07)     | 0.50                          | 0.16             |
| (0.20, 0.10)     | 0.61                          | 0.25             |

(0.10, 0.05) is the joint optimum. (0.10, 0.07) is acceptable and gives
slightly looser σ-spread that may help with the 3 outlier ee cells (worth
checking if you care).

## N-dependence

The same (α_µ, α_σ) does *not* stay calibrated across N ∈ {5, 10, 20, 50,
100}:

| N    | best ee (α_µ, α_σ) | ee fitness | best mumu (α_µ, α_σ) | mumu fitness |
|------|--------------------|------------|----------------------|--------------|
| 5    | (0.50, 0.50)       | 0.01       | (0.50, 0.50)         | 0.23         |
| 10   | (0.20, 0.10)       | 0.02       | (0.50, 0.30)         | 0.10         |
| 20   | (0.10, 0.03)       | 0.20       | (0.10, 0.07)         | 0.09         |
| 50   | (1.00, 0.50)       | 0.31       | (0.15, 0.50)         | 0.04         |
| 100  | (0.15, 0.50)       | 0.60       | (0.05, 0.30)         | 0.09         |

The optimal `α_µ` is roughly stable at 0.1–0.5; the optimal `α_σ` swings
dramatically (0.03 at N=20 → 0.50 at N=50, ee). This reflects that at high
N, the σ-bias is so large that no amount of prior tuning calibrates the
σ-spread to 1. **The N=20 calibration should be regarded as the production
choice; N≠20 calibration is informational.**

## Next steps

1. **Decide on the prior pair.** Default recommendation: (α_µ=0.10, α_σ=0.05).
2. **(Optional) Confirm the 3 ee outlier cells.** Could do a finer scan
   around (0.10, 0.05–0.10) at just those 3 cells, but the σ-spread blow-up
   looks structural (peak-shape-dependent), not prior-tuning.
3. **Run `full_grid/` at the chosen widths** on all 369 masses × 2 channels.
   Target ~4 hours of compute (220k fits at 850 fits/s).
