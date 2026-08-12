# Simultaneous Double Exponential Fit: How It Works

Step-by-step explanation of the simultaneous SR + flavor CR double exponential background fit.

## 1. Why not just fit two exponentials in the SR?

The single-exp guide established that the m_lljj background is a mixture of two exponentials: a steep tt+tW component (c ~ -0.0032) and a shallow DYJets component (c ~ -0.0024). The natural next step is fitting the SR with two exponentials:

```
f_SR(m) = N1 * exp(c1 * m) + N2 * exp(c2 * m)
```

This has 4 free parameters: two slopes (c1, c2) and two yields (N1, N2). But in practice, this fit is **degenerate** in a single region. The fitter can trade N1 for N2 almost freely — making the steep component larger while shrinking the shallow one, and vice versa — without changing the total curve much. This shows up as corr(N1, N2) ~ -1.0 in the correlation matrix, meaning the two yields are almost perfectly anti-correlated. The slopes also get pulled around because they're entangled with the yields. The fit "converges" in the sense that Minuit finds a minimum, but the parameter values are poorly determined and unstable.

## 2. The flavor control region

The flavor control region (FCR) contains emu + mue events — one electron and one muon, instead of two same-flavor leptons. This region is special because of **flavor symmetry**.

### What is flavor symmetry?

Flavor-symmetric backgrounds are processes where the two leptons are produced independently, so the probability of getting ee, mumu, emu, or mue are all equal. The dominant example is tt production, where each top quark decays to a W boson that can produce any lepton flavor independently. For these processes:

```
N(emu) = N(mue) = N(ee) = N(mumu)
```

So the combined emu+mue yield is exactly twice the same-flavor yield:

```
N(emu + mue) = 2 * N(ee) = 2 * N(mumu)
```

### What is NOT flavor-symmetric?

DYJets (Drell-Yan + jets) produces a Z/gamma* that decays to a same-flavor pair: ee or mumu, but never emu or mue. So DYJets contributes to the SR but **not** to the flavor CR.

### The key insight

The FCR isolates the flavor-symmetric component. If we assume the SR background is:

```
SR = [flavor-symmetric (tt+tW)] + [DYJets]
```

then the FCR contains only the flavor-symmetric piece, scaled by 2:

```
FCR = 2 * [flavor-symmetric (tt+tW)]
```

This gives us an independent handle on the steep component's shape and yield.

## 3. The simultaneous model

### SR model

Same double exponential as before — 4 free parameters:

```
f_SR(m) = N1 * exp(c1 * m) + N2 * exp(c2 * m)
```

- c1: steep slope (flavor-symmetric / tt+tW-like)
- c2: shallow slope (DYJets-like)
- N1: yield of the steep component in the SR
- N2: yield of the shallow component in the SR

### FCR model

A single exponential with **shared** parameters:

```
f_FCR(m) = 2 * N1 * exp(c1 * m)
```

- c1: **the same c1 as in the SR** — not a separate parameter
- N1: **the same N1 as in the SR** — scaled by 2

The factor of 2 accounts for emu + mue = 2 * ee (or 2 * mumu).

### Total free parameters: 4

Even though we're fitting two regions, the model has only 4 free parameters: c1, c2, N1, N2. The FCR model is entirely determined by the SR parameters c1 and N1. No new degrees of freedom are introduced — just new data to constrain existing ones.

## 4. The simultaneous likelihood

RooFit's `RooSimultaneous` constructs a combined likelihood by multiplying the likelihoods from each region:

```
L_total = L_SR * L_FCR
```

or equivalently, adding the negative log-likelihoods:

```
-ln(L_total) = -ln(L_SR) + -ln(L_FCR)
```

Each region contributes a standard binned extended likelihood (same as the single-exp case):

```
-ln(L_SR) = sum_i [ mu_i^SR - n_i^SR * ln(mu_i^SR) ]
-ln(L_FCR) = sum_j [ mu_j^FCR - n_j^FCR * ln(mu_j^FCR) ]
```

Minuit minimizes the **combined** NLL by adjusting all 4 parameters simultaneously. There is no sequential "fit the FCR first, then lock c1 and N1, then fit the SR" step. On each iteration, Minuit:

1. Picks trial values for all 4 parameters (c1, c2, N1, N2)
2. Computes the SR prediction using all 4: `N1*exp(c1*m) + N2*exp(c2*m)`
3. Computes the FCR prediction using 2 of them: `2*N1*exp(c1*m)`
4. Evaluates `-ln(L_SR) + -ln(L_FCR)` — a single number
5. Adjusts all 4 parameters to reduce that combined number
6. Repeats until convergence

Both regions pull the parameters simultaneously. If Minuit tries a c1 that fits the SR well but describes the FCR poorly, the total likelihood gets worse, so it backs off. The final values are a compromise that best satisfies both regions at the same time.

## 5. How the degeneracy breaks

In the single-region double-exp fit, the fitter can freely trade N1 and N2 because only their sum is well-constrained by the data. Adding the FCR changes this:

- The FCR data constrains `2 * N1 * exp(c1 * m)` **independently** of N2 and c2.
- This anchors both c1 (the steep slope) and N1 (the steep yield) from a dataset where there's no DYJets to confuse things.
- With c1 and N1 anchored by the FCR, the SR data can cleanly determine c2 and N2 from the residual (the part of the SR that the steep component doesn't explain).

The correlation between N1 and N2 drops from ~-1.0 (completely degenerate) to ~-0.5 (moderate anti-correlation, which is expected since the total SR yield N1+N2 is still somewhat constrained).

## 6. Parameter initialization

The starting values come from the single-exp component fits (see the single-exp guide, Section 9):

| Parameter | Initial value | Range | Motivated by |
|-----------|---------------|-------|--------------|
| c1 | -0.0032 | [-0.02, 0] | tt+tW component slope |
| c2 | -0.0024 | [-0.02, 0] | DYJets component slope |
| N1 | 0.7 * N_SR | [0, 50000] | tt+tW is ~70% of SR |
| N2 | 0.3 * N_SR | [0, 50000] | DYJets is ~30% of SR |

These are reasonable starting points. The simultaneous fit adjusts them from there.

## 7. Goodness of fit

Chi2/ndf is computed **separately** for each region after the simultaneous fit:

- **SR chi2/ndf**: computed from the SR data vs the SR model projection, with ndf = n_bins - 4 (all 4 parameters affect the SR).
- **FCR chi2/ndf**: computed from the FCR data vs the FCR model projection, with ndf = n_bins - 2 (only c1 and N1 affect the FCR).

A good fit should give chi2/ndf ~ 1 in both regions. Large values mean the model doesn't describe that region well.

## 8. Current results and the FCR model limitation

The simultaneous fit converges cleanly (status=0, covQual=3) and the degeneracy is broken (corr(N1,N2) ~ -0.5). But the chi2/ndf values are poor:

| Channel | SR chi2/ndf | FCR chi2/ndf |
|---------|-------------|--------------|
| ee | 22.1 | 8.3 |
| mumu | 4.3 | 8.4 |

### What's actually in the flavor CR?

Running `--decompose-fcr` breaks down the FCR by process:

| Component | Events | Fraction |
|-----------|--------|----------|
| tt+tW | 2617.6 | 95.9% |
| Other | 66.6 | 2.4% |
| Nonprompt | 40.2 | 1.5% |
| DYJets | 5.6 | 0.2% |

The FCR is 96% tt+tW, with virtually no DYJets. This confirms the flavor symmetry argument: Drell-Yan does not leak into the emu+mue region. The small non-tt+tW contamination (~4%) comes from Other and Nonprompt processes, not from DYJets.

### Why the FCR chi2 is still bad

The FCR model is a single exponential:

```
f_FCR(m) = 2 * N1 * exp(c1 * m)
```

But tt+tW itself is not perfectly described by a single exponential. The single-exp component fit of tt+tW alone gives chi2/ndf = 3.59 (see single-exp guide, Section 9). This is because tt+tW combines top pair production and single top (tW), which have slightly different kinematics. A single slope cannot capture this internal structure.

In the simultaneous fit, this mismodelling of the FCR forces c1 and N1 to compromise values that don't perfectly describe either region. The FCR chi2/ndf of ~8 is worse than the standalone tt+tW chi2/ndf of ~3.6 because the simultaneous constraint prevents c1 from freely adjusting to best fit the FCR alone.

### Why the SR chi2 is bad

The SR model is correct in form (two exponentials), but the shared parameters c1 and N1 are distorted by the FCR's pull. The simultaneous fit gives N1 ~ 1370, compared to the true tt+tW SR yield of ~1180 from component fits. The FCR forces `2*N1 ~ 2730` to match its total yield, but the FCR total (2730 events) includes ~110 non-tt+tW events. Since the model has no way to account for this, it absorbs the excess into N1, inflating it by ~16%.

This inflated N1 in the SR leaves too little room for N2 (the shallow/DYJets component), distorting the overall shape and producing poor chi2/ndf.

### Bottom line

The simultaneous approach successfully breaks the N1-N2 degeneracy, validating the strategy. The remaining problems are:

1. **FCR purity**: the ~4% non-tt+tW content gets absorbed into N1 since the model has no way to separate it.
2. **tt+tW internal structure**: a single exponential doesn't fully describe tt+tW, which is a mixture of top pair and single top processes.
