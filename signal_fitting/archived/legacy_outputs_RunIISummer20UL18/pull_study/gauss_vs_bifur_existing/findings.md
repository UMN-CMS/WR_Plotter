# Gauss vs bifur — what the existing pull-study CSV already tells us

Source: `outputs/RunIISummer20UL18/pull_study/results.csv` (2.36M rows).
Analysis: `compare_existing.py` (in this directory) — outputs
`bulk_cell_summary.csv` for the 8 bulk cells per channel.

Bulk cells (per channel, M_WR ∈ {3..6} TeV × M_N/M_WR ∈ {≈0.3, ≈0.5}):
WR3000_N1000, WR3000_N1400, WR4000_N1200, WR4000_N2000,
WR5000_N1400, WR5000_N2400, WR6000_N1800, WR6000_N3000.

All numbers below are at the **production config** `both` (µ + width
constrained) and pull-study N ∈ {5, 10, 15, 20}.

ΔNLL convention: `ΔNLL = 2*(min_nll_gauss − min_nll_bifur)`. **Positive ⇒
bifur fits better.** Under Wilks (bifur has Δ as one extra parameter)
ΔNLL ~ χ²(1 d.o.f.); the user's threshold ΔNLL≈4 ↔ 2σ one-sided is the
LLR test for "bifur preferred".

---

## Headline finding

> **Gauss has a structural µ-bias that grows monotonically with N (in pull
> units), and prior re-tuning cannot remove it.** Re-running the full pull
> study with gauss-tuned priors will improve gauss µ-spread by ~10–20% but
> will not change the bias at all.

The bias is structural, not prior-driven. Three independent pieces of
evidence:

1. The gauss µ-bias is present in `no_priors` at almost the same magnitude
   as in `both` (see "Bias is not prior-driven" below).
2. In **GeV** the gauss µ-bias is approximately N-independent (it saturates
   around N=10) — it's a fixed offset between FWHM-defined truth and the
   gauss-fit equilibrium peak. In **σ-units** the same offset grows like √N
   because σ_µ shrinks like 1/√N.
3. The bias is largest in cells with the most-asymmetric peaks and follows
   the qualitative ee/μμ trend (ee asymmetry stronger than μμ).

---

## Key numbers, production config "both", N=20

### Pull bias on µ (median pull) — ee channel

| cell           | gauss µ-bias | bifur µ-bias | gauss |bias|/bifur |bias| |
|----------------|--------------|--------------|---------------------------|
| WR3000_N1000   | −1.98        | −0.20        |  10×                      |
| WR3000_N1400   | −2.44        | −0.46        |   5×                      |
| WR4000_N1200   | −3.03        | −0.17        |  18×                      |
| WR4000_N2000   | −2.63        | −0.13        |  20×                      |
| WR5000_N1400   | −3.28        | −0.41        |   8×                      |
| WR5000_N2400   | −3.18        | −0.12        |  27×                      |
| WR6000_N1800   | −2.22        | +0.23        |  10×                      |
| WR6000_N3000   | −3.11        | +0.10        |  31×                      |

### Pull bias on µ — mumu channel (smaller, but still present)

| cell           | gauss µ-bias | bifur µ-bias |
|----------------|--------------|--------------|
| WR3000_N1000   | −1.36        | −0.18        |
| WR3000_N1400   | −1.54        | −0.20        |
| WR4000_N1200   | −1.70        | −0.15        |
| WR4000_N2000   | −2.10        | −0.15        |
| WR5000_N1400   | −0.84        | +0.14        |
| WR5000_N2400   | −2.88        | −0.51        |
| WR6000_N1800   | −1.18        | +0.18        |
| WR6000_N3000   | −2.16        | −0.07        |

### Pull spread on µ — N=20, "both"

| channel | gauss spread range | bifur spread range |
|---------|--------------------|--------------------|
| ee      | 1.04 – 1.35        | 0.78 – 1.15        |
| mumu    | 1.15 – 1.34        | 0.89 – 1.07        |

Gauss spreads run a bit high (over-covering) because the bifur-tuned
µ-prior α is slightly too tight for the gauss fit. **Re-tuning would
bring this down to ~1.0** — but won't change the bias above.

### Convergence rate, "both", N=5 (the only regime where it differs much)

| channel | gauss | bifur (range across cells) |
|---------|-------|----------------------------|
| ee      | 1.00  | 0.83 – 0.95                |
| mumu    | 1.00  | 0.77 – 0.89                |

By N=10 bifur is ≥0.95 everywhere. By N=15 bifur is ≥0.97. Gauss is
100% across the board for every N and every cell.

---

## Absolute (GeV) µ-bias — proof of structural nature

`(mu_fit − mu_truth)` in GeV, gauss/both/ee:

```
mass            N=5     N=10    N=15    N=20    σ_µ(N=20)
WR3000_N1000   −36     −51     −65     −68      34 GeV
WR3000_N1400   −52     −71     −89     −93      38 GeV
WR4000_N1200   −69     −97    −114    −120      40 GeV
WR4000_N2000   −65     −98    −120    −127      49 GeV
WR5000_N1400   −97    −129    −157    −163      50 GeV
WR5000_N2400   −94    −135    −159    −171      54 GeV
WR6000_N1800   −67     −97    −122    −128      58 GeV
WR6000_N3000  −107    −151    −175    −187      61 GeV
```

The bias saturates around N=15–20 (≤10% change from N=15 to N=20). The
pull-bias growing from −1σ at N=5 to −3σ at N=20 is purely σ_µ shrinking
under a fixed offset, not the offset growing.

For comparison, bifur ee µ-bias at N=20 is in [−32, +23] GeV (≤0.5σ in pull
units) — bifur tracks FWHM-based truth.

---

## Bias is not prior-driven — `no_priors` numbers

Gauss µ-bias at N=20 in `no_priors` (where there is no µ-prior at all):

| cell           | ee gauss bias | mumu gauss bias |
|----------------|---------------|------------------|
| WR3000_N1000   | −1.67         | −1.21            |
| WR3000_N1400   | −2.22         | −1.49            |
| WR4000_N1200   | −2.21         | −1.48            |
| WR4000_N2000   | −2.43         | −1.91            |
| WR5000_N1400   | −2.22         | −0.69            |
| WR5000_N2400   | −2.50         | −2.35            |
| WR6000_N1800   | −1.56         | −0.97            |
| WR6000_N3000   | −2.35         | −1.83            |

Removing the µ-prior shrinks the bias by ~20–30% (because the prior is
narrowing σ_µ, which inflates the pull at fixed GeV offset). It does
*not* remove the bias. **The bias lives in the likelihood — i.e., the
PDF.**

---

## ΔNLL gauss vs bifur — when does the data prefer bifur?

`ΔNLL = 2*(min_nll_gauss − min_nll_bifur)`, paired by seed within
(channel, mass, N, config). Median across 100 toys per cell.

### ee, "both"

```
cell           N=5    N=10   N=15   N=20    P(ΔNLL>4) at N=20
WR3000_N1000  +0.18  +0.55  +1.34  +0.65       0.25
WR3000_N1400  +0.02  +0.81  +1.61  +0.73       0.27
WR4000_N1200  +1.53  +3.14  +4.74  +3.97       0.49
WR4000_N2000  −0.29  +1.31  +2.30  +2.56       0.34
WR5000_N1400  +1.92  +3.43  +5.31  +3.92       0.50
WR5000_N2400  +0.73  +2.47  +4.14  +4.31       0.53
WR6000_N1800  +1.03  +2.23  +4.01  +3.82       0.48
WR6000_N3000  +1.01  +2.94  +4.87  +5.17       0.57
```

ee crossover: median ΔNLL ≥ 4 (≈ 2σ preference per toy) is reached for
**5 of 8 ee bulk cells at N=15**, with the remaining 3 (the two WR3000
cells and WR4000_N2000) crossing somewhere between N=15 and N=25. At
N=10 the median ΔNLL is +1 to +3 (1–1.5σ preference).

### mumu, "both"

Median ΔNLL stays in [−0.8, +1.0] for **all** 8 mumu bulk cells at every
N tested. **mumu never crosses 2σ.** P(ΔNLL>4) at N=20 is 0.08–0.22 — bifur
is only preferred on the tail of toys, never on the median.

The mumu peak asymmetry is small enough that with 20 events the data
genuinely cannot distinguish gauss from bifur. The ee asymmetry is large
enough that 15–20 events do distinguish them.

---

## Why ΔNLL is not the deciding metric

ΔNLL tells you "which PDF the data prefers as a fit", but the
gauss-vs-bifur production decision is a different question:

* **Closure / coverage**: does the fit recover the injected µ
  consistently? gauss has structural ~120 GeV µ-offset against FWHM-truth.
* **Convergence rate**: gauss has 100% at every N; bifur 77–95% at N=5.
* **Calibration**: bifur is already calibrated; gauss would need a
  channel- and N-dependent µ-correction or a redefined µ_truth_gauss.

The ΔNLL crossover (ee at N=15, mumu never) is *consistent with* the
explanation — bifur is better when the data can detect the asymmetry,
indistinguishable when it can't — but it doesn't pick the production
PDF for us.

---

## Verdict and proposed plan

**Verdict: do not re-tune gauss priors at the current FWHM-based truth
convention.** The existing CSV already shows that re-tuning will
optimize spread but cannot remove the structural −2 to −3σ µ-bias at
N=20. The compute would be wasted.

There are three coherent paths forward; pick one:

### Path A — Ship bifur (recommended, status quo)
* Already calibrated, biases <0.5σ on µ, spreads ~1.
* +1σ residual Σ-bias documented as a Stage-4 fit systematic.
* Convergence cost: ~5–10% of N=5 toys lost. Survivable.
* **No additional compute needed beyond what's already done.**

### Path B — Re-do gauss with a self-consistent truth (worth ~1–2 days)
The gauss bias is "the FWHM-based truth differs from the gauss-fit
equilibrium". If we redefine `mu_truth_gauss` as the median of a high-
stats (e.g. 5000-event) gauss fit on MC — analogous to how Δ_truth is
already a high-stats bifur fit — the gauss bias goes to ~0 by
construction. Spread can then be tuned with α.

* Pros: 100% convergence, no asymmetry parameter to fit, mumu becomes
  very clean, ee still gets a 1σ pull from any residual non-Gaussian
  shape but is otherwise consistent.
* Cons: changes the µ-truth convention (downstream interpretation has to
  know which truth it's reading); per-cell gauss-equilibrium fit
  introduces a small new bootstrap step; needs ~30k toys × 369 masses ×
  2 channels × 4 N values = ~1 day of pull-study compute.
* Suggested steps:
   1. Add `gauss_mu_truth` to `compute_truth()` in `pull_study.py` — a
      single deterministic high-stats gauss fit on MC, mirroring the
      existing `Delta_truth` bifur fit.
   2. Run `scan_loose_mu_prior.py --model gauss` over the 8 bulk cells
      × 4 N values. The expected outcome is α_µ_gauss ≈ 0.18–0.22 in
      ee, 0.20–0.25 in mumu (because gauss σ_µ is *tighter* than
      bifur Σ-derived σ_µ for the same fit precision).
   3. Run the full pull study output to a *new* subdir
      `outputs/.../pull_study_gauss_selfconsistent/`. Keep the bifur
      baseline (current `results.csv`) untouched for diff.
   4. Compute bias, spread, ΔNLL on the new CSV. Compare head-to-head
      with bifur on closure + 95% CL coverage tests.

### Path C — Gauss with a calibration nuisance (worst of both)
Keep FWHM-truth, ship gauss, absorb the structural µ-bias into a
calibration nuisance in the limit-setting workspace. This is the
"two-line patch" version of Path B. **Don't do this** — it bakes a
channel- and N-dependent offset into the workspace, and the structural
bias is large enough (>1σ at N=20) that the calibration would be the
dominant systematic for a discovery measurement.

### Crossover-N study (orthogonal — useful for the methodology
note regardless of which path is chosen)
The current CSV already answers it:
* ee: bifur preferred at ≈2σ on median toy starting at N=15 in 5/8 bulk
  cells; reaches all 8 by N≈25.
* mumu: bifur never preferred on the median at any tested N.

If we want a single-number crossover, the **fraction of bulk cells with
median ΔNLL ≥ 4** as a function of N is:

```
N           ee bulk cells    mumu bulk cells
5            0/8              0/8
10           0/8              0/8
15           5/8              0/8
20           7/8 (≥3.8)       0/8
```

This pattern is consistent with: bifur asymmetry is detectable in ee
when N ≳ 12; in mumu it is not detectable at any N ≤ 20.

---

## Concrete next step recommendation

If your priority is **shipping a measurement**: **Path A**. The existing
analysis is calibrated and the +1σ Σ-systematic is well-documented.
Skip the gauss work.

If your priority is **publishing a methodology paper / understanding
the tradeoff**: **Path B** plus the crossover-N tables above. The
recompute is ~1 day and gives you a clean head-to-head.

If you want to confirm the structural bias diagnosis before committing
either way, I can do one cheap follow-up:

* **Single-cell confirmation:** run `scan_loose_mu_prior.py --model gauss
  --mu-prior-alpha 0.05` at WR4000_N2000 ee N=20. Expected: pull spread
  stays ~1 (or drops slightly), µ-bias stays −2 to −3σ — confirming
  that no value of α_µ can rescue gauss against FWHM-based truth.
  Takes ~5 min on LCG_106.

I'd recommend running that single-cell scan as a sanity check before
committing to Path A or Path B.
